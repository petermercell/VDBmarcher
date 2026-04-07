// ═══════════════════════════════════════════════════════════════════════════
// VDBRenderCUDA.cpp — CUDA frame rendering dispatch (CVDB only)
// [Phase 0] Refactored: buildRenderConfig → single config struct to renderer
// ═══════════════════════════════════════════════════════════════════════════

#include "VDBRenderIop.h"
#include "VDBCUDARenderer.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdio>

using namespace DD::Image;

// ─── Build unified render config from all Iop knobs ─────────────────────────

VDBRenderConfig VDBRenderIop::buildRenderConfig(int fmtW, int fmtH) const
{
    VDBRenderConfig cfg{};

    const Box& bbox = info_.box();
    int bx1 = bbox.x(), by1 = bbox.y();
    int bboxW = bbox.r() - bx1, bboxH = bbox.t() - by1;

    double scale;
    switch (_gpuRenderScale) {
        case 0:  scale = 0.25; break;
        case 1:  scale = 0.50; break;
        case 2:  scale = 0.75; break;
        default: scale = 1.0;  break;
    }

    cfg.renderW     = std::max(1, int(bboxW * scale));
    cfg.renderH     = std::max(1, int(bboxH * scale));
    cfg.fmtW        = std::max(1, int(fmtW * scale));
    cfg.fmtH        = std::max(1, int(fmtH * scale));
    cfg.bboxOffsetX = int(bx1 * scale);
    cfg.bboxOffsetY = int(by1 * scale);

    for (int i = 0; i < 3; ++i) cfg.camOrigin[i] = float(_camOrigin[i]);
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r)
            cfg.camRot[c * 3 + r] = float(_camRot[c][r]);
    cfg.halfW = float(_halfW);
    cfg.halfH = float(_halfW * double(fmtH) / double(fmtW));

    buildVolumeTransform(cfg.volumeInvTransform);

    // Forward transform (local → world) for depth+P AOV
    {
        const double deg2rad = 3.14159265358979323846 / 180.0;
        const double rx=_volRotate[0]*deg2rad, ry=_volRotate[1]*deg2rad, rz=_volRotate[2]*deg2rad;
        const double sx=_volScale[0]*_volUniformScale, sy=_volScale[1]*_volUniformScale, sz=_volScale[2]*_volUniformScale;
        const double crx=std::cos(rx),srx=std::sin(rx),cry=std::cos(ry),sry=std::sin(ry),crz=std::cos(rz),srz=std::sin(rz);
        double R[3][3];
        R[0][0]=cry*crz; R[0][1]=srx*sry*crz-crx*srz; R[0][2]=crx*sry*crz+srx*srz;
        R[1][0]=cry*srz; R[1][1]=srx*sry*srz+crx*crz; R[1][2]=crx*sry*srz-srx*crz;
        R[2][0]=-sry;    R[2][1]=srx*cry;              R[2][2]=crx*cry;
        float* f = cfg.volumeFwdTransform;
        f[0]=float(R[0][0]*sx);f[1]=float(R[1][0]*sx);f[2]=float(R[2][0]*sx);f[3]=0;
        f[4]=float(R[0][1]*sy);f[5]=float(R[1][1]*sy);f[6]=float(R[2][1]*sy);f[7]=0;
        f[8]=float(R[0][2]*sz);f[9]=float(R[1][2]*sz);f[10]=float(R[2][2]*sz);f[11]=0;
        f[12]=float(_volTranslate[0]);f[13]=float(_volTranslate[1]);f[14]=float(_volTranslate[2]);f[15]=1;
    }

    cfg.stepSize       = float(_stepSize);
    cfg.extinction     = float(_extinction);
    cfg.scattering     = float(_scattering);
    cfg.boundaryBlend  = float(_boundaryBlend);
    cfg.extinctionR    = float(_extinctionR);
    cfg.extinctionG    = float(_extinctionG);
    cfg.extinctionB    = float(_extinctionB);

    cfg.shadowSteps         = _gpuShadowSteps;
    cfg.shadowStepScale     = float(_gpuShadowStepScale);

    cfg.phaseG1  = float(_phaseG1);
    cfg.phaseG2  = float(_phaseG2);
    cfg.phaseMix = float(_phaseMix);
    cfg.phaseMode = _phaseMode;

    cfg.powderStrength = _powderEnable ? float(_powderStrength) : 0.0f;

    cfg.msEnable        = _msEnable ? 1 : 0;
    cfg.msOctaves       = _msOctaves;
    cfg.msExtFalloff    = float(_msExtFalloff);
    cfg.msScatterFalloff = float(_msScatterFalloff);

    cfg.gradientMix       = float(_gradientMix);
    cfg.gradientThreshold = float(_gradientThreshold);
    for (int i = 0; i < 3; ++i) cfg.gradNormalBias[i] = float(_gradNormalBias[i]);
    cfg.gradientSmooth    = float(_gradientSmooth);
    cfg.scatterSmooth     = float(_scatterSmooth);
    cfg.densityBlur       = float(_densityBlur);

    cfg.noiseAmplitude = _noiseEnable ? float(_noiseAmplitude) : 0.0f;
    cfg.noiseFrequency = float(_noiseFrequency);
    for (int i = 0; i < 3; ++i) cfg.noiseScale[i] = float(_noiseScale[i]);
    cfg.noiseOctaves = _noiseOctaves;
    for (int i = 0; i < 3; ++i) cfg.noiseOffset[i] = float(_noiseOffset[i]);

    cfg.emissionScale    = float(_emissionScale);
    cfg.temperatureScale = float(_temperatureScale);
    cfg.emSelfIllum      = float(_emSelfIllum);
    cfg.emSelfIllumPhase = _emSelfIllumPhase ? 1 : 0;

    cfg.useBlackbody  = _useBlackbody ? 1 : 0;
    cfg.bbKelvinScale = float(_bbKelvinScale);
    cfg.bbMix         = float(_bbMix);
    for (int i = 0; i < 3; ++i) cfg.bbTint[i] = float(_bbTint[i]);

    for (int i = 0; i < 3; ++i) {
        cfg.rampColors[i]      = float(_rampColor0[i]);
        cfg.rampColors[3 + i]  = float(_rampColor1[i]);
        cfg.rampColors[6 + i]  = float(_rampColor2[i]);
        cfg.rampColors[9 + i]  = float(_rampColor3[i]);
        cfg.rampColors[12 + i] = float(_rampColor4[i]);
    }

    for (int i = 0; i < 3; ++i) cfg.ambientColor[i] = float(_ambientColor[i]);
    cfg.ambientIntensity = float(_ambientIntensity);

    // ── Environment SH ──
    cfg.envSHEnable = (_envEnable && _activeEnvSH.valid && _envIntensity > 1e-6) ? 1 : 0;
    cfg.envIntensity = float(_envIntensity);
    cfg.envShadowEnable = _envShadowEnable ? 1 : 0;
    cfg.envShadowLift[0] = float(_envShadowLift[0]);
    cfg.envShadowLift[1] = float(_envShadowLift[1]);
    cfg.envShadowLift[2] = float(_envShadowLift[2]);
    if (cfg.envSHEnable) {
        std::memcpy(cfg.envSH, _activeEnvSH.coeffs, sizeof(cfg.envSH));
        std::memcpy(cfg.envPeakDir, _activeEnvSH.peakDirection, sizeof(cfg.envPeakDir));
        std::memcpy(cfg.envPeakColor, _activeEnvSH.peakColor, sizeof(cfg.envPeakColor));
    } else {
        std::memset(cfg.envSH, 0, sizeof(cfg.envSH));
        std::memset(cfg.envPeakDir, 0, sizeof(cfg.envPeakDir));
        std::memset(cfg.envPeakColor, 0, sizeof(cfg.envPeakColor));
    }

    cfg.numLights = _numGPULights;
    for (int li = 0; li < _numGPULights; ++li)
        cfg.lights[li] = _gpuLights[li];

    cfg.aovFlags = (_createScatterAOV     ? 1  : 0)
                 | (_createEmissionAOV    ? 2  : 0)
                 | (_createTemperatureAOV ? 4  : 0)
                 | ((_createLight1AOV||_light1Enable) ? 8  : 0)
                 | ((_createLight2AOV||_light2Enable) ? 16 : 0)
                 | ((_createLight3AOV||_light3Enable) ? 32 : 0)
                 | (_createAlbedoAOV      ? 64 : 0)
                 | (_createDepthAOV        ? 128 : 0)
                 | (_createPositionAOV     ? 256 : 0)
                 | (_createPrefAOV         ? 512 : 0)
                 | (_createNormalAOV       ? 1024 : 0)
                 | (_createEnvAOV          ? 2048 : 0);
    cfg.outputStride = 4 + __builtin_popcount(cfg.aovFlags) * 4;
    cfg.profileEnable = _profileEnable ? 1 : 0;

    for (int i = 0; i < 3; ++i) cfg.lightMix[i] = float(_lightMix[i]);
    cfg.envMix = float(_envMix);

    return cfg;
}

// ─── Synchronous GPU render ─────────────────────────────────────────────────

bool VDBRenderIop::renderCUDAFrame(int fmtW, int fmtH)
{
    if (!_cudaRenderer || !_camValid || !_gridValid) return false;

    const Box& bbox = info_.box();
    int bboxW = bbox.r() - bbox.x(), bboxH = bbox.t() - bbox.y();
    if (bboxW <= 0 || bboxH <= 0) return false;

    VDBRenderConfig cfg = buildRenderConfig(fmtW, fmtH);
    _gpuOutputBuffer.resize(size_t(cfg.renderW) * cfg.renderH * cfg.outputStride);

    bool ok = _cudaRenderer->render(cfg, _gpuOutputBuffer.data());

    if (ok) {
        _gpuFrameReady     = true;
        _gpuRenderedWidth  = cfg.renderW;
        _gpuRenderedHeight = cfg.renderH;
        _gpuOutputStride   = cfg.outputStride;
        _screenBboxX1      = bbox.x();
        _screenBboxY1      = bbox.y();
    } else {
        std::printf("[VDBRender] Render failed: %s\n", _cudaRenderer->lastError().c_str());
        _gpuFrameReady = false;
    }
    return ok;
}

// ─── Asynchronous GPU render ────────────────────────────────────────────────

bool VDBRenderIop::renderCUDAFrameAsync(int fmtW, int fmtH)
{
    if (!_cudaRenderer || !_camValid || !_gridValid) return false;

    const Box& bbox = info_.box();
    int bboxW = bbox.r() - bbox.x(), bboxH = bbox.t() - bbox.y();
    if (bboxW <= 0 || bboxH <= 0) return false;

    VDBRenderConfig cfg = buildRenderConfig(fmtW, fmtH);

    bool ok = _cudaRenderer->renderAsync(cfg);
    if (ok) {
        _asyncRenderLaunched = true;
        _asyncBboxX1 = bbox.x();
        _asyncBboxY1 = bbox.y();
    } else {
        _asyncRenderLaunched = false;
    }
    return ok;
}

// ─── Poll async completion ──────────────────────────────────────────────────

bool VDBRenderIop::pollAsyncFrame()
{
    if (!_asyncRenderLaunched || !_cudaRenderer) return false;
    if (!_cudaRenderer->isRenderComplete()) return false;

    int w = _cudaRenderer->asyncWidth();
    int h = _cudaRenderer->asyncHeight();
    int s = _cudaRenderer->asyncStride();
    _gpuOutputBuffer.resize(size_t(w) * h * s);

    if (_cudaRenderer->readbackResult(_gpuOutputBuffer.data(), w * h, s)) {
        _gpuFrameReady     = true;
        _gpuRenderedWidth  = w;
        _gpuRenderedHeight = h;
        _gpuOutputStride   = s;
        _screenBboxX1      = _asyncBboxX1;
        _screenBboxY1      = _asyncBboxY1;
    }
    _asyncRenderLaunched = false;
    return _gpuFrameReady;
}
