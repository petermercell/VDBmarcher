// ═══════════════════════════════════════════════════════════════════════════
// VDBValidate.cpp — Validation, camera caching, screen bbox, CUDA dispatch
// ═══════════════════════════════════════════════════════════════════════════

#include "VDBRenderIop.h"
#include "VDBCUDARenderer.h"
#include "VDBEnvironment.h"
#include <DDImage/Format.h>
#include <cmath>
#include <algorithm>
#include <cstdio>

using namespace DD::Image;

void VDBRenderIop::_validate(bool for_real)
{

    const Format* fullFmt = _formats.fullSizeFormat();
    const Format* proxyFmt = _formats.format();
    if (!fullFmt) fullFmt = proxyFmt;
    if (fullFmt) {
        info_.format(*fullFmt); info_.full_size_format(*fullFmt); info_.set(*fullFmt);
    }

    // Register output channels
    ChannelSet outChannels(Mask_RGBA);
    if (_createScatterAOV) {
        outChannels += chan_scatter_r; outChannels += chan_scatter_g;
        outChannels += chan_scatter_b; outChannels += chan_scatter_a;
    }
    if (_createEmissionAOV) { outChannels+=chan_emission_r; outChannels+=chan_emission_g; outChannels+=chan_emission_b; outChannels+=chan_emission_a; }
    if (_createTemperatureAOV) { outChannels+=chan_temperature_r; outChannels+=chan_temperature_g; outChannels+=chan_temperature_b; outChannels+=chan_temperature_a; }
    if (_createLight1AOV || _light1Enable) { outChannels+=chan_light1_r; outChannels+=chan_light1_g; outChannels+=chan_light1_b; outChannels+=chan_light1_a; }
    if (_createLight2AOV || _light2Enable) { outChannels+=chan_light2_r; outChannels+=chan_light2_g; outChannels+=chan_light2_b; outChannels+=chan_light2_a; }
    if (_createLight3AOV || _light3Enable) { outChannels+=chan_light3_r; outChannels+=chan_light3_g; outChannels+=chan_light3_b; outChannels+=chan_light3_a; }
    if (_createAlbedoAOV) { outChannels+=chan_albedo_r; outChannels+=chan_albedo_g; outChannels+=chan_albedo_b; outChannels+=chan_albedo_a; }
    if (_createDepthAOV) { outChannels+=chan_depth_z; }
    if (_createPositionAOV) { outChannels+=chan_P_x; outChannels+=chan_P_y; outChannels+=chan_P_z; outChannels+=chan_P_a; }
    if (_createPrefAOV) { outChannels+=chan_Pref_x; outChannels+=chan_Pref_y; outChannels+=chan_Pref_z; outChannels+=chan_Pref_a; }
    if (_createNormalAOV) { outChannels+=chan_N_x; outChannels+=chan_N_y; outChannels+=chan_N_z; outChannels+=chan_N_a; }
    if (_createEnvAOV) { outChannels+=chan_env_r; outChannels+=chan_env_g; outChannels+=chan_env_b; outChannels+=chan_env_a; }
    info_.channels(outChannels); set_out_channels(outChannels); info_.turn_on(outChannels);

    // ── Deep info (dual Iop/DeepOp) ──
    if (_deepEnable && fullFmt) {
        ChannelSet deepChannels;
        deepChannels += Mask_Deep;
        deepChannels += Mask_RGBA;
        int w = fullFmt->width(), h = fullFmt->height();
        _deepInfo = DeepInfo(_formats, Box(0, 0, w, h), deepChannels);
    }

    if (!for_real) return;

    // Cache camera
    if (CameraOp* cam = camera()) {
        cam->validate(for_real);
        const auto cw = cam->worldTransform();
        double newOrigin[3] = {cw[3][0], cw[3][1], cw[3][2]};
        double newRot[3][3];
        for (int c=0;c<3;++c) for (int r=0;r<3;++r) newRot[c][r]=static_cast<double>(cw[c][r]);
        double newHalfW = (cam->horizontalAperture()*0.5) / cam->focalLength();

        bool camChanged = !_camValid;
        if (!camChanged) {
            const double eps=1e-6;
            for (int i=0;i<3&&!camChanged;++i) if(std::abs(newOrigin[i]-_camOrigin[i])>eps) camChanged=true;
            if (std::abs(newHalfW-_halfW)>eps) camChanged=true;
            for (int c=0;c<3&&!camChanged;++c) for(int r=0;r<3&&!camChanged;++r)
                if(std::abs(newRot[c][r]-_camRot[c][r])>eps) camChanged=true;
        }
        for (int i=0;i<3;++i) _camOrigin[i]=newOrigin[i];
        for (int c=0;c<3;++c) for(int r=0;r<3;++r) _camRot[c][r]=newRot[c][r];
        _halfW = newHalfW; _camValid = true;
        // NOTE: do NOT set _gpuFrameReady = false on camera change.
        // The old frame stays visible until the new render completes.
    } else { _camValid=false; error("Connect a Camera node to input 0."); return; }

    buildLights();
    loadGrids();

    // ── HDRI environment: project input 1 to L2 SH ──
    if (_envEnable && Op::input(1)) {
        Iop* envIop = dynamic_cast<Iop*>(Op::input(1));
        if (envIop) {
            envIop->validate(for_real);
            Hash envHash = envIop->hash();
            // Re-project only when input image changes
            if (envHash != _cachedEnvInputHash) {
                _cachedEnvInputHash = envHash;
                const Format& ef = envIop->format();
                int srcW = ef.width(), srcH = ef.height();
                if (srcW > 0 && srcH > 0) {
                    envIop->request(0, 0, srcW, srcH, Mask_RGB, 1);

                    // Downsample to 256×128 via box filter.
                    // L2 SH (9 coefficients) can't represent detail finer than this.
                    const int dstW = 256, dstH = 128;
                    std::vector<float> envPixels(dstW * dstH * 3, 0.0f);

                    const float scaleX = float(srcW) / dstW;
                    const float scaleY = float(srcH) / dstH;

                    for (int dy = 0; dy < dstH; ++dy) {
                        int srcY0 = int(dy * scaleY);
                        int srcY1 = std::min(int((dy + 1) * scaleY), srcH);
                        if (srcY1 <= srcY0) srcY1 = srcY0 + 1;
                        int rowCount = srcY1 - srcY0;

                        // Read all source rows for this destination row
                        // Accumulate into destination bins
                        for (int sy = srcY0; sy < srcY1; ++sy) {
                            Row r(0, srcW);
                            envIop->get(sy, 0, srcW, Mask_RGB, r);
                            const float* rr = r[Chan_Red];
                            const float* rg = r[Chan_Green];
                            const float* rb = r[Chan_Blue];
                            for (int dx2 = 0; dx2 < dstW; ++dx2) {
                                int srcX0 = int(dx2 * scaleX);
                                int srcX1 = std::min(int((dx2 + 1) * scaleX), srcW);
                                if (srcX1 <= srcX0) srcX1 = srcX0 + 1;
                                float sr = 0, sg = 0, sb = 0;
                                for (int sx = srcX0; sx < srcX1; ++sx) {
                                    sr += rr ? rr[sx] : 0.0f;
                                    sg += rg ? rg[sx] : 0.0f;
                                    sb += rb ? rb[sx] : 0.0f;
                                }
                                int colCount = srcX1 - srcX0;
                                float invN = 1.0f / float(colCount * rowCount);
                                int idx = (dy * dstW + dx2) * 3;
                                // Accumulate rows — final divide baked into invN per row
                                // (we accumulate rowCount rows, so divide once at end)
                                envPixels[idx + 0] += sr / float(colCount);
                                envPixels[idx + 1] += sg / float(colCount);
                                envPixels[idx + 2] += sb / float(colCount);
                            }
                        }
                        // Divide by number of rows accumulated
                        float invRows = 1.0f / float(rowCount);
                        for (int dx2 = 0; dx2 < dstW; ++dx2) {
                            int idx = (dy * dstW + dx2) * 3;
                            envPixels[idx + 0] *= invRows;
                            envPixels[idx + 1] *= invRows;
                            envPixels[idx + 2] *= invRows;
                        }
                    }

                    _cachedEnvSH = projectHDRIFromPixels(envPixels.data(), dstW, dstH);
                    _cachedEnvRotation = _envRotation + 999.0;  // force rotation update below
                    std::printf("[VDBRender] HDRI SH projected from input (%dx%d → %dx%d), peak=(%.2f,%.2f,%.2f)\n",
                        srcW, srcH, dstW, dstH,
                        _cachedEnvSH.peakDirection[0], _cachedEnvSH.peakDirection[1], _cachedEnvSH.peakDirection[2]);
                }
            }
            // Apply rotation (cheap SH band rotation, no re-read)
            if (_cachedEnvSH.valid) {
                bool needsRotUpdate = (std::abs(_envRotation - _cachedEnvRotation) > 1e-6) || !_activeEnvSH.valid;
                if (needsRotUpdate) {
                    _cachedEnvRotation = _envRotation;
                    _activeEnvSH = _cachedEnvSH;  // copy unrotated
                    rotateSH_Y(_activeEnvSH.coeffs, float(_envRotation));
                    // Re-extract peak from rotated coefficients
                    float dx = 0.2126f*_activeEnvSH.coeffs[3]+0.7152f*_activeEnvSH.coeffs[12]+0.0722f*_activeEnvSH.coeffs[21];
                    float dy = 0.2126f*_activeEnvSH.coeffs[1]+0.7152f*_activeEnvSH.coeffs[10]+0.0722f*_activeEnvSH.coeffs[19];
                    float dz = 0.2126f*_activeEnvSH.coeffs[2]+0.7152f*_activeEnvSH.coeffs[11]+0.0722f*_activeEnvSH.coeffs[20];
                    float dlen = std::sqrt(dx*dx+dy*dy+dz*dz);
                    if (dlen > 1e-8f) {
                        _activeEnvSH.peakDirection[0]=dx/dlen;
                        _activeEnvSH.peakDirection[1]=dy/dlen;
                        _activeEnvSH.peakDirection[2]=dz/dlen;
                    } else {
                        _activeEnvSH.peakDirection[0]=0; _activeEnvSH.peakDirection[1]=1; _activeEnvSH.peakDirection[2]=0;
                    }
                    evalSH(_activeEnvSH.coeffs, _activeEnvSH.peakDirection[0], _activeEnvSH.peakDirection[1], _activeEnvSH.peakDirection[2],
                           _activeEnvSH.peakColor[0], _activeEnvSH.peakColor[1], _activeEnvSH.peakColor[2]);
                    _activeEnvSH.valid = true;
                }
            }
        }
    } else {
        _activeEnvSH.valid = false;
    }

    // Compute tight screen-space bbox
    if (_gridValid && _camValid && fullFmt) {
        const int fmtW=fullFmt->width(), fmtH=fullFmt->height();
        const double halfH = _halfW*static_cast<double>(fmtH)/static_cast<double>(fmtW);
        const double deg2rad = 3.14159265358979323846/180.0;
        const double sx=_volScale[0]*_volUniformScale, sy=_volScale[1]*_volUniformScale, sz=_volScale[2]*_volUniformScale;
        const double rx=_volRotate[0]*deg2rad, ry=_volRotate[1]*deg2rad, rz=_volRotate[2]*deg2rad;
        const double crx=std::cos(rx),srx=std::sin(rx),cry=std::cos(ry),sry=std::sin(ry),crz=std::cos(rz),srz=std::sin(rz);

        double sMinX=1e9,sMinY=1e9,sMaxX=-1e9,sMaxY=-1e9; bool anyVis=false;
        for (int iz=0;iz<=1;++iz) for(int iy=0;iy<=1;++iy) for(int ix=0;ix<=1;++ix) {
            double px=ix?_bboxMax[0]:_bboxMin[0], py=iy?_bboxMax[1]:_bboxMin[1], pz=iz?_bboxMax[2]:_bboxMin[2];
            px*=sx; py*=sy; pz*=sz;
            double ty=py*crx-pz*srx, tz=py*srx+pz*crx; py=ty; pz=tz;
            double txr=px*cry+pz*sry, tzr=-px*sry+pz*cry; px=txr; pz=tzr;
            double txz=px*crz-py*srz, tyz=px*srz+py*crz; px=txz; py=tyz;
            px+=_volTranslate[0]; py+=_volTranslate[1]; pz+=_volTranslate[2];
            double dx=px-_camOrigin[0], dy=py-_camOrigin[1], dz=pz-_camOrigin[2];
            double camZ=_camRot[0][2]*dx+_camRot[1][2]*dy+_camRot[2][2]*dz;
            if (camZ>=0.0) continue;
            double camX=_camRot[0][0]*dx+_camRot[1][0]*dy+_camRot[2][0]*dz;
            double camY=_camRot[0][1]*dx+_camRot[1][1]*dy+_camRot[2][1]*dz;
            double pixX=(camX/(-camZ*_halfW)*0.5+0.5)*fmtW;
            double pixY=(camY/(-camZ*halfH)*0.5+0.5)*fmtH;
            sMinX=std::min(sMinX,pixX); sMinY=std::min(sMinY,pixY);
            sMaxX=std::max(sMaxX,pixX); sMaxY=std::max(sMaxY,pixY); anyVis=true;
        }
        if (anyVis) {
            const int pad=std::max(2,_overscan);
            int bx1=int(std::floor(sMinX))-pad, by1=int(std::floor(sMinY))-pad;
            int bx2=int(std::ceil(sMaxX))+pad, by2=int(std::ceil(sMaxY))+pad;
            if (bx2>bx1 && by2>by1) info_.set(DD::Image::Box(bx1,by1,bx2,by2));
        }
    }

    // GPU rendering
    if (_gridValid && _camValid) {
        const Format& fmt = format();
        int W=fmt.width(), H=fmt.height();
        bool inFrustum = isVolumeInFrustum(W,H);
        if (W>0 && H>0 && !inFrustum) {
            // Off-screen → zero buffer
            double scale; switch(_gpuRenderScale){case 0:scale=0.25;break;case 1:scale=0.50;break;case 2:scale=0.75;break;default:scale=1.0;}
            int rW=std::max(1,int(W*scale)), rH=std::max(1,int(H*scale));
            int aovFlags=(_createScatterAOV?1:0)|(_createEmissionAOV?2:0)|(_createTemperatureAOV?4:0)|((_createLight1AOV||_light1Enable)?8:0)|((_createLight2AOV||_light2Enable)?16:0)|((_createLight3AOV||_light3Enable)?32:0)|(_createAlbedoAOV?64:0)|(_createDepthAOV?128:0)|(_createPositionAOV?256:0)|(_createPrefAOV?512:0)|(_createNormalAOV?1024:0)|(_createEnvAOV?2048:0);
            int stride=4+__builtin_popcount(aovFlags)*4;
            _gpuOutputBuffer.assign(static_cast<size_t>(rW)*rH*stride, 0.0f);
            _gpuFrameReady=true; _gpuRenderedWidth=rW; _gpuRenderedHeight=rH;
            _gpuOutputStride=stride; _screenBboxX1=0; _screenBboxY1=0;
            return;
        }

        if (!_cudaRenderer) {
            _cudaRenderer = std::make_unique<VDBCUDARenderer>();
            if (!_cudaRenderer->initialize()) {
                std::printf("[VDBRender] CUDA init failed: %s\n", _cudaRenderer->lastError().c_str());
                _cudaRenderer.reset();
            }
        }
        if (_cudaRenderer && W>0 && H>0) {
            Hash paramHash;
            paramHash.append(_loadedFrame);  // ← critical: frame change must trigger re-render
            for(int i=0;i<3;++i) paramHash.append(_camOrigin[i]);  // camera position
            for(int c=0;c<3;++c) for(int r=0;r<3;++r) paramHash.append(_camRot[c][r]);  // camera rotation
            paramHash.append(_halfW);  // focal length / aperture
            paramHash.append(W); paramHash.append(H); paramHash.append(_gpuRenderScale);
            paramHash.append(_gpuShadowSteps); paramHash.append(_gpuShadowStepScale);
            
            paramHash.append(_stepSize); paramHash.append(_extinction); paramHash.append(_scattering);
            paramHash.append(_extinctionR); paramHash.append(_extinctionG); paramHash.append(_extinctionB);
            paramHash.append(_powderEnable); paramHash.append(_powderStrength);
            paramHash.append(_msEnable); paramHash.append(_msOctaves);
            paramHash.append(_msExtFalloff); paramHash.append(_msScatterFalloff);
            paramHash.append(_gradientMix); paramHash.append(_gradientThreshold);
            for(int i=0;i<3;++i) paramHash.append(_gradNormalBias[i]);
            paramHash.append(_gradientSmooth);
            paramHash.append(_scatterSmooth);
            paramHash.append(_densityBlur);
            paramHash.append(_noiseEnable); paramHash.append(_noiseAmplitude); paramHash.append(_noiseFrequency); paramHash.append(_noiseOctaves);
            for(int i=0;i<3;++i){ paramHash.append(_noiseScale[i]); paramHash.append(_noiseOffset[i]); }
            paramHash.append(_phaseG1); paramHash.append(_phaseG2); paramHash.append(_phaseMix); paramHash.append(_phaseMode);
            for (int i=0;i<3;++i) { paramHash.append(_lightDir[i]); paramHash.append(_lightColor[i]);
                paramHash.append(_ambientColor[i]);
                paramHash.append(_volTranslate[i]); paramHash.append(_volRotate[i]); paramHash.append(_volScale[i]); }
            paramHash.append(_ambientIntensity);
            paramHash.append(_envEnable);
            paramHash.append(_envIntensity);
            paramHash.append(_envRotation);
            paramHash.append(_envShadowEnable);
            for(int i=0;i<3;++i) paramHash.append(_envShadowLift[i]);
            if (_activeEnvSH.valid) {
                for (int i = 0; i < 27; ++i) paramHash.append(_activeEnvSH.coeffs[i]);
            }
            paramHash.append(_volUniformScale); paramHash.append(_boundaryBlend);
            paramHash.append(_emissionScale); paramHash.append(_temperatureScale);
            paramHash.append(_emSelfIllum); paramHash.append(_emSelfIllumPhase);
            paramHash.append(_useBlackbody); paramHash.append(_bbKelvinScale); paramHash.append(_bbMix);
            for(int i=0;i<3;++i) paramHash.append(_bbTint[i]);
            for (int i=0;i<3;++i) {
                paramHash.append(_rampColor0[i]); paramHash.append(_rampColor1[i]);
                paramHash.append(_rampColor2[i]); paramHash.append(_rampColor3[i]);
                paramHash.append(_rampColor4[i]);
            }
            paramHash.append(_createScatterAOV); paramHash.append(_createAlbedoAOV); paramHash.append(_createDepthAOV); paramHash.append(_createPositionAOV); paramHash.append(_createPrefAOV); paramHash.append(_createNormalAOV); paramHash.append(_createEmissionAOV); paramHash.append(_createTemperatureAOV); paramHash.append(_createEnvAOV);
            paramHash.append(_createLight1AOV);
            paramHash.append(_createLight2AOV); paramHash.append(_createLight3AOV);
            for(int i=0;i<3;++i) paramHash.append(_lightMix[i]);
            paramHash.append(_envMix);
            paramHash.append(_light1Enable);
            for(int i=0;i<3;++i){ paramHash.append(_light1Pos[i]); paramHash.append(_light1Target[i]); paramHash.append(_light1Color[i]); }
            paramHash.append(_light1Intensity); paramHash.append(_light1ConeAngle); paramHash.append(_light1Softness);
            paramHash.append(_light2Enable);
            for(int i=0;i<3;++i){ paramHash.append(_light2Pos[i]); paramHash.append(_light2Target[i]); paramHash.append(_light2Color[i]); }
            paramHash.append(_light2Intensity); paramHash.append(_light2ConeAngle); paramHash.append(_light2Softness);
            paramHash.append(_light3Enable);
            for(int i=0;i<3;++i){ paramHash.append(_light3Pos[i]); paramHash.append(_light3Target[i]); paramHash.append(_light3Color[i]); }
            paramHash.append(_light3Intensity); paramHash.append(_light3ConeAngle); paramHash.append(_light3Softness);
            paramHash.append(_numGPULights);

            pollAsyncFrame();
            if (paramHash != _gpuRenderedParamHash) {
                _gpuRenderedParamHash = paramHash;
                _deepReady = false;  // invalidate deep cache too
                _deepFailed = false;
                // Always sync render — async only benefits camera tumbling,
                // and parameter changes need immediate feedback.
                renderCUDAFrame(W, H);
            }
        }
    }
}

void VDBRenderIop::_request(int, int, int, int, ChannelMask, int) {}
