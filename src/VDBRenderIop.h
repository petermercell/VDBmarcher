#pragma once
// ═══════════════════════════════════════════════════════════════════════════
// VDBRenderIop.h — CVDB-only volume renderer for Nuke
// Zero OpenVDB dependency. Reads .cvdb codebook volumes via CUDA.
// ═══════════════════════════════════════════════════════════════════════════

#include <DDImage/Iop.h>
#include <DDImage/Knobs.h>
#include <DDImage/Row.h>
#include <DDImage/Thread.h>
#include <DDImage/CameraOp.h>
#include <DDImage/LightOp.h>
#include <DDImage/Matrix4.h>
#include <DDImage/Format.h>
#include <DDImage/ViewerContext.h>
#include <DDImage/Channel.h>
#include <DDImage/gl.h>
#include <DDImage/DeepOp.h>
#include <DDImage/DeepPlane.h>

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <climits>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <atomic>
#include <cstring>

#include "CVDBLoader.h"
#include "VDBRenderConfig.h"
#include "VDBEnvironment.h"

class VDBCUDARenderer;

class VDBRenderIop : public DD::Image::Iop, public DD::Image::DeepOp
{
public:
    explicit VDBRenderIop(Node* node);
    ~VDBRenderIop() override;

    void        knobs(DD::Image::Knob_Callback f) override;
    int         knob_changed(DD::Image::Knob* k) override;
    const char* Class()     const override { return CLASS; }
    const char* node_help() const override { return HELP; }

    const char* input_label(int idx, char* buf) const override;
    bool        test_input(int idx, DD::Image::Op* op) const override;
    DD::Image::Op* default_input(int idx) const override;
    int         minimum_inputs() const override { return 1; }
    int         maximum_inputs() const override { return 2; }

    void _validate(bool for_real) override;
    void _request(int x, int y, int r, int t,
                  DD::Image::ChannelMask, int count) override;
    void engine(int y, int x, int r,
                DD::Image::ChannelMask, DD::Image::Row&) override;
    void append(DD::Image::Hash& hash) override;

    // ── DeepOp interface (dual Iop/DeepOp) ──
    DD::Image::Op* op() override { return this; }

    void getDeepRequests(DD::Image::Box box, const DD::Image::ChannelSet& channels,
                         int count,
                         std::vector<DD::Image::RequestData>& requests) override;

    bool doDeepEngine(DD::Image::Box box, const DD::Image::ChannelSet& channels,
                      DD::Image::DeepOutputPlane& plane) override;

    void build_handles(DD::Image::ViewerContext* ctx) override;
    void draw_handle(DD::Image::ViewerContext* ctx) override;

    static const DD::Image::Op::Description desc;
    static const char* CLASS;
    static const char* HELP;

    // AOV channels
    static DD::Image::Channel chan_scatter_r, chan_scatter_g, chan_scatter_b, chan_scatter_a;
    static DD::Image::Channel chan_emission_r, chan_emission_g, chan_emission_b, chan_emission_a;
    static DD::Image::Channel chan_temperature_r, chan_temperature_g, chan_temperature_b, chan_temperature_a;
    static DD::Image::Channel chan_light1_r, chan_light1_g, chan_light1_b, chan_light1_a;
    static DD::Image::Channel chan_light2_r, chan_light2_g, chan_light2_b, chan_light2_a;
    static DD::Image::Channel chan_light3_r, chan_light3_g, chan_light3_b, chan_light3_a;
    static DD::Image::Channel chan_albedo_r, chan_albedo_g, chan_albedo_b, chan_albedo_a;
    static DD::Image::Channel chan_depth_z;
    static DD::Image::Channel chan_P_x, chan_P_y, chan_P_z, chan_P_a;
    static DD::Image::Channel chan_Pref_x, chan_Pref_y, chan_Pref_z, chan_Pref_a;
    static DD::Image::Channel chan_N_x, chan_N_y, chan_N_z, chan_N_a;
    static DD::Image::Channel chan_env_r, chan_env_g, chan_env_b, chan_env_a;

private:
    // ── Knobs ──
    const char* _vdbFilePath   = "";
    const char* _gridName      = "density";
    double      _emissionScale = 0.0;
    bool        _createEmissionAOV = false;
    double      _emSelfIllum = 0.0;
    bool        _emSelfIllumPhase = false;
    double      _temperatureScale = 0.0;
    bool        _createTemperatureAOV = false;
    bool        _useBlackbody = false;
    double      _bbKelvinScale = 6500.0;
    double      _bbMix = 1.0;
    double      _bbTint[3] = { 1.0, 1.0, 1.0 };
    double      _rampColor0[3] = { 0.05, 0.00, 0.00 };
    double      _rampColor1[3] = { 0.85, 0.25, 0.01 };
    double      _rampColor2[3] = { 1.00, 0.70, 0.15 };
    double      _rampColor3[3] = { 1.00, 0.95, 0.70 };
    double      _rampColor4[3] = { 0.75, 0.85, 1.00 };
    double      _stepSize      = 0.05;
    double      _extinction    = 1.0;
    double      _extinctionR   = 1.0;
    double      _extinctionG   = 1.0;
    double      _extinctionB   = 1.0;
    double      _scattering    = 0.5;
    double      _lightDir[3]   = { 0.577, 0.577, 0.577 };  // fallback only
    double      _lightColor[3] = { 1.0,   1.0,   1.0   };  // fallback only

    // ── Internal GPU lights (no LightOp inputs) ──
    bool        _light1Enable = true;
    double      _light1Pos[3] = { 5.0, 8.0, 5.0 };
    double      _light1Target[3] = { 0.0, 0.0, 0.0 };
    double      _light1Color[3] = { 1.0, 1.0, 1.0 };
    double      _light1Intensity = 1.0;
    double      _light1ConeAngle = 90.0;   // degrees, full = 180
    double      _light1Softness = 0.3;

    bool        _light2Enable = false;
    double      _light2Pos[3] = { -5.0, 3.0, -3.0 };
    double      _light2Target[3] = { 0.0, 0.0, 0.0 };
    double      _light2Color[3] = { 0.6, 0.7, 1.0 };
    double      _light2Intensity = 0.5;
    double      _light2ConeAngle = 90.0;
    double      _light2Softness = 0.3;

    bool        _light3Enable = false;
    double      _light3Pos[3] = { 0.0, -2.0, 8.0 };
    double      _light3Target[3] = { 0.0, 0.0, 0.0 };
    double      _light3Color[3] = { 1.0, 0.9, 0.7 };
    double      _light3Intensity = 0.3;
    double      _light3ConeAngle = 90.0;
    double      _light3Softness = 0.3;

    double      _lightDisplayScale = 1.0;  // viewport marker size multiplier

    double      _ambientColor[3] = { 0.2, 0.25, 0.35 };
    double      _ambientIntensity = 0.0;

    // ── HDRI Environment (Phase 2B) ──
    bool        _envEnable = true;
    double      _envIntensity = 1.0;
    double      _envRotation = 0.0;
    bool        _envShadowEnable = false;
    double      _envShadowLift[3] = { 0.0, 0.0, 0.0 };
    EnvSH       _cachedEnvSH;           // unrotated SH from last projection
    EnvSH       _activeEnvSH;           // rotated SH currently on GPU
    DD::Image::Hash _cachedEnvInputHash;
    double      _cachedEnvRotation = 0.0;

    double      _phaseG1  = 0.0;
    double      _phaseG2  = 0.0;
    double      _phaseMix = 1.0;
    int         _phaseMode = 2;
    bool        _createScatterAOV = false;
    bool        _createAlbedoAOV = false;
    bool        _createDepthAOV = false;
    bool        _createPositionAOV = false;
    bool        _createPrefAOV = false;
    bool        _createNormalAOV = false;

    bool        _powderEnable = false;
    double      _powderStrength = 0.0;

    // ── Multi-scatter octaves (Frostbite / Hillaire 2016) ──
    bool        _msEnable = false;
    int         _msOctaves = 4;
    double      _msExtFalloff = 0.5;     // extinction reduction per octave
    double      _msScatterFalloff = 0.5;  // contribution weight per octave

    double      _gradientMix = 0.0;
    double      _gradientThreshold = 0.1;
    double      _gradNormalBias[3] = { 0, 0, 0 };
    double      _gradientSmooth = 1.0;  // FD step in voxels (1=sharp, 2-4=smooth)
    double      _scatterSmooth = 0.0;   // density smooth radius for scattering (0=off)
    double      _densityBlur = 0.0;    // raw density blur radius in voxels (0=off)

    bool        _noiseEnable = false;
    double      _noiseAmplitude = 0.0;
    double      _noiseFrequency = 10.0;
    double      _noiseScale[3] = { 1.0, 1.0, 1.0 };
    int         _noiseOctaves = 4;
    double      _noiseOffset[3] = { 0, 0, 0 };
    bool        _createLight1AOV = false;
    bool        _createLight2AOV = false;
    bool        _createLight3AOV = false;
    bool        _createEnvAOV = false;

    // ── Light Mixer (RGBA only, AOVs stay 100%) ──
    double      _lightMix[3] = { 1.0, 1.0, 1.0 };
    double      _envMix = 1.0;
    int         _displayMode   = 2;
    int         _frameOffset   = 0;
    double      _pointSize     = 3.0;
    DD::Image::FormatPair _formats;

    int         _gpuRenderScale = 1;
    int         _gpuShadowSteps = 8;
    double      _gpuShadowStepScale = 4.0;
    double      _boundaryBlend = 0.0;

    double      _volTranslate[3] = { 0, 0, 0 };
    double      _volRotate[3]    = { 0, 0, 0 };
    double      _volScale[3]     = { 1, 1, 1 };
    double      _volUniformScale = 1.0;
    int         _overscan = 0;

    // ── Volume state ──
    double      _bboxMin[3] = {};
    double      _bboxMax[3] = {};
    bool        _gridValid = false;
    int         _loadedFrame = -1;

    // ── Camera ──
    double      _camOrigin[3] = {};
    double      _camRot[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    double      _halfW = 1.0;
    bool        _camValid = false;

    // ── Lights ──
    GPULight    _gpuLights[3] = {};
    int         _numGPULights = 0;

    // ── GPU state ──
    std::unique_ptr<VDBCUDARenderer> _cudaRenderer;
    std::string              _cudaVolumeLoadedPath;
    int                      _cudaVolumeLoadedFrame = -1;
    std::vector<float>       _gpuOutputBuffer;
    bool                     _gpuFrameReady = false;
    int                      _gpuRenderedWidth  = 0;
    int                      _gpuRenderedHeight = 0;
    int                      _gpuOutputStride   = 4;
    int                      _screenBboxX1 = 0;
    int                      _screenBboxY1 = 0;
    DD::Image::Hash          _gpuRenderedParamHash;
    bool                     _asyncRenderLaunched = false;
    int                      _asyncBboxX1 = 0;
    int                      _asyncBboxY1 = 0;

    // ── Methods ──
    void loadGrids();
    bool loadCVDB(const std::string& path, const std::string& gridName);
    bool renderCUDAFrame(int w, int h);
    bool renderCUDAFrameAsync(int w, int h);
    bool pollAsyncFrame();
    void buildVolumeTransform(float out[16]) const;
    bool isVolumeInFrustum(int w, int h) const;

    // [Phase 0] Build unified render config from all knobs
    VDBRenderConfig buildRenderConfig(int fmtW, int fmtH) const;

    // ── 3D viewport ──
    struct DensityPoint { float x, y, z, density; };
    std::vector<DensityPoint> _previewPoints;
    float                     _maxDensity = 1.0f;
    std::string               _cachedPointsPath;
    int                       _cachedPointsFrame = -1;
    void rebuildPointCloud();

    struct CVDBCpuCache {
        std::vector<float>    codebook;
        std::vector<uint16_t> indices;
        std::vector<int32_t>  origins;
        std::vector<float>    gain_maps;
        uint32_t K = 0, numLeaves = 0;
        int blockSize = 8; float voxelSize = 1.0f; bool hasGain = false;
        float gridOffset[3] = {0,0,0};
        bool valid() const { return K > 0 && numLeaves > 0; }
        void clear() { codebook.clear(); indices.clear(); origins.clear();
                        gain_maps.clear(); K = 0; numLeaves = 0; gridOffset[0]=gridOffset[1]=gridOffset[2]=0; }
    } _cvdbCpu;

    static constexpr int _bboxEdges[12][2] = {
        {0,1},{2,3},{4,5},{6,7},{0,2},{1,3},{4,6},{5,7},{0,4},{1,5},{2,6},{3,7}
    };

    DD::Image::CameraOp* camera() const;
    void buildLights();
    std::string resolveFramePath(int frame) const;

    DD::Image::Lock _loadLock;
    std::string     _loadedPath;
    std::string     _loadedGrid;

    // ── Frame prefetch cache ──
    int                                _cacheRadius = 8;
    std::unordered_map<int, CVDBFile>  _frameCache;
    std::mutex                         _frameCacheMtx;
    std::thread                        _prefetchThread;
    std::atomic<bool>                  _prefetchStop{false};
    std::atomic<int>                   _prefetchCenter{INT_MIN};
    std::string                        _prefetchPathBase;

    void startPrefetch(int centerFrame);
    void stopPrefetch();
    void evictDistantFrames(int centerFrame);
    static void prefetchWorker(VDBRenderIop* self);

    // ── Deep output ──
    bool        _deepEnable = false;
    int         _deepMaxSamples = 32;
    bool        _profileEnable = false;
    std::vector<uint16_t> _deepBuf;
    std::vector<int>      _deepCounts;
    int         _deepRenderedW = 0;
    int         _deepRenderedH = 0;
    bool        _deepReady = false;
    bool        _deepFailed = false;
    std::mutex  _deepRenderMtx;

    bool renderDeepFrame(int w, int h);
    static float halfToFloat(uint16_t v) {
        uint32_t s = (v & 0x8000u) << 16;
        uint32_t e = (v >> 10) & 0x1F;
        uint32_t m = v & 0x3FF;
        uint32_t f;
        if (e == 0) f = s;
        else if (e == 31) f = s | 0x7F800000u | (m << 13);
        else f = s | ((e + 112) << 23) | (m << 13);
        float result;
        std::memcpy(&result, &f, sizeof(float));
        return result;
    }
};
