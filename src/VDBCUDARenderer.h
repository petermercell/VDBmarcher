#pragma once
// ═══════════════════════════════════════════════════════════════════════════
// VDBCUDARenderer.h — CVDB codebook volume GPU renderer
// [Phase 0] Render params via VDBRenderConfig. No public member forwarding.
// ═══════════════════════════════════════════════════════════════════════════

#include "VDBRenderConfig.h"
#include <string>
#include <cstddef>
#include <cstdint>

// ── CUDA error checking macro ───────────────────────────────────────────────
#define CUDA_CHECK(call) do {                                  \
    cudaError_t _e = (call);                                   \
    if (_e != cudaSuccess) {                                   \
        _lastError = cudaGetErrorString(_e);                   \
        return false;                                          \
    }                                                          \
} while(0)

class VDBCUDARenderer
{
public:
    VDBCUDARenderer();
    ~VDBCUDARenderer();
    VDBCUDARenderer(const VDBCUDARenderer&) = delete;
    VDBCUDARenderer& operator=(const VDBCUDARenderer&) = delete;

    bool initialize();
    void shutdown();
    bool isInitialized() const { return _initialized; }

    // Upload codebook volume — builds GPU hash table for O(1) leaf lookup
    bool uploadCodebookVolume(
        const float* codebook, uint32_t K,
        const uint16_t* indices, const int32_t* origins, uint32_t numLeaves,
        const float* gainMaps, const float* residuals,
        int blockSize, float voxelSize,
        const double bboxMin[3], const double bboxMax[3],
        const double gridOffset[3], float normScale = 1.0f);
    void clearCodebook();

    // FP16 codebook uploads (CVD6)
    bool uploadCodebookVolumeFP16(
        const uint16_t* codebookFP16, uint32_t K,
        const uint16_t* indices, const int32_t* origins, uint32_t numLeaves,
        const float* gainMaps,
        int blockSize, float voxelSize,
        const double bboxMin[3], const double bboxMax[3],
        const double gridOffset[3], float normScale = 1.0f);

    // Upload emission/temperature as separate codebook grids
    bool uploadEmissionVolume(
        const float* codebook, uint32_t K,
        const uint16_t* indices, const int32_t* origins, uint32_t numLeaves,
        const float* gainMaps, int blockSize, float voxelSize,
        const double gridOffset[3], float normScale = 1.0f);
    void clearEmission();

    bool uploadTemperatureVolume(
        const float* codebook, uint32_t K,
        const uint16_t* indices, const int32_t* origins, uint32_t numLeaves,
        const float* gainMaps, int blockSize, float voxelSize,
        const double gridOffset[3], float normScale = 1.0f);
    void clearTemperature();

    // FP16 emission/temperature (CVD6)
    bool uploadEmissionVolumeFP16(
        const uint16_t* codebookFP16, uint32_t K,
        const uint16_t* indices, const int32_t* origins, uint32_t numLeaves,
        const float* gainMaps, int blockSize, float voxelSize,
        const double gridOffset[3], float normScale = 1.0f);
    bool uploadTemperatureVolumeFP16(
        const uint16_t* codebookFP16, uint32_t K,
        const uint16_t* indices, const int32_t* origins, uint32_t numLeaves,
        const float* gainMaps, int blockSize, float voxelSize,
        const double gridOffset[3], float normScale = 1.0f);

    bool hasEmission() const { return _emission.valid; }
    bool hasTemperature() const { return _temperature.valid; }

    // ── [Phase 0] Config-based render API ───────────────────────────────
    // All render parameters come from VDBRenderConfig. No member variable
    // forwarding — the config struct is the single source of truth.

    // Synchronous render (blocks until complete)
    bool render(const VDBRenderConfig& cfg, float* outputRGBA);

    // Async render (non-blocking pipeline)
    bool renderAsync(const VDBRenderConfig& cfg);

    bool isRenderComplete() const;
    bool readbackResult(float* outputRGBA, int expectedPixels, int expectedStride) const;
    int  asyncWidth()  const { return _asyncWidth; }
    int  asyncHeight() const { return _asyncHeight; }
    int  asyncStride() const { return _asyncStride; }

    // ── Deep rendering ──────────────────────────────────────────────────
    bool renderDeep(const VDBRenderConfig& cfg, int maxDeepSamples,
                    uint16_t* hostDeepBuf, int* hostDeepCounts);

    const std::string& lastError() const { return _lastError; }

    // Per-grid GPU data (public so free functions in .cu can access)
    struct CodebookGridGPU {
        float*    d_codebook = nullptr;
        void*     d_codebookHalf = nullptr;
        uint16_t* d_indices = nullptr;
        float*    d_gainMaps = nullptr;
        float*    d_residuals = nullptr;
        int64_t*  d_hashKeys = nullptr;
        uint32_t* d_hashVals = nullptr;
        uint32_t  hashTableSize = 0;
        uint32_t  K = 0, numLeaves = 0;
        int       blockSize = 8, blockShift = 3, voxelsPerBlock = 512;
        float     voxelSize = 1.0f;
        float     gridOffset[3] = {0,0,0};
        float     normScale = 1.0f;
        bool      valid = false;
        uint32_t  allocCodebookElems = 0;
        uint32_t  allocLeaves = 0;
        uint32_t  allocHashSize = 0;
        bool      allocHasGain = false;
        bool      allocHasResiduals = false;
        bool      curHasGain = false;
        bool      curHasResiduals = false;
        uint64_t  codebookFingerprint = 0;  // [Phase 3] temporal sharing hash
        uint8_t*  d_macroGrid = nullptr;
        int       macroGridDims[3] = {0,0,0};
        int       macroOrigin[3] = {0,0,0};
        int       macroCellShift = 3;
        uint32_t  allocMacroElems = 0;
        uint32_t  uploadGeneration = 0;
        void free();
    };


private:
    bool        _initialized = false;
    std::string _lastError;

    // Output buffers
    void*   _d_output     = nullptr;   // FP16 on GPU
    float*  _d_outputF32  = nullptr;   // FP32 on GPU (for readback conversion)
    int     _outputWidth  = 0;
    int     _outputHeight = 0;
    int     _lastStride   = 0;
    size_t  _outputF32Capacity = 0;

    float _bboxMin[3] = {};
    float _bboxMax[3] = {};

    CodebookGridGPU _density;
    CodebookGridGPU _emission;
    CodebookGridGPU _temperature;

    // Internal: upload a codebook grid to GPU
    bool uploadGrid(CodebookGridGPU& grid,
                    const float* codebook, uint32_t K,
                    const uint16_t* indices, const int32_t* origins, uint32_t numLeaves,
                    const float* gainMaps, const float* residuals,
                    int blockSize, float voxelSize);

    // FP16 codebook upload (CVD6)
    bool uploadGridFP16(CodebookGridGPU& grid,
                        const uint16_t* codebookFP16, uint32_t K,
                        const uint16_t* indices, const int32_t* origins, uint32_t numLeaves,
                        const float* gainMaps,
                        int blockSize, float voxelSize);

    // [Phase 0] Shared tail of uploadGrid/uploadGridFP16: leaf buffers, hash table, macro-grid
    bool finalizeGrid(CodebookGridGPU& g,
                      const uint16_t* indices, const int32_t* origins, uint32_t N,
                      const float* gainMaps, const float* residuals,
                      bool needGain, bool needResiduals,
                      int blockSize, float voxelSize);

    void ensureOutputBuffer(int w, int h, int stride);
    void ensureF32Buffer(size_t elems);

    // Persistent temp buffer for origins upload
    int32_t* _d_tempOrigins = nullptr;
    uint32_t _tempOriginsCapacity = 0;

    // Async state
    void*   _renderStream = nullptr;
    void*   _renderEvent  = nullptr;
    float*  _h_pinnedOutput = nullptr;
    size_t  _pinnedSize     = 0;
    bool    _asyncInFlight  = false;
    int     _asyncWidth  = 0;
    int     _asyncHeight = 0;
    int     _asyncStride = 0;

    // CUDA Graph state
    void*   _graphExec   = nullptr;
    int     _graphWidth  = 0;
    int     _graphHeight = 0;
    int     _graphStride = 0;
    int     _graphNumTiles = 0;

    void ensurePinnedBuffer(size_t bytes);
    void ensureStream();
    void invalidateGraph();

    // Occupancy-tuned block dimensions
    int _blkDimX = 16;
    int _blkDimY = 16;
};
