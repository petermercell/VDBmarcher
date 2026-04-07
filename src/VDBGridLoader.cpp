// ═══════════════════════════════════════════════════════════════════════════
// VDBGridLoader.cpp — CVDB file loading with multi-grid auto-discovery
// One .cvdb file → auto-finds density, emission, and temperature grids.
// ═══════════════════════════════════════════════════════════════════════════

#include "VDBRenderIop.h"
#include "VDBCUDARenderer.h"
#include "CVDBLoader.h"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <chrono>

using namespace DD::Image;

// ─── Prefetch Cache Methods ─────────────────────────────────────────────────

void VDBRenderIop::stopPrefetch()
{
    _prefetchStop.store(true);
    if (_prefetchThread.joinable()) _prefetchThread.join();
    _prefetchStop.store(false);
    _prefetchCenter.store(INT_MIN);
}

void VDBRenderIop::evictDistantFrames(int centerFrame)
{
    // Called with _frameCacheMtx held. Evict frames outside 2× radius.
    int maxDist = _cacheRadius * 2 + 2;
    std::vector<int> toEvict;
    for (auto& kv : _frameCache)
        if (std::abs(kv.first - centerFrame) > maxDist)
            toEvict.push_back(kv.first);
    for (int f : toEvict) _frameCache.erase(f);
}

void VDBRenderIop::startPrefetch(int centerFrame)
{
    if (_cacheRadius <= 0) return;
    _prefetchCenter.store(centerFrame);
    _prefetchPathBase = std::string(_vdbFilePath ? _vdbFilePath : "");
    if (_prefetchPathBase.empty()) return;

    if (!_prefetchThread.joinable()) {
        _prefetchStop.store(false);
        _prefetchThread = std::thread(prefetchWorker, this);
    }
}

void VDBRenderIop::prefetchWorker(VDBRenderIop* self)
{
    while (!self->_prefetchStop.load()) {
        int center = self->_prefetchCenter.load();
        if (center == INT_MIN) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        int radius = self->_cacheRadius;
        std::string pathBase = self->_prefetchPathBase;
        if (pathBase.empty() || radius <= 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        bool didWork = false;
        // Prefetch forward first (playback direction), then backward (scrubbing)
        for (int pass = 0; pass < 2 && !self->_prefetchStop.load(); ++pass) {
            int dir = (pass == 0) ? 1 : -1;
            int count = (pass == 0) ? radius : radius / 2;
            for (int offset = 1; offset <= count && !self->_prefetchStop.load(); ++offset) {
                int frame = center + dir * offset;

                // Check if already cached
                { std::lock_guard<std::mutex> lock(self->_frameCacheMtx);
                  if (self->_frameCache.count(frame)) continue; }

                // Resolve path for this frame (reuse the path pattern logic)
                std::string path = pathBase;
                // Apply frame number substitution (simplified — handles #### and auto-detect)
                {
                    size_t hashStart = path.find('#');
                    if (hashStart != std::string::npos) {
                        size_t hashEnd = hashStart;
                        while (hashEnd < path.size() && path[hashEnd] == '#') ++hashEnd;
                        int padding = static_cast<int>(hashEnd - hashStart);
                        char buf[64]; std::snprintf(buf, sizeof(buf), "%0*d", padding, frame);
                        path.replace(hashStart, hashEnd - hashStart, buf);
                    } else {
                        size_t pct = path.find('%');
                        if (pct != std::string::npos) {
                            size_t scan = pct + 1;
                            if (scan < path.size() && path[scan] == '0') ++scan;
                            while (scan < path.size() && path[scan] >= '0' && path[scan] <= '9') ++scan;
                            if (scan < path.size() && path[scan] == 'd') {
                                std::string fmt = path.substr(pct, scan - pct + 1);
                                char buf[64]; std::snprintf(buf, sizeof(buf), fmt.c_str(), frame);
                                path.replace(pct, scan - pct + 1, buf);
                            }
                        } else {
                            size_t dot = path.rfind('.');
                            if (dot == std::string::npos) dot = path.size();
                            size_t numEnd = dot, numStart = numEnd;
                            while (numStart > 0 && path[numStart-1] >= '0' && path[numStart-1] <= '9') --numStart;
                            if (numStart < numEnd) {
                                int padding = static_cast<int>(numEnd - numStart);
                                char buf[64]; std::snprintf(buf, sizeof(buf), "%0*d", padding, frame);
                                path.replace(numStart, numEnd - numStart, buf);
                            }
                        }
                    }
                }

                // Check file exists before trying to load
                FILE* test = fopen(path.c_str(), "rb");
                if (!test) continue;
                fclose(test);

                // Load from disk (the slow part — done in background)
                try {
                    CVDBFile cvdb = CVDBFile::load(path);
                    if (!cvdb.grids.empty()) {
                        std::lock_guard<std::mutex> lock(self->_frameCacheMtx);
                        self->_frameCache[frame] = std::move(cvdb);
                        self->evictDistantFrames(center);
                        didWork = true;
                    }
                } catch (...) {
                    // File not found or corrupt — skip silently
                }
            }
        }

        if (!didWork) {
            // All nearby frames already cached — sleep before rechecking
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
}

// ─── Frame Path Resolution ──────────────────────────────────────────────────

std::string VDBRenderIop::resolveFramePath(int frame) const
{
    std::string path(_vdbFilePath ? _vdbFilePath : "");
    if (path.empty()) return path;

    // #### patterns
    size_t hashStart = path.find('#');
    if (hashStart != std::string::npos) {
        size_t hashEnd = hashStart;
        while (hashEnd < path.size() && path[hashEnd] == '#') ++hashEnd;
        int padding = static_cast<int>(hashEnd - hashStart);
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%0*d", padding, frame);
        path.replace(hashStart, hashEnd - hashStart, buf);
        return path;
    }

    // %0Nd patterns
    size_t pct = path.find('%');
    if (pct != std::string::npos) {
        size_t scan = pct + 1;
        if (scan < path.size() && path[scan] == '0') ++scan;
        while (scan < path.size() && path[scan] >= '0' && path[scan] <= '9') ++scan;
        if (scan < path.size() && path[scan] == 'd') {
            std::string fmt = path.substr(pct, scan - pct + 1);
            char buf[64];
            std::snprintf(buf, sizeof(buf), fmt.c_str(), frame);
            path.replace(pct, scan - pct + 1, buf);
            return path;
        }
    }

    // Auto-detect last digit group before extension
    size_t dot = path.rfind('.');
    if (dot == std::string::npos) dot = path.size();
    size_t numEnd = dot, numStart = numEnd;
    while (numStart > 0 && path[numStart-1] >= '0' && path[numStart-1] <= '9') --numStart;
    if (numStart < numEnd) {
        int padding = static_cast<int>(numEnd - numStart);
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%0*d", padding, frame);
        path.replace(numStart, numEnd - numStart, buf);
        return path;
    }
    return path;
}

// ─── Grid Loading ───────────────────────────────────────────────────────────

void VDBRenderIop::loadGrids()
{
    Guard guard(_loadLock);

    int curFrame = static_cast<int>(outputContext().frame()) + _frameOffset;
    std::string path = resolveFramePath(curFrame);
    std::string grid(_gridName ? _gridName : "");

    // Fall back to original path if resolved doesn't exist
    {
        FILE* test = fopen(path.c_str(), "rb");
        if (!test) {
            std::string origPath(_vdbFilePath ? _vdbFilePath : "");
            if (origPath != path) {
                FILE* origTest = fopen(origPath.c_str(), "rb");
                if (origTest) { fclose(origTest); path = origPath; }
            }
        } else { fclose(test); }
    }

    if (!_gridValid || path != _loadedPath || grid != _loadedGrid || curFrame != _loadedFrame) {
        _gridValid = false;
        _cvdbCpu.clear();
        _previewPoints.clear();

        if (!path.empty()) {
            if (loadCVDB(path, grid)) {
                _loadedPath  = path;
                _loadedGrid  = grid;
                _loadedFrame = curFrame;
                // Trigger background prefetch of surrounding frames
                startPrefetch(curFrame);
            }
        }
    }
}

// ─── Helper: match grid name to role ────────────────────────────────────────

static bool isEmissionGrid(const std::string& name) {
    return name == "flames" || name == "emission" || name == "flame" || name == "fire";
}

static bool isTemperatureGrid(const std::string& name) {
    return name == "temperature" || name == "heat" || name == "temp";
}

// ─── CVDB Loading — all grids from one file ─────────────────────────────────

bool VDBRenderIop::loadCVDB(const std::string& path, const std::string& gridName)
{
    std::string cleanPath = path;
    for (char& ch : cleanPath) if (ch == '\\') ch = '/';

    int curFrame = static_cast<int>(outputContext().frame()) + _frameOffset;

    // ── Check prefetch cache first (avoids disk I/O if background thread pre-loaded) ──
    CVDBFile cvdb;
    bool cacheHit = false;
    {
        std::lock_guard<std::mutex> lock(_frameCacheMtx);
        auto it = _frameCache.find(curFrame);
        if (it != _frameCache.end()) {
            cvdb = std::move(it->second);
            _frameCache.erase(it);
            cacheHit = true;
        }
    }

    if (!cacheHit) {
        // Cache miss — load from disk (the slow path)
        try {
            cvdb = CVDBFile::load(cleanPath);
        } catch (const std::exception& e) {
            error("CVDB: %s", e.what());
            return false;
        }
    }

    if (cvdb.grids.empty()) {
        error("CVDB: No grids in %s", cleanPath.c_str());
        return false;
    }

    std::printf("[VDBRender] Loaded %s (%zu grid%s%s)\n",
        cleanPath.c_str(), cvdb.grids.size(), cvdb.grids.size()>1?"s":"",
        cacheHit ? ", cached" : "");

    // Find density grid (by name or first grid)
    std::string targetName = gridName.empty() ? "density" : gridName;
    int densityIdx = -1, emissionIdx = -1, temperatureIdx = -1;

    for (int i = 0; i < static_cast<int>(cvdb.grids.size()); ++i) {
        const std::string& name = cvdb.grids[i].name;
        if (name == targetName) densityIdx = i;
        else if (isEmissionGrid(name)) emissionIdx = i;
        else if (isTemperatureGrid(name)) temperatureIdx = i;
    }

    // If no grid matches the target name, use the first non-emission/non-temperature grid
    if (densityIdx < 0) {
        for (int i = 0; i < static_cast<int>(cvdb.grids.size()); ++i) {
            if (i != emissionIdx && i != temperatureIdx) { densityIdx = i; break; }
        }
    }
    if (densityIdx < 0) {
        error("CVDB: No density grid found in %s", cleanPath.c_str());
        return false;
    }

    // Initialize CUDA renderer
    if (!_cudaRenderer) {
        _cudaRenderer = std::make_unique<VDBCUDARenderer>();
        if (!_cudaRenderer->initialize()) {
            error("CVDB: CUDA init failed: %s", _cudaRenderer->lastError().c_str());
            _cudaRenderer.reset();
            return false;
        }
    }

    // ── Upload density ──
    {
        const CVDBGrid& grid = cvdb.grids[densityIdx];

        // The origins in .cvdb are in INDEX space, but the bbox is in WORLD space.
        // OpenVDB transform: world = index * voxelSize + offset
        // We need to compute the offset so the kernel can convert world→index.
        //
        // Compute index-space extent from origins:
        double ixMin[3]={1e30,1e30,1e30}, ixMax[3]={-1e30,-1e30,-1e30};
        for (uint32_t i = 0; i < grid.num_leaves; ++i) {
            for (int j = 0; j < 3; ++j) {
                double o = grid.origins[i*3+j];
                if (o < ixMin[j]) ixMin[j] = o;
                if (o + grid.block_size > ixMax[j]) ixMax[j] = o + grid.block_size;
            }
        }
        // offset = world_min - index_min * voxelSize
        double gridOffset[3];
        for (int j = 0; j < 3; ++j)
            gridOffset[j] = grid.bbox_min[j] - ixMin[j] * grid.voxel_size[j];

        if (grid.codebook_is_fp16) {
            // CVD6: FP16 codebook — direct upload, no CPU conversion
            if (!_cudaRenderer->uploadCodebookVolumeFP16(
                    grid.codebook_fp16.data(), grid.codebook_k,
                    grid.indices.data(), grid.origins.data(), grid.num_leaves,
                    grid.has_gain ? grid.gain_maps.data() : nullptr,
                    grid.block_size, grid.voxel_size[0],
                    grid.bbox_min, grid.bbox_max, gridOffset, grid.norm_scale)) {
                error("CVDB: Density FP16 upload failed: %s", _cudaRenderer->lastError().c_str());
                return false;
            }
        } else {
            if (!_cudaRenderer->uploadCodebookVolume(
                    grid.codebook.data(), grid.codebook_k,
                    grid.indices.data(), grid.origins.data(), grid.num_leaves,
                    grid.has_gain ? grid.gain_maps.data() : nullptr,
                    grid.has_residual ? grid.residuals.data() : nullptr,
                    grid.block_size, grid.voxel_size[0],
                    grid.bbox_min, grid.bbox_max, gridOffset, grid.norm_scale)) {
                error("CVDB: Density upload failed: %s", _cudaRenderer->lastError().c_str());
                return false;
            }
        }

        // Keep WORLD-SPACE bbox for viewport drawing and screen projection
        for (int i = 0; i < 3; ++i) {
            _bboxMin[i] = grid.bbox_min[i];
            _bboxMax[i] = grid.bbox_max[i];
        }

        // CPU cache for viewport
        if (grid.codebook_is_fp16) {
            // Convert FP16→FP32 for viewport point cloud (small, one-time)
            size_t n = grid.codebook_fp16.size();
            _cvdbCpu.codebook.resize(n);
            for (size_t i = 0; i < n; ++i) {
                uint16_t h = grid.codebook_fp16[i];
                uint32_t s=(h&0x8000u)<<16; uint32_t e=(h>>10)&0x1F; uint32_t m=h&0x3FF;
                uint32_t f; if(e==0)f=s; else if(e==31)f=s|0x7F800000u|(m<<13); else f=s|((e+112)<<23)|(m<<13);
                _cvdbCpu.codebook[i]=*reinterpret_cast<float*>(&f);
            }
        } else {
            _cvdbCpu.codebook = grid.codebook;
        }
        _cvdbCpu.indices   = grid.indices;
        _cvdbCpu.origins   = grid.origins;
        _cvdbCpu.K         = grid.codebook_k;
        _cvdbCpu.numLeaves = grid.num_leaves;
        _cvdbCpu.blockSize = grid.block_size;
        _cvdbCpu.voxelSize = grid.voxel_size[0];
        _cvdbCpu.hasGain   = grid.has_gain;
        if (grid.has_gain) _cvdbCpu.gain_maps = grid.gain_maps;
        else               _cvdbCpu.gain_maps.clear();
        for (int j = 0; j < 3; ++j) _cvdbCpu.gridOffset[j] = float(gridOffset[j]);
        _previewPoints.clear();
    }

    // ── Upload emission (if found) ──
    if (emissionIdx >= 0) {
        const CVDBGrid& grid = cvdb.grids[emissionIdx];
        // Compute emission grid offset (same approach as density)
        double emIxMin[3]={1e30,1e30,1e30};
        for (uint32_t i = 0; i < grid.num_leaves; ++i)
            for (int j = 0; j < 3; ++j)
                if (grid.origins[i*3+j] < emIxMin[j]) emIxMin[j] = grid.origins[i*3+j];
        double emOffset[3];
        for (int j = 0; j < 3; ++j) emOffset[j] = grid.bbox_min[j] - emIxMin[j] * grid.voxel_size[j];
        if (grid.codebook_is_fp16) {
            _cudaRenderer->uploadEmissionVolumeFP16(
                grid.codebook_fp16.data(), grid.codebook_k,
                grid.indices.data(), grid.origins.data(), grid.num_leaves,
                grid.has_gain ? grid.gain_maps.data() : nullptr,
                grid.block_size, grid.voxel_size[0], emOffset, grid.norm_scale);
        } else {
            _cudaRenderer->uploadEmissionVolume(
                grid.codebook.data(), grid.codebook_k,
                grid.indices.data(), grid.origins.data(), grid.num_leaves,
                grid.has_gain ? grid.gain_maps.data() : nullptr,
                grid.block_size, grid.voxel_size[0], emOffset, grid.norm_scale);
        }
    } else {
        _cudaRenderer->clearEmission();
    }

    // ── Upload temperature (if found) ──
    if (temperatureIdx >= 0) {
        const CVDBGrid& grid = cvdb.grids[temperatureIdx];
        double tmpIxMin[3]={1e30,1e30,1e30};
        for (uint32_t i = 0; i < grid.num_leaves; ++i)
            for (int j = 0; j < 3; ++j)
                if (grid.origins[i*3+j] < tmpIxMin[j]) tmpIxMin[j] = grid.origins[i*3+j];
        double tmpOffset[3];
        for (int j = 0; j < 3; ++j) tmpOffset[j] = grid.bbox_min[j] - tmpIxMin[j] * grid.voxel_size[j];
        if (grid.codebook_is_fp16) {
            _cudaRenderer->uploadTemperatureVolumeFP16(
                grid.codebook_fp16.data(), grid.codebook_k,
                grid.indices.data(), grid.origins.data(), grid.num_leaves,
                grid.has_gain ? grid.gain_maps.data() : nullptr,
                grid.block_size, grid.voxel_size[0], tmpOffset, grid.norm_scale);
        } else {
            _cudaRenderer->uploadTemperatureVolume(
                grid.codebook.data(), grid.codebook_k,
                grid.indices.data(), grid.origins.data(), grid.num_leaves,
                grid.has_gain ? grid.gain_maps.data() : nullptr,
                grid.block_size, grid.voxel_size[0], tmpOffset, grid.norm_scale);
        }
    } else {
        _cudaRenderer->clearTemperature();
    }

    _cudaVolumeLoadedPath  = cleanPath;
    _cudaVolumeLoadedFrame = curFrame;
    _gridValid = true;
    _gpuFrameReady = false;

    double diag = std::sqrt(
        (_bboxMax[0]-_bboxMin[0])*(_bboxMax[0]-_bboxMin[0]) +
        (_bboxMax[1]-_bboxMin[1])*(_bboxMax[1]-_bboxMin[1]) +
        (_bboxMax[2]-_bboxMin[2])*(_bboxMax[2]-_bboxMin[2]));
    std::printf("[VDBRender] Ready: %zu grid%s, bbox %.0f, emission=%s, temperature=%s\n",
        cvdb.grids.size(), cvdb.grids.size()>1?"s":"", diag,
        emissionIdx >= 0 ? "yes" : "no",
        temperatureIdx >= 0 ? "yes" : "no");
    return true;
}
