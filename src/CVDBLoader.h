/*
 * CVDBLoader.h — Header-only parser for .cvdb (Codebook VDB) files.
 *
 * Format versions:
 *   CVD1 — 8³ blocks, plain codebook
 *   CVD2 — 8³ blocks + gain maps
 *   CVD3 — 8³ blocks + gain maps + 6³ residuals
 *   CVD4 — 4³ blocks + gain maps
 *   CVD5 — 4³ blocks + gain maps + per-grid norm_scale (normalized to 0-1)
 *   CVD6 — 4³ blocks + gain maps + norm_scale + FP16 codebook (half disk size)
 *
 * Per-grid header (CVD5 adds norm_scale after bbox_max):
 *   name: char[64]
 *   num_leaves: uint32
 *   codebook_k: uint32
 *   index_bytes: uint32 (1 or 2)
 *   voxel_size: float[3]
 *   bbox_min: double[3]
 *   bbox_max: double[3]
 *   norm_scale: float        ← CVD5 only (multiply codebook values by this)
 *   codebook: float[K * vpb]
 *   indices: uint8[N] or uint16[N]
 *   origins: int32[N * 3]
 *   gain_maps: float[N * 8]  ← CVD2/3/4/5
 *   residuals: float[N * 216] ← CVD3 only
 */

#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct CVDBGrid {
    std::string           name;
    uint32_t              num_leaves;
    uint32_t              codebook_k;
    uint32_t              index_bytes;  // 1 or 2
    float                 voxel_size[3];
    double                bbox_min[3];
    double                bbox_max[3];
    float                 norm_scale = 1.0f;  // CVD5/6: multiply codebook by this to restore values
    std::vector<float>    codebook;     // K * vpb floats — populated for CVD1-5
    std::vector<uint16_t> codebook_fp16;// K * vpb raw FP16 — populated for CVD6 only
    bool                  codebook_is_fp16 = false;
    std::vector<uint16_t> indices;      // N indices (widened to uint16 even if stored as uint8)
    std::vector<int32_t>  origins;      // N * 3 int32
    std::vector<float>    gain_maps;    // N * 8 floats (2³ per leaf) — empty if CVD1
    std::vector<float>    residuals;    // N * 216 floats — empty if CVD1/CVD2/CVD4/CVD5
    bool                  has_gain = false;
    bool                  has_residual = false;
    int                   block_size = 8;  // 8 for CVD1/2/3, 4 for CVD4/5/6
    int                   voxels_per_block = 512;
};

struct CVDBFile {
    std::vector<CVDBGrid> grids;

    static CVDBFile load(const std::string& path)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open())
            throw std::runtime_error("Cannot open " + path);

        char magic[4];
        f.read(magic, 4);
        bool isCVD1 = (magic[0]=='C' && magic[1]=='V' && magic[2]=='D' && magic[3]=='1');
        bool isCVD2 = (magic[0]=='C' && magic[1]=='V' && magic[2]=='D' && magic[3]=='2');
        bool isCVD3 = (magic[0]=='C' && magic[1]=='V' && magic[2]=='D' && magic[3]=='3');
        bool isCVD4 = (magic[0]=='C' && magic[1]=='V' && magic[2]=='D' && magic[3]=='4');
        bool isCVD5 = (magic[0]=='C' && magic[1]=='V' && magic[2]=='D' && magic[3]=='5');
        bool isCVD6 = (magic[0]=='C' && magic[1]=='V' && magic[2]=='D' && magic[3]=='6');
        if (!isCVD1 && !isCVD2 && !isCVD3 && !isCVD4 && !isCVD5 && !isCVD6)
            throw std::runtime_error("Invalid .cvdb magic in " + path);

        uint32_t numGrids;
        f.read(reinterpret_cast<char*>(&numGrids), 4);

        CVDBFile result;
        result.grids.resize(numGrids);

        for (uint32_t g = 0; g < numGrids; ++g) {
            auto& grid = result.grids[g];

            char name[64];
            f.read(name, 64);
            grid.name = std::string(name);

            f.read(reinterpret_cast<char*>(&grid.num_leaves), 4);
            f.read(reinterpret_cast<char*>(&grid.codebook_k), 4);
            f.read(reinterpret_cast<char*>(&grid.index_bytes), 4);
            f.read(reinterpret_cast<char*>(grid.voxel_size), 12);
            f.read(reinterpret_cast<char*>(grid.bbox_min), 24);
            f.read(reinterpret_cast<char*>(grid.bbox_max), 24);

            // CVD5/CVD6: read per-grid normalization scale
            if (isCVD5 || isCVD6) {
                f.read(reinterpret_cast<char*>(&grid.norm_scale), 4);
            } else {
                grid.norm_scale = 1.0f;
            }

            // Block size: 4 for CVD4/CVD5/CVD6, 8 for all others
            grid.block_size = (isCVD4 || isCVD5 || isCVD6) ? 4 : 8;
            grid.voxels_per_block = grid.block_size * grid.block_size * grid.block_size;

            // Codebook
            if (isCVD6) {
                // CVD6: codebook stored as FP16 (half the bytes)
                size_t n = grid.codebook_k * grid.voxels_per_block;
                grid.codebook_fp16.resize(n);
                f.read(reinterpret_cast<char*>(grid.codebook_fp16.data()), n * sizeof(uint16_t));
                grid.codebook_is_fp16 = true;
            } else {
                grid.codebook.resize(grid.codebook_k * grid.voxels_per_block);
                f.read(reinterpret_cast<char*>(grid.codebook.data()),
                       grid.codebook_k * grid.voxels_per_block * sizeof(float));
            }

            // Indices
            grid.indices.resize(grid.num_leaves);
            if (grid.index_bytes == 1) {
                std::vector<uint8_t> idx8(grid.num_leaves);
                f.read(reinterpret_cast<char*>(idx8.data()), grid.num_leaves);
                for (uint32_t i = 0; i < grid.num_leaves; ++i)
                    grid.indices[i] = idx8[i];
            } else {
                f.read(reinterpret_cast<char*>(grid.indices.data()),
                       grid.num_leaves * sizeof(uint16_t));
            }

            // Origins
            grid.origins.resize(grid.num_leaves * 3);
            f.read(reinterpret_cast<char*>(grid.origins.data()),
                   grid.num_leaves * 3 * sizeof(int32_t));

            // Gain maps (CVD2/CVD3/CVD4/CVD5/CVD6)
            if (isCVD2 || isCVD3 || isCVD4 || isCVD5 || isCVD6) {
                grid.gain_maps.resize(grid.num_leaves * 8);
                f.read(reinterpret_cast<char*>(grid.gain_maps.data()),
                       grid.num_leaves * 8 * sizeof(float));
                grid.has_gain = true;
            }

            // Residuals (CVD3 only — 6³ per leaf)
            if (isCVD3) {
                grid.residuals.resize(grid.num_leaves * 216);
                f.read(reinterpret_cast<char*>(grid.residuals.data()),
                       grid.num_leaves * 216 * sizeof(float));
                grid.has_residual = true;
            }
        }

        return result;
    }
};
