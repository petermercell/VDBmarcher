#pragma once
// ═══════════════════════════════════════════════════════════════════════════
// VDBEnvironment.h — L2 Spherical Harmonics for HDRI environment lighting
//
// Projects an equirectangular HDRI (passed as float RGB pixels) into 9×3
// real SH coefficients. Extracts peak direction from L1 band.
// Supports Y-axis rotation of SH coefficients (CPU-side, no re-projection).
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdint>

struct EnvSH {
    float coeffs[27];          // 9 SH basis × 3 RGB channels (L0, L1, L2)
    float peakDirection[3];    // dominant light direction (from L1 band)
    float peakColor[3];        // evaluated SH color in peak direction
    bool  valid = false;
};

// Project equirectangular HDRI pixels into L2 SH coefficients.
// pixelsRGB: row-major, bottom-to-top (Nuke convention), 3 floats per pixel.
// width/height: pixel dimensions of the environment map.
// Runs once per input change (~5-50ms for typical resolutions).
EnvSH projectHDRIFromPixels(const float* pixelsRGB, int width, int height);

// Rotate SH coefficients around Y axis (in-place).
// angleDeg: rotation in degrees. Cheap (~30 multiplies).
void rotateSH_Y(float coeffs[27], float angleDeg);

// Evaluate L2 SH at a given direction (unit normal).
// Returns RGB color. Used CPU-side for peak color extraction.
void evalSH(const float coeffs[27], float nx, float ny, float nz,
            float& r, float& g, float& b);
