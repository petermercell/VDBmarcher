// ═══════════════════════════════════════════════════════════════════════════
// VDBEnvironment.cpp — L2 Spherical Harmonics projection & rotation
// ═══════════════════════════════════════════════════════════════════════════

#include "VDBEnvironment.h"
#include <cmath>
#include <cstring>

// Real SH basis constants (with Condon-Shortley phase)
static constexpr float kY00  = 0.282095f;   // 1/(2√π)
static constexpr float kY1n1 = 0.488603f;   // √(3/(4π))
static constexpr float kY10  = 0.488603f;
static constexpr float kY11  = 0.488603f;
static constexpr float kY2n2 = 1.092548f;   // √(15/(4π))
static constexpr float kY2n1 = 1.092548f;
static constexpr float kY20  = 0.315392f;   // √(5/(16π))
static constexpr float kY21  = 1.092548f;
static constexpr float kY22  = 0.546274f;   // √(15/(16π))

void evalSH(const float coeffs[27], float nx, float ny, float nz,
            float& r, float& g, float& b)
{
    float basis[9];
    basis[0] = kY00;
    basis[1] = kY1n1 * ny;
    basis[2] = kY10  * nz;
    basis[3] = kY11  * nx;
    basis[4] = kY2n2 * nx * ny;
    basis[5] = kY2n1 * ny * nz;
    basis[6] = kY20  * (3.0f * nz * nz - 1.0f);
    basis[7] = kY21  * nx * nz;
    basis[8] = kY22  * (nx * nx - ny * ny);

    r = g = b = 0.0f;
    for (int i = 0; i < 9; ++i) {
        r += coeffs[i]      * basis[i];
        g += coeffs[9 + i]  * basis[i];
        b += coeffs[18 + i] * basis[i];
    }
}

EnvSH projectHDRIFromPixels(const float* pixelsRGB, int width, int height)
{
    EnvSH result;
    std::memset(&result, 0, sizeof(result));

    if (!pixelsRGB || width < 2 || height < 2)
        return result;

    const double PI = 3.14159265358979323846;
    double sh[27] = {};

    for (int y = 0; y < height; ++y) {
        // Nuke: y=0 is bottom. Equirect: bottom = south pole (θ=π), top = north (θ=0).
        double theta = PI * (1.0 - (y + 0.5) / height);
        double sinTheta = std::sin(theta);
        double cosTheta = std::cos(theta);
        double solidAngle = (2.0 * PI / width) * (PI / height) * sinTheta;

        for (int x = 0; x < width; ++x) {
            double phi = 2.0 * PI * (x + 0.5) / width;
            float nx = float(sinTheta * std::cos(phi));
            float ny = float(cosTheta);
            float nz = float(sinTheta * std::sin(phi));

            // Mirror flip H+V to match Nuke's HDRI orientation
            int idx = ((height - 1 - y) * width + (width - 1 - x)) * 3;
            float pr = pixelsRGB[idx + 0];
            float pg = pixelsRGB[idx + 1];
            float pb = pixelsRGB[idx + 2];

            float basis[9];
            basis[0] = kY00;
            basis[1] = kY1n1 * ny;
            basis[2] = kY10  * nz;
            basis[3] = kY11  * nx;
            basis[4] = kY2n2 * nx * ny;
            basis[5] = kY2n1 * ny * nz;
            basis[6] = kY20  * (3.0f * nz * nz - 1.0f);
            basis[7] = kY21  * nx * nz;
            basis[8] = kY22  * (nx * nx - ny * ny);

            double w = solidAngle;
            for (int i = 0; i < 9; ++i) {
                double bw = basis[i] * w;
                sh[i]      += pr * bw;
                sh[9 + i]  += pg * bw;
                sh[18 + i] += pb * bw;
            }
        }
    }

    for (int i = 0; i < 27; ++i)
        result.coeffs[i] = float(sh[i]);

    // Peak direction from L1 (luminance-weighted)
    float dx = 0.2126f * result.coeffs[3]  + 0.7152f * result.coeffs[12] + 0.0722f * result.coeffs[21];
    float dy = 0.2126f * result.coeffs[1]  + 0.7152f * result.coeffs[10] + 0.0722f * result.coeffs[19];
    float dz = 0.2126f * result.coeffs[2]  + 0.7152f * result.coeffs[11] + 0.0722f * result.coeffs[20];
    float dlen = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (dlen > 1e-8f) {
        result.peakDirection[0] = dx / dlen;
        result.peakDirection[1] = dy / dlen;
        result.peakDirection[2] = dz / dlen;
    } else {
        result.peakDirection[0] = 0.0f;
        result.peakDirection[1] = 1.0f;
        result.peakDirection[2] = 0.0f;
    }

    evalSH(result.coeffs,
           result.peakDirection[0], result.peakDirection[1], result.peakDirection[2],
           result.peakColor[0], result.peakColor[1], result.peakColor[2]);

    result.valid = true;
    return result;
}

// ── Rotate L2 SH coefficients around Y axis ────────────────────────────
//
// Derived from direct polynomial substitution under Y-rotation:
//   x → cosβ·x + sinβ·z,  y → y,  z → -sinβ·x + cosβ·z
// with normalization correction factors N[m]/N[m'] between SH bands.
// The unnormalized polynomial rotation matrix D is computed from the
// substitutions; the normalized coefficient rotation is:
//   c'[m] = Σ_{m'} D[m,m'] · (N[m]/N[m']) · c[m']
//
// Numerically verified against point-light rotation test cases:
//   Light at (0,0,1) rotated π/4 → (0.707,0,0.707), all 5 L2 coefficients match.
//
void rotateSH_Y(float coeffs[27], float angleDeg)
{
    if (std::abs(angleDeg) < 1e-6f) return;

    const double PI = 3.14159265358979323846;
    double rad = angleDeg * PI / 180.0;
    float c  = float(std::cos(rad));
    float s  = float(std::sin(rad));
    float c2 = float(std::cos(2.0 * rad));
    float s2 = float(std::sin(2.0 * rad));
    const float r3 = 1.7320508f;  // √3

    for (int ch = 0; ch < 3; ++ch) {
        int o = ch * 9;

        // L0: index 0 — isotropic, unchanged

        // L1: {1=y, 2=z, 3=x}. Y unchanged; z and x mix.
        float z1 = coeffs[o + 2];
        float x1 = coeffs[o + 3];
        coeffs[o + 2] =  c * z1 - s * x1;
        coeffs[o + 3] =  s * z1 + c * x1;

        // L2: {4=xy, 5=yz, 6=3z²-1, 7=xz, 8=x²-y²}
        float m0 = coeffs[o + 4];
        float m1 = coeffs[o + 5];
        float m2 = coeffs[o + 6];
        float m3 = coeffs[o + 7];
        float m4 = coeffs[o + 8];

        // xy and yz: same normalization → simple cosβ/sinβ rotation
        coeffs[o + 4] =  c * m0 + s * m1;
        coeffs[o + 5] = -s * m0 + c * m1;

        // {3z²-1, xz, x²-y²}: cross-mix via cos2β/sin2β with norm corrections
        coeffs[o + 6] = (1.0f + 3.0f * c2) * 0.25f * m2
                       - r3 * s2 * 0.5f * m3
                       + r3 * (1.0f - c2) * 0.25f * m4;

        coeffs[o + 7] = r3 * s2 * 0.5f * m2
                       + c2 * m3
                       - s2 * 0.5f * m4;

        coeffs[o + 8] = r3 * (1.0f - c2) * 0.25f * m2
                       + s2 * 0.5f * m3
                       + (3.0f + c2) * 0.25f * m4;
    }
}
