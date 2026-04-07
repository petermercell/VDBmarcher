// ═══════════════════════════════════════════════════════════════════════════
// VDBDeepEngine.cpp — Deep output from dual Iop/DeepOp VDBRenderIop
//
// DeepOp interface: doDeepEngine, getDeepRequests, renderDeepFrame.
// Reuses the same VDBCUDARenderer and grid data as the flat Iop path.
//
// Depth conversion: the GPU kernel stores ray-t (distance along normalized
// ray in volume-local space). doDeepEngine converts to camera-Z depth
// (distance along camera forward axis) so deep compositing with Arnold,
// RenderMan, etc. works correctly.
// ═══════════════════════════════════════════════════════════════════════════

#include "VDBRenderIop.h"
#include "VDBCUDARenderer.h"
#include <DDImage/Format.h>
#include <DDImage/Channel.h>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <algorithm>

using namespace DD::Image;

// ─── getDeepRequests — empty, we are the source ─────────────────────────

void VDBRenderIop::getDeepRequests(Box box, const ChannelSet& channels,
                                    int count,
                                    std::vector<RequestData>& requests)
{
    // No deep input to request from — we generate all samples.
}

// ─── Volume forward transform (local → world) ──────────────────────────
// Needed for depth conversion: ray-t is in local space, we need camera-Z.

static void buildVolumeForwardTransform(const double volTranslate[3],
                                         const double volRotate[3],
                                         const double volScale[3],
                                         double volUniformScale,
                                         float out[16])
{
    const double deg2rad = 3.14159265358979323846 / 180.0;
    const double rx = volRotate[0]*deg2rad, ry = volRotate[1]*deg2rad, rz = volRotate[2]*deg2rad;
    const double sx = volScale[0]*volUniformScale, sy = volScale[1]*volUniformScale, sz = volScale[2]*volUniformScale;
    const double crx=cos(rx),srx=sin(rx),cry=cos(ry),sry=sin(ry),crz=cos(rz),srz=sin(rz);
    double R[3][3];
    R[0][0]=cry*crz; R[0][1]=srx*sry*crz-crx*srz; R[0][2]=crx*sry*crz+srx*srz;
    R[1][0]=cry*srz; R[1][1]=srx*sry*srz+crx*crz; R[1][2]=crx*sry*srz-srx*crz;
    R[2][0]=-sry;    R[2][1]=srx*cry;              R[2][2]=crx*cry;
    // Forward = Translate * Rotate * Scale
    out[0] =float(R[0][0]*sx); out[1] =float(R[1][0]*sx); out[2] =float(R[2][0]*sx); out[3]=0;
    out[4] =float(R[0][1]*sy); out[5] =float(R[1][1]*sy); out[6] =float(R[2][1]*sy); out[7]=0;
    out[8] =float(R[0][2]*sz); out[9] =float(R[1][2]*sz); out[10]=float(R[2][2]*sz); out[11]=0;
    out[12]=float(volTranslate[0]); out[13]=float(volTranslate[1]); out[14]=float(volTranslate[2]); out[15]=1;
}

// Transform a direction by a 4x4 matrix (no translation)
static inline void xformDir(const float m[16], float dx, float dy, float dz,
                             float& ox, float& oy, float& oz)
{
    ox = m[0]*dx + m[4]*dy + m[8]*dz;
    oy = m[1]*dx + m[5]*dy + m[9]*dz;
    oz = m[2]*dx + m[6]*dy + m[10]*dz;
}

// ─── renderDeepFrame — GPU deep render using shared renderer ────────────

bool VDBRenderIop::renderDeepFrame(int w, int h)
{
    if (!_cudaRenderer || !_camValid || !_gridValid) return false;

    const Format& fmt = format();
    int fmtW = fmt.width(), fmtH = fmt.height();

    VDBRenderConfig cfg = buildRenderConfig(fmtW, fmtH);

    // Override for deep: full resolution, no AOVs, stride=4
    cfg.renderW = w;
    cfg.renderH = h;
    cfg.fmtW    = w;
    cfg.fmtH    = h;
    cfg.bboxOffsetX = 0;
    cfg.bboxOffsetY = 0;
    cfg.aovFlags     = 0;
    cfg.outputStride = 4;

    // Allocate host buffers
    size_t pixelCount = size_t(w) * h;
    int maxS = _deepMaxSamples;
    _deepBuf.resize(pixelCount * maxS * 3);
    _deepCounts.resize(pixelCount);

    std::printf("[VDBRender] Deep: %dx%d, %d samples (%.0f MB)\n",
        w, h, maxS,
        pixelCount * maxS * 3 * sizeof(uint16_t) / (1024.0 * 1024.0));

    bool ok = _cudaRenderer->renderDeep(cfg, maxS, _deepBuf.data(), _deepCounts.data());

    if (!ok) {
        std::printf("[VDBRender] Deep render failed: %s\n", _cudaRenderer->lastError().c_str());
        return false;
    }

    _deepRenderedW = w;
    _deepRenderedH = h;
    _deepReady = true;

    // Quick stats
    int maxCount = 0, totalSamples = 0;
    for (size_t i = 0; i < pixelCount; ++i) {
        int n = _deepCounts[i];
        totalSamples += n;
        if (n > maxCount) maxCount = n;
    }
    std::printf("[VDBRender] Deep OK: max=%d/px, avg=%.1f\n",
        maxCount, double(totalSamples) / pixelCount);

    return true;
}

// ─── doDeepEngine — fill DeepOutputPlane from GPU deep buffer ───────────
//
// The GPU kernel stores ray-t: distance along the normalized ray in
// volume-local space. For Nuke deep compositing (and compatibility with
// Arnold/RenderMan deep EXRs), we convert to camera-Z depth:
//
//   camera_z = t * dot(fwdTransform * rd_local, camForward)
//
// This factor is computed per-pixel and applied to all samples of that pixel.

bool VDBRenderIop::doDeepEngine(Box box, const ChannelSet& channels,
                                 DeepOutputPlane& plane)
{
    if (!_deepEnable) return false;

    const Format* fmt = _formats.format();
    if (!fmt) return false;
    int W = fmt->width(), H = fmt->height();

    // Render full frame on first call (cached for subsequent tile calls)
    if (!_deepReady) {
        std::lock_guard<std::mutex> lock(_deepRenderMtx);
        if (!_deepReady) {
            if (!renderDeepFrame(W, H))
                return false;
        }
    }

    int maxS = _deepMaxSamples;

    // ── Precompute depth conversion matrices ──
    // Volume forward transform (local → world)
    float fwd[16];
    buildVolumeForwardTransform(_volTranslate, _volRotate, _volScale, _volUniformScale, fwd);

    // Volume inverse transform (world → local, same as kernel)
    float inv[16];
    buildVolumeTransform(inv);

    // Camera rotation (row-major, kernel layout)
    float cR[9];
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r)
            cR[c * 3 + r] = float(_camRot[c][r]);

    float hW = float(_halfW);
    float hH = float(_halfW * double(H) / double(W));

    // Build output plane
    DeepInPlaceOutputPlane outPlane(channels, box, DeepPixel::eZAscending);

    // Count total samples for reservation
    size_t totalSamples = 0;
    for (Box::iterator it = box.begin(); it != box.end(); ++it) {
        int px = it.x, py = it.y;
        if (px >= 0 && px < W && py >= 0 && py < H) {
            int n = _deepCounts[py * W + px];
            if (n > 0) totalSamples += n;
        }
    }
    outPlane.reserveSamples(totalSamples);

    // Write samples with depth conversion
    for (Box::iterator it = box.begin(); it != box.end(); ++it) {
        int px = it.x, py = it.y;

        if (px < 0 || px >= W || py < 0 || py >= H) {
            outPlane.setSampleCount(it, 0);
            continue;
        }

        int pidx = py * W + px;
        int n = _deepCounts[pidx];

        if (n <= 0) {
            outPlane.setSampleCount(it, 0);
            continue;
        }

        // ── Per-pixel ray-t → camera distance conversion ──
        // Deep front/back must be camera distance (positive) to match
        // ScanlineRender deep output for correct DeepMerge sorting.
        //
        // For each pixel: worldPos = fwdTransform * (localOrig + localDir * t)
        // cameraZ = -dot(worldPos - camOrigin, camZAxis)
        //
        // Since localOrig IS the camera (transformed to local space),
        // the camera distance of the ray origin is 0, so:
        //   cameraDist = t * dDist_dt
        // where dDist_dt = -dot(fwd * rd_local, camZAxis)

        float u = (px + 0.5f) / W * 2.f - 1.f;
        float v = (py + 0.5f) / H * 2.f - 1.f;

        // Camera-local direction (matches kernel: u*halfW, v*halfH, -1)
        float ldx = u * hW, ldy = v * hH, ldz = -1.f;

        // World direction: camRot * ld
        float wdx = cR[0]*ldx + cR[3]*ldy + cR[6]*ldz;
        float wdy = cR[1]*ldx + cR[4]*ldy + cR[7]*ldz;
        float wdz = cR[2]*ldx + cR[5]*ldy + cR[8]*ldz;

        // Local direction: invTransform * wd (direction, no translation)
        float rdx, rdy, rdz;
        xformDir(inv, wdx, wdy, wdz, rdx, rdy, rdz);

        // Normalize (same as kernel)
        float rdLen = std::sqrt(rdx*rdx + rdy*rdy + rdz*rdz);
        if (rdLen > 1e-8f) { float il = 1.f/rdLen; rdx *= il; rdy *= il; rdz *= il; }

        // World direction per unit t: fwdTransform * rd_local
        float wdpt_x, wdpt_y, wdpt_z;
        xformDir(fwd, rdx, rdy, rdz, wdpt_x, wdpt_y, wdpt_z);

        // Camera Z axis = column 2 of rotation matrix (indices 6,7,8)
        float camZx = cR[6], camZy = cR[7], camZz = cR[8];
        // Camera distance per unit t: project world-ray-direction onto camera Z
        float dDist_dt = -(wdpt_x * camZx + wdpt_y * camZy + wdpt_z * camZz);
        if (std::abs(dDist_dt) < 1e-8f) dDist_dt = 1.f;  // degenerate ray safety

        outPlane.setSampleCount(it, n);
        DeepOutputPixel outPixel = outPlane.getPixel(it);

        int base = pidx * maxS * 3;

        for (int s = 0; s < n; ++s) {
            int off = base + s * 3;
            float t_front = halfToFloat(_deepBuf[off + 0]);
            float t_back  = halfToFloat(_deepBuf[off + 1]);
            float a       = halfToFloat(_deepBuf[off + 2]);

            // Convert ray-t to camera distance
            float front = t_front * dDist_dt;
            float back  = t_back  * dDist_dt;

            foreach (z, channels) {
                float& out = outPixel.getWritableOrderedSample(s, z);
                if      (z == Chan_DeepFront) out = front;
                else if (z == Chan_DeepBack)  out = back;
                else if (z == Chan_Alpha)     out = a;
                else if (z == Chan_Red)       out = 0.0f;
                else if (z == Chan_Green)     out = 0.0f;
                else if (z == Chan_Blue)      out = 0.0f;
                else                          out = 0.0f;
            }
        }
    }

    mFnAssert(outPlane.isComplete());
    plane = outPlane;
    return true;
}
