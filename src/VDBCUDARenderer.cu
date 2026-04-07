#include "VDBCUDARenderer.h"
#include "VDBProfiler.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <climits>
#include <vector>

// ═══════════════════════════════════════════════════════════════════════════
// Kernel structs
// ═══════════════════════════════════════════════════════════════════════════

// Pointers for one codebook grid — passed to kernel
struct GridPtrs {
    const int64_t*  hashKeys;
    const uint32_t* hashVals;
    uint32_t        hashMask;
    const uint16_t* indices;
    const float*    codebook;
    const __half*   codebookHalf;
    const float*    gainMaps;
    const float*    residuals;
    int  blockShift, blkDim, vpb;
    float invVoxelSize, voxelSize;
    float gridOffset[3];  // world = index * voxelSize + offset
    float normScale;      // multiply codebook values by this to restore original range
    // Coarse occupancy macro-grid
    const uint8_t* macroGrid;
    int macroGridDims[3], macroOrigin[3], macroCellShift;
};

struct RayMarchParams {
    int   width, height, fmtWidth, fmtHeight, bboxOffsetX, bboxOffsetY;
    float camOrigin[3], camRot[9], halfW, halfH;
    float stepSize, extinction, scattering;
    float extinctionR, extinctionG, extinctionB;  // chromatic extinction (Hillaire 2015)
    int   numLights;
    GPULight lights[3];
    float invTransform[16], bboxMin[3], bboxMax[3];
    int   shadowSteps; float shadowStepScale;
    float emissionScale, temperatureScale;
    float rampColors[15];   // 5 stops × RGB
    int   hasEmission, hasTemperature;
    int   useBlackbody;  float bbKelvinScale;  float bbMix;  float bbTint[3];
    int   aovFlags, outputStride;
    float ambientColor[3]; float ambientIntensity;
    // ── Environment SH ──
    float envSH[27];
    int   envSHEnable;
    float envIntensity;
    float envPeakDir[3];
    float envPeakColor[3];
    int   envShadowEnable;
    float envShadowLift[3];
    float boundaryBlend;
    float phaseG1, phaseG2, phaseMix;  // dual-lobe Henyey-Greenstein
    int   phaseMode;                    // 0=Iso, 1=HG, 2=Dual HG, 3=Mie
    float emSelfIllum;                 // emission self-illumination scale
    int   emSelfIllumPhase;            // 0=isotropic, 1=gradient-directed HG
    float powderStrength;              // Schneider & Vos 2015 powder effect (0=off)
    int   msEnable, msOctaves;         // Frostbite multi-scatter octaves (Hillaire 2016)
    float msExtFalloff, msScatterFalloff;
    float gradientMix, gradientThreshold; // gradient normals Lambertian blend
    float gradNormalBias[3];              // additive normal direction bias
    float gradientSmooth;                 // FD step size in voxels
    float scatterSmooth;                  // density smooth for scattering
    float densityBlur;                    // raw density pre-blur radius
    float noiseAmplitude, noiseFrequency, noiseScale[3], noiseOffset[3]; int noiseOctaves; // procedural fBm
    float fwdTransform[16];  // volume local→world (for depth+P AOV)
    int   tileYOffset; // first global row of this tile (0 = no tiling)
    int   tileHeight;  // rows in this tile (== height when not tiling)

    // Deep output mode
    int   deepMode;          // 0 = flat only, 1 = also write deep samples
    int   maxDeepSamples;    // ring buffer size per pixel

    // Light Mixer (RGBA beauty only, AOVs stay 100%)
    float lightMix[3];
    float envMix;
};

// ═══════════════════════════════════════════════════════════════════════════
// Device helpers
// ═══════════════════════════════════════════════════════════════════════════

__device__ inline float3 make_f3(float x,float y,float z){return make_float3(x,y,z);}
__device__ inline float3 transformPoint(const float m[16],float3 p){
    return make_f3(m[0]*p.x+m[4]*p.y+m[8]*p.z+m[12],m[1]*p.x+m[5]*p.y+m[9]*p.z+m[13],m[2]*p.x+m[6]*p.y+m[10]*p.z+m[14]);}
__device__ inline float3 transformDir(const float m[16],float3 d){
    return make_f3(m[0]*d.x+m[4]*d.y+m[8]*d.z,m[1]*d.x+m[5]*d.y+m[9]*d.z,m[2]*d.x+m[6]*d.y+m[10]*d.z);}
__device__ inline float3 normalize3(float3 v){float l=sqrtf(v.x*v.x+v.y*v.y+v.z*v.z);float i=(l>1e-8f)?1.f/l:0.f;return make_f3(v.x*i,v.y*i,v.z*i);}
__device__ inline bool intersectAABB(float3 o,float3 d,float3 mn,float3 mx,float&tE,float&tX){
    float3 id=make_f3(1.f/d.x,1.f/d.y,1.f/d.z);
    float3 t0=make_f3((mn.x-o.x)*id.x,(mn.y-o.y)*id.y,(mn.z-o.z)*id.z);
    float3 t1=make_f3((mx.x-o.x)*id.x,(mx.y-o.y)*id.y,(mx.z-o.z)*id.z);
    float3 lo=make_f3(fminf(t0.x,t1.x),fminf(t0.y,t1.y),fminf(t0.z,t1.z));
    float3 hi=make_f3(fmaxf(t0.x,t1.x),fmaxf(t0.y,t1.y),fmaxf(t0.z,t1.z));
    tE=fmaxf(fmaxf(lo.x,lo.y),fmaxf(lo.z,0.f));tX=fminf(fminf(hi.x,hi.y),hi.z);return tE<tX;}

__device__ inline float3 rampColor(float t, const float ramp[15]){
    t=fmaxf(0.f,fminf(t,1.f)); float idx=t*4.f; int lo=min(int(idx),3); int hi=lo+1; float f=idx-lo;
    int a=lo*3,b=hi*3;
    return make_f3(ramp[a]+( ramp[b]-ramp[a])*f, ramp[a+1]+(ramp[b+1]-ramp[a+1])*f, ramp[a+2]+(ramp[b+2]-ramp[a+2])*f);}

// ── Blackbody Temperature → linear sRGB (CIE 1931) ──
// Kang et al. (2002) polynomial fit to the Planckian locus in CIE xy,
// then XYZ→linear sRGB via IEC 61966-2-1 matrix.
// Accurate from ~1000K to 40000K. Below 1000K the Kang polynomial
// diverges, so we use a deep-red fallback (physically correct: below
// 1000K visible emission is almost entirely deep red / near-infrared).
// Cost: ~6 multiply-adds + 1 division — comparable to old Helland version.
__device__ inline float3 blackbodyColor(float kelvin){
    // Below ~800K the Kang polynomial produces invalid chromaticity.
    // Fade from black (0K) to deep red (1000K) for sub-valid range.
    if(kelvin < 1000.f){
        float f = fmaxf(kelvin, 0.f) * 0.001f;  // 0→1 over 0→1000K
        float f2 = f * f;                         // quadratic: more physical ramp-up
        return make_f3(f2, f2 * 0.05f, 0.f);      // deep red, trace orange
    }

    float T = fminf(kelvin, 40000.f);
    float invT = 1.f / T;
    float invT2 = invT * invT;
    float invT3 = invT2 * invT;

    // Planckian locus chromaticity x(T) — Kang et al. (2002)
    float x;
    if(T <= 4000.f)
        x = -0.2661239e9f*invT3 - 0.2343589e6f*invT2 + 0.8776956e3f*invT + 0.179910f;
    else
        x = -3.0258469e9f*invT3 + 2.1070379e6f*invT2 + 0.2226347e3f*invT + 0.240390f;

    // Planckian locus chromaticity y(x) — Kang et al. (2002)
    float x2 = x * x, x3 = x2 * x;
    float y;
    if(T <= 2222.f)
        y = -1.1063814f*x3 - 1.34811020f*x2 + 2.18555832f*x - 0.20219683f;
    else if(T <= 4000.f)
        y = -0.9549476f*x3 - 1.37418593f*x2 + 2.09137015f*x - 0.16748867f;
    else
        y =  3.0817580f*x3 - 5.87338670f*x2 + 3.75112997f*x - 0.37001483f;

    // CIE XYZ from chromaticity (Y = 1 normalized luminance)
    float invY = 1.f / fmaxf(y, 1e-6f);
    float X = x * invY;           // X = x/y
    float Z = (1.f - x - y) * invY;  // Z = (1-x-y)/y

    // XYZ → linear sRGB (IEC 61966-2-1, D65 white point)
    // Y = 1.0 is folded into the constant terms
    float R =  3.2406255f * X - 1.5372080f - 0.4986286f * Z;
    float G = -0.9689307f * X + 1.8757561f + 0.0415175f * Z;
    float B =  0.0557101f * X - 0.2040211f + 1.0569959f * Z;

    // Clamp negatives, normalize peak to 1.0
    R = fmaxf(R, 0.f); G = fmaxf(G, 0.f); B = fmaxf(B, 0.f);
    float mx = fmaxf(R, fmaxf(G, B));
    if(mx > 0.f){ float inv = 1.f / mx; R *= inv; G *= inv; B *= inv; }
    return make_f3(R, G, B);
}

// ── Dual-Lobe Henyey-Greenstein Phase Function ──
// Single-lobe HG: (1-g²) / (4π * (1+g²-2g·cosθ)^(3/2))
// The 1/(4π) normalization is folded into the scattering coefficient,
// so we return the *unnormalized* ratio relative to isotropic (= 1.0 when g=0).
__device__ inline float henyeyGreenstein(float cosTheta, float g){
    float g2 = g*g;
    float denom = 1.f + g2 - 2.f*g*cosTheta;
    // rsqrtf is a fast HW intrinsic on NVIDIA GPUs
    float invSqrt = rsqrtf(fmaxf(denom, 1e-8f));
    return (1.f - g2) * (invSqrt * invSqrt * invSqrt);  // (1-g²) / denom^(3/2)
}

// Dual-lobe: mix * HG(g1) + (1-mix) * HG(g2)
// Returns 1.0 when both lobes are isotropic (g1=g2=0), preserving backward compatibility.
__device__ inline float dualLobeHG(float cosTheta, float g1, float g2, float mix){
    return mix * henyeyGreenstein(cosTheta, g1) + (1.f - mix) * henyeyGreenstein(cosTheta, g2);
}

// ── Approximate Mie Phase Function (Jendersie & d'Eon, SIGGRAPH 2023) ──
// Physically correct for spherical particles (water droplets in clouds/fog).
// Captures three key Mie features that dual-lobe HG cannot reproduce:
//   • Sharp forward diffraction peak (silver lining, back-lit glow)
//   • Broad side-scattering minimum
//   • Backward glory ring (subtle peak at cosθ ≈ -1)
// Uses Cornette-Shanks forward lobe (g=0.76, fitted to 10μm cloud droplets)
// plus a narrow exponential glory term.
// Cost: 1 rsqrt + 1 exp + arithmetic ≈ 8 FLOPs. Same as dual-lobe HG.
__device__ inline float miePhase(float cosTheta){
    float c2 = cosTheta * cosTheta;
    // Forward peak: Cornette-Shanks (g=0.76, physically motivated Mie fit)
    // CS(μ,g) = 3(1-g²)/[2(2+g²)] · (1+μ²) / (1+g²-2gμ)^(3/2)
    // g=0.76: g²=0.5776, 3(1-g²)/[2(2+g²)] = 1.2672/5.1552 ≈ 0.2457
    float denom = 1.5776f - 1.52f * cosTheta;  // 1+g²-2g·cosθ
    float is = rsqrtf(fmaxf(denom, 1e-8f));
    float fwd = 0.2457f * (1.f + c2) * (is * is * is);
    // Backward glory: narrow peak at exact backscatter (cosθ = -1)
    // Decays within ~5° — matches Mie glory angular width for cloud droplets.
    float glory = 0.1f * expf(-15.f * (1.f + cosTheta));
    return fwd + glory;
}

// ── Phase function dispatcher ──
// Mode 0=Isotropic, 1=HG (single lobe), 2=Dual HG, 3=Mie
__device__ inline float evalPhase(int mode, float cosTheta, float g1, float g2, float mix){
    switch(mode){
        case 0:  return 1.f;
        case 1:  return henyeyGreenstein(cosTheta, g1);
        case 2:  return dualLobeHG(cosTheta, g1, g2, mix);
        case 3:  return miePhase(cosTheta);
        default: return dualLobeHG(cosTheta, g1, g2, mix);
    }
}

// ── L2 Spherical Harmonics evaluation ──────────────────────────────────
// 27 floats, ~27 MADs per sample — negligible vs shadow rays.
__device__ inline float3 evalEnvSH(const float sh[27], float3 n)
{
    const float Y00  = 0.282095f;
    const float Y1n1 = 0.488603f;
    const float Y10  = 0.488603f;
    const float Y11  = 0.488603f;
    const float Y2n2 = 1.092548f;
    const float Y2n1 = 1.092548f;
    const float Y20  = 0.315392f;
    const float Y21  = 1.092548f;
    const float Y22  = 0.546274f;

    float3 result;
    for (int c = 0; c < 3; ++c) {
        int o = c * 9;
        float v = sh[o+0] * Y00;
        v += sh[o+1] * Y1n1 * n.y;
        v += sh[o+2] * Y10  * n.z;
        v += sh[o+3] * Y11  * n.x;
        v += sh[o+4] * Y2n2 * n.x * n.y;
        v += sh[o+5] * Y2n1 * n.y * n.z;
        v += sh[o+6] * Y20  * (3.0f*n.z*n.z - 1.0f);
        v += sh[o+7] * Y21  * n.x * n.z;
        v += sh[o+8] * Y22  * (n.x*n.x - n.y*n.y);
        (&result.x)[c] = v;
    }
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// 3D Gradient Noise + fBm (procedural detail for volumes)
// ═══════════════════════════════════════════════════════════════════════════

// PCG-quality integer hash — proper avalanche, no grid artifacts
__device__ inline uint32_t pcgHash(uint32_t v){
    v = v * 747796405u + 2891336453u;
    v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (v >> 22u) ^ v;
}

// Hash cascade → gradient dot product (12 classic Perlin gradient directions)
// Uses pcgHash(x + pcgHash(y + pcgHash(z))) for proper 3D mixing
__device__ inline float gradDot3D(int hx, int hy, int hz, float dx, float dy, float dz){
    uint32_t h = pcgHash(uint32_t(hx) + pcgHash(uint32_t(hy) + pcgHash(uint32_t(hz))));
    h &= 15u;
    float u = (h < 8) ? dx : dy;
    float v = (h < 4) ? dy : ((h==12||h==14) ? dx : dz);
    return ((h&1) ? -u : u) + ((h&2) ? -v : v);
}

// 3D gradient noise — C2 continuous, no lookup tables, no grid artifacts
__device__ inline float gradNoise3D(float x, float y, float z){
    int ix=int(floorf(x)), iy=int(floorf(y)), iz=int(floorf(z));
    float fx=x-ix, fy=y-iy, fz=z-iz;
    // Quintic Hermite smoothstep (C2 continuous — no Mach banding)
    float ux=fx*fx*fx*(fx*(fx*6.f-15.f)+10.f);
    float uy=fy*fy*fy*(fy*(fy*6.f-15.f)+10.f);
    float uz=fz*fz*fz*(fz*(fz*6.f-15.f)+10.f);
    float n000=gradDot3D(ix,iy,iz,fx,fy,fz),       n100=gradDot3D(ix+1,iy,iz,fx-1,fy,fz);
    float n010=gradDot3D(ix,iy+1,iz,fx,fy-1,fz),    n110=gradDot3D(ix+1,iy+1,iz,fx-1,fy-1,fz);
    float n001=gradDot3D(ix,iy,iz+1,fx,fy,fz-1),    n101=gradDot3D(ix+1,iy,iz+1,fx-1,fy,fz-1);
    float n011=gradDot3D(ix,iy+1,iz+1,fx,fy-1,fz-1),n111=gradDot3D(ix+1,iy+1,iz+1,fx-1,fy-1,fz-1);
    float n00=n000+(n100-n000)*ux, n10=n010+(n110-n010)*ux;
    float n01=n001+(n101-n001)*ux, n11=n011+(n111-n011)*ux;
    float n0=n00+(n10-n00)*uy, n1=n01+(n11-n01)*uy;
    return n0+(n1-n0)*uz;
}

// Fractal Brownian motion — layered gradient noise with octave rotation
// Rotating coordinates between octaves prevents grid line stacking
__device__ inline float fbm3D(float x, float y, float z, int octaves){
    float sum=0.f, amp=1.f;
    #pragma unroll 8
    for(int i=0; i<octaves; ++i){
        sum += amp * gradNoise3D(x, y, z);
        amp *= 0.5f;
        // Rotate + scale for next octave (breaks grid alignment between layers)
        float nx = 2.f*(0.8f*x + 0.6f*y) + 1.7f;
        float ny = 2.f*(-0.6f*x + 0.8f*y) + 9.2f;
        float nz = 2.f*z + 3.4f;
        x=nx; y=ny; z=nz;
    }
    return sum;
}

// ═══════════════════════════════════════════════════════════════════════════
// Hash table
// ═══════════════════════════════════════════════════════════════════════════

static constexpr int64_t HASH_EMPTY = -9223372036854775807LL-1;
__device__ __host__ inline int64_t packLeafKey(int x,int y,int z){return(int64_t(x)&0xFFFF)|((int64_t(y)&0xFFFF)<<16)|((int64_t(z)&0xFFFF)<<32);}
__device__ __host__ inline uint32_t hashKey(int64_t k){uint64_t h=uint64_t(k);h^=h>>33;h*=0xff51afd7ed558ccdULL;h^=h>>33;h*=0xc4ceb9fe1a85ec53ULL;h^=h>>33;return uint32_t(h);}

// ═══════════════════════════════════════════════════════════════════════════
// Codebook lookups
// ═══════════════════════════════════════════════════════════════════════════

__device__ inline float lookupVoxel(const GridPtrs& g, int vx,int vy,int vz){
    int lx=vx>>g.blockShift,ly=vy>>g.blockShift,lz=vz>>g.blockShift;
    int64_t key=packLeafKey(lx,ly,lz); uint32_t slot=hashKey(key)&g.hashMask;
    for(int p=0;p<64;++p){int64_t k=g.hashKeys[slot];
        if(k==key){uint32_t li=g.hashVals[slot];uint16_t ci=g.indices[li];
            int ox=vx-(lx<<g.blockShift),oy=vy-(ly<<g.blockShift),oz=vz-(lz<<g.blockShift);
            float v=g.codebook[ci*g.vpb+ox*g.blkDim*g.blkDim+oy*g.blkDim+oz];
            if(g.gainMaps){float hb=g.blkDim*0.5f,gc=hb*0.5f-0.5f,gi=1.f/hb;
                float gx=fmaxf(0.f,fminf(1.f,(ox-gc)*gi)),gy=fmaxf(0.f,fminf(1.f,(oy-gc)*gi)),gz=fmaxf(0.f,fminf(1.f,(oz-gc)*gi));
                const float*gm=g.gainMaps+li*8;
                float g00=gm[0]+(gm[1]-gm[0])*gx,g10=gm[2]+(gm[3]-gm[2])*gx,g01=gm[4]+(gm[5]-gm[4])*gx,g11=gm[6]+(gm[7]-gm[6])*gx;
                v*=(g00+(g10-g00)*gy)+((g01+(g11-g01)*gy)-(g00+(g10-g00)*gy))*gz;}
            if(g.residuals&&g.blkDim==8){const int R=6;
                float rx=fmaxf(0.f,fminf(R-1.001f,ox*(R-1)/7.f)),ry=fmaxf(0.f,fminf(R-1.001f,oy*(R-1)/7.f)),rz=fmaxf(0.f,fminf(R-1.001f,oz*(R-1)/7.f));
                int rx0=int(rx),ry0=int(ry),rz0=int(rz),rx1=min(rx0+1,R-1),ry1=min(ry0+1,R-1),rz1=min(rz0+1,R-1);
                float fx=rx-rx0,fy=ry-ry0,fz=rz-rz0;const int RS=R*R;const float*r=g.residuals+li*(R*R*R);
                float r00=r[rx0*RS+ry0*R+rz0]+(r[rx1*RS+ry0*R+rz0]-r[rx0*RS+ry0*R+rz0])*fx;
                float r10=r[rx0*RS+ry1*R+rz0]+(r[rx1*RS+ry1*R+rz0]-r[rx0*RS+ry1*R+rz0])*fx;
                float r01=r[rx0*RS+ry0*R+rz1]+(r[rx1*RS+ry0*R+rz1]-r[rx0*RS+ry0*R+rz1])*fx;
                float r11=r[rx0*RS+ry1*R+rz1]+(r[rx1*RS+ry1*R+rz1]-r[rx0*RS+ry1*R+rz1])*fx;
                float rv0=r00+(r10-r00)*fy,rv1=r01+(r11-r01)*fy; v+=rv0+(rv1-rv0)*fz;}
            return v * g.normScale;}
        if(k==HASH_EMPTY) return 0.f; slot=(slot+1)&g.hashMask;}
    return 0.f;}

__device__ inline float sampleTrilinear(const GridPtrs& g, float ix,float iy,float iz){
    int x0=int(floorf(ix)),y0=int(floorf(iy)),z0=int(floorf(iz)); float fx=ix-x0,fy=iy-y0,fz=iz-z0;
    float c000=lookupVoxel(g,x0,y0,z0),c100=lookupVoxel(g,x0+1,y0,z0),c010=lookupVoxel(g,x0,y0+1,z0),c110=lookupVoxel(g,x0+1,y0+1,z0);
    float c001=lookupVoxel(g,x0,y0,z0+1),c101=lookupVoxel(g,x0+1,y0,z0+1),c011=lookupVoxel(g,x0,y0+1,z0+1),c111=lookupVoxel(g,x0+1,y0+1,z0+1);
    float c00=c000+(c100-c000)*fx,c10=c010+(c110-c010)*fx,c01=c001+(c101-c001)*fx,c11=c011+(c111-c011)*fx;
    float c0=c00+(c10-c00)*fy,c1=c01+(c11-c01)*fy; return c0+(c1-c0)*fz;}

// ── Thread-local leaf cache (no shared memory, no race conditions) ──
// In trilinear sampling, 7 of 8 corners are typically in the same leaf block.
// This single-entry cache eliminates ~87% of hash table probes.
struct TLLeafCache { int64_t key; uint32_t leafIdx; };

__device__ inline float lookupVoxelTL(const GridPtrs& g, int vx,int vy,int vz, TLLeafCache& tc){
    int lx=vx>>g.blockShift,ly=vy>>g.blockShift,lz=vz>>g.blockShift;
    int64_t key=packLeafKey(lx,ly,lz);
    uint32_t li;
    if(key==tc.key){ li=tc.leafIdx; }
    else{
        uint32_t slot=hashKey(key)&g.hashMask;
        for(int p=0;p<64;++p){int64_t k=g.hashKeys[slot];
            if(k==key){li=g.hashVals[slot]; tc.key=key; tc.leafIdx=li; goto found;}
            if(k==HASH_EMPTY) return 0.f; slot=(slot+1)&g.hashMask;}
        return 0.f;
    }
    found:;
    uint16_t ci=g.indices[li];
    int ox=vx-(lx<<g.blockShift),oy=vy-(ly<<g.blockShift),oz=vz-(lz<<g.blockShift);
    float v=g.codebook[ci*g.vpb+ox*g.blkDim*g.blkDim+oy*g.blkDim+oz];
    if(g.gainMaps){float hb=g.blkDim*0.5f,gc=hb*0.5f-0.5f,gi=1.f/hb;
        float gx=fmaxf(0.f,fminf(1.f,(ox-gc)*gi)),gy=fmaxf(0.f,fminf(1.f,(oy-gc)*gi)),gz=fmaxf(0.f,fminf(1.f,(oz-gc)*gi));
        const float*gm=g.gainMaps+li*8;
        float g00=gm[0]+(gm[1]-gm[0])*gx,g10=gm[2]+(gm[3]-gm[2])*gx,g01=gm[4]+(gm[5]-gm[4])*gx,g11=gm[6]+(gm[7]-gm[6])*gx;
        v*=(g00+(g10-g00)*gy)+((g01+(g11-g01)*gy)-(g00+(g10-g00)*gy))*gz;}
    if(g.residuals&&g.blkDim==8){const int R=6;
        float rx=fmaxf(0.f,fminf(R-1.001f,ox*(R-1)/7.f)),ry=fmaxf(0.f,fminf(R-1.001f,oy*(R-1)/7.f)),rz=fmaxf(0.f,fminf(R-1.001f,oz*(R-1)/7.f));
        int rx0=int(rx),ry0=int(ry),rz0=int(rz),rx1=min(rx0+1,R-1),ry1=min(ry0+1,R-1),rz1=min(rz0+1,R-1);
        float fx=rx-rx0,fy=ry-ry0,fz=rz-rz0;const int RS=R*R;const float*r=g.residuals+li*(R*R*R);
        float r00=r[rx0*RS+ry0*R+rz0]+(r[rx1*RS+ry0*R+rz0]-r[rx0*RS+ry0*R+rz0])*fx;
        float r10=r[rx0*RS+ry1*R+rz0]+(r[rx1*RS+ry1*R+rz0]-r[rx0*RS+ry1*R+rz0])*fx;
        float r01=r[rx0*RS+ry0*R+rz1]+(r[rx1*RS+ry0*R+rz1]-r[rx0*RS+ry0*R+rz1])*fx;
        float r11=r[rx0*RS+ry1*R+rz1]+(r[rx1*RS+ry1*R+rz1]-r[rx0*RS+ry1*R+rz1])*fx;
        float rv0=r00+(r10-r00)*fy,rv1=r01+(r11-r01)*fy; v+=rv0+(rv1-rv0)*fz;}
    return v*g.normScale;
}

__device__ inline float sampleTrilinearTL(const GridPtrs& g, float ix,float iy,float iz, TLLeafCache& tc){
    int x0=int(floorf(ix)),y0=int(floorf(iy)),z0=int(floorf(iz)); float fx=ix-x0,fy=iy-y0,fz=iz-z0;
    float c000=lookupVoxelTL(g,x0,y0,z0,tc),c100=lookupVoxelTL(g,x0+1,y0,z0,tc),c010=lookupVoxelTL(g,x0,y0+1,z0,tc),c110=lookupVoxelTL(g,x0+1,y0+1,z0,tc);
    float c001=lookupVoxelTL(g,x0,y0,z0+1,tc),c101=lookupVoxelTL(g,x0+1,y0,z0+1,tc),c011=lookupVoxelTL(g,x0,y0+1,z0+1,tc),c111=lookupVoxelTL(g,x0+1,y0+1,z0+1,tc);
    float c00=c000+(c100-c000)*fx,c10=c010+(c110-c010)*fx,c01=c001+(c101-c001)*fx,c11=c011+(c111-c011)*fx;
    float c0=c00+(c10-c00)*fy,c1=c01+(c11-c01)*fy; return c0+(c1-c0)*fz;}

// Simple nearest-neighbor for emission/temperature (no trilinear needed for glow)
__device__ inline float sampleNearest(const GridPtrs& g, float ix,float iy,float iz){
    return lookupVoxel(g, __float2int_rd(ix), __float2int_rd(iy), __float2int_rd(iz));}

// Empty-space skip
__device__ inline bool isOccupied(const GridPtrs& g,int lx,int ly,int lz){
    int64_t key=packLeafKey(lx,ly,lz);uint32_t slot=hashKey(key)&g.hashMask;
    for(int p=0;p<32;++p){int64_t k=g.hashKeys[slot];if(k==key)return true;if(k==HASH_EMPTY)return false;slot=(slot+1)&g.hashMask;}return false;}

__device__ inline float blockSkipDist(float ix,float iy,float iz,float3 rd,float ivs,int bs){
    float rdx=rd.x*ivs,rdy=rd.y*ivs,rdz=rd.z*ivs;
    int bx=int(floorf(ix))>>bs,by=int(floorf(iy))>>bs,bz=int(floorf(iz))>>bs;float e=1e20f;
    if(fabsf(rdx)>1e-10f){float b=(rdx>0)?float((bx+1)<<bs):float(bx<<bs);float d=(b-ix)/rdx;if(d>1e-6f)e=fminf(e,d);}
    if(fabsf(rdy)>1e-10f){float b=(rdy>0)?float((by+1)<<bs):float(by<<bs);float d=(b-iy)/rdy;if(d>1e-6f)e=fminf(e,d);}
    if(fabsf(rdz)>1e-10f){float b=(rdz>0)?float((bz+1)<<bs):float(bz<<bs);float d=(b-iz)/rdz;if(d>1e-6f)e=fminf(e,d);}
    return e+0.01f;}

// Macro-cell occupancy check (coarse level — each cell covers M×M×M blocks)
__device__ inline bool isMacroOccupied(const GridPtrs& g, int lx, int ly, int lz){
    if(!g.macroGrid) return true;  // no macro-grid → assume occupied (fall through to hash)
    int cs=g.macroCellShift;
    int mx=(lx-g.macroOrigin[0])>>cs, my=(ly-g.macroOrigin[1])>>cs, mz=(lz-g.macroOrigin[2])>>cs;
    if(mx<0||my<0||mz<0||mx>=g.macroGridDims[0]||my>=g.macroGridDims[1]||mz>=g.macroGridDims[2]) return false;
    return g.macroGrid[mz*g.macroGridDims[1]*g.macroGridDims[0]+my*g.macroGridDims[0]+mx]!=0;}

// Macro-cell skip distance: jump to next macro-cell boundary along ray
__device__ inline float macroSkipDist(float ix,float iy,float iz,float3 rd,float ivs,
    int blockShift,int cellShift,const int macroOrigin[3])
{
    float rdx=rd.x*ivs,rdy=rd.y*ivs,rdz=rd.z*ivs;
    int totalShift=blockShift+cellShift;  // voxel→macro-cell shift
    int ox=macroOrigin[0]<<blockShift, oy=macroOrigin[1]<<blockShift, oz=macroOrigin[2]<<blockShift;
    int cx=(int(floorf(ix))-ox)>>totalShift, cy=(int(floorf(iy))-oy)>>totalShift, cz=(int(floorf(iz))-oz)>>totalShift;
    float e=1e20f;
    if(fabsf(rdx)>1e-10f){float b=float(((rdx>0?cx+1:cx)<<totalShift)+ox);float d=(b-ix)/rdx;if(d>1e-6f)e=fminf(e,d);}
    if(fabsf(rdy)>1e-10f){float b=float(((rdy>0?cy+1:cy)<<totalShift)+oy);float d=(b-iy)/rdy;if(d>1e-6f)e=fminf(e,d);}
    if(fabsf(rdz)>1e-10f){float b=float(((rdz>0?cz+1:cz)<<totalShift)+oz);float d=(b-iz)/rdz;if(d>1e-6f)e=fminf(e,d);}
    return e+0.01f;}

// FP16 shadow sampler
__device__ inline __half shadowSampleHalf(const GridPtrs& g, float ix,float iy,float iz){
    int vx=__float2int_rd(ix),vy=__float2int_rd(iy),vz=__float2int_rd(iz);
    int lx=vx>>g.blockShift,ly=vy>>g.blockShift,lz=vz>>g.blockShift;
    int64_t key=packLeafKey(lx,ly,lz);uint32_t slot=hashKey(key)&g.hashMask;
    for(int p=0;p<64;++p){int64_t k=g.hashKeys[slot];
        if(k==key){uint32_t li=g.hashVals[slot];uint16_t ci=g.indices[li];
            int ox=vx-(lx<<g.blockShift),oy=vy-(ly<<g.blockShift),oz=vz-(lz<<g.blockShift);
            return g.codebookHalf[ci*g.vpb+ox*g.blkDim*g.blkDim+oy*g.blkDim+oz];}
        if(k==HASH_EMPTY)return __float2half(0.f);slot=(slot+1)&g.hashMask;}
    return __float2half(0.f);}

// ═══════════════════════════════════════════════════════════════════════════
// Leaf cache — two layers: thread-local (1 entry) + shared memory (32 entries)
// Thread-local: eliminates ~87% of hash probes within trilinear (7/8 corners in same leaf)
// Shared memory: helps neighboring threads that march through the same leaves
// ═══════════════════════════════════════════════════════════════════════════

struct LeafCache { int64_t key; uint32_t leafIdx; };

#define SM_LEAF_CACHE_BITS 5
#define SM_LEAF_CACHE_SIZE (1 << SM_LEAF_CACHE_BITS)  // 32 slots
#define SM_LEAF_CACHE_MASK (SM_LEAF_CACHE_SIZE - 1)

// Resolve a leaf key → leaf index, checking caches first
__device__ inline uint32_t resolveLeaf(const GridPtrs& g, int64_t key,
    LeafCache& tc, int64_t* smKeys, uint32_t* smVals)
{
    // Layer 1: thread-local cache (single entry)
    if(key==tc.key) return tc.leafIdx;
    // Layer 2: shared memory cache (direct-mapped, no sync needed)
    uint32_t smSlot=uint32_t(key)&SM_LEAF_CACHE_MASK;
    if(smKeys[smSlot]==key){uint32_t li=smVals[smSlot]; tc.key=key; tc.leafIdx=li; return li;}
    // Layer 3: global hash table probe
    uint32_t slot=hashKey(key)&g.hashMask;
    for(int p=0;p<64;++p){int64_t k=g.hashKeys[slot];
        if(k==key){uint32_t li=g.hashVals[slot]; tc.key=key; tc.leafIdx=li;
            smKeys[smSlot]=key; smVals[smSlot]=li; return li;}
        if(k==HASH_EMPTY) return UINT32_MAX; slot=(slot+1)&g.hashMask;}
    return UINT32_MAX;
}

// Cached voxel lookup — uses resolveLeaf then reads codebook/gain/residual
__device__ inline float lookupVoxelCached(const GridPtrs& g, int vx,int vy,int vz,
    LeafCache& tc, int64_t* smKeys, uint32_t* smVals)
{
    int lx=vx>>g.blockShift,ly=vy>>g.blockShift,lz=vz>>g.blockShift;
    int64_t key=packLeafKey(lx,ly,lz);
    uint32_t li=resolveLeaf(g,key,tc,smKeys,smVals);
    if(li==UINT32_MAX) return 0.f;
    uint16_t ci=g.indices[li];
    int ox=vx-(lx<<g.blockShift),oy=vy-(ly<<g.blockShift),oz=vz-(lz<<g.blockShift);
    float v=g.codebook[ci*g.vpb+ox*g.blkDim*g.blkDim+oy*g.blkDim+oz];
    if(g.gainMaps){float hb=g.blkDim*0.5f,gc=hb*0.5f-0.5f,gi=1.f/hb;
        float gx=fmaxf(0.f,fminf(1.f,(ox-gc)*gi)),gy=fmaxf(0.f,fminf(1.f,(oy-gc)*gi)),gz=fmaxf(0.f,fminf(1.f,(oz-gc)*gi));
        const float*gm=g.gainMaps+li*8;
        float g00=gm[0]+(gm[1]-gm[0])*gx,g10=gm[2]+(gm[3]-gm[2])*gx,g01=gm[4]+(gm[5]-gm[4])*gx,g11=gm[6]+(gm[7]-gm[6])*gx;
        v*=(g00+(g10-g00)*gy)+((g01+(g11-g01)*gy)-(g00+(g10-g00)*gy))*gz;}
    if(g.residuals&&g.blkDim==8){const int R=6;
        float rx=fmaxf(0.f,fminf(R-1.001f,ox*(R-1)/7.f)),ry=fmaxf(0.f,fminf(R-1.001f,oy*(R-1)/7.f)),rz=fmaxf(0.f,fminf(R-1.001f,oz*(R-1)/7.f));
        int rx0=int(rx),ry0=int(ry),rz0=int(rz),rx1=min(rx0+1,R-1),ry1=min(ry0+1,R-1),rz1=min(rz0+1,R-1);
        float fx=rx-rx0,fy=ry-ry0,fz=rz-rz0;const int RS=R*R;const float*r=g.residuals+li*(R*R*R);
        float r00=r[rx0*RS+ry0*R+rz0]+(r[rx1*RS+ry0*R+rz0]-r[rx0*RS+ry0*R+rz0])*fx;
        float r10=r[rx0*RS+ry1*R+rz0]+(r[rx1*RS+ry1*R+rz0]-r[rx0*RS+ry1*R+rz0])*fx;
        float r01=r[rx0*RS+ry0*R+rz1]+(r[rx1*RS+ry0*R+rz1]-r[rx0*RS+ry0*R+rz1])*fx;
        float r11=r[rx0*RS+ry1*R+rz1]+(r[rx1*RS+ry1*R+rz1]-r[rx0*RS+ry1*R+rz1])*fx;
        float rv0=r00+(r10-r00)*fy,rv1=r01+(r11-r01)*fy; v+=rv0+(rv1-rv0)*fz;}
    return v*g.normScale;
}

// Cached trilinear — same 8-corner interpolation but with leaf cache
__device__ inline float sampleTrilinearCached(const GridPtrs& g, float ix,float iy,float iz,
    LeafCache& tc, int64_t* smKeys, uint32_t* smVals)
{
    int x0=int(floorf(ix)),y0=int(floorf(iy)),z0=int(floorf(iz)); float fx=ix-x0,fy=iy-y0,fz=iz-z0;
    float c000=lookupVoxelCached(g,x0,y0,z0,tc,smKeys,smVals),c100=lookupVoxelCached(g,x0+1,y0,z0,tc,smKeys,smVals);
    float c010=lookupVoxelCached(g,x0,y0+1,z0,tc,smKeys,smVals),c110=lookupVoxelCached(g,x0+1,y0+1,z0,tc,smKeys,smVals);
    float c001=lookupVoxelCached(g,x0,y0,z0+1,tc,smKeys,smVals),c101=lookupVoxelCached(g,x0+1,y0,z0+1,tc,smKeys,smVals);
    float c011=lookupVoxelCached(g,x0,y0+1,z0+1,tc,smKeys,smVals),c111=lookupVoxelCached(g,x0+1,y0+1,z0+1,tc,smKeys,smVals);
    float c00=c000+(c100-c000)*fx,c10=c010+(c110-c010)*fx,c01=c001+(c101-c001)*fx,c11=c011+(c111-c011)*fx;
    float c0=c00+(c10-c00)*fy,c1=c01+(c11-c01)*fy; return c0+(c1-c0)*fz;
}

// Cached FP16 shadow sampler — uses shared memory cache (no thread-local, shadow rays diverge)
__device__ inline __half shadowSampleHalfCached(const GridPtrs& g, float ix,float iy,float iz,
    int64_t* smKeys, uint32_t* smVals)
{
    int vx=__float2int_rd(ix),vy=__float2int_rd(iy),vz=__float2int_rd(iz);
    int lx=vx>>g.blockShift,ly=vy>>g.blockShift,lz=vz>>g.blockShift;
    int64_t key=packLeafKey(lx,ly,lz);
    // Check shared memory cache
    uint32_t smSlot=uint32_t(key)&SM_LEAF_CACHE_MASK;
    uint32_t li;
    if(smKeys[smSlot]==key){ li=smVals[smSlot]; }
    else{
        uint32_t slot=hashKey(key)&g.hashMask;
        bool found=false;
        for(int p=0;p<64;++p){int64_t k=g.hashKeys[slot];
            if(k==key){li=g.hashVals[slot]; smKeys[smSlot]=key; smVals[smSlot]=li; found=true; break;}
            if(k==HASH_EMPTY) return __float2half(0.f); slot=(slot+1)&g.hashMask;}
        if(!found) return __float2half(0.f);
    }
    uint16_t ci=g.indices[li];
    int ox=vx-(lx<<g.blockShift),oy=vy-(ly<<g.blockShift),oz=vz-(lz<<g.blockShift);
    return g.codebookHalf[ci*g.vpb+ox*g.blkDim*g.blkDim+oy*g.blkDim+oz];
}

// ═══════════════════════════════════════════════════════════════════════════
// Hash table build kernel
// ═══════════════════════════════════════════════════════════════════════════

__global__ void buildHashTableKernel(const int32_t* origins,int64_t* hk,uint32_t* hv,uint32_t mask,uint32_t N,int bs){
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N)return;
    int ox=origins[i*3]>>bs,oy=origins[i*3+1]>>bs,oz=origins[i*3+2]>>bs;
    int64_t key=packLeafKey(ox,oy,oz);uint32_t slot=hashKey(key)&mask;
    for(int p=0;p<128;++p){int64_t old=atomicCAS(reinterpret_cast<unsigned long long*>(&hk[slot]),
        (unsigned long long)HASH_EMPTY,(unsigned long long)key);
        if(old==HASH_EMPTY||old==key){hv[slot]=i;return;}slot=(slot+1)&mask;}}

// ═══════════════════════════════════════════════════════════════════════════
// Coarse occupancy macro-grid build kernel
// One thread per leaf → set the containing macro-cell to occupied.
// ═══════════════════════════════════════════════════════════════════════════

__global__ void buildMacroGridKernel(const int32_t* origins, uint8_t* macroGrid,
    int macroOriginX, int macroOriginY, int macroOriginZ,
    int macroDimX, int macroDimY, uint32_t N, int blockShift, int cellShift)
{
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N)return;
    int bx=origins[i*3]>>blockShift, by=origins[i*3+1]>>blockShift, bz=origins[i*3+2]>>blockShift;
    int mx=(bx-macroOriginX)>>cellShift, my=(by-macroOriginY)>>cellShift, mz=(bz-macroOriginZ)>>cellShift;
    macroGrid[mz*macroDimY*macroDimX + my*macroDimX + mx] = 1;
}

// ═══════════════════════════════════════════════════════════════════════════
// [Phase 0] GPU hash table clear — replaces per-frame host alloc + H2D copy
// ═══════════════════════════════════════════════════════════════════════════

__global__ void fillHashEmpty(int64_t* keys, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) keys[i] = HASH_EMPTY;
}

// ═══════════════════════════════════════════════════════════════════════════
// FP32 → FP16 codebook conversion kernel (replaces CPU loop + extra H2D copy)
// ═══════════════════════════════════════════════════════════════════════════

__global__ void convertF32ToF16(const float* __restrict__ src, __half* __restrict__ dst, uint32_t n){
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) dst[i]=__float2half(src[i]);
}

__global__ void convertF16ToF32(const __half* __restrict__ src, float* __restrict__ dst, uint32_t n){
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) dst[i]=__half2float(src[i]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Codebook Ray March Kernel — density + emission + temperature
// ═══════════════════════════════════════════════════════════════════════════

__global__ void __launch_bounds__(256, 3) codebookRayMarchKernel(
    GridPtrs density, GridPtrs emission, GridPtrs temperature,
    __half* __restrict__ output,
    __half* __restrict__ deepBuf,     // [W×H×maxSamples×6] or nullptr
    int*    __restrict__ deepCounts,  // [W×H] or nullptr
    const RayMarchParams P)
{
    // ── Tile-aware pixel coordinates ──
    const int px=blockIdx.x*blockDim.x+threadIdx.x;
    const int localPy=blockIdx.y*blockDim.y+threadIdx.y;
    const int py=localPy+P.tileYOffset;  // global row in full frame
    if(px>=P.width||localPy>=P.tileHeight||py>=P.height)return;
    const int stride=P.outputStride, pidx=(py*P.width+px)*stride;

    // Thread-local leaf cache (private per thread, no races)
    TLLeafCache densTC={HASH_EMPTY,0};

    float u=(px+P.bboxOffsetX+0.5f)/P.fmtWidth*2.f-1.f, v=(py+P.bboxOffsetY+0.5f)/P.fmtHeight*2.f-1.f;
    float3 ld=make_f3(u*P.halfW,v*P.halfH,-1.f);
    float3 wd=make_f3(P.camRot[0]*ld.x+P.camRot[3]*ld.y+P.camRot[6]*ld.z,
                       P.camRot[1]*ld.x+P.camRot[4]*ld.y+P.camRot[7]*ld.z,
                       P.camRot[2]*ld.x+P.camRot[5]*ld.y+P.camRot[8]*ld.z);
    float3 cam=make_f3(P.camOrigin[0],P.camOrigin[1],P.camOrigin[2]);
    float3 orig=transformPoint(P.invTransform,cam), rd=normalize3(transformDir(P.invTransform,wd));

    float3 bmin=make_f3(P.bboxMin[0],P.bboxMin[1],P.bboxMin[2]);
    float3 bmax=make_f3(P.bboxMax[0],P.bboxMax[1],P.bboxMax[2]);
    float tE,tX;
    if(!intersectAABB(orig,rd,bmin,bmax,tE,tX)){__half z=__float2half(0.f);for(int i=0;i<stride;++i)output[pidx+i]=z;return;}

    const float baseStep=P.stepSize*density.voxelSize, shadowStep=density.voxelSize*P.shadowStepScale;
    const int nShadow=P.shadowSteps;
    const float ext=P.extinction, scat=P.scattering;
    const float emScale=P.emissionScale, tmpScale=P.temperatureScale;
    const int nLights=P.numLights;

    // Adaptive step size: enlarge step in low-density regions
    const float maxStep=baseStep*4.f;
    const float adaptThreshLo=0.01f;
    const float adaptThreshHi=0.05f;
    float curStep=baseStep;

    float3 llp[3];   // light position in local space
    float3 lld[3];   // light direction in local space (emission dir, pos→target)
    float  lIntensity[3];
    float  lConeAngle[3];
    float  lConeSoftness[3];
    int    lEnabled[3]={0,0,0};
    for(int li=0;li<nLights;++li){
        lEnabled[li]=P.lights[li].enabled;
        if(!lEnabled[li]) continue;
        float3 lw=make_f3(P.lights[li].position[0],P.lights[li].position[1],P.lights[li].position[2]);
        llp[li]=transformPoint(P.invTransform,lw);
        // Transform direction to local space (no translation — direction only)
        float dw0=P.lights[li].direction[0], dw1=P.lights[li].direction[1], dw2=P.lights[li].direction[2];
        float ldx=P.invTransform[0]*dw0+P.invTransform[4]*dw1+P.invTransform[8]*dw2;
        float ldy=P.invTransform[1]*dw0+P.invTransform[5]*dw1+P.invTransform[9]*dw2;
        float ldz=P.invTransform[2]*dw0+P.invTransform[6]*dw1+P.invTransform[10]*dw2;
        float ll=rsqrtf(fmaxf(ldx*ldx+ldy*ldy+ldz*ldz,1e-12f));
        lld[li]=make_f3(ldx*ll,ldy*ll,ldz*ll);
        lIntensity[li]=P.lights[li].intensity;
        lConeAngle[li]=P.lights[li].coneAngle;
        lConeSoftness[li]=P.lights[li].coneSoftness;
    }

    float Tr=1.f, Tg=1.f, Tb=1.f, scR=0,scG=0,scB=0, alb=0;
    float plR[3]={},plG[3]={},plB[3]={};
    float envScR=0,envScG=0,envScB=0;
    float emR=0,emG=0,emB=0,emA=0, tpR=0,tpG=0,tpB=0,tpA=0;

    // Deep mode: track running totals to compute per-step deltas
    float prevScR=0,prevScG=0,prevScB=0;
    float prevEmR=0,prevEmG=0,prevEmB=0;
    float prevTpR=0,prevTpG=0,prevTpB=0;

    // Depth + Position AOV: record first significant density hit
    float firstHitDepth=0.f;
    float firstHitPx=0.f, firstHitPy=0.f, firstHitPz=0.f;
    float firstHitPrefX=0.f, firstHitPrefY=0.f, firstHitPrefZ=0.f;
    float firstHitNx=0.f, firstHitNy=0.f, firstHitNz=0.f;
    bool  gotFirstHit=false;

    // Chromatic extinction (Hillaire 2015): per-channel camera-ray transmittance
    const float extR=P.extinctionR, extG=P.extinctionG, extB=P.extinctionB;

    float h1=fmodf(sinf(px*12.9898f+py*78.233f)*43758.5453f,1.f);
    float h2=fmodf(sinf(px*39.3468f+py*11.135f)*28947.7143f,1.f);
    float jit=fmodf(h1+h2*0.5f,1.f); if(jit<0)jit+=1.f;
    float t=tE+baseStep*jit;

    while(t<tX && fmaxf(Tr, fmaxf(Tg, Tb))>0.002f){
        float3 wp=make_f3(orig.x+rd.x*t,orig.y+rd.y*t,orig.z+rd.z*t);
        float ix=(wp.x-density.gridOffset[0])*density.invVoxelSize;
        float iy=(wp.y-density.gridOffset[1])*density.invVoxelSize;
        float iz=(wp.z-density.gridOffset[2])*density.invVoxelSize;

        // Two-level empty-space skip:
        // Level 1: coarse macro-grid (each cell = M×M×M blocks) — skips large empty regions
        // Level 2: block hash table probe — skips individual empty blocks
        {int lx=int(floorf(ix))>>density.blockShift,ly=int(floorf(iy))>>density.blockShift,lz=int(floorf(iz))>>density.blockShift;
         if(!isMacroOccupied(density,lx,ly,lz)){
             t+=fmaxf(macroSkipDist(ix,iy,iz,rd,density.invVoxelSize,density.blockShift,density.macroCellShift,density.macroOrigin),baseStep);
             curStep=baseStep;continue;}
         if(!isOccupied(density,lx,ly,lz)){t+=fmaxf(blockSkipDist(ix,iy,iz,rd,density.invVoxelSize,density.blockShift),baseStep);curStep=baseStep;continue;}}

        // Boundary blend
        if(P.boundaryBlend>0.f){float bd=float(density.blkDim),zone=P.boundaryBlend;
            float lx=ix-floorf(ix/bd)*bd,ly=iy-floorf(iy/bd)*bd,lz=iz-floorf(iz/bd)*bd;
            float dx=fminf(lx,bd-lx),dy=fminf(ly,bd-ly),dz=fminf(lz,bd-lz);
            float jh1=fmodf(sinf(t*127.1f+px*311.7f+py*97.3f)*43758.5453f,1.f);
            float jh2=fmodf(sinf(t*269.5f+px*67.1f+py*183.3f)*28947.7143f,1.f);
            float jh3=fmodf(sinf(t*419.2f+py*311.7f+px*53.9f)*16825.3217f,1.f);
            float s=fminf(zone,bd*0.5f);
            if(dx<zone)ix+=(jh1-0.5f)*(zone-dx)/zone*s*0.5f;
            if(dy<zone)iy+=(jh2-0.5f)*(zone-dy)/zone*s*0.5f;
            if(dz<zone)iz+=(jh3-0.5f)*(zone-dz)/zone*s*0.5f;}

        float dens=sampleTrilinearTL(density,ix,iy,iz,densTC);

        // Procedural fBm detail noise — adds fine-scale structure not in the VDB
        if(P.noiseAmplitude>0.f){
            // Irrational offsets decorrelate the noise lattice from the voxel grid
            float nx=wp.x*P.noiseFrequency*P.noiseScale[0]+P.noiseOffset[0]+127.1f;
            float ny=wp.y*P.noiseFrequency*P.noiseScale[1]+P.noiseOffset[1]+311.7f;
            float nz=wp.z*P.noiseFrequency*P.noiseScale[2]+P.noiseOffset[2]+74.7f;
            float n=fbm3D(nx,ny,nz,P.noiseOctaves);
            // Scale noise by existing density so it only adds detail where volume exists
            dens+=n*P.noiseAmplitude*fmaxf(dens,0.f);
            dens=fmaxf(dens,0.f);
        }

        if(dens<1e-5f){
            // Empty sample → grow step for next iteration
            curStep=fminf(curStep*2.f, maxStep);
            t+=curStep; continue;}

        // Non-trivial density — reset to baseStep before accumulation if dense
        if(dens>=adaptThreshHi) curStep=baseStep;

        // First-hit recording for depth + P AOV
        if(!gotFirstHit && (P.aovFlags&(128|256|512|1024))){
            gotFirstHit=true;
            // Object-space position (volume local coordinates)
            firstHitPx=wp.x; firstHitPy=wp.y; firstHitPz=wp.z;
            // Pref: bbox-normalized 0-1 coordinates (stable across animation)
            float bszX=P.bboxMax[0]-P.bboxMin[0], bszY=P.bboxMax[1]-P.bboxMin[1], bszZ=P.bboxMax[2]-P.bboxMin[2];
            firstHitPrefX=(bszX>1e-6f) ? (wp.x-P.bboxMin[0])/bszX : 0.5f;
            firstHitPrefY=(bszY>1e-6f) ? (wp.y-P.bboxMin[1])/bszY : 0.5f;
            firstHitPrefZ=(bszZ>1e-6f) ? (wp.z-P.bboxMin[2])/bszZ : 0.5f;
            // depth.Z = camera distance (positive in front of camera)
            float3 worldP=transformPoint(P.fwdTransform,wp);
            float dx=worldP.x-P.camOrigin[0], dy=worldP.y-P.camOrigin[1], dz=worldP.z-P.camOrigin[2];
            // Camera Z axis = column 2 of rotation matrix (indices 6,7,8)
            firstHitDepth=-(dx*P.camRot[6]+dy*P.camRot[7]+dz*P.camRot[8]);
            // N: gradient normal at first hit (world space)
            if(P.aovFlags&1024){
                int gix=int(floorf(ix)),giy=int(floorf(iy)),giz=int(floorf(iz));
                int gs=int(P.gradientSmooth);
                float gx=lookupVoxelTL(density,gix+gs,giy,giz,densTC)-lookupVoxelTL(density,gix-gs,giy,giz,densTC);
                float gy=lookupVoxelTL(density,gix,giy+gs,giz,densTC)-lookupVoxelTL(density,gix,giy-gs,giz,densTC);
                float gz=lookupVoxelTL(density,gix,giy,giz+gs,densTC)-lookupVoxelTL(density,gix,giy,giz-gs,densTC);
                float gMag=sqrtf(gx*gx+gy*gy+gz*gz);
                if(gMag>1e-8f){
                    float inv=1.f/gMag;
                    // Outward normal in local space, then transform to world
                    float3 nLocal=make_f3(-gx*inv,-gy*inv,-gz*inv);
                    float3 nWorld=normalize3(transformDir(P.fwdTransform,nLocal));
                    firstHitNx=nWorld.x; firstHitNy=nWorld.y; firstHitNz=nWorld.z;
                }
            }
        }

        // ── Gradient normal for Lambertian blend (computed on SHARP data) ──
        float3 gradN={0,0,0};
        float gradBlend=0.f;
        if(P.gradientMix>0.f){
            int gix=int(floorf(ix)),giy=int(floorf(iy)),giz=int(floorf(iz));
            int gs=int(P.gradientSmooth);
            float gx=lookupVoxelTL(density,gix+gs,giy,giz,densTC)-lookupVoxelTL(density,gix-gs,giy,giz,densTC);
            float gy=lookupVoxelTL(density,gix,giy+gs,giz,densTC)-lookupVoxelTL(density,gix,giy-gs,giz,densTC);
            float gz=lookupVoxelTL(density,gix,giy,giz+gs,densTC)-lookupVoxelTL(density,gix,giy,giz-gs,densTC);
            float gradMag=sqrtf(gx*gx+gy*gy+gz*gz);
            if(gradMag>1e-8f){
                float invMag=1.f/gradMag;
                gradN=make_f3(-gx*invMag,-gy*invMag,-gz*invMag);
                gradN.x+=P.gradNormalBias[0]; gradN.y+=P.gradNormalBias[1]; gradN.z+=P.gradNormalBias[2];
                float nLen=sqrtf(gradN.x*gradN.x+gradN.y*gradN.y+gradN.z*gradN.z);
                if(nLen>1e-8f){float ni=1.f/nLen; gradN.x*=ni; gradN.y*=ni; gradN.z*=ni;}
                float st=fminf(gradMag/fmaxf(P.gradientThreshold,1e-6f),1.f);
                gradBlend=P.gradientMix*st*st*(3.f-2.f*st);
            }
        }

        // ── Density blur (after gradient, affects extinction + scattering + powder) ──
        if(P.densityBlur>0.f){
            int db=int(P.densityBlur), db2=db*2;
            int gix=int(floorf(ix)),giy=int(floorf(iy)),giz=int(floorf(iz));
            float bd = dens * 4.f;
            bd += 2.f*(lookupVoxelTL(density,gix+db,giy,giz,densTC)
                      +lookupVoxelTL(density,gix-db,giy,giz,densTC)
                      +lookupVoxelTL(density,gix,giy+db,giz,densTC)
                      +lookupVoxelTL(density,gix,giy-db,giz,densTC)
                      +lookupVoxelTL(density,gix,giy,giz+db,densTC)
                      +lookupVoxelTL(density,gix,giy,giz-db,densTC));
            bd += lookupVoxelTL(density,gix+db2,giy,giz,densTC)
                 +lookupVoxelTL(density,gix-db2,giy,giz,densTC)
                 +lookupVoxelTL(density,gix,giy+db2,giz,densTC)
                 +lookupVoxelTL(density,gix,giy-db2,giz,densTC)
                 +lookupVoxelTL(density,gix,giy,giz+db2,densTC)
                 +lookupVoxelTL(density,gix,giy,giz-db2,densTC);
            dens = fmaxf(bd * (1.f/22.f), 0.f);
        }

        // ── Scatter smooth (stacks on top of density blur if both enabled) ──
        float sig_s;
        float shadeDens = dens;
        if(P.scatterSmooth>0.f){
            int ss=int(P.scatterSmooth);
            int ss2=ss*2;
            int gix=int(floorf(ix)),giy=int(floorf(iy)),giz=int(floorf(iz));
            float sd = dens * 4.f;
            sd += 2.f * (lookupVoxelTL(density,gix+ss,giy,giz,densTC)
                       + lookupVoxelTL(density,gix-ss,giy,giz,densTC)
                       + lookupVoxelTL(density,gix,giy+ss,giz,densTC)
                       + lookupVoxelTL(density,gix,giy-ss,giz,densTC)
                       + lookupVoxelTL(density,gix,giy,giz+ss,densTC)
                       + lookupVoxelTL(density,gix,giy,giz-ss,densTC));
            sd += lookupVoxelTL(density,gix+ss2,giy,giz,densTC)
                + lookupVoxelTL(density,gix-ss2,giy,giz,densTC)
                + lookupVoxelTL(density,gix,giy+ss2,giz,densTC)
                + lookupVoxelTL(density,gix,giy-ss2,giz,densTC)
                + lookupVoxelTL(density,gix,giy,giz+ss2,densTC)
                + lookupVoxelTL(density,gix,giy,giz-ss2,densTC);
            shadeDens = fmaxf(sd * (1.f/22.f), 0.f);
        }
        sig_s = shadeDens * scat;
        alb+=sig_s*curStep;

        // Lighting
        for(int li=0;li<nLights;++li){
            if(!lEnabled[li]) continue;
            // Cone attenuation: check if sample is inside the light's cone
            float toSampleX=wp.x-llp[li].x, toSampleY=wp.y-llp[li].y, toSampleZ=wp.z-llp[li].z;
            float tsDist=rsqrtf(fmaxf(toSampleX*toSampleX+toSampleY*toSampleY+toSampleZ*toSampleZ,1e-12f));
            float tsDirX=toSampleX*tsDist, tsDirY=toSampleY*tsDist, tsDirZ=toSampleZ*tsDist;
            float cosAngle=tsDirX*lld[li].x+tsDirY*lld[li].y+tsDirZ*lld[li].z;
            float cosCone=cosf(lConeAngle[li]);
            if(cosAngle<cosCone) continue;  // outside the cone

            // Smooth cone falloff at edge
            float coneAtten=1.f;
            if(lConeSoftness[li]>0.f){
                float softRange=lConeSoftness[li]*(1.f-cosCone);  // softness zone
                float innerCos=cosCone+softRange;
                if(cosAngle<innerCos){
                    float t=(cosAngle-cosCone)/fmaxf(softRange,1e-6f);
                    coneAtten=t*t*(3.f-2.f*t);  // smoothstep
                }
            }

            // Shadow ray direction: opposite of emission direction (toward light)
            float3 ldir = make_f3(-lld[li].x, -lld[li].y, -lld[li].z);
            float sJ=fmodf(sinf(t*91.17f+px*7.31f+py*13.97f)*43758.5453f,1.f); if(sJ<0)sJ+=1.f;
            float shadowOD;
            float lT;
            {
                __half hOD=__float2half(0.f);const __half hES=__float2half(ext*shadowStep*density.normScale);const __half hTh=__float2half(4.6f);
                for(int ls=0;ls<nShadow;++ls){float sOff=(ls+sJ)*shadowStep;
                    float3 sp=make_f3(wp.x+ldir.x*sOff,wp.y+ldir.y*sOff,wp.z+ldir.z*sOff);
                    float six=(sp.x-density.gridOffset[0])*density.invVoxelSize;
                    float siy=(sp.y-density.gridOffset[1])*density.invVoxelSize;
                    float siz=(sp.z-density.gridOffset[2])*density.invVoxelSize;
                    {int sl=int(floorf(six))>>density.blockShift,sm=int(floorf(siy))>>density.blockShift,sn=int(floorf(siz))>>density.blockShift;
                     if(!isOccupied(density,sl,sm,sn))continue;}
                    __half hD=shadowSampleHalf(density,six,siy,siz);
                    if(__hle(hD,__float2half(1e-6f)))continue;
                    hOD=__hadd(hOD,__hmul(hD,hES));if(__hge(hOD,hTh))break;}
                shadowOD=__half2float(hOD);
                lT=expf(-shadowOD);
            }
            float cosTheta=rd.x*ldir.x+rd.y*ldir.y+rd.z*ldir.z;
            float phase=evalPhase(P.phaseMode,cosTheta,P.phaseG1,P.phaseG2,P.phaseMix);
            if(gradBlend>0.f){
                float nDotL=fmaxf(0.f,gradN.x*ldir.x+gradN.y*ldir.y+gradN.z*ldir.z);
                phase+=(nDotL-phase)*gradBlend;
            }
            float powder=1.f;
            if(P.powderStrength>0.f) powder=1.f-expf(-dens*2.f*curStep*P.powderStrength);
            float base_sc=sig_s*lT*curStep*phase*powder*lIntensity[li]*coneAtten;
            float lcR=P.lights[li].color[0],lcG=P.lights[li].color[1],lcB=P.lights[li].color[2];
            float r=base_sc*Tr*lcR,g=base_sc*Tg*lcG,b=base_sc*Tb*lcB;
            scR+=r*P.lightMix[li];scG+=g*P.lightMix[li];scB+=b*P.lightMix[li];plR[li]+=r;plG[li]+=g;plB[li]+=b;

            // Frostbite multi-scatter octaves (Hillaire 2016)
            // Each octave approximates an additional light bounce:
            //   - Extinction drops by extFalloff^n → light penetrates deeper
            //   - Phase function flattens (g → g*extFalloff^n) → more isotropic
            //   - Contribution drops by scatterFalloff^n → energy conservation
            // Reuses the shadow optical depth — no extra ray marching.
            if(P.msEnable){
                float aE=P.msExtFalloff;
                float aC=P.msScatterFalloff;
                float extMul=aE;
                float contMul=aC;
                #pragma unroll 4
                for(int oct=0; oct<P.msOctaves; ++oct){
                    float msT=expf(-shadowOD*extMul);
                    // Phase becomes isotropic as octave increases
                    float g1_eff=P.phaseG1*extMul;
                    float g2_eff=P.phaseG2*extMul;
                    float msPhase=evalPhase(P.phaseMode,cosTheta,g1_eff,g2_eff,P.phaseMix);
                    if(gradBlend>0.f){
                        float nDotL=fmaxf(0.f,gradN.x*ldir.x+gradN.y*ldir.y+gradN.z*ldir.z);
                        msPhase+=(nDotL-msPhase)*gradBlend;
                    }
                    float ms_sc=sig_s*msT*curStep*msPhase*powder*lIntensity[li]*coneAtten*contMul;
                    float ms_r=ms_sc*Tr*lcR, ms_g=ms_sc*Tg*lcG, ms_b=ms_sc*Tb*lcB;
                    scR+=ms_r*P.lightMix[li]; scG+=ms_g*P.lightMix[li]; scB+=ms_b*P.lightMix[li];
                    plR[li]+=ms_r; plG[li]+=ms_g; plB[li]+=ms_b;
                    extMul*=aE;
                    contMul*=aC;
                }
            }
            }

        // Ambient light (constant, isotropic — no shadow ray, no phase function)
        if(P.ambientIntensity>0.f){
            float base_amb=sig_s*P.ambientIntensity*curStep;
            scR+=base_amb*Tr*P.ambientColor[0]; scG+=base_amb*Tg*P.ambientColor[1]; scB+=base_amb*Tb*P.ambientColor[2];}

        // Environment SH lighting (Phase 2B — HDRI spherical harmonics)
        if (P.envSHEnable) {
            // Use gradient normal at cloud surfaces, L0-only fallback in deep interior
            float3 envN;
            if (gradBlend > 0.f && (gradN.x*gradN.x+gradN.y*gradN.y+gradN.z*gradN.z) > 0.5f) {
                envN = normalize3(transformDir(P.fwdTransform, gradN));
            } else {
                envN = make_f3(0.f, 1.f, 0.f);
            }

            float3 env = evalEnvSH(P.envSH, envN);
            float envScale = sig_s * curStep * P.envIntensity;

            float envShadow = 1.0f;
            if (P.envShadowEnable) {
                float3 peakDir = make_f3(-P.envPeakDir[0], -P.envPeakDir[1], -P.envPeakDir[2]);
                float3 ldir;
                {
                    float pdx = P.invTransform[0]*peakDir.x + P.invTransform[4]*peakDir.y + P.invTransform[8]*peakDir.z;
                    float pdy = P.invTransform[1]*peakDir.x + P.invTransform[5]*peakDir.y + P.invTransform[9]*peakDir.z;
                    float pdz = P.invTransform[2]*peakDir.x + P.invTransform[6]*peakDir.y + P.invTransform[10]*peakDir.z;
                    float plen = rsqrtf(fmaxf(pdx*pdx+pdy*pdy+pdz*pdz, 1e-12f));
                    ldir = make_f3(pdx*plen, pdy*plen, pdz*plen);
                }
                float sJ = fmodf(sinf(t*91.17f+px*7.31f+py*13.97f)*43758.5453f, 1.f);
                if (sJ < 0) sJ += 1.f;
                __half hOD = __float2half(0.f);
                const __half hES = __float2half(ext * shadowStep * density.normScale);
                const __half hTh = __float2half(4.6f);
                for (int ls = 0; ls < nShadow; ++ls) {
                    float sOff = (ls + sJ) * shadowStep;
                    float3 sp = make_f3(wp.x+ldir.x*sOff, wp.y+ldir.y*sOff, wp.z+ldir.z*sOff);
                    float six = (sp.x - density.gridOffset[0]) * density.invVoxelSize;
                    float siy = (sp.y - density.gridOffset[1]) * density.invVoxelSize;
                    float siz = (sp.z - density.gridOffset[2]) * density.invVoxelSize;
                    int sl = int(floorf(six)) >> density.blockShift;
                    int sm = int(floorf(siy)) >> density.blockShift;
                    int sn = int(floorf(siz)) >> density.blockShift;
                    if (!isOccupied(density, sl, sm, sn)) continue;
                    __half hD = shadowSampleHalf(density, six, siy, siz);
                    if (__hle(hD, __float2half(1e-6f))) continue;
                    hOD = __hadd(hOD, __hmul(hD, hES));
                    if (__hge(hOD, hTh)) break;
                }
                envShadow = expf(-__half2float(hOD));
            }

            float envShadowR = fmaxf(envShadow, P.envShadowLift[0]);
            float envShadowG = fmaxf(envShadow, P.envShadowLift[1]);
            float envShadowB = fmaxf(envShadow, P.envShadowLift[2]);

            float eR = envScale * Tr * fmaxf(env.x, 0.f) * envShadowR;
            float eG = envScale * Tg * fmaxf(env.y, 0.f) * envShadowG;
            float eB = envScale * Tb * fmaxf(env.z, 0.f) * envShadowB;
            envScR += eR; envScG += eG; envScB += eB;
            scR += eR * P.envMix; scG += eG * P.envMix; scB += eB * P.envMix;
        }

        // Self-illumination phase: density gradient → HG direction
        // Forward-difference gradient (3 lookups, TL-cached so near-free).
        // Gradient points toward increasing density; we negate to get the
        // "outward" direction (toward thinner regions) as the effective
        // light direction — emission scatters outward from dense cores.
        float siPhase = 1.f;  // 1.0 = isotropic (default when phase disabled)
        if(P.emSelfIllumPhase && P.emSelfIllum>0.f){
            int ix0=int(floorf(ix)),iy0=int(floorf(iy)),iz0=int(floorf(iz));
            float gx=lookupVoxelTL(density,ix0+1,iy0,iz0,densTC)-dens;
            float gy=lookupVoxelTL(density,ix0,iy0+1,iz0,densTC)-dens;
            float gz=lookupVoxelTL(density,ix0,iy0,iz0+1,densTC)-dens;
            float glen=gx*gx+gy*gy+gz*gz;
            if(glen>1e-12f){
                float invG=rsqrtf(glen);
                // cosθ = dot(viewRay, -gradientDir):  negative because
                // outward = opposite to gradient direction
                float cosTheta=-(rd.x*gx+rd.y*gy+rd.z*gz)*invG;
                siPhase=evalPhase(P.phaseMode,cosTheta,P.phaseG1,P.phaseG2,P.phaseMix);
            }
            // glen ≈ 0 → uniform density region → isotropic (siPhase stays 1.0)
        }

        // Emission (separate codebook grid)
        if(P.hasEmission && emScale>0.f){
            float eix=(wp.x-emission.gridOffset[0])*emission.invVoxelSize;
            float eiy=(wp.y-emission.gridOffset[1])*emission.invVoxelSize;
            float eiz=(wp.z-emission.gridOffset[2])*emission.invVoxelSize;
            float emVal;
            if(P.scatterSmooth>0.f){
                int ss=int(P.scatterSmooth); int ss2=ss*2;
                int gex=int(floorf(eix)),gey=int(floorf(eiy)),gez=int(floorf(eiz));
                float ed = lookupVoxel(emission,gex,gey,gez) * 4.f;
                ed += 2.f*(lookupVoxel(emission,gex+ss,gey,gez)
                          +lookupVoxel(emission,gex-ss,gey,gez)
                          +lookupVoxel(emission,gex,gey+ss,gez)
                          +lookupVoxel(emission,gex,gey-ss,gez)
                          +lookupVoxel(emission,gex,gey,gez+ss)
                          +lookupVoxel(emission,gex,gey,gez-ss));
                ed += lookupVoxel(emission,gex+ss2,gey,gez)
                     +lookupVoxel(emission,gex-ss2,gey,gez)
                     +lookupVoxel(emission,gex,gey+ss2,gez)
                     +lookupVoxel(emission,gex,gey-ss2,gez)
                     +lookupVoxel(emission,gex,gey,gez+ss2)
                     +lookupVoxel(emission,gex,gey,gez-ss2);
                emVal = fmaxf(ed * (1.f/22.f), 0.f);
            } else {
                emVal=sampleNearest(emission,eix,eiy,eiz);
            }
            if(emVal>0.f){float norm=fminf(emVal*emScale,1.f);float3 ec=rampColor(norm,P.rampColors);
                float es_base=norm*curStep;
                emR+=ec.x*es_base*Tr;emG+=ec.y*es_base*Tg;emB+=ec.z*es_base*Tb;emA+=es_base*Tg;
                // Self-illumination: emission as co-located light source
                if(P.emSelfIllum>0.f){
                    float si_base=sig_s*norm*curStep*P.emSelfIllum*siPhase;
                    scR+=ec.x*si_base*Tr; scG+=ec.y*si_base*Tg; scB+=ec.z*si_base*Tb;}}}

        // Temperature (separate codebook grid)
        if(P.hasTemperature && tmpScale>0.f){
            float tix=(wp.x-temperature.gridOffset[0])*temperature.invVoxelSize;
            float tiy=(wp.y-temperature.gridOffset[1])*temperature.invVoxelSize;
            float tiz=(wp.z-temperature.gridOffset[2])*temperature.invVoxelSize;
            float tpVal;
            if(P.scatterSmooth>0.f){
                int ss=int(P.scatterSmooth); int ss2=ss*2;
                int gtx=int(floorf(tix)),gty=int(floorf(tiy)),gtz=int(floorf(tiz));
                float td = lookupVoxel(temperature,gtx,gty,gtz) * 4.f;
                td += 2.f*(lookupVoxel(temperature,gtx+ss,gty,gtz)
                          +lookupVoxel(temperature,gtx-ss,gty,gtz)
                          +lookupVoxel(temperature,gtx,gty+ss,gtz)
                          +lookupVoxel(temperature,gtx,gty-ss,gtz)
                          +lookupVoxel(temperature,gtx,gty,gtz+ss)
                          +lookupVoxel(temperature,gtx,gty,gtz-ss));
                td += lookupVoxel(temperature,gtx+ss2,gty,gtz)
                     +lookupVoxel(temperature,gtx-ss2,gty,gtz)
                     +lookupVoxel(temperature,gtx,gty+ss2,gtz)
                     +lookupVoxel(temperature,gtx,gty-ss2,gtz)
                     +lookupVoxel(temperature,gtx,gty,gtz+ss2)
                     +lookupVoxel(temperature,gtx,gty,gtz-ss2);
                tpVal = fmaxf(td * (1.f/22.f), 0.f);
            } else {
                tpVal=sampleNearest(temperature,tix,tiy,tiz);
            }
            if(tpVal>0.f){float norm=fminf(tpVal*tmpScale,1.f);
                float3 rc=rampColor(norm,P.rampColors);
                float3 tc=rc;
                if(P.useBlackbody){float3 bb=blackbodyColor(norm*P.bbKelvinScale);
                    bb.x*=P.bbTint[0]; bb.y*=P.bbTint[1]; bb.z*=P.bbTint[2];
                    float m=P.bbMix;
                    tc=make_f3(rc.x+(bb.x-rc.x)*m, rc.y+(bb.y-rc.y)*m, rc.z+(bb.z-rc.z)*m);}
                float ts_base=norm*curStep;
                tpR+=tc.x*ts_base*Tr;tpG+=tc.y*ts_base*Tg;tpB+=tc.z*ts_base*Tb;tpA+=ts_base*Tg;
                // Self-illumination from temperature
                if(P.emSelfIllum>0.f){
                    float si_base=sig_s*norm*curStep*P.emSelfIllum*siPhase;
                    scR+=tc.x*si_base*Tr; scG+=tc.y*si_base*Tg; scB+=tc.z*si_base*Tb;}}}

        // Chromatic transmittance update (Hillaire 2015)
        Tr*=expf(-dens*extR*curStep);
        Tg*=expf(-dens*extG*curStep);
        Tb*=expf(-dens*extB*curStep);

        // ── Deep sample output (alongside normal flat accumulation) ──
        // Alpha-only deep: stores front, back, alpha per step slab.
        // RGBA beauty comes from the flat render. Deep is for holdout/compositing.
        // When deepBuf == nullptr (flat path), this block is skipped entirely.
        if (P.deepMode && deepBuf) {
            float stepAlpha = 1.0f - expf(-dens * P.extinction * curStep);

            if (stepAlpha > 1e-6f) {
                int pidx_deep = py * P.width + px;
                int idx = atomicAdd(&deepCounts[pidx_deep], 1);
                if (idx < P.maxDeepSamples) {
                    int dbase = pidx_deep * P.maxDeepSamples * 3;
                    int doff = dbase + idx * 3;
                    deepBuf[doff + 0] = __float2half(t);               // front depth
                    deepBuf[doff + 1] = __float2half(t + curStep);     // back depth
                    deepBuf[doff + 2] = __float2half(stepAlpha);       // alpha
                }
            }
            prevScR=scR; prevScG=scG; prevScB=scB;
            prevEmR=emR; prevEmG=emG; prevEmB=emB;
            prevTpR=tpR; prevTpG=tpG; prevTpB=tpB;
        }

        // Post-accumulation: grow step if density is thin
        if(dens<adaptThreshLo) curStep=fminf(curStep*2.f, maxStep);

        t+=curStep;
    }

    float alpha=1.f-Tg;  // green channel = base for compositing alpha
    output[pidx]=__float2half(scR+emR+tpR); output[pidx+1]=__float2half(scG+emG+tpG);
    output[pidx+2]=__float2half(scB+emB+tpB); output[pidx+3]=__float2half(alpha);

    int off=4;
    if(P.aovFlags&1){output[pidx+off]=__float2half(scR);output[pidx+off+1]=__float2half(scG);
        output[pidx+off+2]=__float2half(scB);output[pidx+off+3]=__float2half(alpha);off+=4;}
    if(P.aovFlags&2){output[pidx+off]=__float2half(emR);output[pidx+off+1]=__float2half(emG);
        output[pidx+off+2]=__float2half(emB);output[pidx+off+3]=__float2half(fminf(emA,1.f));off+=4;}
    if(P.aovFlags&4){output[pidx+off]=__float2half(tpR);output[pidx+off+1]=__float2half(tpG);
        output[pidx+off+2]=__float2half(tpB);output[pidx+off+3]=__float2half(fminf(tpA,1.f));off+=4;}
    for(int li=0;li<3;++li)if(P.aovFlags&(8<<li)){
        output[pidx+off]=__float2half(plR[li]);output[pidx+off+1]=__float2half(plG[li]);
        output[pidx+off+2]=__float2half(plB[li]);output[pidx+off+3]=__float2half(alpha);off+=4;}
    if(P.aovFlags&64){__half a=__float2half(alb),al=__float2half(alpha);
        output[pidx+off]=a;output[pidx+off+1]=a;output[pidx+off+2]=a;output[pidx+off+3]=al;off+=4;}
    if(P.aovFlags&128){output[pidx+off]=__float2half(firstHitDepth);
        output[pidx+off+1]=__float2half(0.f);output[pidx+off+2]=__float2half(0.f);
        output[pidx+off+3]=__float2half(gotFirstHit?1.f:0.f);off+=4;}
    if(P.aovFlags&256){output[pidx+off]=__float2half(firstHitPx);
        output[pidx+off+1]=__float2half(firstHitPy);output[pidx+off+2]=__float2half(firstHitPz);
        output[pidx+off+3]=__float2half(gotFirstHit?1.f:0.f);off+=4;}
    if(P.aovFlags&512){output[pidx+off]=__float2half(firstHitPrefX);
        output[pidx+off+1]=__float2half(firstHitPrefY);output[pidx+off+2]=__float2half(firstHitPrefZ);
        output[pidx+off+3]=__float2half(gotFirstHit?1.f:0.f);off+=4;}
    if(P.aovFlags&1024){output[pidx+off]=__float2half(firstHitNx);
        output[pidx+off+1]=__float2half(firstHitNy);output[pidx+off+2]=__float2half(firstHitNz);
        output[pidx+off+3]=__float2half(gotFirstHit?1.f:0.f);off+=4;}
    if(P.aovFlags&2048){output[pidx+off]=__float2half(envScR);
        output[pidx+off+1]=__float2half(envScG);output[pidx+off+2]=__float2half(envScB);
        output[pidx+off+3]=__float2half(alpha);off+=4;}
}

// ═══════════════════════════════════════════════════════════════════════════
// Deep slab merge kernel — compacts consecutive low-contribution slabs
// One thread per pixel. Runs in-place on the deep buffer.
// ═══════════════════════════════════════════════════════════════════════════

__global__ void mergeDeepSlabs(
    __half* deepBuf, int* counts,
    int width, int height, int maxSamples, float mergeThresh)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    int pidx = py * width + px;
    int n = min(counts[pidx], maxSamples);
    if (n <= 1) return;

    int base = pidx * maxSamples * 3;
    int out = 0;

    for (int i = 0; i < n; ) {
        float frontT = __half2float(deepBuf[base + i*3 + 0]);
        float backT  = __half2float(deepBuf[base + i*3 + 1]);
        float mA     = __half2float(deepBuf[base + i*3 + 2]);

        // Merge forward while contribution stays below threshold
        int j = i + 1;
        while (j < n && __half2float(deepBuf[base + j*3 + 2]) < mergeThresh) {
            backT = __half2float(deepBuf[base + j*3 + 1]);
            mA   += __half2float(deepBuf[base + j*3 + 2]);
            ++j;
        }

        // Write compacted slab
        deepBuf[base + out*3 + 0] = __float2half(frontT);
        deepBuf[base + out*3 + 1] = __float2half(backT);
        deepBuf[base + out*3 + 2] = __float2half(mA);
        ++out;
        i = j;
    }
    counts[pidx] = out;
}

// ═══════════════════════════════════════════════════════════════════════════
// Host implementation
// ═══════════════════════════════════════════════════════════════════════════

VDBCUDARenderer::VDBCUDARenderer(){}
VDBCUDARenderer::~VDBCUDARenderer(){shutdown();}

bool VDBCUDARenderer::initialize(){
    if(_initialized)return true;
    int dc=0;cudaError_t e=cudaGetDeviceCount(&dc);if(e!=cudaSuccess||dc==0){_lastError="No CUDA";return false;}
    cudaDeviceProp p;cudaGetDeviceProperties(&p,0);
    if(p.major<7||(p.major==7&&p.minor<5)){_lastError="CC 7.5+ needed";return false;}

    // ── Occupancy-based block size tuning ──
    int minGridSize=0, optBlockSize=256;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize,&optBlockSize,
        codebookRayMarchKernel,0/*dynSharedMem*/,0/*blockSizeLimit=no limit*/);
    // Factorize 1D → 2D: prefer wider X (better coalescing), keep Y ≥ 4
    // Target roughly square: try 16-wide first, fall back to 8-wide
    if(optBlockSize>=256)      {_blkDimX=16;_blkDimY=16;}
    else if(optBlockSize>=192) {_blkDimX=16;_blkDimY=12;}
    else if(optBlockSize>=128) {_blkDimX=16;_blkDimY=8;}
    else if(optBlockSize>=64)  {_blkDimX=8; _blkDimY=8;}
    else                       {_blkDimX=8; _blkDimY=4;}

    _initialized=true;
    std::printf("[VDBRender] GPU: %s (%.0f MB, CC %d.%d)\n",p.name,p.totalGlobalMem/(1024.*1024.),p.major,p.minor);
    return true;}

void VDBCUDARenderer::CodebookGridGPU::free(){
    if(d_codebook){cudaFree(d_codebook);d_codebook=nullptr;}
    if(d_codebookHalf){cudaFree(d_codebookHalf);d_codebookHalf=nullptr;}
    if(d_indices){cudaFree(d_indices);d_indices=nullptr;}
    if(d_gainMaps){cudaFree(d_gainMaps);d_gainMaps=nullptr;}
    if(d_residuals){cudaFree(d_residuals);d_residuals=nullptr;}
    if(d_hashKeys){cudaFree(d_hashKeys);d_hashKeys=nullptr;}
    if(d_hashVals){cudaFree(d_hashVals);d_hashVals=nullptr;}
    if(d_macroGrid){cudaFree(d_macroGrid);d_macroGrid=nullptr;}
    hashTableSize=0;K=0;numLeaves=0;valid=false;
    allocCodebookElems=0;allocLeaves=0;allocHashSize=0;allocMacroElems=0;
    allocHasGain=false;allocHasResiduals=false;
    curHasGain=false;curHasResiduals=false;
    codebookFingerprint=0;}

void VDBCUDARenderer::invalidateGraph(){
    if(_graphExec){cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(_graphExec));_graphExec=nullptr;}
    _graphWidth=_graphHeight=_graphStride=_graphNumTiles=0;}

void VDBCUDARenderer::shutdown(){
    if(!_initialized)return;
    if(_asyncInFlight&&_renderEvent){cudaEventSynchronize(static_cast<cudaEvent_t>(_renderEvent));_asyncInFlight=false;}
    invalidateGraph();
    if(_renderStream){cudaStreamDestroy(static_cast<cudaStream_t>(_renderStream));_renderStream=nullptr;}
    if(_renderEvent){cudaEventDestroy(static_cast<cudaEvent_t>(_renderEvent));_renderEvent=nullptr;}
    if(_h_pinnedOutput){cudaFreeHost(_h_pinnedOutput);_h_pinnedOutput=nullptr;_pinnedSize=0;}
    if(_d_output){cudaFree(_d_output);_d_output=nullptr;}
    if(_d_outputF32){cudaFree(_d_outputF32);_d_outputF32=nullptr;_outputF32Capacity=0;}
    if(_d_tempOrigins){cudaFree(_d_tempOrigins);_d_tempOrigins=nullptr;_tempOriginsCapacity=0;}
    _density.free();_emission.free();_temperature.free();
    _initialized=false;}

// ── Shared grid upload ──────────────────────────────────────────────────

// [Phase 3] Codebook fingerprint for temporal sharing.
// Samples a few positions from the codebook data to produce a 64-bit hash.
// If the fingerprint matches between frames, the codebook is identical
// (shared across the sequence) → skip the expensive GPU upload.
// FNV-1a on first 16 + last 16 bytes + total size → practically unique.
static uint64_t codebookFingerprint(const void* data, size_t byteSize) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint64_t h = 14695981039346656037ULL; // FNV offset basis
    auto feed = [&](uint8_t b){ h ^= b; h *= 1099511628211ULL; };
    // Hash total size
    for (int i = 0; i < 8; ++i) feed(uint8_t(byteSize >> (i*8)));
    // Hash first 16 bytes
    size_t n = byteSize < 16 ? byteSize : 16;
    for (size_t i = 0; i < n; ++i) feed(p[i]);
    // Hash last 16 bytes
    if (byteSize > 16) {
        size_t start = byteSize - 16;
        for (size_t i = 0; i < 16; ++i) feed(p[start + i]);
    }
    // Hash 8 evenly spaced samples from the middle
    if (byteSize > 64) {
        size_t stride = byteSize / 10;
        for (int i = 1; i <= 8; ++i) {
            size_t off = i * stride;
            feed(p[off]); feed(p[off+1]); feed(p[off+2]); feed(p[off+3]);
        }
    }
    return h;
}

bool VDBCUDARenderer::uploadGrid(CodebookGridGPU& g,
    const float* codebook,uint32_t K,const uint16_t* indices,const int32_t* origins,uint32_t N,
    const float* gainMaps,const float* residuals,int blockSize,float voxelSize)
{
    invalidateGraph();
    int vpb=blockSize*blockSize*blockSize;
    uint32_t cbElems=(uint32_t)K*vpb;
    uint32_t hashSize=1; while(hashSize<N*2) hashSize<<=1;
    bool needGain=(gainMaps!=nullptr), needResiduals=(residuals!=nullptr);

    // ── Codebook: realloc only if K×vpb changed ──
    if(cbElems!=g.allocCodebookElems){
        if(g.d_codebook)cudaFree(g.d_codebook);
        if(g.d_codebookHalf)cudaFree(g.d_codebookHalf);
        g.d_codebook=nullptr; g.d_codebookHalf=nullptr;
        cudaError_t e=cudaMalloc(&g.d_codebook,(size_t)cbElems*sizeof(float));
        if(e!=cudaSuccess){_lastError="Codebook malloc fail";g.free();return false;}
        e=cudaMalloc(&g.d_codebookHalf,(size_t)cbElems*2);
        if(e!=cudaSuccess){_lastError="FP16 malloc fail";g.free();return false;}
        g.allocCodebookElems=cbElems;
        g.codebookFingerprint=0; // force upload after realloc
    }
    // [Phase 3] Temporal codebook sharing: skip upload if codebook is identical to previous frame
    uint64_t fp=codebookFingerprint(codebook,(size_t)cbElems*sizeof(float));
    if(fp!=g.codebookFingerprint){
        cudaMemcpy(g.d_codebook,codebook,(size_t)cbElems*sizeof(float),cudaMemcpyHostToDevice);
        // FP16 shadow codebook — convert on GPU (avoids CPU loop + extra H2D copy)
        convertF32ToF16<<<(cbElems+255)/256,256>>>(g.d_codebook,static_cast<__half*>(g.d_codebookHalf),cbElems);
        g.codebookFingerprint=fp;
    } else {
        // Codebook unchanged — reuse existing GPU data
    }

    // ── Per-leaf buffers: realloc only if capacity insufficient ──
    if(N>g.allocLeaves){
        // Grow with 25% headroom to avoid realloc on every frame
        uint32_t newCap=N+N/4;
        if(g.d_indices)cudaFree(g.d_indices); g.d_indices=nullptr;
        if(g.d_gainMaps)cudaFree(g.d_gainMaps); g.d_gainMaps=nullptr;
        if(g.d_residuals)cudaFree(g.d_residuals); g.d_residuals=nullptr;
        cudaMalloc(&g.d_indices,newCap*sizeof(uint16_t));
        if(needGain) cudaMalloc(&g.d_gainMaps,(size_t)newCap*8*sizeof(float));
        if(needResiduals) cudaMalloc(&g.d_residuals,(size_t)newCap*216*sizeof(float));
        g.allocLeaves=newCap; g.allocHasGain=needGain; g.allocHasResiduals=needResiduals;
    } else {
        // Capacity sufficient — just ensure gain/residual buffers exist if needed
        if(needGain && !g.allocHasGain){
            if(g.d_gainMaps)cudaFree(g.d_gainMaps);
            cudaMalloc(&g.d_gainMaps,(size_t)g.allocLeaves*8*sizeof(float));
            g.allocHasGain=true;
        }
        if(needResiduals && !g.allocHasResiduals){
            if(g.d_residuals)cudaFree(g.d_residuals);
            cudaMalloc(&g.d_residuals,(size_t)g.allocLeaves*216*sizeof(float));
            g.allocHasResiduals=true;
        }
    }
    cudaMemcpy(g.d_indices,indices,N*sizeof(uint16_t),cudaMemcpyHostToDevice);
    if(needGain) cudaMemcpy(g.d_gainMaps,gainMaps,(size_t)N*8*sizeof(float),cudaMemcpyHostToDevice);
    if(needResiduals) cudaMemcpy(g.d_residuals,residuals,(size_t)N*216*sizeof(float),cudaMemcpyHostToDevice);
    g.curHasGain=needGain; g.curHasResiduals=needResiduals;

    // ── Hash table: realloc only if size increased ──
    if(hashSize>g.allocHashSize){
        if(g.d_hashKeys)cudaFree(g.d_hashKeys);
        if(g.d_hashVals)cudaFree(g.d_hashVals);
        g.d_hashKeys=nullptr; g.d_hashVals=nullptr;
        cudaMalloc(&g.d_hashKeys,hashSize*sizeof(int64_t));
        cudaMalloc(&g.d_hashVals,hashSize*sizeof(uint32_t));
        g.allocHashSize=hashSize;
    }
    g.hashTableSize=hashSize;
    // [Phase 0] GPU hash clear — replaces per-frame host alloc + H2D copy
    fillHashEmpty<<<(hashSize+255)/256,256>>>(g.d_hashKeys,hashSize);
    cudaMemset(g.d_hashVals,0,hashSize*4);

    // ── Build hash table using persistent temp origins buffer ──
    if(N>_tempOriginsCapacity){
        if(_d_tempOrigins)cudaFree(_d_tempOrigins);
        uint32_t newCap=N+N/4;
        cudaMalloc(&_d_tempOrigins,newCap*12);
        _tempOriginsCapacity=newCap;
    }
    cudaMemcpy(_d_tempOrigins,origins,N*12,cudaMemcpyHostToDevice);
    buildHashTableKernel<<<(N+255)/256,256>>>(_d_tempOrigins,g.d_hashKeys,g.d_hashVals,hashSize-1,N,
        (blockSize==4)?2:3);
    cudaDeviceSynchronize();

    // ── Build coarse occupancy macro-grid ──
    {
        int bs=(blockSize==4)?2:3;
        int bMin[3]={INT_MAX,INT_MAX,INT_MAX}, bMax[3]={INT_MIN,INT_MIN,INT_MIN};
        for(uint32_t i=0;i<N;++i) for(int j=0;j<3;++j){
            int b=origins[i*3+j]>>bs;
            if(b<bMin[j]) bMin[j]=b; if(b>bMax[j]) bMax[j]=b;
        }
        int cs=3; // 8 blocks per macro-cell side
        for(int j=0;j<3;++j){ g.macroOrigin[j]=bMin[j]; g.macroGridDims[j]=((bMax[j]-bMin[j])>>cs)+1; }
        g.macroCellShift=cs;
        uint32_t macroElems=uint32_t(g.macroGridDims[0])*g.macroGridDims[1]*g.macroGridDims[2];
        if(macroElems>g.allocMacroElems){
            if(g.d_macroGrid)cudaFree(g.d_macroGrid);
            cudaMalloc(&g.d_macroGrid,macroElems); g.allocMacroElems=macroElems;
        }
        cudaMemset(g.d_macroGrid,0,macroElems);
        buildMacroGridKernel<<<(N+255)/256,256>>>(_d_tempOrigins,g.d_macroGrid,
            bMin[0],bMin[1],bMin[2],g.macroGridDims[0],g.macroGridDims[1],N,bs,cs);
        cudaDeviceSynchronize();
    }

    g.K=K;g.numLeaves=N;g.voxelSize=voxelSize;g.blockSize=blockSize;
    g.blockShift=(blockSize==4)?2:3;g.voxelsPerBlock=vpb;
    g.valid=true;
    ++g.uploadGeneration;
    return true;}

// ── FP16 codebook upload (CVD6) ─────────────────────────────────────────
// Uploads FP16 directly to d_codebookHalf, then GPU-converts to FP32 for d_codebook.
// Saves: 50% H2D bandwidth for codebook, zero CPU conversion.

bool VDBCUDARenderer::uploadGridFP16(CodebookGridGPU& g,
    const uint16_t* codebookFP16,uint32_t K,const uint16_t* indices,const int32_t* origins,uint32_t N,
    const float* gainMaps,int blockSize,float voxelSize)
{
    invalidateGraph();
    int vpb=blockSize*blockSize*blockSize;
    uint32_t cbElems=(uint32_t)K*vpb;
    uint32_t hashSize=1; while(hashSize<N*2) hashSize<<=1;
    bool needGain=(gainMaps!=nullptr);

    // ── Codebook: FP16 first, then GPU-convert to FP32 ──
    if(cbElems!=g.allocCodebookElems){
        if(g.d_codebook)cudaFree(g.d_codebook);
        if(g.d_codebookHalf)cudaFree(g.d_codebookHalf);
        g.d_codebook=nullptr; g.d_codebookHalf=nullptr;
        cudaMalloc(&g.d_codebookHalf,(size_t)cbElems*sizeof(uint16_t));
        cudaMalloc(&g.d_codebook,(size_t)cbElems*sizeof(float));
        g.allocCodebookElems=cbElems;
        g.codebookFingerprint=0; // force upload after realloc
    }
    // [Phase 3] Temporal codebook sharing: skip upload if codebook is identical to previous frame
    uint64_t fp=codebookFingerprint(codebookFP16,(size_t)cbElems*sizeof(uint16_t));
    if(fp!=g.codebookFingerprint){
        // Upload FP16 directly — half the bytes of FP32
        cudaMemcpy(g.d_codebookHalf,codebookFP16,(size_t)cbElems*sizeof(uint16_t),cudaMemcpyHostToDevice);
        // GPU-convert FP16 → FP32 for primary trilinear sampler
        convertF16ToF32<<<(cbElems+255)/256,256>>>(
            static_cast<const __half*>(g.d_codebookHalf),g.d_codebook,cbElems);
        g.codebookFingerprint=fp;
    } else {
        // Codebook unchanged — reuse existing GPU data
    }

    // ── Per-leaf buffers (same as FP32 path) ──
    if(N>g.allocLeaves){
        uint32_t newCap=N+N/4;
        if(g.d_indices)cudaFree(g.d_indices); g.d_indices=nullptr;
        if(g.d_gainMaps)cudaFree(g.d_gainMaps); g.d_gainMaps=nullptr;
        if(g.d_residuals)cudaFree(g.d_residuals); g.d_residuals=nullptr;
        cudaMalloc(&g.d_indices,newCap*sizeof(uint16_t));
        if(needGain) cudaMalloc(&g.d_gainMaps,(size_t)newCap*8*sizeof(float));
        g.allocLeaves=newCap; g.allocHasGain=needGain; g.allocHasResiduals=false;
    } else {
        if(needGain && !g.allocHasGain){
            if(g.d_gainMaps)cudaFree(g.d_gainMaps);
            cudaMalloc(&g.d_gainMaps,(size_t)g.allocLeaves*8*sizeof(float));
            g.allocHasGain=true;
        }
    }
    cudaMemcpy(g.d_indices,indices,N*sizeof(uint16_t),cudaMemcpyHostToDevice);
    if(needGain) cudaMemcpy(g.d_gainMaps,gainMaps,(size_t)N*8*sizeof(float),cudaMemcpyHostToDevice);
    g.curHasGain=needGain; g.curHasResiduals=false;

    // ── Hash table ──
    if(hashSize>g.allocHashSize){
        if(g.d_hashKeys)cudaFree(g.d_hashKeys);
        if(g.d_hashVals)cudaFree(g.d_hashVals);
        g.d_hashKeys=nullptr; g.d_hashVals=nullptr;
        cudaMalloc(&g.d_hashKeys,hashSize*sizeof(int64_t));
        cudaMalloc(&g.d_hashVals,hashSize*sizeof(uint32_t));
        g.allocHashSize=hashSize;
    }
    g.hashTableSize=hashSize;
    // [Phase 0] GPU hash clear
    fillHashEmpty<<<(hashSize+255)/256,256>>>(g.d_hashKeys,hashSize);
    cudaMemset(g.d_hashVals,0,hashSize*4);
    if(N>_tempOriginsCapacity){
        if(_d_tempOrigins)cudaFree(_d_tempOrigins);
        uint32_t newCap=N+N/4;
        cudaMalloc(&_d_tempOrigins,newCap*12);
        _tempOriginsCapacity=newCap;
    }
    cudaMemcpy(_d_tempOrigins,origins,N*12,cudaMemcpyHostToDevice);
    buildHashTableKernel<<<(N+255)/256,256>>>(_d_tempOrigins,g.d_hashKeys,g.d_hashVals,hashSize-1,N,
        (blockSize==4)?2:3);
    cudaDeviceSynchronize();

    // ── Build coarse occupancy macro-grid ──
    {
        int bs=(blockSize==4)?2:3;
        int bMin[3]={INT_MAX,INT_MAX,INT_MAX}, bMax[3]={INT_MIN,INT_MIN,INT_MIN};
        for(uint32_t i=0;i<N;++i) for(int j=0;j<3;++j){
            int b=origins[i*3+j]>>bs;
            if(b<bMin[j]) bMin[j]=b; if(b>bMax[j]) bMax[j]=b;
        }
        int cs=3;
        for(int j=0;j<3;++j){ g.macroOrigin[j]=bMin[j]; g.macroGridDims[j]=((bMax[j]-bMin[j])>>cs)+1; }
        g.macroCellShift=cs;
        uint32_t macroElems=uint32_t(g.macroGridDims[0])*g.macroGridDims[1]*g.macroGridDims[2];
        if(macroElems>g.allocMacroElems){
            if(g.d_macroGrid)cudaFree(g.d_macroGrid);
            cudaMalloc(&g.d_macroGrid,macroElems); g.allocMacroElems=macroElems;
        }
        cudaMemset(g.d_macroGrid,0,macroElems);
        buildMacroGridKernel<<<(N+255)/256,256>>>(_d_tempOrigins,g.d_macroGrid,
            bMin[0],bMin[1],bMin[2],g.macroGridDims[0],g.macroGridDims[1],N,bs,cs);
        cudaDeviceSynchronize();
    }

    g.K=K;g.numLeaves=N;g.voxelSize=voxelSize;g.blockSize=blockSize;
    g.blockShift=(blockSize==4)?2:3;g.voxelsPerBlock=vpb;
    g.valid=true;
    ++g.uploadGeneration;
    return true;}

// ── Public upload methods ───────────────────────────────────────────────

bool VDBCUDARenderer::uploadCodebookVolume(const float* cb,uint32_t K,const uint16_t* idx,const int32_t* orig,uint32_t N,
    const float* gm,const float* res,int bs,float vs,const double bmin[3],const double bmax[3],const double gridOffset[3],float normScale){
    if(!_initialized||!cb||!idx||!orig||N==0)return false;
    if(!uploadGrid(_density,cb,K,idx,orig,N,gm,res,bs,vs))return false;
    for(int i=0;i<3;++i){_bboxMin[i]=float(bmin[i]);_bboxMax[i]=float(bmax[i]);}
    if(gridOffset) for(int i=0;i<3;++i) _density.gridOffset[i]=float(gridOffset[i]);
    else for(int i=0;i<3;++i) _density.gridOffset[i]=0.f;
    _density.normScale=normScale;
    return true;}

bool VDBCUDARenderer::uploadEmissionVolume(const float* cb,uint32_t K,const uint16_t* idx,const int32_t* orig,uint32_t N,
    const float* gm,int bs,float vs,const double gridOffset[3],float normScale){
    if(!_initialized||!cb||N==0)return false;
    if(!uploadGrid(_emission,cb,K,idx,orig,N,gm,nullptr,bs,vs))return false;
    if(gridOffset) for(int i=0;i<3;++i) _emission.gridOffset[i]=float(gridOffset[i]);
    else for(int i=0;i<3;++i) _emission.gridOffset[i]=0.f;
    _emission.normScale=normScale;
    return true;}

bool VDBCUDARenderer::uploadTemperatureVolume(const float* cb,uint32_t K,const uint16_t* idx,const int32_t* orig,uint32_t N,
    const float* gm,int bs,float vs,const double gridOffset[3],float normScale){
    if(!_initialized||!cb||N==0)return false;
    if(!uploadGrid(_temperature,cb,K,idx,orig,N,gm,nullptr,bs,vs))return false;
    if(gridOffset) for(int i=0;i<3;++i) _temperature.gridOffset[i]=float(gridOffset[i]);
    else for(int i=0;i<3;++i) _temperature.gridOffset[i]=0.f;
    _temperature.normScale=normScale;
    return true;}

// ── FP16 public upload methods (CVD6) ───────────────────────────────────

bool VDBCUDARenderer::uploadCodebookVolumeFP16(const uint16_t* cb,uint32_t K,const uint16_t* idx,const int32_t* orig,uint32_t N,
    const float* gm,int bs,float vs,const double bmin[3],const double bmax[3],const double gridOffset[3],float normScale){
    if(!_initialized||!cb||!idx||!orig||N==0)return false;
    if(!uploadGridFP16(_density,cb,K,idx,orig,N,gm,bs,vs))return false;
    for(int i=0;i<3;++i){_bboxMin[i]=float(bmin[i]);_bboxMax[i]=float(bmax[i]);}
    if(gridOffset) for(int i=0;i<3;++i) _density.gridOffset[i]=float(gridOffset[i]);
    else for(int i=0;i<3;++i) _density.gridOffset[i]=0.f;
    _density.normScale=normScale;
    return true;}

bool VDBCUDARenderer::uploadEmissionVolumeFP16(const uint16_t* cb,uint32_t K,const uint16_t* idx,const int32_t* orig,uint32_t N,
    const float* gm,int bs,float vs,const double gridOffset[3],float normScale){
    if(!_initialized||!cb||N==0)return false;
    if(!uploadGridFP16(_emission,cb,K,idx,orig,N,gm,bs,vs))return false;
    if(gridOffset) for(int i=0;i<3;++i) _emission.gridOffset[i]=float(gridOffset[i]);
    else for(int i=0;i<3;++i) _emission.gridOffset[i]=0.f;
    _emission.normScale=normScale;
    return true;}

bool VDBCUDARenderer::uploadTemperatureVolumeFP16(const uint16_t* cb,uint32_t K,const uint16_t* idx,const int32_t* orig,uint32_t N,
    const float* gm,int bs,float vs,const double gridOffset[3],float normScale){
    if(!_initialized||!cb||N==0)return false;
    if(!uploadGridFP16(_temperature,cb,K,idx,orig,N,gm,bs,vs))return false;
    if(gridOffset) for(int i=0;i<3;++i) _temperature.gridOffset[i]=float(gridOffset[i]);
    else for(int i=0;i<3;++i) _temperature.gridOffset[i]=0.f;
    _temperature.normScale=normScale;
    return true;}

void VDBCUDARenderer::clearCodebook(){_density.free();invalidateGraph();}
void VDBCUDARenderer::clearEmission(){_emission.free();invalidateGraph();}
void VDBCUDARenderer::clearTemperature(){_temperature.free();invalidateGraph();}

// ── Output buffer ───────────────────────────────────────────────────────

void VDBCUDARenderer::ensureOutputBuffer(int w,int h,int s){
    if(_d_output&&_outputWidth==w&&_outputHeight==h&&s==_lastStride)return;
    if(_d_output)cudaFree(_d_output);size_t b=size_t(w)*h*s*sizeof(__half);
    cudaMalloc(&_d_output,b);cudaMemset(_d_output,0,b);_outputWidth=w;_outputHeight=h;_lastStride=s;
    invalidateGraph();}

// [Phase 0] FP32 device buffer for GPU readback conversion
void VDBCUDARenderer::ensureF32Buffer(size_t elems){
    if(_d_outputF32&&_outputF32Capacity>=elems)return;
    if(_d_outputF32)cudaFree(_d_outputF32);
    cudaMalloc(&_d_outputF32,elems*sizeof(float));
    _outputF32Capacity=elems;}

// ── Build GridPtrs from CodebookGridGPU ─────────────────────────────────

static GridPtrs makeGridPtrs(const VDBCUDARenderer::CodebookGridGPU& g){
    GridPtrs p={};
    if(!g.valid)return p;
    p.hashKeys=g.d_hashKeys;p.hashVals=g.d_hashVals;p.hashMask=g.hashTableSize-1;
    p.indices=g.d_indices;p.codebook=g.d_codebook;p.codebookHalf=static_cast<const __half*>(g.d_codebookHalf);
    p.gainMaps=g.curHasGain ? g.d_gainMaps : nullptr;
    p.residuals=g.curHasResiduals ? g.d_residuals : nullptr;
    p.blockShift=g.blockShift;p.blkDim=g.blockSize;p.vpb=g.voxelsPerBlock;
    p.invVoxelSize=1.f/g.voxelSize;p.voxelSize=g.voxelSize;
    for(int i=0;i<3;++i) p.gridOffset[i]=g.gridOffset[i];
    p.normScale=g.normScale;
    p.macroGrid=g.d_macroGrid;
    for(int i=0;i<3;++i){ p.macroGridDims[i]=g.macroGridDims[i]; p.macroOrigin[i]=g.macroOrigin[i]; }
    p.macroCellShift=g.macroCellShift;
    return p;}

// [Phase 0] Build RayMarchParams from VDBRenderConfig + renderer internal state.
// This is the ONLY place where config → kernel params mapping happens.
static RayMarchParams buildParamsFromConfig(const VDBRenderConfig& cfg,
    float voxelSize, const float bmin[3], const float bmax[3],
    int hasEmission, int hasTemperature)
{
    RayMarchParams P = {};
    P.width = cfg.renderW;  P.height = cfg.renderH;
    P.fmtWidth = cfg.fmtW;  P.fmtHeight = cfg.fmtH;
    P.bboxOffsetX = cfg.bboxOffsetX;  P.bboxOffsetY = cfg.bboxOffsetY;

    memcpy(P.camOrigin, cfg.camOrigin, sizeof(P.camOrigin));
    memcpy(P.camRot, cfg.camRot, sizeof(P.camRot));
    P.halfW = cfg.halfW;  P.halfH = cfg.halfH;

    P.stepSize = cfg.stepSize;  P.extinction = cfg.extinction;  P.scattering = cfg.scattering;
    P.extinctionR = cfg.extinctionR;  P.extinctionG = cfg.extinctionG;  P.extinctionB = cfg.extinctionB;
    P.boundaryBlend = cfg.boundaryBlend;

    P.shadowSteps = cfg.shadowSteps;  P.shadowStepScale = cfg.shadowStepScale;

    P.phaseG1 = cfg.phaseG1;  P.phaseG2 = cfg.phaseG2;
    P.phaseMix = cfg.phaseMix;  P.phaseMode = cfg.phaseMode;

    P.powderStrength = cfg.powderStrength;
    P.emSelfIllum = cfg.emSelfIllum;  P.emSelfIllumPhase = cfg.emSelfIllumPhase;

    P.msEnable = cfg.msEnable;  P.msOctaves = cfg.msOctaves;
    P.msExtFalloff = cfg.msExtFalloff;  P.msScatterFalloff = cfg.msScatterFalloff;

    P.gradientMix = cfg.gradientMix;  P.gradientThreshold = cfg.gradientThreshold;
    memcpy(P.gradNormalBias, cfg.gradNormalBias, sizeof(P.gradNormalBias));
    P.gradientSmooth = cfg.gradientSmooth;
    P.scatterSmooth = cfg.scatterSmooth;
    P.densityBlur = cfg.densityBlur;

    P.noiseAmplitude = cfg.noiseAmplitude;  P.noiseFrequency = cfg.noiseFrequency;
    memcpy(P.noiseScale, cfg.noiseScale, sizeof(P.noiseScale));
    memcpy(P.noiseOffset, cfg.noiseOffset, sizeof(P.noiseOffset));
    P.noiseOctaves = cfg.noiseOctaves;

    P.emissionScale = cfg.emissionScale;  P.temperatureScale = cfg.temperatureScale;
    P.useBlackbody = cfg.useBlackbody;  P.bbKelvinScale = cfg.bbKelvinScale;
    P.bbMix = cfg.bbMix;  memcpy(P.bbTint, cfg.bbTint, sizeof(P.bbTint));

    memcpy(P.rampColors, cfg.rampColors, sizeof(P.rampColors));

    P.ambientIntensity = cfg.ambientIntensity;
    memcpy(P.ambientColor, cfg.ambientColor, sizeof(P.ambientColor));

    P.envSHEnable = cfg.envSHEnable;
    P.envIntensity = cfg.envIntensity;
    P.envShadowEnable = cfg.envShadowEnable;
    memcpy(P.envShadowLift, cfg.envShadowLift, sizeof(P.envShadowLift));
    memcpy(P.envSH, cfg.envSH, sizeof(P.envSH));
    memcpy(P.envPeakDir, cfg.envPeakDir, sizeof(P.envPeakDir));
    memcpy(P.envPeakColor, cfg.envPeakColor, sizeof(P.envPeakColor));

    P.numLights = cfg.numLights;
    for (int i = 0; i < cfg.numLights; ++i)
        P.lights[i] = cfg.lights[i];

    P.aovFlags = cfg.aovFlags;  P.outputStride = cfg.outputStride;
    P.hasEmission = hasEmission;  P.hasTemperature = hasTemperature;

    memcpy(P.invTransform, cfg.volumeInvTransform, sizeof(P.invTransform));
    memcpy(P.fwdTransform, cfg.volumeFwdTransform, sizeof(P.fwdTransform));
    for (int i = 0; i < 3; ++i) { P.bboxMin[i] = bmin[i]; P.bboxMax[i] = bmax[i]; }

    P.tileYOffset = 0;  P.tileHeight = cfg.renderH;

    P.deepMode = 0;
    P.maxDeepSamples = 0;

    memcpy(P.lightMix, cfg.lightMix, sizeof(P.lightMix));
    P.envMix = cfg.envMix;

    return P;
}

// ── Render ──────────────────────────────────────────────────────────────

// ── [Phase 0] Render — config-based, GPU readback conversion ────────────

bool VDBCUDARenderer::render(const VDBRenderConfig& cfg, float* out)
{
    if(!_initialized||!_density.valid){_lastError="Not ready";return false;}
    int w=cfg.renderW, h=cfg.renderH, os=cfg.outputStride;

    VDBProfiler prof;
    prof.enabled = cfg.profileEnable != 0;
    prof.reset();

    prof.begin("Build params");
    ensureOutputBuffer(w,h,os);
    auto P=buildParamsFromConfig(cfg, _density.voxelSize, _bboxMin, _bboxMax,
                                 _emission.valid?1:0, _temperature.valid?1:0);
    GridPtrs gd=makeGridPtrs(_density),ge=makeGridPtrs(_emission),gt=makeGridPtrs(_temperature);
    prof.end();

    prof.end_gpu();

    dim3 blk(_blkDimX,_blkDimY),grd((w+_blkDimX-1)/_blkDimX,(h+_blkDimY-1)/_blkDimY);

    prof.begin_gpu("Ray march kernel");
    codebookRayMarchKernel<<<grd,blk>>>(gd,ge,gt,static_cast<__half*>(_d_output),nullptr,nullptr,P);
    cudaError_t e=cudaGetLastError();if(e!=cudaSuccess){_lastError=cudaGetErrorString(e);return false;}
    prof.end_gpu();

    // GPU FP16→FP32 conversion + readback
    size_t n=size_t(w)*h*os;
    ensureF32Buffer(n);

    prof.begin_gpu("FP16 → FP32 convert");
    convertF16ToF32<<<(uint32_t(n)+255)/256,256>>>(static_cast<const __half*>(_d_output),_d_outputF32,uint32_t(n));
    prof.end_gpu();

    prof.begin_gpu("GPU → CPU readback");
    cudaMemcpy(out,_d_outputF32,n*sizeof(float),cudaMemcpyDeviceToHost);
    prof.end_gpu();

    prof.print_summary(w, h);
    return true;
}

// ── Deep Render ────────────────────────────────────────────────────────

bool VDBCUDARenderer::renderDeep(const VDBRenderConfig& cfg, int maxDeepSamples,
                                  uint16_t* hostDeepBuf, int* hostDeepCounts)
{
    if (!_initialized || !_density.valid) {
        _lastError = "Not ready";
        return false;
    }

    VDBProfiler prof;
    prof.enabled = cfg.profileEnable != 0;
    prof.reset();

    int w = cfg.renderW, h = cfg.renderH;
    int os = cfg.outputStride;
    size_t pixelCount = size_t(w) * h;
    size_t deepBufElems = pixelCount * maxDeepSamples * 3;
    size_t deepBytes = deepBufElems * sizeof(__half) + pixelCount * sizeof(int);

    cudaGetLastError();

    size_t freeMem = 0, totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);
    if (deepBytes > freeMem * 0.9) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "Deep buffer needs %.0f MB but only %.0f MB free (of %.0f MB total). "
            "Reduce resolution or max samples.",
            deepBytes / (1024.0*1024.0), freeMem / (1024.0*1024.0), totalMem / (1024.0*1024.0));
        _lastError = buf;
        return false;
    }

    prof.begin_gpu("Deep alloc + memset");
    __half* d_deepBuf = nullptr;
    int*    d_deepCounts = nullptr;
    cudaError_t e;
    e = cudaMalloc(&d_deepBuf, deepBufElems * sizeof(__half));
    if (e != cudaSuccess) {
        _lastError = std::string("Deep buf malloc failed: ") + cudaGetErrorString(e);
        return false;
    }
    e = cudaMalloc(&d_deepCounts, pixelCount * sizeof(int));
    if (e != cudaSuccess) {
        cudaFree(d_deepBuf);
        _lastError = std::string("Deep counts malloc failed: ") + cudaGetErrorString(e);
        return false;
    }
    cudaMemset(d_deepCounts, 0, pixelCount * sizeof(int));
    ensureOutputBuffer(w, h, os);
    prof.end_gpu();

    prof.begin("Build params");
    auto P = buildParamsFromConfig(cfg, _density.voxelSize, _bboxMin, _bboxMax,
                                   _emission.valid ? 1 : 0, _temperature.valid ? 1 : 0);
    P.deepMode = 1;
    P.maxDeepSamples = maxDeepSamples;
    GridPtrs gd = makeGridPtrs(_density);
    GridPtrs ge = makeGridPtrs(_emission);
    GridPtrs gt = makeGridPtrs(_temperature);
    prof.end();

    prof.end_gpu();

    dim3 blk(_blkDimX, _blkDimY);
    dim3 grd((w + _blkDimX - 1) / _blkDimX, (h + _blkDimY - 1) / _blkDimY);

    prof.begin_gpu("Deep ray march kernel");
    codebookRayMarchKernel<<<grd, blk>>>(
        gd, ge, gt,
        static_cast<__half*>(_d_output),
        d_deepBuf, d_deepCounts, P);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        _lastError = std::string("Deep kernel launch: ") + cudaGetErrorString(e);
        cudaFree(d_deepBuf); cudaFree(d_deepCounts);
        return false;
    }
    prof.end_gpu();

    prof.begin_gpu("Deep merge slabs");
    float mergeThresh = 0.5f / std::max(1, maxDeepSamples);
    dim3 mBlk(16, 16);
    dim3 mGrd((w + 15) / 16, (h + 15) / 16);
    mergeDeepSlabs<<<mGrd, mBlk>>>(
        d_deepBuf, d_deepCounts, w, h, maxDeepSamples, mergeThresh);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        _lastError = std::string("Merge kernel: ") + cudaGetErrorString(e);
        cudaFree(d_deepBuf); cudaFree(d_deepCounts);
        return false;
    }
    prof.end_gpu();

    prof.begin_gpu("Deep GPU → CPU readback");
    cudaMemcpy(hostDeepBuf, d_deepBuf,
               deepBufElems * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostDeepCounts, d_deepCounts,
               pixelCount * sizeof(int), cudaMemcpyDeviceToHost);
    prof.end_gpu();

    cudaFree(d_deepBuf);
    cudaFree(d_deepCounts);

    prof.print_summary(w, h);
    return true;
}

// ── Async ───────────────────────────────────────────────────────────────

void VDBCUDARenderer::ensureStream(){if(!_renderStream){cudaStream_t s;cudaStreamCreate(&s);_renderStream=s;
    cudaEvent_t e;cudaEventCreateWithFlags(&e,cudaEventDisableTiming);_renderEvent=e;}}
void VDBCUDARenderer::ensurePinnedBuffer(size_t b){if(_h_pinnedOutput&&_pinnedSize>=b)return;
    if(_h_pinnedOutput){cudaFreeHost(_h_pinnedOutput);_h_pinnedOutput=nullptr;}
    if(cudaMallocHost(&_h_pinnedOutput,b)!=cudaSuccess){_h_pinnedOutput=nullptr;_pinnedSize=0;return;}_pinnedSize=b;
    invalidateGraph();}

bool VDBCUDARenderer::renderAsync(const VDBRenderConfig& cfg)
{
    if(!_initialized||!_density.valid){_lastError="Not ready";return false;}
    int w=cfg.renderW, h=cfg.renderH, os=cfg.outputStride;
    ensureStream();auto stream=static_cast<cudaStream_t>(_renderStream);auto event=static_cast<cudaEvent_t>(_renderEvent);
    if(_asyncInFlight){cudaEventSynchronize(event);_asyncInFlight=false;}
    ensureOutputBuffer(w,h,os);size_t bytes=size_t(w)*h*os*2;ensurePinnedBuffer(bytes);
    if(!_h_pinnedOutput){_lastError="Pin fail";return false;}

    auto P=buildParamsFromConfig(cfg, _density.voxelSize, _bboxMin, _bboxMax,
                                 _emission.valid?1:0, _temperature.valid?1:0);
    GridPtrs gd=makeGridPtrs(_density),ge=makeGridPtrs(_emission),gt=makeGridPtrs(_temperature);


    const int TILE_H=64;
    int numTiles=(h+TILE_H-1)/TILE_H;
    __half* d_out=static_cast<__half*>(_d_output);

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for(int tile=0;tile<numTiles;++tile){
        int yStart=tile*TILE_H;
        int thisTileH=(yStart+TILE_H<=h)?TILE_H:(h-yStart);
        P.tileYOffset=yStart; P.tileHeight=thisTileH;
        dim3 blk(_blkDimX,_blkDimY),grd((w+_blkDimX-1)/_blkDimX,(thisTileH+_blkDimY-1)/_blkDimY);
        codebookRayMarchKernel<<<grd,blk,0,stream>>>(gd,ge,gt,d_out,nullptr,nullptr,P);
        size_t rowHalfs=size_t(w)*os;
        size_t tileOffset=size_t(yStart)*rowHalfs*sizeof(__half);
        size_t tileBytes=size_t(thisTileH)*rowHalfs*sizeof(__half);
        cudaMemcpyAsync(reinterpret_cast<char*>(_h_pinnedOutput)+tileOffset,
                        reinterpret_cast<char*>(d_out)+tileOffset,
                        tileBytes,cudaMemcpyDeviceToHost,stream);
    }
    cudaGraph_t newGraph;
    cudaStreamEndCapture(stream, &newGraph);

    bool structMatch = _graphExec && _graphWidth==w && _graphHeight==h
                       && _graphStride==os && _graphNumTiles==numTiles;
    if(structMatch){
        cudaGraphExecUpdateResultInfo updateResult;
        cudaError_t upd=cudaGraphExecUpdate(static_cast<cudaGraphExec_t>(_graphExec),newGraph,&updateResult);
        if(upd!=cudaSuccess || updateResult.result!=cudaGraphExecUpdateSuccess){
            cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(_graphExec));
            _graphExec=nullptr; structMatch=false;
        }
    }
    if(!structMatch){
        if(_graphExec){cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(_graphExec));_graphExec=nullptr;}
        cudaGraphExec_t exec;
        cudaError_t inst=cudaGraphInstantiate(&exec,newGraph,0);
        if(inst!=cudaSuccess){cudaGraphDestroy(newGraph);_lastError="Graph instantiate fail";return false;}
        _graphExec=exec; _graphWidth=w; _graphHeight=h; _graphStride=os; _graphNumTiles=numTiles;
    }
    cudaGraphDestroy(newGraph);

    cudaGraphLaunch(static_cast<cudaGraphExec_t>(_graphExec),stream);
    cudaError_t e=cudaGetLastError();if(e!=cudaSuccess){_lastError=cudaGetErrorString(e);return false;}
    cudaEventRecord(event,stream);_asyncInFlight=true;_asyncWidth=w;_asyncHeight=h;_asyncStride=os;return true;
}

bool VDBCUDARenderer::isRenderComplete()const{if(!_asyncInFlight)return false;return cudaEventQuery(static_cast<cudaEvent_t>(_renderEvent))==cudaSuccess;}

bool VDBCUDARenderer::readbackResult(float* out,int np,int os)const{
    if(!_h_pinnedOutput||!_asyncInFlight)return false;size_t n=size_t(np)*os;if(n*2>_pinnedSize)return false;
    const uint16_t*s=reinterpret_cast<const uint16_t*>(_h_pinnedOutput);
    for(size_t i=0;i<n;++i){uint16_t v=s[i];uint32_t sg=(v&0x8000u)<<16;uint32_t ex=(v>>10)&0x1F;uint32_t m=v&0x3FF;
        uint32_t f;if(ex==0)f=sg;else if(ex==31)f=sg|0x7F800000u|(m<<13);else f=sg|((ex+112)<<23)|(m<<13);
        out[i]=*reinterpret_cast<float*>(&f);}return true;}
