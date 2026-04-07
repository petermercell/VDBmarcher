#pragma once
// ═══════════════════════════════════════════════════════════════════════════
// VDBRenderConfig.h — Single render configuration struct
// Built once per render in the Iop, passed by const-ref to the GPU renderer.
// Eliminates the 3-copy parameter forwarding: knobs → members → buildParams.
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdint>

struct GPULight {
    float position[3];     // world-space position
    float direction[3];    // normalized direction (position → target)
    float color[3];        // RGB color
    float intensity;       // scalar multiplier
    float coneAngle;       // half-angle in radians (beam spread)
    float coneSoftness;    // edge falloff (0 = hard edge, 1 = fully soft)
    int   enabled;
};

struct VDBRenderConfig {
    // ── Camera ──
    float camOrigin[3];
    float camRot[9];       // 3×3 row-major
    float halfW, halfH;

    // ── Render region ──
    int renderW, renderH;
    int fmtW, fmtH;
    int bboxOffsetX, bboxOffsetY;

    // ── Volume ──
    float stepSize;
    float extinction;
    float scattering;
    float volumeInvTransform[16];
    float volumeFwdTransform[16];
    float boundaryBlend;

    // ── Chromatic extinction (Hillaire 2015) ──
    float extinctionR, extinctionG, extinctionB;

    // ── Shadow ──
    int   shadowSteps;
    float shadowStepScale;

    // ── Phase function ──
    float phaseG1, phaseG2, phaseMix;
    int   phaseMode;         // 0=Iso, 1=HG, 2=Dual HG, 3=Mie

    // ── Powder effect (Schneider & Vos 2015) ──
    float powderStrength;

    // ── Multi-scatter octaves (Frostbite / Hillaire 2016) ──
    int   msEnable;
    int   msOctaves;
    float msExtFalloff;       // extinction reduction per octave
    float msScatterFalloff;   // contribution weight per octave

    // ── Gradient normals ──
    float gradientMix;
    float gradientThreshold;
    float gradNormalBias[3];
    float gradientSmooth;          // FD step size in voxels (1=sharp, 2-8=smooth)
    float scatterSmooth;           // density smooth radius for scattering (0=off)
    float densityBlur;             // raw density blur radius in voxels (0=off)

    // ── Procedural fBm noise ──
    float noiseAmplitude;
    float noiseFrequency;
    float noiseScale[3];
    float noiseOffset[3];
    int   noiseOctaves;

    // ── Emission ──
    float emissionScale;
    float temperatureScale;
    float emSelfIllum;
    int   emSelfIllumPhase;

    // ── Blackbody ──
    int   useBlackbody;
    float bbKelvinScale;
    float bbMix;
    float bbTint[3];

    // ── Ramp colors (5 stops × RGB) ──
    float rampColors[15];

    // ── Ambient ──
    float ambientColor[3];
    float ambientIntensity;

    // ── Environment SH (Phase 2B) ──
    float envSH[27];              // L2 SH coefficients (9 × 3 RGB)
    int   envSHEnable;            // HDRI loaded and valid
    float envIntensity;
    float envPeakDir[3];          // dominant direction for shadow rays
    float envPeakColor[3];        // color in dominant direction
    int   envShadowEnable;        // cast shadow ray along peak direction
    float envShadowLift[3];       // minimum shadow color (per-channel lift)

    // ── Lights ──
    int      numLights;
    GPULight lights[3];

    // ── AOV / output ──
    int   aovFlags;
    int   outputStride;
    int   profileEnable;

    // ── Light Mixer (RGBA only, AOVs stay 100%) ──
    float lightMix[3];
    float envMix;
};
