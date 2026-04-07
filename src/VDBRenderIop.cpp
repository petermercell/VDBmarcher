#include "VDBRenderIop.h"
#include "VDBCUDARenderer.h"
#include <DDImage/Knob.h>
#include <DDImage/Format.h>
#include <cmath>
#include <algorithm>
#include <cstdio>

using namespace DD::Image;

const char* VDBRenderIop::CLASS = "VDBRender";
const char* VDBRenderIop::HELP  =
    "Ray-march a CVDB codebook volume into RGBA.\n"
    "Reads .cvdb files directly on the GPU via CUDA.\n"
    "Connect a Camera node to input 0.\n"
    "Optionally connect an equirectangular HDRI to input 1 (env).\n"
    "3 built-in GPU lights with 3D viewport handles.";

Channel VDBRenderIop::chan_scatter_r = getChannel("scatter.red");
Channel VDBRenderIop::chan_scatter_g = getChannel("scatter.green");
Channel VDBRenderIop::chan_scatter_b = getChannel("scatter.blue");
Channel VDBRenderIop::chan_scatter_a = getChannel("scatter.alpha");
Channel VDBRenderIop::chan_light1_r = getChannel("light1.red");
Channel VDBRenderIop::chan_light1_g = getChannel("light1.green");
Channel VDBRenderIop::chan_light1_b = getChannel("light1.blue");
Channel VDBRenderIop::chan_light1_a = getChannel("light1.alpha");
Channel VDBRenderIop::chan_light2_r = getChannel("light2.red");
Channel VDBRenderIop::chan_light2_g = getChannel("light2.green");
Channel VDBRenderIop::chan_light2_b = getChannel("light2.blue");
Channel VDBRenderIop::chan_light2_a = getChannel("light2.alpha");
Channel VDBRenderIop::chan_light3_r = getChannel("light3.red");
Channel VDBRenderIop::chan_light3_g = getChannel("light3.green");
Channel VDBRenderIop::chan_light3_b = getChannel("light3.blue");
Channel VDBRenderIop::chan_light3_a = getChannel("light3.alpha");
Channel VDBRenderIop::chan_emission_r = getChannel("emission.red");
Channel VDBRenderIop::chan_emission_g = getChannel("emission.green");
Channel VDBRenderIop::chan_emission_b = getChannel("emission.blue");
Channel VDBRenderIop::chan_emission_a = getChannel("emission.alpha");
Channel VDBRenderIop::chan_temperature_r = getChannel("temperature.red");
Channel VDBRenderIop::chan_temperature_g = getChannel("temperature.green");
Channel VDBRenderIop::chan_temperature_b = getChannel("temperature.blue");
Channel VDBRenderIop::chan_temperature_a = getChannel("temperature.alpha");
Channel VDBRenderIop::chan_albedo_r = getChannel("albedo.red");
Channel VDBRenderIop::chan_albedo_g = getChannel("albedo.green");
Channel VDBRenderIop::chan_albedo_b = getChannel("albedo.blue");
Channel VDBRenderIop::chan_albedo_a = getChannel("albedo.alpha");
Channel VDBRenderIop::chan_depth_z  = getChannel("depth.Z");
Channel VDBRenderIop::chan_P_x     = getChannel("P.red");
Channel VDBRenderIop::chan_P_y     = getChannel("P.green");
Channel VDBRenderIop::chan_P_z     = getChannel("P.blue");
Channel VDBRenderIop::chan_P_a     = getChannel("P.alpha");
Channel VDBRenderIop::chan_Pref_x  = getChannel("Pref.red");
Channel VDBRenderIop::chan_Pref_y  = getChannel("Pref.green");
Channel VDBRenderIop::chan_Pref_z  = getChannel("Pref.blue");
Channel VDBRenderIop::chan_Pref_a  = getChannel("Pref.alpha");
Channel VDBRenderIop::chan_N_x     = getChannel("N.red");
Channel VDBRenderIop::chan_N_y     = getChannel("N.green");
Channel VDBRenderIop::chan_N_z     = getChannel("N.blue");
Channel VDBRenderIop::chan_N_a     = getChannel("N.alpha");
Channel VDBRenderIop::chan_env_r   = getChannel("env.red");
Channel VDBRenderIop::chan_env_g   = getChannel("env.green");
Channel VDBRenderIop::chan_env_b   = getChannel("env.blue");
Channel VDBRenderIop::chan_env_a   = getChannel("env.alpha");

VDBRenderIop::VDBRenderIop(Node* node) : Iop(node)
{
    inputs(2);
}

VDBRenderIop::~VDBRenderIop() { stopPrefetch(); }

void VDBRenderIop::knobs(Knob_Callback f)
{
    Format_knob(f, &_formats, "format", "Output Format");
    Int_knob(f, &_frameOffset, "frame_offset", "Frame Offset");
    ClearFlags(f, Knob::STARTLINE);

    Divider(f, "File");
    File_knob(f, &_vdbFilePath, "file", "CVDB File");
    Tooltip(f, "Path to a .cvdb codebook volume.\n\n"
               "Sequences:\n"
               "  smoke.####.cvdb  or  smoke.%04d.cvdb\n"
               "  Or just pick any file — auto-detects frame numbers.");

    Divider(f, "Volume Transform");
    XYZ_knob(f, _volTranslate, "vol_translate", "Translate");
    XYZ_knob(f, _volRotate, "vol_rotate", "Rotate");
    XYZ_knob(f, _volScale, "vol_scale", "Scale");
    SetFlags(f, Knob::STARTLINE);
    Double_knob(f, &_volUniformScale, "vol_uniform_scale", "Uniform Scale");

    Divider(f, "GPU");
    static const char* renderScaleOptions[] = {
        "Quarter (fast)", "Half", "3/4", "Full", nullptr
    };
    Enumeration_knob(f, &_gpuRenderScale, renderScaleOptions, "gpu_render_scale", "Render Scale");
    Int_knob(f, &_gpuShadowSteps, "gpu_shadow_steps", "Shadow Steps");
    SetRange(f, 1, 32);
    Double_knob(f, &_gpuShadowStepScale, "gpu_shadow_step_scale", "Shadow Step Scale");
    SetRange(f, 1.0, 8.0);
    Int_knob(f, &_cacheRadius, "cache_radius", "Prefetch Frames");
    SetRange(f, 0, 32);
    Tooltip(f, "Number of frames to prefetch into RAM in each direction.\n"
               "0 = disabled. 8 = cache 16 frames total.\n"
               "Eliminates disk I/O stalls during sequence playback.");
    Divider(f, "CVDB");
    Double_knob(f, &_boundaryBlend, "boundary_blend", "Boundary Blend");
    Tooltip(f, "Block boundary jitter zone in voxels. 0=off, 1-3=subtle, 5+=aggressive.");
    SetRange(f, 0.0, 10.0);
    Int_knob(f, &_overscan, "overscan", "Overscan");
    SetRange(f, 0, 500);

    Divider(f, "3D Viewport");
    static const char* displayModes[] = { "Off", "Bounding Box", "Bbox + Points", nullptr };
    Enumeration_knob(f, &_displayMode, displayModes, "display_mode", "3D Display");
    Double_knob(f, &_pointSize, "point_size", "Point Size");
    SetRange(f, 1.0, 20.0);
    Divider(f, "About");
    Text_knob(f, "    ",
        "VDBRender \xe2\x80\x94 Codebook Volume Renderer for Nuke\n"
        "\n"
        "Original idea and concept: Marten Blumen\n"
        "GPU rewrite: Peter Mercell & Claude (Anthropic)");

    // ── Density Tab ──
    Tab_knob(f, "Density");
    Bool_knob(f, &_createScatterAOV, "create_scatter_aov", "Scatter");
    Bool_knob(f, &_createAlbedoAOV, "create_albedo_aov", "Albedo");
    ClearFlags(f, Knob::STARTLINE);
    Bool_knob(f, &_createDepthAOV, "create_depth_aov", "Depth");
    ClearFlags(f, Knob::STARTLINE);
    Bool_knob(f, &_createPositionAOV, "create_position_aov", "P");
    ClearFlags(f, Knob::STARTLINE);
    Bool_knob(f, &_createPrefAOV, "create_pref_aov", "Pref");
    ClearFlags(f, Knob::STARTLINE);
    Bool_knob(f, &_createNormalAOV, "create_normal_aov", "N");
    ClearFlags(f, Knob::STARTLINE);
    String_knob(f, &_gridName, "grid_name", "Grid Name");
    Tooltip(f, "Name of the density grid inside the CVDB file.\n"
               "Default: 'density'. Leave empty to use the first grid.");
    Divider(f, "Volume");
    Double_knob(f, &_stepSize, "step_size", "Step Size");
    Tooltip(f, "Smaller = more detail but slower. Try 0.1-0.5.");
    Double_knob(f, &_extinction, "extinction", "Extinction");
    Double_knob(f, &_scattering, "scattering", "Scattering");
    Divider(f, "Chromatic Extinction");
    Double_knob(f, &_extinctionR, "extinction_r", "Extinction R");
    Tooltip(f, "Per-channel extinction (Hillaire 2015).\n"
               "Produces depth-dependent colour shifts.\n"
               "Higher blue = distant smoke appears warmer/redder.\n"
               "When R=G=B, matches the scalar Extinction knob.\n"
               "Presets: Smoke (4, 5, 7). Dust (3.5, 5, 8). Fire (4, 5.5, 7).");
    SetRange(f, 0.0, 20.0);
    Double_knob(f, &_extinctionG, "extinction_g", "Extinction G");
    SetRange(f, 0.0, 20.0);
    Double_knob(f, &_extinctionB, "extinction_b", "Extinction B");
    SetRange(f, 0.0, 20.0);
    Divider(f, "Procedural Noise");
    Bool_knob(f, &_noiseEnable, "noise_enable", "Enable");
    Double_knob(f, &_noiseAmplitude, "noise_amplitude", "Amplitude");
    Tooltip(f, "fBm noise added to density at each sample point.\n"
               "0 = disabled (default). 0.1-0.5 = subtle wisp detail.\n"
               "Creates fine-scale structure not present in the VDB data.");
    SetRange(f, 0.0, 1.0);
    Double_knob(f, &_noiseFrequency, "noise_frequency", "Frequency");
    Tooltip(f, "Noise spatial frequency. Higher = finer detail.\n"
               "5-10 = large cloud wisps. 20-50 = fine turbulence.");
    SetRange(f, 1.0, 100.0);
    XYZ_knob(f, _noiseScale, "noise_scale", "Scale");
    Tooltip(f, "Per-axis frequency multiplier.\n"
               "(1,1,1) = uniform. (1,1,0.5) = stretched vertically.\n"
               "(2,2,1) = compressed horizontally. Use to match volume aspect ratio.");
    Int_knob(f, &_noiseOctaves, "noise_octaves", "Octaves");
    Tooltip(f, "Number of fBm layers (1-8).\n"
               "Each octave doubles the frequency and halves the amplitude.\n"
               "4 = good default. 6-8 = maximum detail (slower).");
    SetRange(f, 1, 8);
    XYZ_knob(f, _noiseOffset, "noise_offset", "Offset");
    Tooltip(f, "World-space offset for the noise field.\n"
               "Keyframe this to animate drifting detail (cloud wisps, smoke turbulence).");

    // ── Phase Tab ──
    Tab_knob(f, "Phase");
    Divider(f, "Phase Function");
    static const char* phaseModeOptions[] = {
        "Isotropic", "HG (single)", "Dual HG", "Mie", nullptr
    };
    Enumeration_knob(f, &_phaseMode, phaseModeOptions, "phase_mode", "Phase Mode");
    Tooltip(f, "Isotropic: uniform scattering (cheapest).\n"
               "HG: single Henyey-Greenstein lobe (uses g1 only).\n"
               "Dual HG: two-lobe blend (default, uses g1/g2/mix).\n"
               "Mie: approximate Mie scattering (Jendersie & d'Eon 2023).\n"
               "  Physically correct for water droplets (clouds, fog).\n"
               "  Produces sharp silver linings and backward glory.\n"
               "  Ignores g1/g2/mix — uses built-in Mie coefficients.");
    Double_knob(f, &_phaseG1, "phase_g1", "Forward (g1)");
    Tooltip(f, "Henyey-Greenstein forward lobe asymmetry.\n"
               "0 = isotropic (uniform scattering).\n"
               "Positive = forward scattering (silver lining, back-lit glow).\n"
               "Typical: 0.3-0.8 for clouds, 0.0 for smoke.");
    SetRange(f, -0.99, 0.99);
    Double_knob(f, &_phaseG2, "phase_g2", "Backward (g2)");
    Tooltip(f, "Henyey-Greenstein backward lobe asymmetry.\n"
               "Negative values scatter light back toward the light source.\n"
               "Typical: -0.3 to -0.5 for a subtle bright inner core.");
    SetRange(f, -0.99, 0.99);
    Double_knob(f, &_phaseMix, "phase_mix", "Lobe Mix");
    Tooltip(f, "Blend between the two lobes.\n"
               "1.0 = forward lobe only (g1).\n"
               "0.0 = backward lobe only (g2).\n"
               "Typical: 0.5-0.8 for clouds.");
    SetRange(f, 0.0, 1.0);
    Divider(f, "Gradient Normals");
    Double_knob(f, &_gradientMix, "gradient_mix", "Gradient Normal Mix");
    Tooltip(f, "Blend between volumetric scattering and Lambertian surface shading.\n"
               "Uses the density gradient as a surface normal at cloud edges.\n"
               "0 = disabled (pure volumetric, default).\n"
               "0.3-0.5 = subtle edge definition.\n"
               "0.7-1.0 = strong sculpted silhouettes (cumulus clouds).\n"
               "Weak gradients in cloud interiors remain soft and volumetric.");
    SetRange(f, 0.0, 1.0);
    Double_knob(f, &_gradientThreshold, "gradient_threshold", "Gradient Threshold");
    Tooltip(f, "Gradient magnitude for full Lambertian blend.\n"
               "Below this threshold, the blend ramps smoothly from 0.\n"
               "Lower = more aggressive surface shading (affects thin regions).\n"
               "Higher = only strong edges get surface-like lighting.");
    SetRange(f, 0.01, 1.0);
    XYZ_knob(f, _gradNormalBias, "gradient_normal_bias", "Normal Bias");
    Tooltip(f, "Additive bias to the gradient normal direction.\n"
               "Pushes normals toward the specified direction before N·L.\n"
               "The vector magnitude controls bias strength:\n"
               "  (0,0,0) = no bias (pure gradient normal).\n"
               "  (0,1,0) = push normals upward (top-lit cloud look).\n"
               "  (0,2,0) = stronger upward push.\n"
               "Useful for simulating directional ambient or environment lighting.");
    Double_knob(f, &_gradientSmooth, "gradient_smooth", "Gradient Smooth");
    Tooltip(f, "Finite difference step size for gradient normals (in voxels).\n"
               "1 = sharp (current voxel neighbors, picks up block artifacts).\n"
               "2-4 = smooth normals (wider sampling, softens cloud surfaces).\n"
               "6-8 = very smooth (large-scale shape only).\n"
               "Also affects the N AOV and gradient-directed emission.");
    SetRange(f, 1.0, 8.0);
    Double_knob(f, &_scatterSmooth, "scatter_smooth", "Scatter Smooth");
    Tooltip(f, "Density smoothing radius for scattering (in voxels).\n"
               "0 = off (sharp density, shows codebook block edges).\n"
               "2-4 = smooth scattering response (hides block artifacts).\n"
               "Smooths density, emission, and temperature grids for lighting.\n"
               "Cloud shape and alpha stay sharp.\n"
               "Cost: 6 extra voxel lookups per ray step per grid.");
    SetRange(f, 0.0, 8.0);
    Double_knob(f, &_densityBlur, "density_blur", "Density Blur");
    Tooltip(f, "Pre-blur the raw density before all shading.\n"
               "0 = off. 1-3 = subtle smoothing. 4-8 = heavy blur.\n"
               "Blurs the density field at the source — affects everything:\n"
               "cloud shape, extinction, scattering, powder, alpha.\n"
               "Stacks with Scatter Smooth (which only affects lighting).\n"
               "Use Density Blur alone for overall smoothing,\n"
               "or combine both for maximum softness.");
    SetRange(f, 0.0, 8.0);

    // ── Grids Tab (Emission + Temperature) ──
    Tab_knob(f, "Grids");
    Divider(f, "Emission");
    Bool_knob(f, &_createEmissionAOV, "create_emission_aov", "Create Emission Channels");
    Bool_knob(f, &_emSelfIllumPhase, "emission_self_illum_phase", "Directional (use phase)");
    ClearFlags(f, Knob::STARTLINE);
    Tooltip(f, "Apply the dual-lobe phase function to self-illumination.\n"
               "Uses a density gradient to determine scattering direction:\n"
               "  OFF: isotropic — emission scatters equally in all directions (fast).\n"
               "  ON:  gradient-directed — emission preferentially scatters outward\n"
               "       from dense regions, using the same g1/g2/mix as external lights.\n"
               "       Produces realistic silver-lining/glow-through-smoke effect.\n"
               "       Cost: ~3 extra voxel lookups per step (cache-friendly).");
    Double_knob(f, &_emissionScale, "emission_scale", "Emission Scale");
    Tooltip(f, "Auto-detected from .cvdb (looks for 'flames' or 'emission' grid).\n"
               "0 = disabled. For flames (0-1): try 1-5. For temperature (0-3000): try 0.001.");
    SetFlags(f, Knob::SLIDER | Knob::LOG_SLIDER);
    Double_knob(f, &_emSelfIllum, "emission_self_illum", "Self Illumination");
    Tooltip(f, "How much emission/temperature light scatters into surrounding density.\n"
               "0 = disabled (additive glow only, cheapest).\n"
               ">0 = emission acts as a local light source illuminating the volume.\n"
               "No shadow rays needed — light is co-located with the sample.\n"
               "Typical: 0.5-2.0 for fire lighting up smoke.");
    SetFlags(f, Knob::SLIDER | Knob::LOG_SLIDER);
    SetRange(f, 0.0, 10.0);
    Divider(f, "Emission Color Ramp");
    Color_knob(f, _rampColor0, "ramp_color_0", "Dark (0.0)");
    Color_knob(f, _rampColor1, "ramp_color_1", "Low (0.25)");
    Color_knob(f, _rampColor2, "ramp_color_2", "Mid (0.5)");
    Color_knob(f, _rampColor3, "ramp_color_3", "High (0.75)");
    Color_knob(f, _rampColor4, "ramp_color_4", "Peak (1.0)");
    Divider(f, "Temperature");
    Bool_knob(f, &_createTemperatureAOV, "create_temperature_aov", "Create Temperature Channels");
    Bool_knob(f, &_useBlackbody, "use_blackbody", "Blackbody Color");
    ClearFlags(f, Knob::STARTLINE);
    Tooltip(f, "Replace the emission color ramp with physically-based blackbody radiation.\n"
               "Maps the normalized temperature value (0-1) through the Kelvin Scale\n"
               "to produce accurate incandescent colors (deep red → orange → yellow → white → blue-white).\n"
               "When OFF, temperature uses the 5-stop emission ramp above.");
    Double_knob(f, &_temperatureScale, "temperature_scale", "Temperature Scale");
    Tooltip(f, "Auto-detected from .cvdb (looks for 'temperature' or 'heat' grid).\n0 = disabled.");
    SetFlags(f, Knob::SLIDER | Knob::LOG_SLIDER);
    Double_knob(f, &_bbKelvinScale, "bb_kelvin_scale", "Kelvin Scale");
    Tooltip(f, "Peak Kelvin temperature corresponding to norm=1.0.\n"
               "The normalized temperature range 0→1 maps to 0→this value in Kelvin.\n"
               "  1500 = candle/match (deep red-orange)\n"
               "  3000 = typical fire (orange-yellow)\n"
               "  6500 = daylight white (default)\n"
               "  10000 = hot blue-white plasma");
    SetRange(f, 500.0, 20000.0);
    Double_knob(f, &_bbMix, "bb_mix", "Blackbody Mix");
    Tooltip(f, "Blend between emission ramp and blackbody color.\n"
               "0 = ramp only (blackbody disabled).\n"
               "0.2 = subtle physical tint over artist ramp.\n"
               "1.0 = pure blackbody (default when enabled).");
    SetRange(f, 0.0, 1.0);
    Color_knob(f, _bbTint, "bb_tint", "Blackbody Tint");
    Tooltip(f, "Multiplies the blackbody color before mixing with the ramp.\n"
               "Default white (1,1,1) = no tint.\n"
               "Warm tint (1, 0.9, 0.7) = push blue-white toward amber.\n"
               "Cool tint (0.8, 0.9, 1) = enhance blue at high temperatures.");

    // ── Lighting Tab ──
    Tab_knob(f, "Lighting");
    Double_knob(f, &_lightDisplayScale, "light_display_scale", "Display Scale");
    SetRange(f, 0.1, 20);
    Tooltip(f, "Size multiplier for light markers in the 3D viewport.");
    Divider(f, "");

    // ── Light 1 (key light, enabled by default) ──
    Bool_knob(f, &_light1Enable, "light1_enable", "Light 1");
    SetFlags(f, Knob::STARTLINE);
    XYZ_knob(f, _light1Pos, "light1_position", "Position");
    XYZ_knob(f, _light1Target, "light1_target", "Target");
    Color_knob(f, _light1Color, "light1_color", "Color");
    Double_knob(f, &_light1Intensity, "light1_intensity", "Intensity");
    SetRange(f, 0, 20);
    Double_knob(f, &_light1ConeAngle, "light1_cone_angle", "Cone Angle");
    SetRange(f, 1, 180);
    Tooltip(f, "Beam spread half-angle in degrees. 180 = hemisphere (wide). 10 = tight spot.");
    Double_knob(f, &_light1Softness, "light1_softness", "Softness");
    
    SetRange(f, 0, 1);
    Tooltip(f, "Cone edge falloff. 0 = hard cutoff. 1 = fully smooth.");
    Divider(f, "");

    // ── Light 2 (fill, disabled by default) ──
    Bool_knob(f, &_light2Enable, "light2_enable", "Light 2");
    SetFlags(f, Knob::STARTLINE);
    XYZ_knob(f, _light2Pos, "light2_position", "Position");
    XYZ_knob(f, _light2Target, "light2_target", "Target");
    Color_knob(f, _light2Color, "light2_color", "Color");
    Double_knob(f, &_light2Intensity, "light2_intensity", "Intensity");
    SetRange(f, 0, 20);
    Double_knob(f, &_light2ConeAngle, "light2_cone_angle", "Cone Angle");
    SetRange(f, 1, 180);
    Double_knob(f, &_light2Softness, "light2_softness", "Softness");
    SetRange(f, 0, 1);
    Divider(f, "");

    // ── Light 3 (rim, disabled by default) ──
    Bool_knob(f, &_light3Enable, "light3_enable", "Light 3");
    SetFlags(f, Knob::STARTLINE);
    XYZ_knob(f, _light3Pos, "light3_position", "Position");
    XYZ_knob(f, _light3Target, "light3_target", "Target");
    Color_knob(f, _light3Color, "light3_color", "Color");
    Double_knob(f, &_light3Intensity, "light3_intensity", "Intensity");
    SetRange(f, 0, 20);
    Double_knob(f, &_light3ConeAngle, "light3_cone_angle", "Cone Angle");
    SetRange(f, 1, 180);
    Double_knob(f, &_light3Softness, "light3_softness", "Softness");
    SetRange(f, 0, 1);
    Divider(f, "");

    // ── Fallback direction (used when no lights enabled) ──
    XYZ_knob(f, _lightDir, "light_dir", "Fallback Direction");
    Tooltip(f, "Light direction used when all 3 lights are disabled.");
    Color_knob(f, _lightColor, "light_color", "Fallback Color");

    // ── Ambient ──
    Divider(f, "Ambient");
    Color_knob(f, _ambientColor, "ambient_color", "Ambient Color");
    Tooltip(f, "Constant ambient light color. Provides fill lighting so volumes\n"
               "don't appear to sit in a void. Default is a slight blue sky tint.\n"
               "When CUDAEnvLight is connected, auto-filled from HDRI residual.");
    Double_knob(f, &_ambientIntensity, "ambient_intensity", "Ambient Intensity");
    Tooltip(f, "Strength of ambient fill light.\n"
               "0 = disabled (no ambient). 0.1-0.3 = subtle fill.\n"
               "0.5-1.0 = strong ambient (overcast sky look).");
    SetRange(f, 0.0, 2.0);

    Divider(f, "Environment (HDRI)");
    Bool_knob(f, &_envEnable, "env_enable", "Enable");
    Tooltip(f, "Enable HDRI environment lighting from input 'env'.\n"
               "Connect an equirectangular HDRI image to the env input.");
    Double_knob(f, &_envIntensity, "env_intensity", "Env Intensity");
    Tooltip(f, "Strength of HDRI environment lighting.\n"
               "Connect an equirectangular HDRI to input 'env'.\n"
               "Projected to L2 spherical harmonics for diffuse fill.\n"
               "0 = disabled. 1 = match HDRI brightness.");
    SetRange(f, 0.0, 10.0);
    Double_knob(f, &_envRotation, "env_rotation", "Env Rotation");
    Tooltip(f, "Rotate the environment map around Y axis (degrees).\n"
               "Rotates the SH coefficients — no re-read of the input.");
    SetRange(f, 0.0, 360.0);
    Bool_knob(f, &_envShadowEnable, "env_shadow", "Shadow from Peak");
    ClearFlags(f, Knob::STARTLINE);
    Tooltip(f, "Cast shadow rays along the dominant HDRI direction.\n"
               "Uses the same multi-step shadow rays as the cone lights.\n"
               "Adds one virtual directional light from the brightest part of the HDRI.");
    Color_knob(f, _envShadowLift, "env_shadow_lift", "Shadow Lift");
    Tooltip(f, "Minimum shadow color for the environment peak shadow.\n"
               "Black (0,0,0) = full shadow depth.\n"
               "Lift to retain ambient fill or tint shadows in occluded regions.");
    Bool_knob(f, &_createEnvAOV, "create_env_aov", "Environment AOV");
    ClearFlags(f, Knob::STARTLINE);
    Tooltip(f, "Output the HDRI environment lighting contribution as a separate channel layer (env.rgba).");

    Divider(f, "Powder Effect");
    Bool_knob(f, &_powderEnable, "powder_enable", "Enable");
    Double_knob(f, &_powderStrength, "powder_strength", "Powder Strength");
    Tooltip(f, "Schneider & Vos (2015) powder effect.\n"
               "Brightens dense interiors, darkens thin edges.\n"
               "Creates the characteristic bright-core look in clouds.\n"
               "0 = disabled (current behaviour).\n"
               "1.5-3.0 = typical for clouds.\n"
               "3.0-5.0 = fire/explosions illuminating smoke.");
    SetFlags(f, Knob::SLIDER | Knob::LOG_SLIDER);
    SetRange(f, 0.0, 10.0);
    Divider(f, "Multi-Scatter (Frostbite)");
    Bool_knob(f, &_msEnable, "ms_enable", "Enable");
    Tooltip(f, "Frostbite multi-scatter octave model (Hillaire 2016).\n"
               "Approximates multiple light bounces inside the volume.\n"
               "Each octave uses reduced extinction (light penetrates deeper)\n"
               "and a more isotropic phase function. No extra shadow rays needed —\n"
               "reuses the primary shadow optical depth scaled per octave.");
    Int_knob(f, &_msOctaves, "ms_octaves", "Octaves");
    Tooltip(f, "Number of multi-scatter octaves (1-8).\n"
               "Each octave approximates one additional bounce.\n"
               "4 = good default for clouds. 6-8 = diminishing returns.");
    SetRange(f, 1, 8);
    Double_knob(f, &_msExtFalloff, "ms_ext_falloff", "Extinction Falloff");
    Tooltip(f, "How much extinction drops per octave.\n"
               "0.5 = each octave halves the shadow density (default).\n"
               "Lower = light penetrates much deeper per octave.\n"
               "Higher = more conservative, subtle fill.");
    SetRange(f, 0.0, 1.0);
    Double_knob(f, &_msScatterFalloff, "ms_scatter_falloff", "Scatter Falloff");
    Tooltip(f, "Energy contribution per octave.\n"
               "0.5 = each octave contributes half the previous (default).\n"
               "Lower = higher octaves contribute very little.\n"
               "Higher = brighter cloud interiors.");
    SetRange(f, 0.0, 1.0);

    Divider(f, "Light Mixer");
    Double_knob(f, &_lightMix[0], "light1_mix", "Light 1 Mix");
    SetRange(f, 0.0, 2.0);
    Tooltip(f, "Light 1 contribution to RGBA beauty. 0 = muted, 1 = full, >1 = boost.\n"
               "Per-light AOV channels always stay at full strength.");
    Double_knob(f, &_lightMix[1], "light2_mix", "Light 2 Mix");
    SetRange(f, 0.0, 2.0);
    Double_knob(f, &_lightMix[2], "light3_mix", "Light 3 Mix");
    SetRange(f, 0.0, 2.0);
    Double_knob(f, &_envMix, "env_mix", "Env Mix");
    SetRange(f, 0.0, 2.0);
    Tooltip(f, "HDRI environment contribution to RGBA beauty.\n"
               "The env AOV channel always stays at full strength.");

    // ── Deep Tab ──
    Tab_knob(f, "Deep");
    Bool_knob(f, &_deepEnable, "deep_enable", "Enable Deep Output");
    Tooltip(f, "When enabled, this node also outputs deep data.\n"
               "Connect to DeepMerge or DeepWrite downstream.");
    Int_knob(f, &_deepMaxSamples, "deep_max_samples", "Max Samples");
    SetRange(f, 8, 128);
    Tooltip(f, "Maximum deep slabs per pixel.\n"
               "32 = good default (~190 MB VRAM at 1080p).\n"
               "64 = complex layered volumes (~380 MB).");
    Divider(f, "");
    Bool_knob(f, &_profileEnable, "profile_enable", "Profiler");
    Tooltip(f, "Print per-frame timing breakdown to the terminal.\n"
               "Shows per-frame timing breakdown: ray march, conversion, readback.\n"
               "Disable for production — adds cudaDeviceSynchronize barriers.");
}

int VDBRenderIop::knob_changed(Knob* k)
{
    if (k->is("file") || k->is("grid_name") || k->is("frame_offset")) {
        _gridValid = false; _gpuFrameReady = false;
        _deepReady = false; _deepFailed = false;
        _previewPoints.clear(); _cvdbCpu.clear();
        stopPrefetch();
        { std::lock_guard<std::mutex> lock(_frameCacheMtx); _frameCache.clear(); }
        return 1;
    }
    if (k->is("display_mode") || k->is("point_size")) {
        _previewPoints.clear(); return 1;
    }
    if (k->is("gpu_render_scale") || k->is("gpu_shadow_steps") ||
        k->is("gpu_shadow_step_scale") || k->is("boundary_blend") ||
        k->is("step_size") || k->is("extinction") || k->is("scattering") ||
        k->is("extinction_r") || k->is("extinction_g") || k->is("extinction_b") ||
        k->is("powder_enable") || k->is("powder_strength") ||
        k->is("ms_enable") || k->is("ms_octaves") || k->is("ms_ext_falloff") || k->is("ms_scatter_falloff") ||
        k->is("noise_enable") || k->is("noise_amplitude") || k->is("noise_frequency") || k->is("noise_scale") || k->is("noise_octaves") || k->is("noise_offset") ||
        k->is("phase_g1") || k->is("phase_g2") || k->is("phase_mix") || k->is("phase_mode") ||
        k->is("gradient_mix") || k->is("gradient_threshold") || k->is("gradient_normal_bias") || k->is("gradient_smooth") || k->is("scatter_smooth") || k->is("density_blur") ||
        k->is("light_dir") || k->is("light_color") ||
        k->is("light1_enable") || k->is("light1_position") || k->is("light1_color") ||
        k->is("light1_intensity") || k->is("light1_cone_angle") || k->is("light1_softness") ||
        k->is("light1_target") ||
        k->is("light2_enable") || k->is("light2_position") || k->is("light2_color") ||
        k->is("light2_intensity") || k->is("light2_cone_angle") || k->is("light2_softness") ||
        k->is("light2_target") ||
        k->is("light3_enable") || k->is("light3_position") || k->is("light3_color") ||
        k->is("light3_intensity") || k->is("light3_cone_angle") || k->is("light3_softness") ||
        k->is("light3_target") ||
        k->is("ambient_color") || k->is("ambient_intensity") ||
        k->is("env_enable") || k->is("env_intensity") || k->is("env_rotation") || k->is("env_shadow") || k->is("env_shadow_lift") ||
        k->is("vol_translate") || k->is("vol_rotate") ||
        k->is("vol_scale") || k->is("vol_uniform_scale") || k->is("overscan") ||
        k->is("create_scatter_aov") || k->is("create_albedo_aov") ||
        k->is("create_depth_aov") || k->is("create_position_aov") ||
        k->is("create_pref_aov") || k->is("create_normal_aov") ||
        k->is("emission_scale") || k->is("temperature_scale") ||
        k->is("emission_self_illum") || k->is("emission_self_illum_phase") ||
        k->is("create_emission_aov") || k->is("create_temperature_aov") ||
        k->is("use_blackbody") || k->is("bb_kelvin_scale") || k->is("bb_mix") || k->is("bb_tint") ||
        k->is("ramp_color_0") || k->is("ramp_color_1") || k->is("ramp_color_2") ||
        k->is("ramp_color_3") || k->is("ramp_color_4") ||
        k->is("create_env_aov") ||
        k->is("light1_mix") || k->is("light2_mix") || k->is("light3_mix") || k->is("env_mix")) {
        _deepReady = false;
        _deepFailed = false;
        return 1;
    }
    if (k->is("deep_enable") || k->is("deep_max_samples")) {
        _deepReady = false;
        _deepFailed = false;
        return 1;
    }
    return Iop::knob_changed(k);
}

CameraOp* VDBRenderIop::camera() const { return dynamic_cast<CameraOp*>(Op::input(0)); }

const char* VDBRenderIop::input_label(int idx, char*) const
{
    if (idx == 0) return "cam";
    if (idx == 1) return "env";
    return "";
}

bool VDBRenderIop::test_input(int idx, Op* op) const
{
    if (!op) return true;
    if (idx == 0) return dynamic_cast<CameraOp*>(op) != nullptr;
    if (idx == 1) return dynamic_cast<Iop*>(op) != nullptr;
    return false;
}

Op* VDBRenderIop::default_input(int) const { return nullptr; }

void VDBRenderIop::append(Hash& hash)
{
    hash.append(outputContext().frame()); hash.append(_frameOffset);
    hash.append(_stepSize); hash.append(_extinction); hash.append(_scattering);
    hash.append(_extinctionR); hash.append(_extinctionG); hash.append(_extinctionB);
    hash.append(_powderEnable); hash.append(_powderStrength);
    hash.append(_msEnable); hash.append(_msOctaves);
    hash.append(_msExtFalloff); hash.append(_msScatterFalloff);
    hash.append(_phaseG1); hash.append(_phaseG2); hash.append(_phaseMix); hash.append(_phaseMode);
    hash.append(_gradientMix); hash.append(_gradientThreshold);
    for(int i=0;i<3;++i) hash.append(_gradNormalBias[i]);
    hash.append(_gradientSmooth);
    hash.append(_scatterSmooth);
    hash.append(_densityBlur);
    hash.append(_noiseEnable); hash.append(_noiseAmplitude); hash.append(_noiseFrequency); hash.append(_noiseOctaves);
    for(int i=0;i<3;++i){ hash.append(_noiseScale[i]); hash.append(_noiseOffset[i]); }
    for (int i = 0; i < 3; ++i) {
        hash.append(_lightDir[i]); hash.append(_lightColor[i]);
        hash.append(_ambientColor[i]);
        hash.append(_volTranslate[i]); hash.append(_volRotate[i]); hash.append(_volScale[i]);
    }
    hash.append(_ambientIntensity);
    hash.append(_envEnable);
    hash.append(_envIntensity);
    hash.append(_envRotation);
    hash.append(_envShadowEnable);
    for(int i=0;i<3;++i) hash.append(_envShadowLift[i]);
    hash.append(_light1Enable);
    for(int i=0;i<3;++i){ hash.append(_light1Pos[i]); hash.append(_light1Target[i]); hash.append(_light1Color[i]); }
    hash.append(_light1Intensity); hash.append(_light1ConeAngle); hash.append(_light1Softness);
    hash.append(_light2Enable);
    for(int i=0;i<3;++i){ hash.append(_light2Pos[i]); hash.append(_light2Target[i]); hash.append(_light2Color[i]); }
    hash.append(_light2Intensity); hash.append(_light2ConeAngle); hash.append(_light2Softness);
    hash.append(_light3Enable);
    for(int i=0;i<3;++i){ hash.append(_light3Pos[i]); hash.append(_light3Target[i]); hash.append(_light3Color[i]); }
    hash.append(_light3Intensity); hash.append(_light3ConeAngle); hash.append(_light3Softness);
    hash.append(_volUniformScale); hash.append(_gpuRenderScale);
    hash.append(_gpuShadowSteps); hash.append(_gpuShadowStepScale);
    
    hash.append(_boundaryBlend); hash.append(_overscan);
    hash.append(_emissionScale); hash.append(_temperatureScale);
    hash.append(_emSelfIllum); hash.append(_emSelfIllumPhase);
    hash.append(_useBlackbody); hash.append(_bbKelvinScale); hash.append(_bbMix);
    for(int i=0;i<3;++i) hash.append(_bbTint[i]);
    for (int i=0;i<3;++i) {
        hash.append(_rampColor0[i]); hash.append(_rampColor1[i]); hash.append(_rampColor2[i]);
        hash.append(_rampColor3[i]); hash.append(_rampColor4[i]);
    }
    hash.append(_createEmissionAOV); hash.append(_createTemperatureAOV);
    hash.append(_createScatterAOV); hash.append(_createAlbedoAOV);
    hash.append(_createDepthAOV);
    hash.append(_createPositionAOV);
    hash.append(_createPrefAOV);
    hash.append(_createNormalAOV);
    hash.append(_createEnvAOV);
    for(int i=0;i<3;++i) hash.append(_lightMix[i]);
    hash.append(_envMix);
    hash.append(_deepEnable); hash.append(_deepMaxSamples);
}

static Op* build(Node* node) { return new VDBRenderIop(node); }
const Op::Description VDBRenderIop::desc(VDBRenderIop::CLASS, build);
