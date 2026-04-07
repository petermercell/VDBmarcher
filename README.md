# VDBRender — GPU Codebook Volume Renderer for Nuke

![VDBRender](https://raw.githubusercontent.com/petermercell/VDBmarcher/main/CVDB.jpg)

A CUDA-accelerated volume renderer for [Foundry Nuke](https://www.foundry.com/products/nuke) that ray marches compressed [CVDB](https://github.com/petermercell/CVDB) codebook volumes directly on the GPU. Volumes are pre-compressed from [OpenVDB](https://www.openvdb.org/) using the vdb2cvdb converter — at render time, the plugin samples the codebook natively with no decompression step.

## Disclaimer

**This project is experimental and not production-ready. It may never be. Use at your own risk. Quality is not bad, but not for AAA VFX projects.**

## Quick Start — Converting VDB to CVDB

VDBRender reads `.cvdb` files, not `.vdb` directly. Use the [vdb2cvdb](https://github.com/petermercell/CVDB) converter to compress your OpenVDB volumes:

```bash
# Single file
./vdb2cvdb explosion.vdb -k 8192 -i 50 -s 400000

# Batch sequence
./vdb2cvdb /path/to/vdb_sequence/ -o /path/to/output/ -k 8192 -i 50 -s 400000
```

| Flag | Description | Default |
|------|-------------|---------|
| `-k` | Codebook size (number of centroids) | 4096 |
| `-i` | K-means iterations | 50 |
| `-s` | Training subsample size (0 = all) | 0 |
| `-o` | Output directory (batch mode) | — |

For animated sequences with temporal coherence, see the [CVDB temporal encoder](https://github.com/petermercell/CVDB).

## Origin

This project is a complete GPU rewrite of [VDBmarcher](https://github.com/bratgot/VDBmarcher) by **Marten Blumen**, who conceived the idea of a dedicated ray marching renderer for codebook-compressed VDB volumes inside Nuke. VDBRender rebuilds the concept from scratch in CUDA/C++ with a production feature set. GPU rewrite by **Peter Mercell** and **Claude** (Anthropic).

## Features

### Rendering
- CUDA ray marching with codebook + gain map sampling (CVD1–CVD6, including FP16)
- GPU spatial hash table for O(1) leaf lookup
- Coarse occupancy macro-grid for empty-space skipping
- Adaptive step size (enlarges in low-density regions)
- Bilinear upscale from quarter/half/three-quarter resolution
- Asynchronous render pipeline with CUDA streams and pinned readback
- Deep output (alpha-only deep samples with ray-t → camera-Z conversion)

### Lighting
- 3 built-in spot/directional lights with cone angle and soft falloff
- HDRI environment lighting via L2 spherical harmonics (projected from equirectangular input)
- Environment Y-axis rotation (SH rotation, no re-projection)
- Shadow rays along HDRI peak direction with per-channel shadow lift
- Ambient fill light
- Light Mixer — per-light and per-env blend into RGBA beauty (AOV channels stay at full strength)

### Shading
- Chromatic extinction (Hillaire 2015) — per-channel R/G/B transmittance
- Dual-lobe Henyey-Greenstein phase function
- Approximate Mie scattering (Jendersie & d'Eon 2023)
- Powder effect (Schneider & Vos 2015)
- Multi-scatter octaves (Frostbite / Hillaire 2016)
- Density gradient normals with Lambertian surface blend
- Scatter smooth (22-tap blur for density, emission, and temperature grids)
- Density blur (pre-blur before all shading)
- Procedural fBm noise injection

### Emission & Temperature
- Separate codebook grids for emission and temperature channels
- 5-stop color ramp with artist control
- Physically-based blackbody radiation with Kelvin scale and tint
- Self-illumination (emission as co-located light source, optional gradient-directed phase)

### AOVs
- Scatter, Emission, Temperature, Albedo
- Per-light (light1/light2/light3)
- Environment (HDRI contribution)
- Depth, Position (P), Reference Position (Pref), Surface Normal (N)

### Nuke Integration
- Dual Iop/DeepOp node (flat + deep output from a single node)
- Camera input with automatic focal length and transform extraction
- HDRI environment input (equirectangular, auto-projected to SH)
- Frame sequence support with background prefetch cache
- 3D viewport display (bounding box + density point cloud)
- Per-light 3D viewport markers

## File Format

VDBRender reads `.cvdb` files produced by [CVDB](https://github.com/petermercell/CVDB) (vdb2cvdb converter). Supported format versions: CVD1 through CVD6. Multiple grids per file (density, flames, temperature).

## Requirements

- **Nuke** 15.0+ (NDK/DDImage)
- **CUDA Toolkit** 12.x
- **GPU** with compute capability ≥ 7.5 (Turing+), tested on RTX A5000
- **CMake** ≥ 3.18

OpenVDB is not required at runtime — VDBRender reads pre-compressed `.cvdb` files directly. To convert your OpenVDB volumes to CVDB format, use the [vdb2cvdb](https://github.com/petermercell/CVDB) converter, which depends on the excellent [OpenVDB](https://www.openvdb.org/) library by the Academy Software Foundation.

## Building

```bash
mkdir build && cd build
cmake .. -DNUKE_ROOT=/path/to/Nuke15.0v1
make -j$(nproc)
```

The build produces `VDBRender.so` — copy it to your `~/.nuke` or Nuke plugin path.

## Source Files

| File | Description |
|------|-------------|
| `VDBRenderIop.h / .cpp` | Main Nuke Iop — knobs, UI tabs, channel registration |
| `VDBRenderCUDA.cpp` | Render config builder, sync/async GPU dispatch |
| `VDBCUDARenderer.h / .cu` | CUDA renderer — hash table, ray march kernel, light mixer |
| `VDBEngine.cpp` | Scanline engine — GPU buffer bilinear upscale to Nuke rows |
| `VDBDeepEngine.cpp` | Deep output — ray-t to camera-Z conversion |
| `VDBValidate.cpp` | Node validation, camera extraction, grid upload |
| `VDBGridLoader.cpp` | CVDB file loading and frame sequence resolution |
| `VDBLighting.cpp` | Light population from knobs (3 lights + fallback) |
| `VDBEnvironment.h / .cpp` | L2 spherical harmonics projection, Y-rotation, evaluation |
| `VDBViewport.cpp` | 3D viewport — bounding box, point cloud, light markers |
| `VDBRenderConfig.h` | Unified render configuration struct |
| `CVDBLoader.h` | Header-only .cvdb file parser (CVD1–CVD6) |
| `VDBProfiler.h` | Lightweight GPU/CPU pipeline profiler |

## UI Tabs

| Tab | Contents |
|-----|----------|
| **VDBRender** | File path, volume transform, GPU settings, 3D viewport, about |
| **Density** | Grid name, step size, extinction, scattering, chromatic extinction, procedural noise |
| **Phase** | Phase function (Iso/HG/Dual HG/Mie), gradient normals, scatter smooth, density blur |
| **Grids** | Emission (scale, self-illumination, color ramp) and Temperature (blackbody, Kelvin) |
| **Lighting** | 3 spot lights, ambient, HDRI environment (SH, rotation, shadow), powder, multi-scatter, light mixer |
| **Deep** | Deep output enable, max samples, profiler |

## Acknowledgments

- **Marten Blumen** — original idea and concept ([VDBmarcher](https://github.com/bratgot/VDBmarcher))
- **Claude** (Anthropic) — GPU rewrite collaboration
- **Enzo Crema** — [VQVDB](https://github.com/ZephirFXEC/VQVDB), which inspired the CVDB compression format

## Support

VDBRender is free and open source. If you find it useful, consider supporting development on **[Patreon](https://www.patreon.com/cw/PeterMercell)**.

## License

[MIT](LICENSE)
