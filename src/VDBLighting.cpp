// ═══════════════════════════════════════════════════════════════════════════
// VDBLighting.cpp — Internal GPU light population from knobs
// ═══════════════════════════════════════════════════════════════════════════

#include "VDBRenderIop.h"
#include <cmath>

void VDBRenderIop::buildLights()
{
    struct LightDef {
        bool*   enable;
        double* pos;
        double* target;
        double* color;
        double* intensity;
        double* coneAngle;
        double* softness;
    };

    LightDef defs[3] = {
        { &_light1Enable, _light1Pos, _light1Target, _light1Color, &_light1Intensity, &_light1ConeAngle, &_light1Softness },
        { &_light2Enable, _light2Pos, _light2Target, _light2Color, &_light2Intensity, &_light2ConeAngle, &_light2Softness },
        { &_light3Enable, _light3Pos, _light3Target, _light3Color, &_light3Intensity, &_light3ConeAngle, &_light3Softness },
    };

    // Always populate all 3 slots to preserve AOV index mapping
    _numGPULights = 3;
    for (int i = 0; i < 3; ++i) {
        GPULight& gl = _gpuLights[i];
        gl.enabled = *defs[i].enable ? 1 : 0;
        for (int c = 0; c < 3; ++c) {
            gl.position[c] = static_cast<float>(defs[i].pos[c]);
            gl.color[c]    = static_cast<float>(defs[i].color[c]);
        }
        // Direction: position → target (normalized)
        float dx = static_cast<float>(defs[i].target[0] - defs[i].pos[0]);
        float dy = static_cast<float>(defs[i].target[1] - defs[i].pos[1]);
        float dz = static_cast<float>(defs[i].target[2] - defs[i].pos[2]);
        float dlen = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (dlen > 1e-8f) { dx /= dlen; dy /= dlen; dz /= dlen; }
        gl.direction[0] = dx; gl.direction[1] = dy; gl.direction[2] = dz;
        gl.intensity = static_cast<float>(*defs[i].intensity);
        gl.coneAngle = static_cast<float>(*defs[i].coneAngle * 3.14159265358979323846 / 180.0);
        gl.coneSoftness = static_cast<float>(*defs[i].softness);
    }

    // If no lights enabled, set up fallback directional on slot 0
    if (!_light1Enable && !_light2Enable && !_light3Enable) {
        float lx = static_cast<float>(_lightDir[0]);
        float ly = static_cast<float>(_lightDir[1]);
        float lz = static_cast<float>(_lightDir[2]);
        float len = std::sqrt(lx*lx + ly*ly + lz*lz);
        if (len > 1e-8f) { lx /= len; ly /= len; lz /= len; }
        float cx = static_cast<float>((_bboxMin[0] + _bboxMax[0]) * 0.5);
        float cy = static_cast<float>((_bboxMin[1] + _bboxMax[1]) * 0.5);
        float cz = static_cast<float>((_bboxMin[2] + _bboxMax[2]) * 0.5);
        float dist = 100000.0f;
        GPULight& gl = _gpuLights[0];
        gl.position[0] = cx + lx * dist;
        gl.position[1] = cy + ly * dist;
        gl.position[2] = cz + lz * dist;
        gl.color[0] = static_cast<float>(_lightColor[0]);
        gl.color[1] = static_cast<float>(_lightColor[1]);
        gl.color[2] = static_cast<float>(_lightColor[2]);
        gl.direction[0] = -lx; gl.direction[1] = -ly; gl.direction[2] = -lz;
        gl.intensity = 1.0f;
        gl.coneAngle = 3.14159f;
        gl.coneSoftness = 0.0f;
        gl.enabled = 1;
    }
}
