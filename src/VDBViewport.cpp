// ═══════════════════════════════════════════════════════════════════════════
// VDBViewport.cpp — 3D viewport drawing, volume transforms, frustum culling
// ═══════════════════════════════════════════════════════════════════════════

#include "VDBRenderIop.h"
#include <cmath>
#include <algorithm>

using namespace DD::Image;

void VDBRenderIop::build_handles(ViewerContext* ctx)
{
    if (_displayMode == 0 || !_gridValid) return;
    add_draw_handle(ctx);

    // Light position + target 3D handles
    if (_light1Enable) {
        Knob* k = knob("light1_position"); if (k) k->build_handle(ctx);
        k = knob("light1_target"); if (k) k->build_handle(ctx);
    }
    if (_light2Enable) {
        Knob* k = knob("light2_position"); if (k) k->build_handle(ctx);
        k = knob("light2_target"); if (k) k->build_handle(ctx);
    }
    if (_light3Enable) {
        Knob* k = knob("light3_position"); if (k) k->build_handle(ctx);
        k = knob("light3_target"); if (k) k->build_handle(ctx);
    }
}

void VDBRenderIop::rebuildPointCloud()
{
    _previewPoints.clear();
    _maxDensity = 1.0f;
    if (_displayMode < 2 || !_cvdbCpu.valid()) return;

    const uint32_t N = _cvdbCpu.numLeaves;
    const int bs = _cvdbCpu.blockSize;
    const int vpb = bs*bs*bs;
    const float vs = _cvdbCpu.voxelSize;
    const int targetPoints = 24000;
    const float threshold = 0.001f;
    float maxD = 0.0f;

    int samplesPerLeaf = std::max(1, targetPoints/static_cast<int>(N));
    samplesPerLeaf = std::min(samplesPerLeaf, 8);
    int leafStride = 1;
    if (N*samplesPerLeaf > uint32_t(targetPoints*2))
        leafStride = std::max(1, int(N*samplesPerLeaf/targetPoints));

    struct Off { int x,y,z; };
    Off offs[8]; int nSamp=0;
    if (samplesPerLeaf==1) { int h=bs/2; offs[0]={h,h,h}; nSamp=1; }
    else { int lo=bs/4, hi=bs*3/4;
        for(int sz=0;sz<2&&nSamp<samplesPerLeaf;++sz)
          for(int sy=0;sy<2&&nSamp<samplesPerLeaf;++sy)
            for(int sx=0;sx<2&&nSamp<samplesPerLeaf;++sx)
              offs[nSamp++]={sx?hi:lo, sy?hi:lo, sz?hi:lo};
    }

    _previewPoints.reserve(std::min(uint32_t(targetPoints*2), N*samplesPerLeaf));
    for (uint32_t li=0; li<N; li+=leafStride) {
        int ox=_cvdbCpu.origins[li*3], oy=_cvdbCpu.origins[li*3+1], oz=_cvdbCpu.origins[li*3+2];
        uint16_t cbIdx = _cvdbCpu.indices[li];
        for (int si=0; si<nSamp; ++si) {
            int lx=offs[si].x, ly=offs[si].y, lz=offs[si].z;
            float d = _cvdbCpu.codebook[cbIdx*vpb + lx*bs*bs + ly*bs + lz];
            if (_cvdbCpu.hasGain && !_cvdbCpu.gain_maps.empty()) {
                int gx=(lx>=bs/2)?1:0, gy=(ly>=bs/2)?1:0, gz=(lz>=bs/2)?1:0;
                d *= _cvdbCpu.gain_maps[li*8 + gx + gy*2 + gz*4];
            }
            if (d > threshold) {
                float wx=(ox+lx+0.5f)*vs + _cvdbCpu.gridOffset[0];
                float wy=(oy+ly+0.5f)*vs + _cvdbCpu.gridOffset[1];
                float wz=(oz+lz+0.5f)*vs + _cvdbCpu.gridOffset[2];
                _previewPoints.push_back({wx,wy,wz,d});
                if (d>maxD) maxD=d;
            }
        }
    }
    _maxDensity = (maxD>0.0f) ? maxD : 1.0f;
}

void VDBRenderIop::draw_handle(ViewerContext* ctx)
{
    if (_displayMode==0 || !_gridValid) return;
    glPushAttrib(GL_CURRENT_BIT|GL_LINE_BIT|GL_ENABLE_BIT|GL_POINT_BIT);
    glDisable(GL_LIGHTING);

    glPushMatrix();
    {
        const double deg2rad=3.14159265358979323846/180.0;
        const double rx=_volRotate[0]*deg2rad, ry=_volRotate[1]*deg2rad, rz=_volRotate[2]*deg2rad;
        const double sx=_volScale[0]*_volUniformScale, sy=_volScale[1]*_volUniformScale, sz=_volScale[2]*_volUniformScale;
        const double crx=cos(rx),srx=sin(rx),cry=cos(ry),sry=sin(ry),crz=cos(rz),srz=sin(rz);
        double R[3][3];
        R[0][0]=cry*crz; R[0][1]=srx*sry*crz-crx*srz; R[0][2]=crx*sry*crz+srx*srz;
        R[1][0]=cry*srz; R[1][1]=srx*sry*srz+crx*crz; R[1][2]=crx*sry*srz-srx*crz;
        R[2][0]=-sry;    R[2][1]=srx*cry;              R[2][2]=crx*cry;
        float fwd[16];
        fwd[0]=float(R[0][0]*sx);fwd[1]=float(R[1][0]*sx);fwd[2]=float(R[2][0]*sx);fwd[3]=0;
        fwd[4]=float(R[0][1]*sy);fwd[5]=float(R[1][1]*sy);fwd[6]=float(R[2][1]*sy);fwd[7]=0;
        fwd[8]=float(R[0][2]*sz);fwd[9]=float(R[1][2]*sz);fwd[10]=float(R[2][2]*sz);fwd[11]=0;
        fwd[12]=float(_volTranslate[0]);fwd[13]=float(_volTranslate[1]);fwd[14]=float(_volTranslate[2]);fwd[15]=1;
        glMultMatrixf(fwd);
    }

    if (_displayMode>=1) {
        float c[8][3]; int ci=0;
        for(int iz=0;iz<=1;++iz)for(int iy=0;iy<=1;++iy)for(int ix=0;ix<=1;++ix){
            c[ci][0]=float(ix?_bboxMax[0]:_bboxMin[0]);
            c[ci][1]=float(iy?_bboxMax[1]:_bboxMin[1]);
            c[ci][2]=float(iz?_bboxMax[2]:_bboxMin[2]); ++ci;}
        glLineWidth(1.5f); glColor3f(0,1,0);
        glBegin(GL_LINES);
        for(int e=0;e<12;++e){int i0=_bboxEdges[e][0],i1=_bboxEdges[e][1];
            glVertex3fv(c[i0]); glVertex3fv(c[i1]);}
        glEnd();
        float cx=float((_bboxMin[0]+_bboxMax[0])*0.5), cy=float((_bboxMin[1]+_bboxMax[1])*0.5), cz=float((_bboxMin[2]+_bboxMax[2])*0.5);
        float csz=float(std::max({_bboxMax[0]-_bboxMin[0],_bboxMax[1]-_bboxMin[1],_bboxMax[2]-_bboxMin[2]})*0.05);
        glColor3f(1,1,0); glBegin(GL_LINES);
        glVertex3f(cx-csz,cy,cz);glVertex3f(cx+csz,cy,cz);
        glVertex3f(cx,cy-csz,cz);glVertex3f(cx,cy+csz,cz);
        glVertex3f(cx,cy,cz-csz);glVertex3f(cx,cy,cz+csz); glEnd();
    }

    if (_displayMode>=2) {
        int curFrame=int(outputContext().frame())+_frameOffset;
        std::string curPath=resolveFramePath(curFrame);
        if (_previewPoints.empty()||_cachedPointsPath!=curPath||_cachedPointsFrame!=curFrame) {
            rebuildPointCloud(); _cachedPointsPath=curPath; _cachedPointsFrame=curFrame; }
        if (!_previewPoints.empty()) {
            glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE);
            glPointSize(static_cast<float>(std::max(1.0,_pointSize))); glEnable(GL_POINT_SMOOTH);
            glBegin(GL_POINTS); float inv=1.0f/_maxDensity;
            for (const auto& pt:_previewPoints) {
                float t=std::min(pt.density*inv,1.0f);
                glColor4f(0.3f+0.7f*t, 0.3f+0.7f*t, 0.3f+0.7f*t, 0.15f+0.85f*t);
                glVertex3f(pt.x,pt.y,pt.z);
            } glEnd(); glDisable(GL_BLEND);
        }
    }
    glPopMatrix();

    // ── Light markers (drawn in world space, outside volume transform) ──
    {
        struct { bool enabled; double* pos; double* target; double* color; double coneAngle; const char* label; } lts[3] = {
            { _light1Enable, _light1Pos, _light1Target, _light1Color, _light1ConeAngle, "L1" },
            { _light2Enable, _light2Pos, _light2Target, _light2Color, _light2ConeAngle, "L2" },
            { _light3Enable, _light3Pos, _light3Target, _light3Color, _light3ConeAngle, "L3" },
        };

        float ds = float(_lightDisplayScale);

        for (int i = 0; i < 3; ++i) {
            if (!lts[i].enabled) continue;
            float px = float(lts[i].pos[0]);
            float py = float(lts[i].pos[1]);
            float pz = float(lts[i].pos[2]);
            float cr = float(lts[i].color[0]);
            float cg = float(lts[i].color[1]);
            float cb = float(lts[i].color[2]);

            float sz = 0.4f * ds;

            // ── Light icon: cross + diamond ──
            glLineWidth(2.5f);
            glColor3f(cr, cg, cb);
            glBegin(GL_LINES);
            glVertex3f(px-sz, py, pz); glVertex3f(px+sz, py, pz);
            glVertex3f(px, py-sz, pz); glVertex3f(px, py+sz, pz);
            glVertex3f(px, py, pz-sz); glVertex3f(px, py, pz+sz);
            glEnd();

            // Diamond
            glBegin(GL_LINE_LOOP);
            glVertex3f(px, py+sz, pz);
            glVertex3f(px+sz, py, pz);
            glVertex3f(px, py-sz, pz);
            glVertex3f(px-sz, py, pz);
            glEnd();
            glBegin(GL_LINE_LOOP);
            glVertex3f(px, py, pz+sz);
            glVertex3f(px+sz, py, pz);
            glVertex3f(px, py, pz-sz);
            glVertex3f(px-sz, py, pz);
            glEnd();

            // ── Direction cone: from light toward target ──
            {
                float tgx = float(lts[i].target[0]);
                float tgy = float(lts[i].target[1]);
                float tgz = float(lts[i].target[2]);

                // Small target marker (crosshair)
                float tsz = sz * 0.4f;
                glLineWidth(1.5f);
                glColor4f(cr, cg, cb, 0.5f);
                glBegin(GL_LINES);
                glVertex3f(tgx-tsz, tgy, tgz); glVertex3f(tgx+tsz, tgy, tgz);
                glVertex3f(tgx, tgy-tsz, tgz); glVertex3f(tgx, tgy+tsz, tgz);
                glVertex3f(tgx, tgy, tgz-tsz); glVertex3f(tgx, tgy, tgz+tsz);
                glEnd();

                float dx = tgx - px, dy = tgy - py, dz = tgz - pz;
                float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (dist > 1e-4f) {
                    float idx = 1.f / dist;
                    float ndx = dx*idx, ndy = dy*idx, ndz = dz*idx;

                    // Build tangent frame
                    float upx=0,upy=1,upz=0;
                    if (std::abs(ndy) > 0.9f) { upx=1; upy=0; }
                    float tx = upy*ndz - upz*ndy;
                    float ty = upz*ndx - upx*ndz;
                    float tz = upx*ndy - upy*ndx;
                    float tlen = std::sqrt(tx*tx + ty*ty + tz*tz);
                    if (tlen > 1e-6f) { tx/=tlen; ty/=tlen; tz/=tlen; }
                    float bx = ndy*tz - ndz*ty;
                    float by = ndz*tx - ndx*tz;
                    float bz = ndx*ty - ndy*tx;

                    glEnable(GL_BLEND);
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

                    // Arrowhead at 30% along the line
                    float ax = px + dx*0.3f, ay = py + dy*0.3f, az = pz + dz*0.3f;
                    float arrowSz = sz * 0.6f;
                    glColor4f(cr, cg, cb, 0.8f);
                    glBegin(GL_LINES);
                    glVertex3f(ax, ay, az);
                    glVertex3f(ax - ndx*arrowSz + tx*arrowSz*0.5f,
                               ay - ndy*arrowSz + ty*arrowSz*0.5f,
                               az - ndz*arrowSz + tz*arrowSz*0.5f);
                    glVertex3f(ax, ay, az);
                    glVertex3f(ax - ndx*arrowSz - tx*arrowSz*0.5f,
                               ay - ndy*arrowSz - ty*arrowSz*0.5f,
                               az - ndz*arrowSz - tz*arrowSz*0.5f);
                    glVertex3f(ax, ay, az);
                    glVertex3f(ax - ndx*arrowSz + bx*arrowSz*0.5f,
                               ay - ndy*arrowSz + by*arrowSz*0.5f,
                               az - ndz*arrowSz + bz*arrowSz*0.5f);
                    glVertex3f(ax, ay, az);
                    glVertex3f(ax - ndx*arrowSz - bx*arrowSz*0.5f,
                               ay - ndy*arrowSz - by*arrowSz*0.5f,
                               az - ndz*arrowSz - bz*arrowSz*0.5f);
                    glEnd();

                    // Cone rays: 8 spread lines matching actual cone angle
                    float coneAngleRad = float(lts[i].coneAngle * 3.14159265358979 / 180.0);
                    float coneLen = dist * 0.5f;
                    float coneRad = coneLen * std::tan(std::min(coneAngleRad, 1.5f));
                    glLineWidth(1.0f);
                    glColor4f(cr, cg, cb, 0.2f);
                    glBegin(GL_LINES);
                    for (int r = 0; r < 8; ++r) {
                        float angle = float(r) * 0.7853981f;  // 2π/8
                        float cx2 = std::cos(angle) * coneRad;
                        float cy2 = std::sin(angle) * coneRad;
                        float ex = px + ndx*coneLen + tx*cx2 + bx*cy2;
                        float ey = py + ndy*coneLen + ty*cx2 + by*cy2;
                        float ez = pz + ndz*coneLen + tz*cx2 + bz*cy2;
                        glVertex3f(px, py, pz);
                        glVertex3f(ex, ey, ez);
                    }
                    glEnd();

                    // Cone ring at end
                    glColor4f(cr, cg, cb, 0.15f);
                    glBegin(GL_LINE_LOOP);
                    for (int r = 0; r < 16; ++r) {
                        float angle = float(r) * 0.3926991f;  // 2π/16
                        float cx2 = std::cos(angle) * coneRad;
                        float cy2 = std::sin(angle) * coneRad;
                        glVertex3f(px + ndx*coneLen + tx*cx2 + bx*cy2,
                                   py + ndy*coneLen + ty*cx2 + by*cy2,
                                   pz + ndz*coneLen + tz*cx2 + bz*cy2);
                    }
                    glEnd();

                    glDisable(GL_BLEND);
                }
            }

            // XYZ_knob handles (position + target)
            {
                static const char* pnames[] = {"light1_position","light2_position","light3_position"};
                static const char* tnames[] = {"light1_target","light2_target","light3_target"};
                Knob* k = knob(pnames[i]);
                if (k) k->draw_handle(ctx);
                k = knob(tnames[i]);
                if (k) k->draw_handle(ctx);
            }
        }
    }

    glPopAttrib();
}

// ─── Volume Transform ───────────────────────────────────────────────────

void VDBRenderIop::buildVolumeTransform(float out[16]) const
{
    const double deg2rad=3.14159265358979323846/180.0;
    const double rx=_volRotate[0]*deg2rad, ry=_volRotate[1]*deg2rad, rz=_volRotate[2]*deg2rad;
    const double sx=_volScale[0]*_volUniformScale, sy=_volScale[1]*_volUniformScale, sz=_volScale[2]*_volUniformScale;
    const double isx=(std::abs(sx)>1e-12)?1.0/sx:1e12;
    const double isy=(std::abs(sy)>1e-12)?1.0/sy:1e12;
    const double isz=(std::abs(sz)>1e-12)?1.0/sz:1e12;
    const double crx=cos(rx),srx=sin(rx),cry=cos(ry),sry=sin(ry),crz=cos(rz),srz=sin(rz);
    double F[3][3];
    F[0][0]=cry*crz;F[0][1]=srx*sry*crz-crx*srz;F[0][2]=crx*sry*crz+srx*srz;
    F[1][0]=cry*srz;F[1][1]=srx*sry*srz+crx*crz;F[1][2]=crx*sry*srz-srx*crz;
    F[2][0]=-sry;F[2][1]=srx*cry;F[2][2]=crx*cry;
    const double is[3]={isx,isy,isz};
    for(int col=0;col<3;++col){for(int row=0;row<3;++row) out[col*4+row]=float(is[row]*F[col][row]); out[col*4+3]=0;}
    double tx=-_volTranslate[0],ty=-_volTranslate[1],tz=-_volTranslate[2];
    out[12]=float(isx*(F[0][0]*tx+F[1][0]*ty+F[2][0]*tz));
    out[13]=float(isy*(F[0][1]*tx+F[1][1]*ty+F[2][1]*tz));
    out[14]=float(isz*(F[0][2]*tx+F[1][2]*ty+F[2][2]*tz));
    out[15]=1.0f;
}

// ─── Frustum Culling ────────────────────────────────────────────────────

bool VDBRenderIop::isVolumeInFrustum(int outputW, int outputH) const
{
    if (!_camValid||!_gridValid) return true;
    const double halfH=_halfW*double(outputH)/double(outputW);
    const double deg2rad=3.14159265358979323846/180.0;
    const double sx=_volScale[0]*_volUniformScale,sy=_volScale[1]*_volUniformScale,sz=_volScale[2]*_volUniformScale;
    const double rx=_volRotate[0]*deg2rad,ry=_volRotate[1]*deg2rad,rz=_volRotate[2]*deg2rad;
    const double crx=cos(rx),srx=sin(rx),cry=cos(ry),sry=sin(ry),crz=cos(rz),srz=sin(rz);
    for(int iz=0;iz<=1;++iz)for(int iy=0;iy<=1;++iy)for(int ix=0;ix<=1;++ix){
        double px=ix?_bboxMax[0]:_bboxMin[0],py=iy?_bboxMax[1]:_bboxMin[1],pz=iz?_bboxMax[2]:_bboxMin[2];
        px*=sx;py*=sy;pz*=sz;
        double ty=py*crx-pz*srx,tz=py*srx+pz*crx;py=ty;pz=tz;
        double txr=px*cry+pz*sry,tzr=-px*sry+pz*cry;px=txr;pz=tzr;
        double txz=px*crz-py*srz,tyz=px*srz+py*crz;px=txz;py=tyz;
        px+=_volTranslate[0];py+=_volTranslate[1];pz+=_volTranslate[2];
        double dx=px-_camOrigin[0],dy=py-_camOrigin[1],dz=pz-_camOrigin[2];
        double camX=_camRot[0][0]*dx+_camRot[1][0]*dy+_camRot[2][0]*dz;
        double camY=_camRot[0][1]*dx+_camRot[1][1]*dy+_camRot[2][1]*dz;
        double camZ=_camRot[0][2]*dx+_camRot[1][2]*dy+_camRot[2][2]*dz;
        if(camZ>=0) continue;
        double ndcX=camX/(-camZ*_halfW),ndcY=camY/(-camZ*halfH);
        if(ndcX>=-2&&ndcX<=2&&ndcY>=-2&&ndcY<=2) return true;
    }
    return false;
}
