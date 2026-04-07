// ═══════════════════════════════════════════════════════════════════════════
// VDBEngine.cpp — Scanline engine: GPU buffer bilinear upscale
// ═══════════════════════════════════════════════════════════════════════════

#include "VDBRenderIop.h"
#include <cmath>
#include <algorithm>

using namespace DD::Image;

void VDBRenderIop::engine(int y, int x, int r, ChannelMask channels, Row& row)
{
    if (_gpuFrameReady && _gpuRenderedWidth > 0 && _gpuRenderedHeight > 0) {
        const int srcW = _gpuRenderedWidth, srcH = _gpuRenderedHeight;
        const int stride = _gpuOutputStride;
        const int bx1 = _screenBboxX1, by1 = _screenBboxY1;

        float* rOut=row.writable(Chan_Red), *gOut=row.writable(Chan_Green);
        float* bOut=row.writable(Chan_Blue), *aOut=row.writable(Chan_Alpha);

        float *scR=0,*scG=0,*scB=0,*scA=0;
        float *emR=0,*emG=0,*emB=0,*emA=0;
        float *tpR=0,*tpG=0,*tpB=0,*tpA=0;
        float *l1R=0,*l1G=0,*l1B=0,*l1A=0;
        float *l2R=0,*l2G=0,*l2B=0,*l2A=0;
        float *l3R=0,*l3G=0,*l3B=0,*l3A=0;
        float *abR=0,*abG=0,*abB=0,*abA=0;
        int scOff=-1, emOff=-1, tpOff=-1, l1Off=-1, l2Off=-1, l3Off=-1, abOff=-1, off=4;
        if (_createScatterAOV) { scOff=off; off+=4;
            scR=row.writable(chan_scatter_r); scG=row.writable(chan_scatter_g);
            scB=row.writable(chan_scatter_b); scA=row.writable(chan_scatter_a); }
        if (_createEmissionAOV) { emOff=off; off+=4;
            emR=row.writable(chan_emission_r); emG=row.writable(chan_emission_g);
            emB=row.writable(chan_emission_b); emA=row.writable(chan_emission_a); }
        if (_createTemperatureAOV) { tpOff=off; off+=4;
            tpR=row.writable(chan_temperature_r); tpG=row.writable(chan_temperature_g);
            tpB=row.writable(chan_temperature_b); tpA=row.writable(chan_temperature_a); }
        if (_createLight1AOV || _light1Enable) { l1Off=off; off+=4;
            l1R=row.writable(chan_light1_r); l1G=row.writable(chan_light1_g);
            l1B=row.writable(chan_light1_b); l1A=row.writable(chan_light1_a); }
        if (_createLight2AOV || _light2Enable) { l2Off=off; off+=4;
            l2R=row.writable(chan_light2_r); l2G=row.writable(chan_light2_g);
            l2B=row.writable(chan_light2_b); l2A=row.writable(chan_light2_a); }
        if (_createLight3AOV || _light3Enable) { l3Off=off; off+=4;
            l3R=row.writable(chan_light3_r); l3G=row.writable(chan_light3_g);
            l3B=row.writable(chan_light3_b); l3A=row.writable(chan_light3_a); }
        if (_createAlbedoAOV) { abOff=off; off+=4;
            abR=row.writable(chan_albedo_r); abG=row.writable(chan_albedo_g);
            abB=row.writable(chan_albedo_b); abA=row.writable(chan_albedo_a); }
        float *dpZ=0;
        float *posX=0,*posY=0,*posZ=0,*posA=0;
        float *prX=0,*prY=0,*prZ=0,*prA=0;
        int dpOff=-1, posOff=-1, prOff=-1;
        if (_createDepthAOV) { dpOff=off; off+=4;
            dpZ=row.writable(chan_depth_z); }
        if (_createPositionAOV) { posOff=off; off+=4;
            posX=row.writable(chan_P_x); posY=row.writable(chan_P_y);
            posZ=row.writable(chan_P_z); posA=row.writable(chan_P_a); }
        if (_createPrefAOV) { prOff=off; off+=4;
            prX=row.writable(chan_Pref_x); prY=row.writable(chan_Pref_y);
            prZ=row.writable(chan_Pref_z); prA=row.writable(chan_Pref_a); }
        float *nX=0,*nY=0,*nZ=0,*nA=0;
        int nOff=-1;
        if (_createNormalAOV) { nOff=off; off+=4;
            nX=row.writable(chan_N_x); nY=row.writable(chan_N_y);
            nZ=row.writable(chan_N_z); nA=row.writable(chan_N_a); }
        float *evR=0,*evG=0,*evB=0,*evA=0;
        int evOff=-1;
        if (_createEnvAOV) { evOff=off; off+=4;
            evR=row.writable(chan_env_r); evG=row.writable(chan_env_g);
            evB=row.writable(chan_env_b); evA=row.writable(chan_env_a); }

        auto zeroPixel = [&](int ix) {
            rOut[ix]=gOut[ix]=bOut[ix]=aOut[ix]=0.0f;
            if(scOff>=0){scR[ix]=scG[ix]=scB[ix]=scA[ix]=0.0f;}
            if(emOff>=0){emR[ix]=emG[ix]=emB[ix]=emA[ix]=0.0f;}
            if(tpOff>=0){tpR[ix]=tpG[ix]=tpB[ix]=tpA[ix]=0.0f;}
            if(l1Off>=0){l1R[ix]=l1G[ix]=l1B[ix]=l1A[ix]=0.0f;}
            if(l2Off>=0){l2R[ix]=l2G[ix]=l2B[ix]=l2A[ix]=0.0f;}
            if(l3Off>=0){l3R[ix]=l3G[ix]=l3B[ix]=l3A[ix]=0.0f;}
            if(abOff>=0){abR[ix]=abG[ix]=abB[ix]=abA[ix]=0.0f;}
            if(dpOff>=0){dpZ[ix]=0.0f;}
            if(posOff>=0){posX[ix]=posY[ix]=posZ[ix]=posA[ix]=0.0f;}
            if(prOff>=0){prX[ix]=prY[ix]=prZ[ix]=prA[ix]=0.0f;}
            if(nOff>=0){nX[ix]=nY[ix]=nZ[ix]=nA[ix]=0.0f;}
            if(evOff>=0){evR[ix]=evG[ix]=evB[ix]=evA[ix]=0.0f;}
        };

        const Box& bi=info_.box();
        int bboxW=bi.r()-bi.x(), bboxH=bi.t()-bi.y();
        const float scaleX=static_cast<float>(srcW)/std::max(1,bboxW);
        const float scaleY=static_cast<float>(srcH)/std::max(1,bboxH);

        for (int ix=x; ix<r; ++ix) {
            float srcXf=(ix-bx1+0.5f)*scaleX-0.5f;
            float srcYf=(y -by1+0.5f)*scaleY-0.5f;
            if (srcXf<-0.5f||srcXf>=srcW||srcYf<-0.5f||srcYf>=srcH) { zeroPixel(ix); continue; }

            int x0=int(std::floor(srcXf)), y0=int(std::floor(srcYf));
            int x1=x0+1, y1=y0+1; float fx=srcXf-x0, fy=srcYf-y0;
            x0=std::max(0,std::min(x0,srcW-1)); x1=std::max(0,std::min(x1,srcW-1));
            y0=std::max(0,std::min(y0,srcH-1)); y1=std::max(0,std::min(y1,srcH-1));

            const float* p00=&_gpuOutputBuffer[(size_t(y0)*srcW+x0)*stride];
            const float* p10=&_gpuOutputBuffer[(size_t(y0)*srcW+x1)*stride];
            const float* p01=&_gpuOutputBuffer[(size_t(y1)*srcW+x0)*stride];
            const float* p11=&_gpuOutputBuffer[(size_t(y1)*srcW+x1)*stride];
            float w00=(1-fx)*(1-fy), w10=fx*(1-fy), w01=(1-fx)*fy, w11=fx*fy;

            float tmp[52];
            for (int c=0; c<stride; ++c) tmp[c]=w00*p00[c]+w10*p10[c]+w01*p01[c]+w11*p11[c];
            rOut[ix]=tmp[0]; gOut[ix]=tmp[1]; bOut[ix]=tmp[2]; aOut[ix]=tmp[3];
            if(scOff>=0){scR[ix]=tmp[scOff];scG[ix]=tmp[scOff+1];scB[ix]=tmp[scOff+2];scA[ix]=tmp[scOff+3];}
            if(emOff>=0){emR[ix]=tmp[emOff];emG[ix]=tmp[emOff+1];emB[ix]=tmp[emOff+2];emA[ix]=tmp[emOff+3];}
            if(tpOff>=0){tpR[ix]=tmp[tpOff];tpG[ix]=tmp[tpOff+1];tpB[ix]=tmp[tpOff+2];tpA[ix]=tmp[tpOff+3];}
            if(l1Off>=0){l1R[ix]=tmp[l1Off];l1G[ix]=tmp[l1Off+1];l1B[ix]=tmp[l1Off+2];l1A[ix]=tmp[l1Off+3];}
            if(l2Off>=0){l2R[ix]=tmp[l2Off];l2G[ix]=tmp[l2Off+1];l2B[ix]=tmp[l2Off+2];l2A[ix]=tmp[l2Off+3];}
            if(l3Off>=0){l3R[ix]=tmp[l3Off];l3G[ix]=tmp[l3Off+1];l3B[ix]=tmp[l3Off+2];l3A[ix]=tmp[l3Off+3];}
            if(abOff>=0){abR[ix]=tmp[abOff];abG[ix]=tmp[abOff+1];abB[ix]=tmp[abOff+2];abA[ix]=tmp[abOff+3];}
            if(dpOff>=0){
                // Depth: nearest-neighbor to avoid edge bleeding
                int nx=std::max(0,std::min(int(std::round(srcXf)),srcW-1));
                int ny=std::max(0,std::min(int(std::round(srcYf)),srcH-1));
                const float* pNN=&_gpuOutputBuffer[(size_t(ny)*srcW+nx)*stride];
                dpZ[ix]=pNN[dpOff];
            }
            if(posOff>=0){posX[ix]=tmp[posOff];posY[ix]=tmp[posOff+1];posZ[ix]=tmp[posOff+2];posA[ix]=tmp[posOff+3];}
            if(prOff>=0){prX[ix]=tmp[prOff];prY[ix]=tmp[prOff+1];prZ[ix]=tmp[prOff+2];prA[ix]=tmp[prOff+3];}
            if(nOff>=0){nX[ix]=tmp[nOff];nY[ix]=tmp[nOff+1];nZ[ix]=tmp[nOff+2];nA[ix]=tmp[nOff+3];}
            if(evOff>=0){evR[ix]=tmp[evOff];evG[ix]=tmp[evOff+1];evB[ix]=tmp[evOff+2];evA[ix]=tmp[evOff+3];}
        }
        return;
    }

    // No GPU frame ready → black
    foreach (z, channels) {
        float* ptr = row.writable(z);
        for (int ix=x; ix<r; ++ix) ptr[ix] = 0.0f;
    }
}
