/**
 * MSA.cu
 *
 * Aligned-sequence device arrays and distance construction. Allocates 4-bit
 * compressed MSA, seq lengths; computes pairwise distances on GPU (uncorrected,
 * Jukes-Cantor, Tajima-Nei, K2P, Tamura, Jin-Nei) from compressed columns.
 */

#include "mash_placement.cuh"

#include <stdio.h>
#include <queue>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <chrono>
#include <iostream>
#include <cub/cub.cuh>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <array>
#include <cmath>

__constant__ float d_aaSimMatrix[20][20];

namespace {
float entropyGammaFromEnvironment() {
    const char* value = std::getenv("DIPPER_ENTROPY_GAMMA");
    if (!value) return 2.0f;
    char* end = nullptr;
    float gamma = std::strtof(value, &end);
    return (end != value && std::isfinite(gamma) && gamma >= 0.0f) ? gamma : 2.0f;
}

std::array<float, 400> buildProteinSimilarityMatrix() {
    // BLOSUM62 in its standard A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V order.
    static constexpr int blosum[20][20] = {
        {4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0},
        {-1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3},
        {-2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3},
        {-2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3},
        {0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1},
        {-1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2},
        {-1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2},
        {0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3},
        {-2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3},
        {-1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3},
        {-1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1},
        {-1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2},
        {-1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1},
        {-2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1},
        {-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2},
        {1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2},
        {0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0},
        {-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3},
        {-2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1},
        {0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4}
    };
    static constexpr int dipperToStandard[20] = {0,4,7,16,3,6,13,8,9,11,10,12,2,14,5,1,15,19,17,18};
    std::array<float, 400> matrix{};
    for (int a = 0; a < 20; ++a) {
        for (int b = 0; b < 20; ++b) {
            const int score = blosum[dipperToStandard[a]][dipperToStandard[b]];
            const int diagonalA = blosum[dipperToStandard[a]][dipperToStandard[a]];
            const int diagonalB = blosum[dipperToStandard[b]][dipperToStandard[b]];
            matrix[a * 20 + b] = score > 0
                ? std::min(1.0f, static_cast<float>(score) / std::sqrt(static_cast<float>(diagonalA * diagonalB)))
                : 0.0f;
        }
    }
    for (int i = 0; i < 20; ++i) matrix[i * 20 + i] = 1.0f;
    return matrix;
}

float gammaAlphaFromEnvironment() {
    const char* value = std::getenv("DIPPER_GAMMA_ALPHA");
    if (!value) return 0.8f;
    char* end = nullptr;
    float alpha = std::strtof(value, &end);
    return (end != value && std::isfinite(alpha) && alpha > 0.0f) ? alpha : 0.8f;
}

bool binaryProteinFromEnvironment() {
    const char* value = std::getenv("DIPPER_BINARY_AA");
    return value && value[0] == '1';
}
}

void MashPlacement::MSADeviceArrays::allocateDeviceArrays(uint64_t ** h_compressedSeqs, uint64_t * h_seqLengths, size_t num, Param& params)
{
    cudaError_t err;

    numSequences = int(num);
    seqLen = h_seqLengths[0];
    // Allocate memory
    err = cudaMalloc(&d_seqLengths, 1ll*numSequences*sizeof(uint64_t));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Gpu_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    /* Flatten data */
    uint64_t flatStringLength=0;
    for (size_t i =0; i<numSequences; i++) flatStringLength+= params.isProtein ? (h_seqLengths[i]+7)/8 : (h_seqLengths[i]+15)/16;
    uint64_t * h_flattenCompressSeqs = new uint64_t[flatStringLength];
    flatStringLength=0;
    for (size_t i =0; i<numSequences; i++) 
    {
        uint64_t flatStringLengthLocal = params.isProtein ? (h_seqLengths[i]+7)/8 : (h_seqLengths[i]+15)/16;
        flatStringLength+=flatStringLengthLocal;
        for (size_t j=0; j<flatStringLengthLocal;j++)  
        {
            h_flattenCompressSeqs[j] = h_compressedSeqs[i][j];
            // if (i==9) printf("%u\n",h_flattenCompressSeqs[j]); 
        }
        h_flattenCompressSeqs += flatStringLengthLocal;
    }
    h_flattenCompressSeqs -= flatStringLength;


    err = cudaMalloc(&d_compressedSeqs, 1ll*flatStringLength*sizeof(uint64_t));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Gpu_ERROR: cudaMalloc failed!\n");
        exit(1);
    }


    // Transfer data

    err = cudaMemcpy(d_seqLengths, h_seqLengths, 1ll*numSequences*sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "Gpu_ERROR: cudaMemCpy failed!\n");
        exit(1);
    }

    err = cudaMemcpy(d_compressedSeqs, h_flattenCompressSeqs, 1ll*flatStringLength*sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "Gpu_ERROR: cudaMemCpy failed!\n");
        exit(1);
    }

    // Protein distances use one weight per alignment column.  The weights
    // are computed once from the complete MSA and reused by every pairwise
    // GPU calculation.  DNA keeps a null pointer and the kernel uses w=1.
    if (params.isProtein) {
        std::array<float, 400> similarity = buildProteinSimilarityMatrix();
        if (params.proteinBinary || binaryProteinFromEnvironment()) {
            similarity.fill(0.0f);
            for (int i = 0; i < 20; ++i) similarity[i * 20 + i] = 1.0f;
        }
        err = cudaMemcpyToSymbol(d_aaSimMatrix, similarity.data(), similarity.size() * sizeof(float));
        if (err != cudaSuccess) {
            fprintf(stderr, "Gpu_ERROR: cudaMemcpyToSymbol for amino-acid matrix failed!\n");
            exit(1);
        }
        const float gamma = params.entropyGamma;
        const int length = seqLen;
        std::vector<std::array<uint32_t, 20>> counts(length);
        for (auto& column : counts) column.fill(0);
        for (size_t row = 0; row < numSequences; ++row) {
            const uint64_t words = (h_seqLengths[row] + 7) / 8;
            for (uint64_t word = 0; word < words; ++word) {
                const uint64_t packed = h_compressedSeqs[row][word];
                for (int b = 0; b < 8; ++b) {
                    const int column = static_cast<int>(word * 8 + b);
                    if (column >= length) break;
                    const uint8_t aa = static_cast<uint8_t>((packed >> (b * 8)) & 0xffu);
                    if (aa < 20) ++counts[column][aa];
                }
            }
        }
        std::vector<float> weights(length, 1.0f);
        if (gamma != 0.0f) {
            constexpr float invLog20 = 1.0f / 2.995732273553991f;
            for (int column = 0; column < length; ++column) {
                uint32_t total = 0;
                for (uint32_t count : counts[column]) total += count;
                if (total == 0) continue;
                float entropy = 0.0f;
                for (uint32_t count : counts[column]) {
                    if (count == 0) continue;
                    const float frequency = static_cast<float>(count) / total;
                    entropy -= frequency * std::log(frequency);
                }
                entropy = std::min(1.0f, std::max(0.0f, entropy * invLog20));
                weights[column] = std::exp(-gamma * entropy);
            }
        }
        const size_t paddedLength = ((static_cast<size_t>(length) + 7) / 8) * 8;
        std::vector<float> paddedWeights(paddedLength, 0.0f);
        std::copy(weights.begin(), weights.end(), paddedWeights.begin());
        err = cudaMalloc(&d_siteWeights, paddedLength * sizeof(float));
        if (err != cudaSuccess) {
            fprintf(stderr, "Gpu_ERROR: cudaMalloc for site weights failed!\n");
            exit(1);
        }
        err = cudaMemcpy(d_siteWeights, paddedWeights.data(), paddedLength * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "Gpu_ERROR: cudaMemcpy for site weights failed!\n");
            exit(1);
        }
    }

    cudaDeviceSynchronize();
}

void MashPlacement::MSADeviceArrays::deallocateDeviceArrays(){
    cudaFree(d_compressedSeqs);
    cudaFree(d_seqLengths);
    cudaFree(d_siteWeights);
    // cudaFree(d_hashList);
    // cudaFree(d_mashDist);
}

#define DIST_UNCORRECTED 1
#define DIST_JUKESCANTOR 2
#define DIST_TAJIMANEI 3
#define DIST_KIMURA2P 4
#define DIST_TAMURA 5
#define DIST_JINNEI 6

/** Count useful (non-gap) sites and matches between two compressed rows; DNA 4-bit. */
__device__ void calculateParams(int tarRowId, int curRowId, int seqLen, uint64_t * compressedSeqs, int & useful, int & match){
    int compLen=(seqLen+15)/16;
    long long px=1ll*curRowId*compLen, py=1ll*tarRowId*compLen;
    for(int i=0;i<compLen;i++){
        long long vt=compressedSeqs[px+i], vc=compressedSeqs[py+i];
        for(int j=0;j<16&&i*16+j<seqLen;j++){
            int et=(vt>>(j*4))&15, ec=(vc>>(j*4))&15;
            // Pairwise deletion: a site is informative only when both
            // sequences contain an unambiguous nucleotide.
            if(et<4&&ec<4) {
                useful++;
                if(et==ec) match++;
            }
        }
    }
}

/** Parallel count useful/matches over compressed rows; supports DNA (4-bit) and protein (8-bit). */
__device__ void calculateParamsParallel(int tarRowId, int curRowId, int seqLen, uint64_t * compressedSeqs, float * siteWeights, float & useful, float & match, bool isProtein){
    int tx=threadIdx.x, bs=blockDim.x, bx=blockIdx.x;
    int compLen=isProtein ? (seqLen+7)/8 : (seqLen+15)/16;
    long long px=1ll*curRowId*compLen, py=1ll*tarRowId*compLen;

    // create a shared memory array to store results
    __shared__ float sharedUseful[1024];
    __shared__ float sharedMatch[1024];
    sharedUseful[tx] = 0;
    sharedMatch[tx] = 0;

    for (int i=tx; i<compLen; i+=1024) {
        long long vt=compressedSeqs[px+i], vc=compressedSeqs[py+i];
        if (isProtein) {
            for(int j=0;j<8&&i*8+j<seqLen;j++){
                int et=(vt>>(j*8))&255, ec=(vc>>(j*8))&255;
                if(et<20&&ec<20) {
                    const float weight = siteWeights ? siteWeights[i * 8 + j] : 1.0f;
                    const float similarity = d_aaSimMatrix[et][ec];
                    sharedUseful[tx] += weight;
                    sharedMatch[tx] += weight * similarity;
                }
            }
        } else {
            for(int j=0;j<16&&i*16+j<seqLen;j++){
                int et=(vt>>(j*4))&15, ec=(vc>>(j*4))&15;
                if(et<4&&ec<4) {
                    sharedUseful[tx] += 1.0f;
                    if(et==ec) sharedMatch[tx] += 1.0f;
                }
            }
        }
    }
    __syncthreads();

    // reduce the results in shared memory
    for (int stride = bs / 2; stride > 0; stride /= 2) {
        if (tx < stride) {
            sharedUseful[tx] += sharedUseful[tx + stride];
            sharedMatch[tx] += sharedMatch[tx + stride];
        }
        __syncthreads();
    }

    // write the final results to the first thread
    if (tx == 0) {
        useful = sharedUseful[0];
        match = sharedMatch[0];
    }
    __syncthreads();
}

/** Tajima-Nei: base counts frac[4], pair counts pr[4], total and match. */
__device__ void calculateParams_TJ(int tarRowId, int curRowId, int seqLen, uint64_t * compressedSeqs, int * frac, int &tot, int &match, int * pr){
    int compLen=(seqLen+15)/16;
    long long px=1ll*curRowId*compLen, py=1ll*tarRowId*compLen;
    for(int i=0;i<compLen;i++){
        long long vt=compressedSeqs[px+i], vc=compressedSeqs[py+i];
        for(int j=0;j<16&&i*16+j<seqLen;j++){
            int et=(vt>>(j*4))&15, ec=(vc>>(j*4))&15;
            if(et>=4||ec>=4) continue;
            frac[ec]++, frac[et]++, tot++;
            if(ec>et){
                int temp=ec;
                ec=et,et=temp;
            }
            if(ec==et) match++;
            if(ec==0&&et==2) pr[0]++;
            else if(ec==0&&et==3) pr[1]++;
            else if(ec==1&&et==2) pr[2]++;
            else if(ec==1&&et==3) pr[3]++;
        }
    }
}

/** Parallel Tajima-Nei parameter computation. */
__device__ void calculateParamsParallel_TJ(int tarRowId, int curRowId, int seqLen, uint64_t * compressedSeqs, int * frac, int &tot, int &match, int * pr){
    int tx=threadIdx.x, bs=blockDim.x, bx=blockIdx.x;
    int compLen=(seqLen+15)/16;
    long long px=1ll*curRowId*compLen, py=1ll*tarRowId*compLen;

    __shared__ int sharedP0[128];
    __shared__ int sharedP1[128];
    __shared__ int sharedP2[128];
    __shared__ int sharedP3[128];
    __shared__ int sharedFrac0[128];
    __shared__ int sharedFrac1[128];
    __shared__ int sharedFrac2[128];
    __shared__ int sharedFrac3[128];
    __shared__ int sharedMatch[128];
    __shared__ int sharedTotal[128];
    if (tx < 128) {
        sharedMatch[tx] = 0;
        sharedTotal[tx] = 0;
        sharedP0[tx] = 0;
        sharedP1[tx] = 0;
        sharedP2[tx] = 0;
        sharedP3[tx] = 0;
        sharedFrac0[tx] = 0;
        sharedFrac1[tx] = 0;
        sharedFrac2[tx] = 0;
        sharedFrac3[tx] = 0;

        for (int i=tx; i<compLen; i+=128) {
            long long vt=compressedSeqs[px+i], vc=compressedSeqs[py+i];
            for(int j=0;j<16&&i*16+j<seqLen;j++){
                int et=(vt>>(j*4))&15, ec=(vc>>(j*4))&15;
                if(et>=4||ec>=4) continue;
                if (ec == 0) sharedFrac0[tx]++;
                else if (ec == 1) sharedFrac1[tx]++;
                else if (ec == 2) sharedFrac2[tx]++;
                else if (ec == 3) sharedFrac3[tx]++;
                if (et == 0) sharedFrac0[tx]++;
                else if (et == 1) sharedFrac1[tx]++;
                else if (et == 2) sharedFrac2[tx]++;
                else if (et == 3) sharedFrac3[tx]++;
                sharedTotal[tx]++;
                if(ec>et){
                    int temp=ec;
                    ec=et,et=temp;
                }
                if(ec==et) sharedMatch[tx]++;
                if(ec==0&&et==2) sharedP0[tx]++;
                else if(ec==0&&et==3) sharedP1[tx]++;
                else if(ec==1&&et==2) sharedP2[tx]++;
                else if(ec==1&&et==3) sharedP3[tx]++;
            }
        }
    }
    __syncthreads();

    // reduction
    for(int stride=128/2; stride>0; stride/=2){
        if(tx<stride){
            sharedP0[tx] += sharedP0[tx + stride];
            sharedP1[tx] += sharedP1[tx + stride];
            sharedP2[tx] += sharedP2[tx + stride];
            sharedP3[tx] += sharedP3[tx + stride];
            sharedFrac0[tx] += sharedFrac0[tx + stride];
            sharedFrac1[tx] += sharedFrac1[tx + stride];
            sharedFrac2[tx] += sharedFrac2[tx + stride];
            sharedFrac3[tx] += sharedFrac3[tx + stride];
            sharedMatch[tx] += sharedMatch[tx + stride];
            sharedTotal[tx] += sharedTotal[tx + stride];
        }
        __syncthreads();
    }

    

    // write the final results to the first thread
    if (tx == 0) {
        frac[0] = sharedFrac0[0];
        frac[1] = sharedFrac1[0];
        frac[2] = sharedFrac2[0];
        frac[3] = sharedFrac3[0];
        match = sharedMatch[0];
        tot = sharedTotal[0];
        pr[0] = sharedP0[0];
        pr[1] = sharedP1[0];
        pr[2] = sharedP2[0];
        pr[3] = sharedP3[0];
    }
    __syncthreads();
}

/** Kimura 2-parameter: transitions p, transversions q, total. */
__device__ void calculateParams_K2P(int tarRowId, int curRowId, int seqLen, uint64_t * compressedSeqs, int &p, int &q, int &tot){
    int compLen=(seqLen+15)/16;
    long long px=1ll*curRowId*compLen, py=1ll*tarRowId*compLen;
    for(int i=0;i<compLen;i++){
        long long vt=compressedSeqs[px+i], vc=compressedSeqs[py+i];
        for(int j=0;j<16&&i*16+j<seqLen;j++){
            int et=(vt>>(j*4))&15, ec=(vc>>(j*4))&15;
            if(et>=4||ec>=4) continue;
            tot++;
            if(et==ec) continue;
            if(et%2==ec%2) p++;
            else q++;
        }
    }
}

/** Parallel K2P parameter computation. */
__device__ void calculateParamsParallel_K2P(int tarRowId, int curRowId, int seqLen, uint64_t * compressedSeqs, int &p, int &q, int &tot){
    int tx=threadIdx.x, bs=blockDim.x, bx=blockIdx.x;
    int compLen=(seqLen+15)/16;
    long long px=1ll*curRowId*compLen, py=1ll*tarRowId*compLen;

    // create a shared memory array to store results
    __shared__ int sharedP[1024];
    __shared__ int sharedQ[1024];
    __shared__ int sharedTot[1024];
    sharedP[tx] = 0;
    sharedQ[tx] = 0;
    sharedTot[tx] = 0;

    for (int i=tx; i<compLen; i+=1024) {
        long long vt=compressedSeqs[px+i], vc=compressedSeqs[py+i];
        for(int j=0;j<16&&i*16+j<seqLen;j++){
            int et=(vt>>(j*4))&15, ec=(vc>>(j*4))&15;
            if(et>=4||ec>=4) continue;
            sharedTot[tx]++;
            if(et==ec) continue;
            if(et%2==ec%2) sharedP[tx]++;
            else sharedQ[tx]++;
        }
    }
    __syncthreads();
    // reduce the results in shared memory
    for (int stride = 1024 / 2; stride > 0; stride /= 2) {
        if (tx < stride) {
            sharedP[tx] += sharedP[tx + stride];
            sharedQ[tx] += sharedQ[tx + stride];
            sharedTot[tx] += sharedTot[tx + stride];
        }
        __syncthreads();
    }
    // write the final results to the first thread
    if (tx == 0) {
        p = sharedP[0];
        q = sharedQ[0];
        tot = sharedTot[0];
    }
    __syncthreads();
}

/** Tamura: p, q, tot, and GC counts gc1, gc2. */
__device__ void calculateParams_TAMURA(int tarRowId, int curRowId, int seqLen, uint64_t * compressedSeqs, int &p, int &q, int &tot, int &gc1, int &gc2){
    int compLen=(seqLen+15)/16;
    long long px=1ll*curRowId*compLen, py=1ll*tarRowId*compLen;
    for(int i=0;i<compLen;i++){
        long long vt=compressedSeqs[px+i], vc=compressedSeqs[py+i];
        for(int j=0;j<16&&i*16+j<seqLen;j++){
            int et=(vt>>(j*4))&15, ec=(vc>>(j*4))&15;
            if(et>=4||ec>=4) continue;
            tot++;
            if(et==ec) continue;
            if(et%2==ec%2) p++;
            else q++;
            if(ec==1||ec==2) gc1++;
            if(et==1||et==2) gc2++;
        }
    }
}

/** Parallel Tamura parameter computation. */
__device__ void calculateParamsParallel_TAMURA(int tarRowId, int curRowId, int seqLen, uint64_t * compressedSeqs, int &p, int &q, int &tot, int &gc1, int &gc2){
    int tx=threadIdx.x, bs=blockDim.x, bx=blockIdx.x;
    int compLen=(seqLen+15)/16;
    long long px=1ll*curRowId*compLen, py=1ll*tarRowId*compLen;

    // create a shared memory array to store results
    __shared__ int sharedP[512];
    __shared__ int sharedQ[512];
    __shared__ int sharedTot[512];
    __shared__ int sharedGC1[512];
    __shared__ int sharedGC2[512];
    if (tx < 512) {
        sharedP[tx] = 0;
        sharedQ[tx] = 0;
        sharedTot[tx] = 0;
        sharedGC1[tx] = 0;
        sharedGC2[tx] = 0;
        for (int i=tx; i<compLen; i+=512) {
            long long vt=compressedSeqs[px+i], vc=compressedSeqs[py+i];
            for(int j=0;j<16&&i*16+j<seqLen;j++){
                int et=(vt>>(j*4))&15, ec=(vc>>(j*4))&15;
                if(et>=4||ec>=4) continue;
                sharedTot[tx]++;
                if(et==ec) continue;
                if(et%2==ec%2) sharedP[tx]++;
                else sharedQ[tx]++;
                if(ec==1||ec==2) sharedGC1[tx]++;
                if(et==1||et==2) sharedGC2[tx]++;
            }
        }
    }
    __syncthreads();
    // reduce the results in shared memory
    for (int stride = 512 / 2; stride > 0; stride /= 2) {
        if (tx < stride) {
            sharedP[tx] += sharedP[tx + stride];
            sharedQ[tx] += sharedQ[tx + stride];
            sharedTot[tx] += sharedTot[tx + stride];
            sharedGC1[tx] += sharedGC1[tx + stride];
            sharedGC2[tx] += sharedGC2[tx + stride];
        }
        __syncthreads();
    }
    // write the final results to the first thread
    if (tx == 0) {
        p = sharedP[0];
        q = sharedQ[0];
        tot = sharedTot[0];
        gc1 = sharedGC1[0];
        gc2 = sharedGC2[0];
    }
    __syncthreads();

}

/** Build distance row rowId: one block per target row; dispatch by distanceType (uncorrected, JC, TJ, K2P, Tamura, Jin-Nei). */
__global__ void MSADistConstruction(
    int rowId,
    uint64_t * compressedSeqs,
    double * dist,
    int seqLen,
    int numSequences,
    int distanceType,
    bool isProtein,
    float * siteWeights,
    float gammaAlpha
){
    int tx=threadIdx.x, bs=blockDim.x, bx=blockIdx.x;
    int idx=tx+bs*bx;
    // if(idx>=rowId) return;
    for (int blockID = bx; blockID < rowId; blockID += gridDim.x) {
        if (blockID >= rowId) return; // Ensure we don't access out of bounds
        // printf("bx: %d, rowId: %d\n", blockID, rowId);
        if(distanceType==DIST_UNCORRECTED||distanceType==DIST_JUKESCANTOR){
            float useful=0.0f, match=0.0f;
            calculateParamsParallel(rowId, blockID, seqLen, compressedSeqs, siteWeights, useful, match, isProtein);
            // calculateParams(rowId, idx, seqLen, compressedSeqs, useful, match);
            if (tx == 0) {   
                const double priorWeight = isProtein ? 3.0 : 0.0;
                const double priorMatch = isProtein ? 0.15 * priorWeight : 0.0;
                const double effectiveUseful = double(useful) + priorWeight;
                const double effectiveMatch = double(match) + priorMatch;
                double uncor = (effectiveUseful > 0.0)
                    ? (1.0 - effectiveMatch / effectiveUseful) : 0.999;
                uncor = fmin(0.999999, fmax(0.0, uncor));
                if(distanceType==DIST_UNCORRECTED) dist[blockID]=uncor;
                else {
                    if (isProtein) {
                        const double alpha = gammaAlpha;
                        const double ratio = min(0.98, max(0.0, uncor / 0.92));
                        const double gammaDist = alpha * (pow(1.0 - ratio, -1.0 / alpha) - 1.0);
                        dist[blockID] = min(3.5, max(0.0, gammaDist));
                    }
                    else dist[blockID]=-0.75*log(1.0-uncor/0.75);
                }
            }
        }
        else if(distanceType==DIST_TAJIMANEI){
            int frac[4]={},pr[4]={},tot=0,match=0;
            double fr[4]={};
            // calculateParams_TJ(rowId, idx, seqLen, compressedSeqs, frac, tot, match, pr);
            calculateParamsParallel_TJ(rowId, blockID, seqLen, compressedSeqs, frac, tot, match, pr);
            // __syncthreads();
            if (tx == 0) {
                for(int i=0;i<4;i++) fr[i]=double(frac[i])/tot/2.0;
                double h=0;
                h+=0.5*pr[0]*fr[0]*fr[2];
                h+=0.5*pr[1]*fr[0]*fr[3];
                h+=0.5*pr[2]*fr[1]*fr[2];
                h+=0.5*pr[3]*fr[1]*fr[3];
                double D=double(tot-match)/tot;
                double b=0.5*(1.0-fr[0]*fr[0]-fr[2]*fr[2]+D*D/h);
                dist[blockID]=-b*log(1.0-D/b);
            }
        }
        else if(distanceType==DIST_KIMURA2P||distanceType==DIST_JINNEI){
            int p=0,q=0,tot=0;
            // calculateParams_K2P(rowId, idx, seqLen, compressedSeqs, p, q, tot);
            calculateParamsParallel_K2P(rowId, blockID, seqLen, compressedSeqs, p, q, tot);
            if (tx == 0) {
                double pp=double(p)/tot,qq=double(q)/tot;
                if(distanceType==DIST_KIMURA2P) dist[blockID]=-0.5*log((1-2*pp-qq)*sqrt(1-2*qq));
                else dist[blockID]=0.5*(1.0/(1-2*pp-qq)+0.5/(1-qq*2)-1.5);
            }
        }
        else if(distanceType==DIST_TAMURA){
            int p=0,q=0,tot=0,gc1=0,gc2=0;
            // calculateParams_TAMURA(rowId, idx, seqLen, compressedSeqs, p, q, tot, gc1, gc2);
            calculateParamsParallel_TAMURA(rowId, blockID, seqLen, compressedSeqs, p, q, tot, gc1, gc2);
            if (tx == 0) {
                double pp=double(p)/tot,qq=double(q)/tot, c=double(gc1)/tot+double(gc2)/tot-2*double(gc1)*double(gc2)/tot/tot;
                dist[blockID]=-c*log(1-pp/c-qq)-0.5*(1-c)*log(1-2*qq);
            }
        }
        else dist[blockID]=0.0;
    }
}

/** Launch MSADistConstruction for row rowId; threadNum 1024 used internally for distance logic. */
void MashPlacement::MSADeviceArrays::distConstructionOnGpu(Param& params, int rowId, double* d_mashDist) const {
    int threadNum = 1024, blockNum = 1024;
    MSADistConstruction <<<blockNum, threadNum>>> (
        rowId, 
        d_compressedSeqs, 
        d_mashDist, 
        seqLen,
        numSequences,
        params.distanceType,
        params.isProtein,
        d_siteWeights,
        params.gammaAlpha
    );
}
