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
#include <cstring>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

__constant__ float d_aaSimMatrix[20][20];

namespace {
float entropyGammaFromEnvironment() {
    const char* value = std::getenv("DIPPER_ENTROPY_GAMMA");
    if (!value) return 0.25f;
    char* end = nullptr;
    float gamma = std::strtof(value, &end);
    return (end != value && std::isfinite(gamma) && gamma >= 0.0f) ? gamma : 0.25f;
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
    constexpr float kBlosumShift = 1.5f;
    std::array<float, 400> matrix{};
    for (int a = 0; a < 20; ++a) {
        for (int b = 0; b < 20; ++b) {
            const float score = static_cast<float>(blosum[dipperToStandard[a]][dipperToStandard[b]]) + kBlosumShift;
            const float diagonalA = static_cast<float>(blosum[dipperToStandard[a]][dipperToStandard[a]]) + kBlosumShift;
            const float diagonalB = static_cast<float>(blosum[dipperToStandard[b]][dipperToStandard[b]]) + kBlosumShift;
            matrix[a * 20 + b] = score > 0.0f
                ? std::min(1.0f, score / std::sqrt(diagonalA * diagonalB))
                : 0.0f;
        }
    }
    for (int i = 0; i < 20; ++i) matrix[i * 20 + i] = 1.0f;
    return matrix;
}

float gammaAlphaFromEnvironment() {
    const char* value = std::getenv("DIPPER_GAMMA_ALPHA");
    if (!value) return 3.0f;
    char* end = nullptr;
    float alpha = std::strtof(value, &end);
    return (end != value && std::isfinite(alpha) && alpha > 0.0f) ? alpha : 3.0f;
}

bool binaryProteinFromEnvironment() {
    const char* value = std::getenv("DIPPER_BINARY_AA");
    return value && value[0] == '1';
}
}

void MashPlacement::initializeProteinSimilarityMatrix(const Param& params) {
    std::array<float, 400> similarity = buildProteinSimilarityMatrix();
    if (params.proteinBinary || binaryProteinFromEnvironment()) {
        similarity.fill(0.0f);
        for (int i = 0; i < 20; ++i) similarity[i * 20 + i] = 1.0f;
    }
    const cudaError_t err = cudaMemcpyToSymbol(
        d_aaSimMatrix,
        similarity.data(),
        similarity.size() * sizeof(float)
    );
    if (err != cudaSuccess) {
        fprintf(stderr, "Gpu_ERROR: cudaMemcpyToSymbol for amino-acid matrix failed!\n");
        exit(1);
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
        initializeProteinSimilarityMatrix(params);
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
        MashPlacement::g_msaContext.h_compressedSeqs = h_flattenCompressSeqs;
        MashPlacement::g_msaContext.h_siteWeights = weights;
        MashPlacement::g_msaContext.seqLen = seqLen;
        MashPlacement::g_msaContext.totalNumSequences = numSequences;
        MashPlacement::g_msaContext.params = params;
        MashPlacement::g_msaContext.valid = true;
    } else {
        MashPlacement::g_msaContext.valid = false;
    }

    cudaDeviceSynchronize();
}

void MashPlacement::MSADeviceArrays::deallocateDeviceArrays(){
    cudaFree(d_compressedSeqs);
    cudaFree(d_seqLengths);
    cudaFree(d_siteWeights);
    MashPlacement::g_msaContext.valid = false;
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

__device__ inline double safeLogDistanceArg(double value) {
    return log(fmax(value, 1.0e-12));
}

__device__ inline double safeDistanceRatio(double value) {
    return fmin(0.999999999, fmax(0.0, value));
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
    float gammaAlpha,
    float proteinSaturation,
    float priorWeight
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
                const double effectivePrior = isProtein ? priorWeight : 0.0;
                const double priorMatch = isProtein ? 0.15 * effectivePrior : 0.0;
                const double effectiveUseful = double(useful) + effectivePrior;
                const double effectiveMatch = double(match) + priorMatch;
                double uncor = (effectiveUseful > 0.0)
                    ? (1.0 - effectiveMatch / effectiveUseful) : 0.999;
                uncor = fmin(0.999999, fmax(0.0, uncor));
                if(distanceType==DIST_UNCORRECTED) dist[blockID]=uncor;
                else {
                    if (isProtein) {
                        const double alpha = gammaAlpha;
                        const double ratio = min(0.98, max(0.0, uncor / proteinSaturation));
                        const double gammaDist = alpha * (pow(1.0 - ratio, -1.0 / alpha) - 1.0);
                        dist[blockID] = min(3.5, max(0.0, gammaDist));
                    }
                else dist[blockID]=-0.75*safeLogDistanceArg(1.0-uncor/0.75);
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
                if (tot == 0) { dist[blockID] = 0.999999; continue; }
                for(int i=0;i<4;i++) fr[i]=double(frac[i])/tot/2.0;
                double h=0;
                h+=0.5*pr[0]*fr[0]*fr[2];
                h+=0.5*pr[1]*fr[0]*fr[3];
                h+=0.5*pr[2]*fr[1]*fr[2];
                h+=0.5*pr[3]*fr[1]*fr[3];
                double D=double(tot-match)/tot;
                double b=0.5*(1.0-fr[0]*fr[0]-fr[2]*fr[2]+D*D/h);
                dist[blockID]=-b*safeLogDistanceArg(1.0-D/b);
            }
        }
        else if(distanceType==DIST_KIMURA2P||distanceType==DIST_JINNEI){
            int p=0,q=0,tot=0;
            // calculateParams_K2P(rowId, idx, seqLen, compressedSeqs, p, q, tot);
            calculateParamsParallel_K2P(rowId, blockID, seqLen, compressedSeqs, p, q, tot);
            if (tx == 0) {
                if (tot == 0) { dist[blockID] = 0.999999; continue; }
                double pp=double(p)/tot,qq=double(q)/tot;
                if(distanceType==DIST_KIMURA2P) dist[blockID]=-0.5*safeLogDistanceArg((1-2*pp-qq)*sqrt(fmax(0.0,1-2*qq)));
                else dist[blockID]=0.5*(1.0/(1-2*pp-qq)+0.5/(1-qq*2)-1.5);
            }
        }
        else if(distanceType==DIST_TAMURA){
            int p=0,q=0,tot=0,gc1=0,gc2=0;
            // calculateParams_TAMURA(rowId, idx, seqLen, compressedSeqs, p, q, tot, gc1, gc2);
            calculateParamsParallel_TAMURA(rowId, blockID, seqLen, compressedSeqs, p, q, tot, gc1, gc2);
            if (tx == 0) {
                if (tot == 0) { dist[blockID] = 0.999999; continue; }
                double pp=double(p)/tot,qq=double(q)/tot, c=double(gc1)/tot+double(gc2)/tot-2*double(gc1)*double(gc2)/tot/tot;
                c = fmax(c, 1.0e-12);
                dist[blockID]=-c*safeLogDistanceArg(1-pp/c-qq)-0.5*(1-c)*safeLogDistanceArg(1-2*qq);
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
        params.gammaAlpha,
        params.proteinSaturation,
        params.lowOverlapPriorWeight
    );
}

namespace {

struct BinaryTreeNode {
    int origId = -1;
    int child[2] = {-1, -1}; // >= 0: internal node index (0..M-1); < 0: -(leafIdx + 1)
    double length[2] = {0.0, 0.0};
};

struct PrecomputedQuartet {
    double dAB = 0.0, dCD = 0.0, dAC = 0.0, dBD = 0.0, dAD = 0.0, dBC = 0.0;
    bool valid = false;
};

void runProfileBMENNI(std::vector<BinaryTreeNode>& nodes, int maxRounds) {
    if (!MashPlacement::g_msaContext.valid || !MashPlacement::g_msaContext.params.isProtein) return;
    const int M = static_cast<int>(nodes.size());
    if (M < 4 || maxRounds <= 0) return;

    const int seqLen = MashPlacement::g_msaContext.seqLen;
    if (seqLen <= 0) return;
    const uint64_t* h_compressedSeqs = MashPlacement::g_msaContext.h_compressedSeqs;
    if (!h_compressedSeqs) return;

    const auto& params = MashPlacement::g_msaContext.params;
    const auto& siteWeights = MashPlacement::g_msaContext.h_siteWeights;

    // 1. Build 20x20 similarity matrix and exact Cholesky factor Q (S = Q * Q^T).
    std::array<float, 400> sim = buildProteinSimilarityMatrix();
    if (params.proteinBinary || binaryProteinFromEnvironment()) {
        sim.fill(0.0f);
        for (int i = 0; i < 20; ++i) sim[i * 20 + i] = 1.0f;
    }
    float Q[20][20] = {};
    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = sim[i * 20 + j];
            for (int k = 0; k < j; ++k) sum -= static_cast<double>(Q[i][k]) * Q[j][k];
            if (i == j) {
                Q[i][i] = (sum > 1e-9) ? static_cast<float>(std::sqrt(sum)) : 0.0f;
            } else if (Q[j][j] > 1e-9f) {
                Q[i][j] = static_cast<float>(sum / Q[j][j]);
            }
        }
    }

    const int dimP = 20 * seqLen;
    const int totalDim = 21 * seqLen;
    const int wordsPerSeq = (seqLen + 7) / 8;

    // Precompute lookup table: for each (col, aa in 0..20), 20 floats for P and 1 float for O.
    std::vector<float> leafLookupP(static_cast<size_t>(seqLen) * 21 * 20, 0.0f);
    std::vector<float> leafLookupO(static_cast<size_t>(seqLen) * 21, 0.0f);
    for (int col = 0; col < seqLen; ++col) {
        const float w = (col < static_cast<int>(siteWeights.size())) ? siteWeights[col] : 1.0f;
        const float sw = std::sqrt(std::max(0.0f, w));
        for (int aa = 0; aa < 20; ++aa) {
            float* dst = &leafLookupP[(col * 21 + aa) * 20];
            for (int k = 0; k < 20; ++k) dst[k] = sw * Q[aa][k];
            leafLookupO[col * 21 + aa] = sw;
        }
    }

    auto fillLeafProfile = [&](int leafIdx, float* dst) {
        const uint64_t* row = h_compressedSeqs + 1ULL * leafIdx * wordsPerSeq;
        float* dstP = dst;
        float* dstO = dst + dimP;
        for (int w = 0; w < wordsPerSeq; ++w) {
            const uint64_t packed = row[w];
            for (int b = 0; b < 8; ++b) {
                const int col = w * 8 + b;
                if (col >= seqLen) break;
                int aa = static_cast<int>((packed >> (b * 8)) & 0xffu);
                if (aa > 20) aa = 20;
                std::memcpy(dstP + col * 20, &leafLookupP[(col * 21 + aa) * 20], 20 * sizeof(float));
                dstO[col] = leafLookupO[col * 21 + aa];
            }
        }
    };

    std::vector<float> PO(static_cast<size_t>(M) * totalDim, 0.0f);
    std::vector<float> PO_out(static_cast<size_t>(M) * totalDim, 0.0f);

    auto getNodeProfile = [&](int nodeCode, std::vector<float>& scratch) -> const float* {
        if (nodeCode >= 0) {
            return &PO[static_cast<size_t>(nodeCode) * totalDim];
        } else {
            fillLeafProfile(-nodeCode - 1, scratch.data());
            return scratch.data();
        }
    };

    auto averageProfiles = [&](float* dst, const float* p0, const float* p1) {
        for (int k = 0; k < totalDim; ++k) {
            dst[k] = 0.5f * (p0[k] + p1[k]);
        }
    };

    auto toEvolDist = [&](double match, double useful) -> double {
        const double effPrior = params.lowOverlapPriorWeight;
        const double effUseful = useful + effPrior;
        const double effMatch = match + 0.15 * effPrior;
        double uncor = (effUseful > 0.0) ? (1.0 - effMatch / effUseful) : 0.999;
        uncor = std::min(0.999999, std::max(0.0, uncor));
        if (params.distanceType == DIST_UNCORRECTED) return uncor;
        const double alpha = params.gammaAlpha;
        const double ratio = std::min(0.98, std::max(0.0, uncor / params.proteinSaturation));
        return std::min(3.5, std::max(0.0, alpha * (std::pow(1.0 - ratio, -1.0 / alpha) - 1.0)));
    };

    auto evalQuartet = [&](const float* pA, const float* pB, const float* pC, const float* pD) -> PrecomputedQuartet {
        float mAB = 0.0f, mCD = 0.0f, mAC = 0.0f, mBD = 0.0f, mAD = 0.0f, mBC = 0.0f;
        for (int k = 0; k < dimP; ++k) {
            const float a = pA[k];
            const float b = pB[k];
            const float c = pC[k];
            const float d = pD[k];
            mAB += a * b;
            mCD += c * d;
            mAC += a * c;
            mBD += b * d;
            mAD += a * d;
            mBC += b * c;
        }
        float uAB = 0.0f, uCD = 0.0f, uAC = 0.0f, uBD = 0.0f, uAD = 0.0f, uBC = 0.0f;
        const float* oA = pA + dimP;
        const float* oB = pB + dimP;
        const float* oC = pC + dimP;
        const float* oD = pD + dimP;
        for (int k = 0; k < seqLen; ++k) {
            const float a = oA[k];
            const float b = oB[k];
            const float c = oC[k];
            const float d = oD[k];
            uAB += a * b;
            uCD += c * d;
            uAC += a * c;
            uBD += b * d;
            uAD += a * d;
            uBC += b * c;
        }
        PrecomputedQuartet q;
        q.dAB = toEvolDist(mAB, uAB);
        q.dCD = toEvolDist(mCD, uCD);
        q.dAC = toEvolDist(mAC, uAC);
        q.dBD = toEvolDist(mBD, uBD);
        q.dAD = toEvolDist(mAD, uAD);
        q.dBC = toEvolDist(mBC, uBC);
        q.valid = true;
        return q;
    };

    std::vector<int> order;
    order.reserve(M);
    std::vector<int> parent(M, -1);
    std::vector<int> depth(M, 0);
    std::vector<std::vector<int>> levels;

    auto rebuildTraversal = [&]() {
        order.clear();
        levels.clear();
        parent[0] = -1;
        depth[0] = 0;
        std::vector<int> stk = {0};
        while (!stk.empty()) {
            const int u = stk.back();
            stk.pop_back();
            order.push_back(u);
            const int du = depth[u];
            if (du >= static_cast<int>(levels.size())) levels.resize(du + 1);
            levels[du].push_back(u);
            for (int iv = 0; iv < 2; ++iv) {
                const int c = nodes[u].child[iv];
                if (c >= 0) {
                    parent[c] = u;
                    depth[c] = du + 1;
                    stk.push_back(c);
                }
            }
        }
        std::reverse(order.begin(), order.end());
    };

    auto computeBottomUp = [&]() {
        for (int d = static_cast<int>(levels.size()) - 1; d >= 0; --d) {
            const auto& lvl = levels[d];
            tbb::parallel_for(tbb::blocked_range<size_t>(0, lvl.size()), [&](const tbb::blocked_range<size_t>& r) {
                std::vector<float> buf0(totalDim), buf1(totalDim);
                for (size_t idx = r.begin(); idx < r.end(); ++idx) {
                    const int u = lvl[idx];
                    const float* p0 = getNodeProfile(nodes[u].child[0], buf0);
                    const float* p1 = getNodeProfile(nodes[u].child[1], buf1);
                    averageProfiles(&PO[static_cast<size_t>(u) * totalDim], p0, p1);
                }
            });
        }
    };

    auto computeTopDown = [&]() {
        {
            std::vector<float> buf0(totalDim), buf1(totalDim);
            const int c0 = nodes[0].child[0];
            const int c1 = nodes[0].child[1];
            const float* p0 = getNodeProfile(c0, buf0);
            const float* p1 = getNodeProfile(c1, buf1);
            if (c0 >= 0) std::memcpy(&PO_out[static_cast<size_t>(c0) * totalDim], p1, totalDim * sizeof(float));
            if (c1 >= 0) std::memcpy(&PO_out[static_cast<size_t>(c1) * totalDim], p0, totalDim * sizeof(float));
        }
        for (size_t d = 1; d < levels.size(); ++d) {
            const auto& lvl = levels[d];
            tbb::parallel_for(tbb::blocked_range<size_t>(0, lvl.size()), [&](const tbb::blocked_range<size_t>& r) {
                std::vector<float> buf0(totalDim), buf1(totalDim);
                for (size_t idx = r.begin(); idx < r.end(); ++idx) {
                    const int u = lvl[idx];
                    const int c0 = nodes[u].child[0];
                    const int c1 = nodes[u].child[1];
                    if (c0 < 0 && c1 < 0) continue;
                    const float* pOutU = &PO_out[static_cast<size_t>(u) * totalDim];
                    if (c0 >= 0) {
                        const float* p1 = getNodeProfile(c1, buf1);
                        averageProfiles(&PO_out[static_cast<size_t>(c0) * totalDim], pOutU, p1);
                    }
                    if (c1 >= 0) {
                        const float* p0 = getNodeProfile(c0, buf0);
                        averageProfiles(&PO_out[static_cast<size_t>(c1) * totalDim], pOutU, p0);
                    }
                }
            });
        }
    };

    rebuildTraversal();
    computeBottomUp();

    std::vector<uint8_t> active(M, 1);
    std::vector<uint8_t> nextActive(M, 0);
    std::vector<int> touched(M, -1);
    std::vector<PrecomputedQuartet> preQuartets(M);
    std::vector<float> scratchB(totalDim), scratchC(totalDim), scratchD(totalDim);

    for (int round = 0; round < maxRounds; ++round) {
        computeTopDown();

        // Parallel quartet evaluation across all active internal edges (u -> v).
        tbb::parallel_for(tbb::blocked_range<int>(1, M), [&](const tbb::blocked_range<int>& r) {
            std::vector<float> bBuf(totalDim), cBuf(totalDim), dBuf(totalDim);
            for (int v = r.begin(); v < r.end(); ++v) {
                if (!active[v]) {
                    preQuartets[v].valid = false;
                    continue;
                }
                const int u = parent[v];
                if (u <= 0) {
                    preQuartets[v].valid = false;
                    continue;
                }
                const int iv = (nodes[u].child[0] == v) ? 0 : 1;
                const int B = nodes[u].child[1 - iv];
                const int C = nodes[v].child[0];
                const int D = nodes[v].child[1];
                const float* pA = &PO_out[static_cast<size_t>(u) * totalDim];
                const float* pB = getNodeProfile(B, bBuf);
                const float* pC = getNodeProfile(C, cBuf);
                const float* pD = getNodeProfile(D, dBuf);
                preQuartets[v] = evalQuartet(pA, pB, pC, pD);
            }
        });

        std::fill(nextActive.begin(), nextActive.end(), 0);
        int swaps = 0;

        auto markNeighborhoodActive = [&](int u, int v) {
            auto markNode = [&](int x) {
                if (x >= 0 && x < M) {
                    nextActive[x] = 1;
                    if (nodes[x].child[0] >= 0) nextActive[nodes[x].child[0]] = 1;
                    if (nodes[x].child[1] >= 0) nextActive[nodes[x].child[1]] = 1;
                }
            };
            markNode(u);
            markNode(v);
            if (parent[u] >= 0) markNode(parent[u]);
            if (nodes[v].child[0] >= 0) markNode(nodes[v].child[0]);
            if (nodes[v].child[1] >= 0) markNode(nodes[v].child[1]);
        };

        // Sequential postorder walk (identical order to reversed(order) in prototype: top-down / reverse-postorder).
        for (int idx = M - 1; idx >= 0; --idx) {
            const int u = order[idx];
            if (u == 0) continue;
            for (int iv = 0; iv < 2; ++iv) {
                const int v = nodes[u].child[iv];
                if (v < 0) continue;
                const int B = nodes[u].child[1 - iv];
                const int C = nodes[v].child[0];
                const int D = nodes[v].child[1];

                const bool dirtyInRound =
                    (touched[u] == round) || (touched[v] == round) ||
                    (B >= 0 && touched[B] == round) ||
                    (C >= 0 && touched[C] == round) ||
                    (D >= 0 && touched[D] == round);

                if (!active[v] && !dirtyInRound) continue;

                PrecomputedQuartet q;
                if (!dirtyInRound && preQuartets[v].valid) {
                    q = preQuartets[v];
                } else {
                    const float* pA = &PO_out[static_cast<size_t>(u) * totalDim];
                    const float* pB = getNodeProfile(B, scratchB);
                    const float* pC = getNodeProfile(C, scratchC);
                    const float* pD = getNodeProfile(D, scratchD);
                    q = evalQuartet(pA, pB, pC, pD);
                }

                const double s0 = q.dAB + q.dCD;
                const double s1 = q.dAC + q.dBD;
                const double s2 = q.dAD + q.dBC;

                if (s1 < s0 - 1e-7 && s1 <= s2) {
                    const double lenB = nodes[u].length[1 - iv];
                    const double lenC = nodes[v].length[0];
                    nodes[u].child[1 - iv] = C;
                    nodes[u].length[1 - iv] = lenC;
                    nodes[v].child[0] = B;
                    nodes[v].length[0] = lenB;
                    if (C >= 0) parent[C] = u;
                    if (B >= 0) parent[B] = v;

                    const double newLenUV = 0.25 * (q.dAB + q.dAD + q.dBC + q.dCD) - 0.5 * s1;
                    nodes[u].length[iv] = std::max(1e-6, newLenUV);

                    const float* pA = &PO_out[static_cast<size_t>(u) * totalDim];
                    const float* pB = getNodeProfile(B, scratchB);
                    const float* pC = getNodeProfile(C, scratchC);
                    const float* pD = getNodeProfile(D, scratchD);

                    float* poV = &PO[static_cast<size_t>(v) * totalDim];
                    float* poU = &PO[static_cast<size_t>(u) * totalDim];
                    float* poOutV = &PO_out[static_cast<size_t>(v) * totalDim];

                    averageProfiles(poV, pB, pD);
                    averageProfiles(poU, poV, pC);
                    if (C >= 0) averageProfiles(&PO_out[static_cast<size_t>(C) * totalDim], pA, poV);
                    averageProfiles(poOutV, pA, pC);
                    if (B >= 0) averageProfiles(&PO_out[static_cast<size_t>(B) * totalDim], poOutV, pD);
                    if (D >= 0) averageProfiles(&PO_out[static_cast<size_t>(D) * totalDim], poOutV, pB);

                    touched[u] = round;
                    touched[v] = round;
                    if (B >= 0) touched[B] = round;
                    if (C >= 0) touched[C] = round;
                    if (D >= 0) touched[D] = round;
                    markNeighborhoodActive(u, v);
                    ++swaps;
                } else if (s2 < s0 - 1e-7 && s2 < s1) {
                    const double lenB = nodes[u].length[1 - iv];
                    const double lenD = nodes[v].length[1];
                    nodes[u].child[1 - iv] = D;
                    nodes[u].length[1 - iv] = lenD;
                    nodes[v].child[1] = B;
                    nodes[v].length[1] = lenB;
                    if (D >= 0) parent[D] = u;
                    if (B >= 0) parent[B] = v;

                    const double newLenUV = 0.25 * (q.dAB + q.dAC + q.dBC + q.dCD) - 0.5 * s2;
                    nodes[u].length[iv] = std::max(1e-6, newLenUV);

                    const float* pA = &PO_out[static_cast<size_t>(u) * totalDim];
                    const float* pB = getNodeProfile(B, scratchB);
                    const float* pC = getNodeProfile(C, scratchC);
                    const float* pD = getNodeProfile(D, scratchD);

                    float* poV = &PO[static_cast<size_t>(v) * totalDim];
                    float* poU = &PO[static_cast<size_t>(u) * totalDim];
                    float* poOutV = &PO_out[static_cast<size_t>(v) * totalDim];

                    averageProfiles(poV, pC, pB);
                    averageProfiles(poU, poV, pD);
                    if (D >= 0) averageProfiles(&PO_out[static_cast<size_t>(D) * totalDim], pA, poV);
                    averageProfiles(poOutV, pA, pD);
                    if (C >= 0) averageProfiles(&PO_out[static_cast<size_t>(C) * totalDim], poOutV, pB);
                    if (B >= 0) averageProfiles(&PO_out[static_cast<size_t>(B) * totalDim], poOutV, pC);

                    touched[u] = round;
                    touched[v] = round;
                    if (B >= 0) touched[B] = round;
                    if (C >= 0) touched[C] = round;
                    if (D >= 0) touched[D] = round;
                    markNeighborhoodActive(u, v);
                    ++swaps;
                } else if (nodes[u].length[iv] <= 0.0) {
                    const double bmeLen = 0.25 * (q.dAC + q.dAD + q.dBC + q.dBD) - 0.5 * s0;
                    nodes[u].length[iv] = std::max(1e-6, bmeLen);
                }
            }
        }

        if (swaps == 0) break;
        rebuildTraversal();
        computeBottomUp();
        active.swap(nextActive);
    }
}

} // namespace

void MashPlacement::refineNJTreeWithProfileNNI(
    int rootNode,
    int numLeaves,
    std::vector<std::vector<std::pair<int, double>>>& tree,
    int maxRounds
) {
    if (!g_msaContext.valid || !g_msaContext.params.isProtein || numLeaves < 5) return;

    std::vector<BinaryTreeNode> nodes;
    nodes.reserve(numLeaves - 1);
    bool validBinary = true;

    std::function<int(int)> buildDfs = [&](int u) -> int {
        if (tree[u].empty()) {
            return -(u + 1);
        }
        if (tree[u].size() != 2) {
            validBinary = false;
            return -1;
        }
        const int myIdx = static_cast<int>(nodes.size());
        nodes.push_back(BinaryTreeNode{u, {-1, -1}, {tree[u][0].second, tree[u][1].second}});
        const int c0 = buildDfs(tree[u][0].first);
        const int c1 = buildDfs(tree[u][1].first);
        nodes[myIdx].child[0] = c0;
        nodes[myIdx].child[1] = c1;
        return myIdx;
    };

    buildDfs(rootNode);
    if (!validBinary || nodes.size() < 4) return;

    runProfileBMENNI(nodes, maxRounds);

    for (const auto& nd : nodes) {
        const int u = nd.origId;
        tree[u].clear();
        for (int k = 0; k < 2; ++k) {
            const int c = nd.child[k];
            const int origC = (c >= 0) ? nodes[c].origId : (-c - 1);
            tree[u].push_back({origC, nd.length[k]});
        }
    }
}

void MashPlacement::refineAdjacencyTreeWithProfileNNI(
    int rootNode,
    int totalSlots,
    int* h_head,
    int* h_e,
    int* h_nxt,
    int* h_belong,
    double* h_len,
    int* outEdgeCount,
    int maxRounds
) {
    if (!g_msaContext.valid || !g_msaContext.params.isProtein) return;

    std::vector<BinaryTreeNode> nodes;
    std::vector<int> leafNodes;
    bool validBinary = true;

    std::function<int(int, int)> buildDfs = [&](int u, int from) -> int {
        int outEdges[2] = {-1, -1};
        int deg = 0;
        for (int i = h_head[u]; i != -1; i = h_nxt[i]) {
            if (h_e[i] == from) continue;
            if (deg < 2) outEdges[deg] = i;
            ++deg;
        }
        if (deg == 0) {
            leafNodes.push_back(u);
            return -(u + 1);
        }
        if (deg != 2) {
            validBinary = false;
            return -1;
        }
        const int myIdx = static_cast<int>(nodes.size());
        nodes.push_back(BinaryTreeNode{u, {-1, -1}, {h_len[outEdges[0]], h_len[outEdges[1]]}});
        const int c0 = buildDfs(h_e[outEdges[0]], u);
        const int c1 = buildDfs(h_e[outEdges[1]], u);
        nodes[myIdx].child[0] = c0;
        nodes[myIdx].child[1] = c1;
        return myIdx;
    };

    buildDfs(rootNode, -1);
    if (!validBinary || nodes.size() < 4) return;

    runProfileBMENNI(nodes, maxRounds);

    for (const auto& nd : nodes) h_head[nd.origId] = -1;
    for (int lf : leafNodes) h_head[lf] = -1;

    int edgeCount = 0;
    for (const auto& nd : nodes) {
        const int origU = nd.origId;
        for (int k = 0; k < 2; ++k) {
            const int c = nd.child[k];
            const int origC = (c >= 0) ? nodes[c].origId : (-c - 1);
            const double w = nd.length[k];

            // origC -> origU
            h_e[edgeCount] = origU;
            h_len[edgeCount] = w;
            if (h_belong) h_belong[edgeCount] = origC;
            h_nxt[edgeCount] = h_head[origC];
            h_head[origC] = edgeCount++;

            // origU -> origC
            h_e[edgeCount] = origC;
            h_len[edgeCount] = w;
            if (h_belong) h_belong[edgeCount] = origU;
            h_nxt[edgeCount] = h_head[origU];
            h_head[origU] = edgeCount++;
        }
    }
    if (outEdgeCount) *outEdgeCount = edgeCount;
}
