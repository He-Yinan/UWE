// ------------------------------------------------------------------
// Faster R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Shaoqing Ren
// ------------------------------------------------------------------

// compile using python3 setup.py build_ext --inplace

#include "feat_dist_gpu.hpp"
#include <vector>
#include <iostream>
#include <cstdio>
#include <iostream>
using namespace std;

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

__device__ inline float cosine_sim(float const * const a, float const * const b, int const feat_dim) {
  float sim = 0.0;
  float norm_a = 0.0;
  float norm_b = 0.0;
  for (int i = 0; i < feat_dim; i++)
  {
    sim += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }

  norm_a = sqrtf(norm_a);
  norm_b = sqrtf(norm_b);
  return sim / (norm_a * norm_b);
}

__device__ inline float kl_diverg(float const * const a, float const * const b, int const feat_dim) {
  float dist = 0.0, kl1 = 0.0, kl2 = 0.0;

  for (int i = 0; i < feat_dim; i++)
  {
    kl1 += a[i] * log(a[i] / b[i]);
    kl2 += b[i] * log(b[i] / a[i]); 
  }

  return (kl1 + kl2)/2;
}

__device__ inline float euclidean_dist(float const * const a, float const * const b, int const feat_dim) {
  float dist = 0.0;
  for (int i = 0; i < feat_dim; i++)
  {
    dist += (a[i] - b[i]) * (a[i] - b[i]);
  }

  return sqrtf(dist);
}

__global__ void feat_dist_kernel(float* dist_dev, const float *feat_dev, const float* sel_feat_dev, const int feat_num, const int feat_dim, const int metric) {
//   extern __shared__ float s_feat_sel [];

//   for(int i = threadIdx.x; i < feat_dim; i += blockDim.x) s_feat_sel[i] = sel_feat_dev[i];
//    __syncthreads();

  uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < feat_num)
  {
      if (metric == 0)
         dist_dev[idx] = euclidean_dist(feat_dev + idx*feat_dim, sel_feat_dev, feat_dim);
      else if (metric == 1)
         dist_dev[idx] = cosine_sim(feat_dev + idx*feat_dim, sel_feat_dev, feat_dim);
      else if (metric == 2)
         dist_dev[idx] = kl_diverg(feat_dev + idx*feat_dim, sel_feat_dev, feat_dim);
  }
}

void _set_device(int device_id) {
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == device_id) {
    return;
  }
  // The call to cudaSetDevice must come before any calls to Get, which
  // may perform initialization using the GPU.
  CUDA_CHECK(cudaSetDevice(device_id));
}

void _feat_dist_gpu(float* dist_host, const float* feat_host, const float* sel_feat, int feat_num, int feat_dim, int metric, int device_id) {
  
    _set_device(device_id);

    float* feat_dev = NULL;
    float* sel_feat_dev = NULL;
    float* dist_dev = NULL;

    // int id;
    // size_t free, total;

    // unsigned long long needed = (unsigned long long)feat_num * (unsigned long long)feat_dim * (unsigned long long)sizeof(float);

    // cudaGetDevice( &id );
    // cudaMemGetInfo( &free, &total );
    // cout << "GPU " << id << " memory: free=" << free << ", total=" << total << endl;
    // cout << "needed " << needed << endl;
    // cout << "feat_num " << feat_num << endl;
    // cout << "feat_dim " << feat_dim << endl;
    // cout << "sizeof float " << sizeof(float) << endl;

    CUDA_CHECK(cudaMalloc(&feat_dev,
                        (unsigned long long)feat_num * feat_dim * sizeof(float)));

    // cudaMemGetInfo( &free, &total );
    // cout << "GPU " << id << " memory: free=" << free << ", total=" << total << endl;

    CUDA_CHECK(cudaMemcpy(feat_dev,
                        feat_host,
                        (unsigned long long)feat_num * feat_dim * sizeof(float),
                        cudaMemcpyHostToDevice));

    // cudaMemGetInfo( &free, &total );
    // cout << "GPU " << id << " memory: free=" << free << ", total=" << total << endl;

    CUDA_CHECK(cudaMalloc(&sel_feat_dev,
                        (unsigned long long)feat_dim * sizeof(float)));
    
    // cudaMemGetInfo( &free, &total );
    // cout << "GPU " << id << " memory: free=" << free << ", total=" << total << endl;

    CUDA_CHECK(cudaMemcpy(sel_feat_dev,
                        sel_feat,
                        (unsigned long long)feat_dim * sizeof(float),
                        cudaMemcpyHostToDevice));

    // cudaMemGetInfo( &free, &total );
    // cout << "GPU " << id << " memory: free=" << free << ", total=" << total << endl;

    CUDA_CHECK(cudaMalloc(&dist_dev,
                        (unsigned long long)feat_num * sizeof(float)));

    // cudaMemGetInfo( &free, &total );
    // cout << "GPU " << id << " memory: free=" << free << ", total=" << total << endl;


    int const threadsPerBlock = 512;
    dim3 threads(threadsPerBlock);
    dim3 blocks(DIVUP(feat_num, threadsPerBlock));


    clock_t begin = clock();
    // feat_dist_kernel<<<blocks, threads, sizeof(float)*feat_dim>>>(dist_dev,
    //                                 feat_dev,
    //                                 sel_feat_dev,
    //                                 feat_num,
    //                                 feat_dim,
    //                                 metric
    //                                 );

    feat_dist_kernel<<<blocks, threads>>>(dist_dev,
                                feat_dev,
                                sel_feat_dev,
                                feat_num,
                                feat_dim,
                                metric
                                );

    cudaDeviceSynchronize();
    clock_t end = clock();
    //printf("feat_dist_kernel time: %f ms\n", double(end - begin)/CLOCKS_PER_SEC * 1000.0);


    CUDA_CHECK(cudaMemcpy(dist_host,
                        dist_dev,
                        sizeof(float) * feat_num,
                        cudaMemcpyDeviceToHost));


    CUDA_CHECK(cudaFree(feat_dev));
    CUDA_CHECK(cudaFree(dist_dev));
}