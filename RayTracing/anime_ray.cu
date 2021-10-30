/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include "cuda.h"
#include "./common/book.h"
#include "./common/cpu_bitmap.h"
#include "./sphere.cuh"
        
#define DIM 1024                                // 图像宽度
#define CAM_SHAKE_WIDTH 15                      // 相机晃动幅度
#define CAM_FRAME_PER_SHAKE 2                   // 相机晃动频率，单位为 "多少帧晃动一次"
#define ENABLE_CAMERA_SHAKE false               // 是否启用相机晃动
#define THREAD_DIM 64


#define rnd( x ) (x * rand() / RAND_MAX)

int* shifts;                        // 小球当前帧下的x,y偏移量
double* angles;                     // 小球当前移动的角度
curandStateXORWOW* states;          // 生成小球当前帧下的位移量的随机数状态

__constant__ Sphere s[SPHERES];     // 分配在constant内存上的小球
Sphere* globalSpheres;              // 分配在global内存上的小球
cudaEvent_t start, stop;            // 用于记录每一帧的计算时间的event实例

__global__ void kernel( unsigned char *ptr, int c_shift_x, int c_shift_y, int* shifts, Sphere* global_s) {

    // block内使用shared内存来预读取和存储所有小球的运动情况
    // 用于加速光线追踪的hit操作在读取小球运动情况时的速度
    __shared__ int shared_shifts[SPHERES * SPHERE_SHIFT_DATA_WIDTH]; 

    const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    const int thread_per_block = blockDim.x * blockDim.y;
    int cur_thread_id = thread_id;
    
    while (cur_thread_id < SPHERES) {
        shared_shifts[SPHERE_SHIFT_DATA_WIDTH * cur_thread_id] = shifts[SPHERE_SHIFT_DATA_WIDTH * cur_thread_id];
        shared_shifts[SPHERE_SHIFT_DATA_WIDTH * cur_thread_id + 1] = shifts[SPHERE_SHIFT_DATA_WIDTH * cur_thread_id + 1];
        shared_shifts[SPHERE_SHIFT_DATA_WIDTH * cur_thread_id + 2] = shifts[SPHERE_SHIFT_DATA_WIDTH * cur_thread_id + 2];
        shared_shifts[SPHERE_SHIFT_DATA_WIDTH * cur_thread_id + 3] = shifts[SPHERE_SHIFT_DATA_WIDTH * cur_thread_id + 3];
        cur_thread_id += thread_per_block;
    }
    // block内同步写入操作，随后再各自并行运行各自的hit读操作
    __syncthreads();

    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float   ox = (x - DIM/2 + c_shift_x);
    float   oy = (y - DIM/2 + c_shift_y);

    float   r=0, g=0, b=0;
    float   maxz = -INF;
    for(int i=0; i<SPHERES; i++) {
        float   n;
        float   t = s[i].hit(ox, oy, &n, shared_shifts); //global_s[i].hit(ox, oy, &n, shared_shifts); //shifts);
        // 检查光线是否和小球发生碰撞
        // 只会取距离最近的小球
        if (t > maxz) {
            float fscale = n;
            r = s[i].r * fscale;//s[i].r * fscale;global_s[i].r * fscale;        // 光线沿法线方向射出，在z轴方向的分量需要乘上sinθ
            g = s[i].g * fscale;//s[i].g * fscale;global_s[i].g * fscale;
            b = s[i].b * fscale;//s[i].b * fscale;global_s[i].b * fscale;
            maxz = t;                   
        }
    } 

    ptr[offset*4 + 0] = (int)(r * 255);
    ptr[offset*4 + 1] = (int)(g * 255);
    ptr[offset*4 + 2] = (int)(b * 255);
    ptr[offset*4 + 3] = 255;
}

// globals needed by the update routine
struct DataBlock {
    unsigned char* dev_bitmap;      // 真正的数据存放位置，指向GPU上的内存
    CPUAnimBitmap* bitmap;          // 用于生成动画的bitmap结构
};

//####################################################################################
// 生成一个动画帧的回调函数
//####################################################################################
void generate_frame(DataBlock* d) {
    HANDLE_ERROR(cudaEventRecord(start, 0));

    static int c_shift_x = 0, c_shift_y = 0, camera_frame_count_loop = 0;
    if (ENABLE_CAMERA_SHAKE && (camera_frame_count_loop = ++camera_frame_count_loop % CAM_FRAME_PER_SHAKE) == 0) {
        int x_rd_val = rnd(CAM_SHAKE_WIDTH), y_rd_val = rnd(CAM_SHAKE_WIDTH);
        // 每一帧中camera的x，y偏移量，用于造成动画的移动效果
        c_shift_x += (x_rd_val - CAM_SHAKE_WIDTH / 2);
        c_shift_y += (y_rd_val - CAM_SHAKE_WIDTH / 2);
    }

    dim3    blocks(DIM / THREAD_DIM, DIM / THREAD_DIM);
    dim3    threads(THREAD_DIM, THREAD_DIM);
    //dim3    blocks(DIM / 32, DIM / 32);
    //dim3    threads(32, 32);

    // 函数静态变量控制小球偏移量更新的频率
    static int sphere_frame_count_loop = 0;
    // 每次要绘制前，检查当前帧是否应该更新小球的随机偏移量
    if (ENABLE_SPHERE_SHAKE && (sphere_frame_count_loop = ++sphere_frame_count_loop % SPHERE_FRAME_PER_SHAKE) == 0) {
        if (SPHERE_SHAKE_TYPE == 0) 
            updateSphereShiftsWithAxisMove <<< SPHERE_BLOCK, 1 >>> (states, shifts);
        else {
            updateSphereShiftsWithCurveMove <<< SPHERE_BLOCK, 1 >>> (states, shifts, angles);
            updateSphereCurveSpeedAngle <<< SPHERE_BLOCK, 1 >>> (states, shifts, angles);
        }
    }
    kernel <<<blocks,threads>>> (d->dev_bitmap, c_shift_x, c_shift_y, shifts, globalSpheres);

    HANDLE_ERROR(cudaMemcpy(d->bitmap->get_ptr(),
        d->dev_bitmap,
        d->bitmap->image_size(),
        cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                        start, stop ) );
    printf( "Time to generate a frame:  %3.1f ms\n", elapsedTime );
}

//####################################################################################
// 后续清除扫尾用回调函数
// clean up memory allocated on the GPU
//####################################################################################
void cleanup(DataBlock* d) {
    HANDLE_ERROR(cudaFree(d->dev_bitmap));
    HANDLE_ERROR(cudaFree(shifts));
    HANDLE_ERROR(cudaFree(states));
    HANDLE_ERROR(cudaFree(angles));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    // global memory执行代码
    //--------------------------------------------------------
    // 释放存储在global的小球数据
    //HANDLE_ERROR(cudaFree(globalSpheres));
    //--------------------------------------------------------
}

//####################################################################################
// 在constant内存上分配小球数据
//####################################################################################
void allocateSpheresOnConstant() {
    // allocate temp memory, initialize it, copy to constant
    // memory on the GPU, then free our temp memory
    Sphere* temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
    for (int i = 0; i < SPHERES; i++) {
        temp_s[i].r = rnd(1.0f);
        temp_s[i].g = rnd(1.0f);
        temp_s[i].b = rnd(1.0f);
        temp_s[i].x = rnd(1000.0f) - 500;         // x,y,z 在初始化时已经中心化过了
        temp_s[i].y = rnd(1000.0f) - 500;
        temp_s[i].z = rnd(1000.0f) - 500;
        temp_s[i].radius = rnd(20.0f) + 8;
        temp_s[i].idx = i;
    }
    // 因为是拷贝到constant内存中，使用ToSymbol版本
    HANDLE_ERROR(cudaMemcpyToSymbol(s, temp_s,
        sizeof(Sphere) * SPHERES));

    // 立刻释放host端的内存
    free(temp_s);
}

void allocateSpheresOnGlobal() {
    // allocate temp memory, initialize it, copy to constant
// memory on the GPU, then free our temp memory
    Sphere* temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
    for (int i = 0; i < SPHERES; i++) {
        temp_s[i].r = rnd(1.0f);
        temp_s[i].g = rnd(1.0f);
        temp_s[i].b = rnd(1.0f);
        temp_s[i].x = rnd(1000.0f) - 500;         // x,y,z 在初始化时已经中心化过了
        temp_s[i].y = rnd(1000.0f) - 500;
        temp_s[i].z = rnd(1000.0f) - 500;
        temp_s[i].radius = rnd(20.0f) + 8;
        temp_s[i].idx = i;
    }
    // 因为是拷贝到constant内存中，使用ToSymbol版本
    HANDLE_ERROR(cudaMemcpy(globalSpheres, temp_s,
        sizeof(Sphere) * SPHERES,
        cudaMemcpyHostToDevice));

    // 立刻释放host端的内存
    free(temp_s);
}

int main( void ) {
    // 数据初始化
    DataBlock   data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;

    // capture the start time
    //cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    //HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    // 展示画板
    //CPUBitmap bitmap( DIM, DIM, &data );
    //unsigned char   *dev_bitmap;

    // allocate memory on the GPU for the output bitmap
    // GPU展示画板内存分配
    HANDLE_ERROR( cudaMalloc( (void**)&data.dev_bitmap,
                              bitmap.image_size() ) );
    // 分配随机数状态内存
    HANDLE_ERROR(cudaMalloc((void**)&states,
        sizeof(curandStateXORWOW) * SPHERES));
    // 分配每一个小球的x,y偏移量数组
    HANDLE_ERROR(cudaMalloc((void**)&shifts,
        sizeof(int) * SPHERES * SPHERE_SHIFT_DATA_WIDTH));
    // 分配每一个小球的角度方向值
    HANDLE_ERROR(cudaMalloc((void**)&angles,
        sizeof(double) * SPHERES));

    // global memory执行代码
    //--------------------------------------------------------
    // 分配每一个小球的角度方向值
    //HANDLE_ERROR(cudaMalloc((void**)&globalSpheres,
    //    sizeof(Sphere) * SPHERES));
    //allocateSpheresOnGlobal();
    //--------------------------------------------------------


    // 将小球的数据分配到constant内存上
    allocateSpheresOnConstant();

    // 初始化小球的状态
    initSpheres <<<SPHERE_BLOCK, 1>>> (states, shifts, angles);

   // 开始动画
    bitmap.anim_and_exit((void (*)(void*))generate_frame,
                         (void (*)(void*))cleanup);
}

