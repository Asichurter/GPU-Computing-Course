/*
    动态场景光线追踪

    小球的数据结构，更新、计算相关和控制小球随机移动的宏的头文件
*/

#include <cuda_runtime.h>
#include "./common/cpu_anim.h"

#define INF     2e10f
#define SPHERE_SHIFT_DATA_WIDTH 4               // int类型的shift数组中，一个小球对应的数据长度。[0]是x偏移量，[1]是y偏移量，[2]是切向速度，[3]是角速度方向
#define ENABLE_SPHERE_SHAKE true                // 是否启用小球随机移动
#define SPHERE_SHAKE_TYPE 1                     // 小球随机移动的类型，0为x,y水平移动，1为旋转移动
#define SPHERE_FRAME_PER_SHAKE 4                // 小球随机移动频率，单位为“多少帧随机移动一次”
#define SPHERE_MAX_SPEED 18                     // 小球最大切向移动速度
#define SPHERE_UPDATE_CURVE_PROB 1              // 小球选择更新自己的曲线移动方向和速度的概率值
#define SPHERE_SHAKE_WIDTH 35                   // 小球随机移动幅度

#define PI 3.1415926535898                      // π
#define ANGLE_VEL PI/12                         // 小球移动角速度

#define SPHERES 500             // 小球数量
#define SPHERE_BLOCK 128        // 启动小球相关的global函数时（初始化和更新），启用的block数量（thread默认为1）

#define dev_rnd(x,s)(curand(&s) % 1000000 * 1.0 / 1000000 * x)      // device上的随机数生成，返回的是double类型


struct Sphere {
    float   r, b, g;
    float   radius;
    float   x, y, z;
    int idx;            // 小球下标，方便索引其偏移量状态

    __device__ float hit(float ox, float oy, float* n, int* shifts) {
        int x_shift = shifts[SPHERE_SHIFT_DATA_WIDTH * idx], y_shift = shifts[SPHERE_SHIFT_DATA_WIDTH * idx + 1];
        float dx = ox - (x + x_shift);
        float dy = oy - (y + y_shift);
        if (dx * dx + dy * dy < radius * radius) {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / sqrtf(radius * radius);
            return dz + z;
        }
        return -INF;
    }
};

//####################################################################################
// 初始化小球随机偏移量的随机数状态
//####################################################################################
__global__ void initSpheres(curandStateXORWOW* states, int* shifts, double* angles) {
    int i = blockIdx.x;
    while (i < SPHERES) {
        curand_init(i, 0, 0, &states[i]);
        shifts[SPHERE_SHIFT_DATA_WIDTH * i] = shifts[SPHERE_SHIFT_DATA_WIDTH * i + 1] = 0;
        shifts[SPHERE_SHIFT_DATA_WIDTH * i + 2] = (i % 5 + 1) * 5;                // 初始化切线方向移动速度
        shifts[SPHERE_SHIFT_DATA_WIDTH * i + 3] = (i % 2) * 2 - 1;  // 初始化角速度方向（+/-）
        angles[i] = 0.0;                                            // 初始化角度

        i += SPHERE_BLOCK;
    }
}

//####################################################################################
// 根据小球当前的偏移量随机数状态，更新小球的随机偏移量
//####################################################################################
__global__ void updateSphereShiftsWithAxisMove(curandStateXORWOW* states, int* shifts) {
    int i = blockIdx.x;
    while (i < SPHERES) {
        int x_shift = dev_rnd(SPHERE_SHAKE_WIDTH, states[i]); //curand(&states[i]) % 1000000 * 1.0 / 1000000 * SPHERE_SHAKE_WIDTH;
        int y_shift = dev_rnd(SPHERE_SHAKE_WIDTH, states[i]); //curand(&states[i]) % 1000000 * 1.0 / 1000000 * SPHERE_SHAKE_WIDTH;
        shifts[SPHERE_SHIFT_DATA_WIDTH * i] = x_shift;
        shifts[SPHERE_SHIFT_DATA_WIDTH * i + 1] = y_shift;

        i += SPHERE_BLOCK;
    }
}

//####################################################################################
// 根据小球当前的速度角度和速度值，更新x,y偏移量，并且根据角速度更新角度值
//####################################################################################
__global__ void updateSphereShiftsWithCurveMove(curandStateXORWOW* states, int* shifts, double* angles) {
    int i = blockIdx.x;
    while (i < SPHERES) {
        int speed = shifts[SPHERE_SHIFT_DATA_WIDTH * i + 2];
        int x_shift = speed * cosf(angles[i]);                  //dev_rnd(SPHERE_SHAKE_WIDTH, states[i]); //curand(&states[i]) % 1000000 * 1.0 / 1000000 * SPHERE_SHAKE_WIDTH;
        int y_shift = speed * sinf(angles[i]);                  //dev_rnd(SPHERE_SHAKE_WIDTH, states[i]); //curand(&states[i]) % 1000000 * 1.0 / 1000000 * SPHERE_SHAKE_WIDTH;
        shifts[SPHERE_SHIFT_DATA_WIDTH * i] += x_shift;
        shifts[SPHERE_SHIFT_DATA_WIDTH * i + 1] += y_shift;

        angles[i] = fmod(angles[i] + ANGLE_VEL * shifts[SPHERE_SHIFT_DATA_WIDTH * i + 3], 2 * PI);   // 根据角速度和角速度方向，更新角度

        i += SPHERE_BLOCK;
    }
}

//####################################################################################
// 随机更新小球的角速度方向，象限和速度值
//####################################################################################
__global__ void updateSphereCurveSpeedAngle(curandStateXORWOW* states, int* shifts, double* angles) {
    int i = blockIdx.x;
    while (i < SPHERES) {
        int updateProbVal = dev_rnd(10, states[i]);
        if (updateProbVal >= SPHERE_UPDATE_CURVE_PROB) {
            i += SPHERE_BLOCK;
            continue;
        }

        shifts[i * SPHERE_SHIFT_DATA_WIDTH + 2] = dev_rnd(SPHERE_MAX_SPEED, states[i]);         // 更新速度
        shifts[i * SPHERE_SHIFT_DATA_WIDTH + 3] = ((int)dev_rnd(2, states[i])) * 2 - 1;         // 更新角速度方向(+/-)
        angles[i] = fmod(angles[i] + ((int)dev_rnd(2, states[i])) * PI, 2 * PI);                // 更新角度的象限，+π可以使得曲线交界处平滑

        i += SPHERE_BLOCK;
    }
}
