#ifndef CHECK_H
#define CHECK_H

#include <cuda_runtime.h>

#include "bvh.cuh"
#include "tri_contact.cuh"
#include "utils.cuh"

__global__ void checkMortons(unsigned int* mortons, unsigned int num, unsigned int* flag)
{
	for (int i = 0; i < num - 1; i++) {
		if (mortons[i] > mortons[i + 1]) {
			atomicAdd(flag, 1);
		}
	}
}

__global__ void testFunc(int* res)
{
    unsigned long long int nums[] = { 1,2,4,5,19, 24, 25, 30 };
    Range range = determineRange(nums, 8, 6);
    int split = findSplit(nums, range.x, range.y);
    res[0] = range.x;
    res[1] = range.y;
    res[2] = split;
}

__global__ void checkTriangleIdx(Node* ns, vec3f* vs, unsigned int num, unsigned int maxv, unsigned int* count)
{
	// NOTE: Here assumes kernel launches by 1D <BLCOK,THREAD>
	// NOTE: 此处假设kernel以<BLOCK,THREAD>的1D方式启动
	const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // 当前线程在所有线程下的坐标
	const int total_thread = blockDim.x * gridDim.x;                // 一个grid内的所有线程数量，作为递增量

	for (int i = t_idx; i < num; i+=total_thread) {
		if (ns[i].triangle == NULL) {
			atomicAdd(count, 1);
			continue;
		}
		for (int j = 0; j < 3; j++) {
			unsigned int vidx = ns[i].triangle->vIdx[j];
			if (vidx >= maxv) atomicAdd(count, 1);
			else {
				vec3f vec = vs[vidx];
				double p = vec.x;
			}
		}
	}
}

__global__ void checkTriContact(Triangle* ts, vec3f* vs, unsigned int* count)
{
	// NOTE: Here assumes kernel launches by 1D <BLCOK,THREAD>
	// NOTE: 此处假设kernel以<BLOCK,THREAD>的1D方式启动
	const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // 当前线程在所有线程下的坐标
	const int total_thread = blockDim.x * gridDim.x;                // 一个grid内的所有线程数量，作为递增量

	for (int i = t_idx; i < 100000; i+=total_thread) {
		if (checkTriangleContactHelper(ts + i, ts + i + i, vs)) atomicAdd(count, 1);
	}
}

__global__ void checkInternalNodes(Node* nodes, unsigned int numObjects, unsigned int* nullParentNum, unsigned int* wrongBoundNum, unsigned int* nullChildNum, unsigned int* notInternalCount, unsigned int* uninitBoxCount)
{
	// NOTE: Here assumes kernel launches by 1D <BLCOK,THREAD>
	// NOTE: 此处假设kernel以<BLOCK,THREAD>的1D方式启动
	const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // 当前线程在所有线程下的坐标
	const int total_thread = blockDim.x * gridDim.x;                // 一个grid内的所有线程数量，作为递增量

	for (int i = t_idx; i < numObjects; i += total_thread) {
		//if (nodes[i].isLeaf()) atomicAdd(notInternalCount, 1);
		if (nodes[i].bounded != 2) atomicAdd(wrongBoundNum, 1);
		if (nodes[i].parent == NULL) atomicAdd(nullParentNum, 1);
		if (nodes[i].childA == NULL) atomicAdd(nullChildNum, 1);
		if (nodes[i].childB == NULL) atomicAdd(nullChildNum, 1);
		if (nodes[i].box.selfCheck() == 0) atomicAdd(uninitBoxCount, 1);
	}
}

__global__ void checkLeafNodes(Node* nodes, unsigned int numObjects, unsigned int* nullParentNum, unsigned int* nullTriangleNum, unsigned int* notLeafCount, unsigned int* illegalboxCount)
{
	// NOTE: Here assumes kernel launches by 1D <BLCOK,THREAD>
	// NOTE: 此处假设kernel以<BLOCK,THREAD>的1D方式启动
	const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // 当前线程在所有线程下的坐标
	const int total_thread = blockDim.x * gridDim.x;                // 一个grid内的所有线程数量，作为递增量

	for (int i = t_idx; i < numObjects; i += total_thread) {
		//Node* lnode = &nodes[i];
		//if (!lnode->isLeaf()) atomicAdd(notLeafCount, 1);
		if (nodes[i].parent == NULL) atomicAdd(nullParentNum, 1);
		if (nodes[i].triangle == NULL || nodes[i].triangle->selfCheck() > 0) atomicAdd(nullTriangleNum, 1);
		if (nodes[i].box.selfCheck() == 0)
			atomicAdd(illegalboxCount, 1);
	}
}

__global__ void checkBoxOverlapFunc(Node* root, Node* leaf, unsigned int* overlap)
{
	if (checkBoxOverlap(&root->box, &leaf->box)) atomicAdd(overlap, 1);
}

__global__ void checkLeafVisitedFunc(Node* leaves, unsigned int num, unsigned int* count)
{
	for (int i = 0; i < num; i++) {
		if (leaves[i].visited == 0) atomicAdd(count, 1);
	}
}

__device__ void neighborCountCheck(Triangle* queryTriangle, Node* leaves, unsigned int* count, unsigned int num)
{
	for (int i = 0; i < num; i++) {
		atomicAdd(count, queryTriangle->neighborCount(leaves[i].triangle));
	}
}

__global__ void checkDirectComp(Node* leaves, vec3f* vertexes, unsigned int numObjects, unsigned int* count)
{
	// NOTE: Here assumes kernel launches by 1D <BLCOK,THREAD>
	// NOTE: 此处假设kernel以<BLOCK,THREAD>的1D方式启动
	//const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // 当前线程在所有线程下的坐标
	//const int total_thread = blockDim.x * gridDim.x;                // 一个grid内的所有线程数量，作为递增量

	// NOTE: Here assumes kernel launches by 2D <BLOCK_GRID,THREAD_GRID>
	// NOTE: 此处假设kernel以<BLOCK_GRID,THREAD_GRID>的2D方式启动
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int t_idx = x + y * blockDim.x * gridDim.x;
	unsigned int total_thread = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
	unsigned int i = t_idx;

	for (int i = t_idx; i < numObjects; i += total_thread) {
		for (int j = 0; j < numObjects; j++) {
			//atomicAdd(count, 1);
			if (leaves[i].triangle->neighborCount(leaves[j].triangle) < 1) {
				//checkTriangleContactHelper(leaves[i].triangle, leaves[j].triangle, vertexes);
				atomicAdd(count, checkTriangleContactHelper(leaves[i].triangle, leaves[j].triangle, vertexes));
			}
		}
	}
}


#endif // !CHECK_H
