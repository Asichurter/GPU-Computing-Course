#ifndef COLLISION_H
#define COLLISION_H

#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>

#include "bvh.cuh"
#include "vec3f.cuh"
#include "box.cuh"
#include "tri_contact.cuh"

__device__ void pushPairCollision(unsigned int* colList, unsigned int v1, unsigned v2) 
{
    // 将冲突列表中的第一个位置视为当前元素的数量计数
    // 一次性预留出2个元素的空间
    //unsigned int idx = atomicAdd(&colList[0], 2);     
    //colList[idx + 1] = v1;
    //colList[idx + 2] = v2;
}

__device__ void traverseRecursive(
    Node* node,
    Box* queryBox,
    Triangle* queryTriangle,
    vec3f* vertexes,
    unsigned int& colcount,
    unsigned int* collist)
{
    Node* childL = node->childA;
    Node* childR = node->childB;
    bool overlapL = true; // checkBoxOverlap(queryBox, &(childL->box));
    bool overlapR = true; // checkBoxOverlap(queryBox, &(childR->box));

    // Query overlaps a leaf node => report collision.
    // 如果发现目标节点是叶节点，而且box发生了碰撞，则在三角形层次上进行碰撞检测
    if (overlapL && childL->isLeaf) {
        Triangle* leafTriangle = childL->triangle;
        colcount++;
        if (checkTriangleContactHelper(queryTriangle, leafTriangle, vertexes)) {
            //colcount++;
            //if (layer == 20) atomicAdd(count, 1);
            //pushPairCollision(colList, queryTriangle->ID, leafTriangle->ID);
            
        }
    }
    // 如果发现目标节点是叶节点，而且box发生了碰撞，则在三角形层次上进行碰撞检测
    if (overlapR && childR->isLeaf) {
        Triangle* leaf_triangle = childR->triangle;
        colcount++;
        if (checkTriangleContactHelper(queryTriangle, leaf_triangle, vertexes)) {
            //colcount++;
            //if (layer == 20) atomicAdd(count, 1);
            //pushPairCollision(colList, queryTriangle->ID, leaf_triangle->ID);
        }
    }

    // Query overlaps an internal node => traverse.
    // 检查左右子节点是否需要继续递归下去检查
    // 仅限于检查节点有碰撞而且非叶节点的情况
    bool traverseL = (overlapL && !childL->isLeaf);
    bool traverseR = (overlapR && !childR->isLeaf);

    if (traverseL) traverseRecursive(childL, queryBox, queryTriangle, vertexes, colcount, collist);
    if (traverseR) traverseRecursive(childR, queryBox, queryTriangle, vertexes, colcount, collist);
}

__device__ void findCollisionIterative(Node* root, Triangle* qt, Box* qb, vec3f* vs, unsigned int* count, unsigned int* colTris)
{
	Node* stack[32];
	unsigned int sptr = 0;
	stack[sptr++] = NULL;
	Node* node = root;
	unsigned int counter = 0;

	while (node != NULL) {
		Node* childL = node->childA;
		Node* childR = node->childB;

		int overlapL = checkBoxOverlap(qb, &childL->box); // qb->overlap(&childL->box); 
		int overlapR = checkBoxOverlap(qb, &childR->box); // qb->overlap(&childL->box);

		//atomicAdd(count, 1);

		if (overlapL > 0) {
			if (childL->isLeaf) {
				//atomicAdd(&colTris[0], 1);
				childL->visited = 1;
				if (qt->neighborCount(childL->triangle) < 1) {
					if (checkTriangleContactHelper(qt, childL->triangle, vs) > 0) {
						int curIdx = atomicAdd(count, 1);
						colTris[2 * curIdx] = qt->ID;
						colTris[2 * curIdx + 1] = childL->triangle->ID;
					}
				}
				//if (checkTriangleContactHelper(qt, childL->triangle, vs) > 0) {
				//	//unsigned int oldIdx = atomicAdd(count, 2);
				//	//colTris[oldIdx].set(qt);
				//	//colTris[oldIdx + 1].set(childL->triangle);
				//	//pushPairCollision(colList, queryTriangle->ID, leaf_triangle->ID);
				//}
				//atomicAdd(count, 1);
			}
			else {
				stack[sptr++] = childL;
			}
		}

		if (overlapR > 0) {
			if (childR->isLeaf) {
				//atomicAdd(&colTris[0], 1);
				childR->visited = 1;
				if (qt->neighborCount(childR->triangle) < 1) {
					if (checkTriangleContactHelper(qt, childR->triangle, vs) > 0) {
						int curIdx = atomicAdd(count, 1);
						colTris[2 * curIdx] = qt->ID;
						colTris[2 * curIdx + 1] = childR->triangle->ID;
					}
				}
				//if (qt->neighborCount(childR->triangle) >= 2) atomicAdd(count, 1);
				//if (checkTriangleContactHelper(qt, childR->triangle, vs) > 0) {
				//	//unsigned int oldIdx = atomicAdd(count, 2);
				//	//colTris[oldIdx].set(qt);
				//	//colTris[oldIdx + 1].set(childL->triangle);
				//	//pushPairCollision(colList, queryTriangle->ID, leaf_triangle->ID);
				//}
			}
			else {
				stack[sptr++] = childR;
			}
		}

		//if (overlapL > 0 && overlapR > 0) atomicAdd(count, 1);

		node = stack[--sptr];
	}
}

__global__ void findCollisions(Node* root, Node* leaves, vec3f* vs, unsigned int numObjects, unsigned int* count, unsigned int* colTris)
{
	// NOTE: 此处假设kernel以<BLOCK_GRID,THREAD_GRID>的2D方式启动
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int t_idx = x + y * blockDim.x * gridDim.x;
	unsigned int total_thread = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
	unsigned int thread_per_block = blockDim.x * blockDim.y;
	//unsigned int t_idx_of_block = thread_idx
	unsigned int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int i = t_idx;

	//__shared__ Node* queryNodes[];

	// TODO: 1. 将root或者部分内部节点写入constant加速 2. 将query的叶节点读入shared中加速

	// NOTE: 此处假设kernel以<BLOCK,THREAD>的1D方式启动
	//const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // 当前线程在所有线程下的坐标
	//const int total_thread = blockDim.x * gridDim.x;                // 一个grid内的所有线程数量，作为递增量
	//unsigned int i = t_idx;

	for (int i = t_idx; i < numObjects; i += total_thread) {
		//atomicAdd(count, 1);
		//if (i == 0)
		findCollisionIterative(root, leaves[i].triangle, &leaves[i].box, vs, count, colTris);
	}
}

#endif // !COLLISION_H