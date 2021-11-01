/*
* This code mainly references an Nvidia blog:
* https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
*/


#ifndef COLLISION_H
#define COLLISION_H

#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>

#include "bvh.cuh"
#include "vec3f.cuh"
#include "box.cuh"
#include "tri_contact.cuh"

__device__ void findCollisionIterative(Node* root, Triangle* qt, Box* qb, vec3f* vs, unsigned int* count, unsigned int* colTris)
{
	Node* stack[32];
	unsigned int sptr = 0;		// stack pointer, always point to top of the stack (one position ahead)
	stack[sptr++] = NULL;
	Node* node = root;

	while (node != NULL) {
		Node* childL = node->childA;
		Node* childR = node->childB;

		// Check box overlap as indicator of triangle contact
		int overlapL = checkBoxOverlap(qb, &childL->box); 
		int overlapR = checkBoxOverlap(qb, &childR->box);

		// Check leaf side child if overlapped
		if (overlapL > 0) {
			if (childL->isLeaf) {
				childL->visited = 1;
				if (qt->neighborCount(childL->triangle) < 1) {
					if (checkTriangleContactHelper(qt, childL->triangle, vs) > 0) {
						int curIdx = atomicAdd(count, 1);
						colTris[2 * curIdx] = qt->ID;
						colTris[2 * curIdx + 1] = childL->triangle->ID;
					}
				}
			}
			else {
				stack[sptr++] = childL;
			}
		}

		// Check right side child if overlapped
		if (overlapR > 0) {
			if (childR->isLeaf) {
				childR->visited = 1;
				if (qt->neighborCount(childR->triangle) < 1) {
					if (checkTriangleContactHelper(qt, childR->triangle, vs) > 0) {
						int curIdx = atomicAdd(count, 1);
						colTris[2 * curIdx] = qt->ID;
						colTris[2 * curIdx + 1] = childR->triangle->ID;
					}
				}
			}
			else {
				stack[sptr++] = childR;
			}
		}

		// Pop out whatever
		node = stack[--sptr];
	}
}

__global__ void findCollisions(Node* root, Node* leaves, vec3f* vs, unsigned int numObjects, unsigned int* count, unsigned int* colTris)
{
	// NOTE: Here assumes kernel launches by 2D <BLOCK_GRID,THREAD_GRID>
	// NOTE: 此处假设kernel以<BLOCK_GRID,THREAD_GRID>的2D方式启动
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int t_idx = x + y * blockDim.x * gridDim.x;
	unsigned int total_thread = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
	unsigned int thread_per_block = blockDim.x * blockDim.y;
	unsigned int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int i = t_idx;

	for (int i = t_idx; i < numObjects; i += total_thread) {
		findCollisionIterative(root, leaves[i].triangle, &leaves[i].box, vs, count, colTris);
	}
}

// NOT USED, FOR STACK LIMITATION
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
	bool overlapL = true;
	bool overlapR = true;

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

#endif // !COLLISION_H