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
	// NOTE: 此处假设kernel以<BLOCK,THREAD>的1D方式启动
	const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // 当前线程在所有线程下的坐标
	const int total_thread = blockDim.x * gridDim.x;                // 一个grid内的所有线程数量，作为递增量

	for (int i = t_idx; i < 100000; i+=total_thread) {
		if (checkTriangleContactHelper(ts + i, ts + i + i, vs)) atomicAdd(count, 1);
	}
}

__global__ void checkInternalNodes(Node* nodes, unsigned int numObjects, unsigned int* nullParentNum, unsigned int* wrongBoundNum, unsigned int* nullChildNum, unsigned int* notInternalCount, unsigned int* uninitBoxCount)
{
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

__device__ void _selfCheckCollisionIterative(Node* root, Triangle* qt, Box* qb, vec3f* vs, unsigned int* count, Triangle* colTris)
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

			if (overlapL > 0) {
				if (childL->isLeaf) {
					childL->visited = 1;
					if (qt->neighborCount(childL->triangle) < 1) {
						//checkTriangleContactHelper(qt, childL->triangle, vs);
						atomicAdd(count, checkTriangleContactHelper(qt, childL->triangle, vs));
					}

					//if (checkTriangleContactHelper(qt, childL->triangle, vs) > 0) {
					//	//unsigned int oldIdx = atomicAdd(count, 2);
					//	//colTris[oldIdx].set(qt);
					//	//colTris[oldIdx + 1].set(childL->triangle);
					//	//pushPairCollision(colList, queryTriangle->ID, leaf_triangle->ID);
					//}
					//atomicAdd(count, 1);
				}
				else stack[sptr++] = childL;
			}

			if (overlapR > 0) {
				if (childR->isLeaf) {
					childR->visited = 1;
					if (qt->neighborCount(childR->triangle) < 1) {
						//checkTriangleContactHelper(qt, childR->triangle, vs);
						atomicAdd(count, checkTriangleContactHelper(qt, childR->triangle, vs));
					}

					//if (qt->neighborCount(childR->triangle) >= 2) atomicAdd(count, 1);
					//if (checkTriangleContactHelper(qt, childR->triangle, vs) > 0) {
					//	//unsigned int oldIdx = atomicAdd(count, 2);
					//	//colTris[oldIdx].set(qt);
					//	//colTris[oldIdx + 1].set(childL->triangle);
					//	//pushPairCollision(colList, queryTriangle->ID, leaf_triangle->ID);
					//}
				}
				else stack[sptr++] = childR;
			}

			//stack[sptr++] = childL;
			//stack[sptr++] = childR;

			node = stack[--sptr];
		}
}

__global__ void selfCheckCollisionIterative(Node* root, Node* leaves, vec3f* vs, unsigned int numObjects, unsigned int* count, Triangle* colTris)
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
		_selfCheckCollisionIterative(root, leaves[i].triangle, &leaves[i].box, vs, count, colTris);
	}
}

__global__ void checkBoxOverlapFunc(Node* root, Node* leaf, unsigned int* overlap)
{
	//if (leaf->box.overlap(&root->box) > 0) atomicAdd(overlap, 1);
	if (checkBoxOverlap(&root->box, &leaf->box)) atomicAdd(overlap, 1);
	//if (checkBoxOverlap(&leaf->box, &root->box) > 0) atomicAdd(overlap, 1);
}

__global__ void checkLeafVisitedFunc(Node* leaves, unsigned int num, unsigned int* count)
{
	for (int i = 0; i < num; i++) {
		if (leaves[i].visited == 0) atomicAdd(count, 1);
	}
}

//__global__ void checkChildBoxOverlapFunc(Node* root, )

__device__ void _selfCheckCollisionRecursive(Node* node, Box* queryBox, Triangle* queryTriangle, vec3f* vertexes, unsigned int* count, int layer) {
	Node* childL = node->childA;
	Node* childR = node->childB;
	int overlapL = queryBox->overlapTest(&(childL->box));
	int overlapR = queryBox->overlapTest(&(childR->box));

		//Query overlaps a leaf node => report collision.
		//如果发现目标节点是叶节点，而且box发生了碰撞，则在三角形层次上进行碰撞检测
	//if (overlapL && childL->isLeaf) {
	//	Triangle* leafTriangle = childL->triangle;
	//	if (checkTriangleContactHelper(queryTriangle, leafTriangle, vertexes)) {
	//		layer++;
	//		//atomicAdd(count, 1);
	//		//colcount++;
	//		//if (layer == 20) atomicAdd(count, 1);
	//		//pushPairCollision(colList, queryTriangle->ID, leafTriangle->ID);
	//	}
	//}

	//if (overlapR && childR->isLeaf) {
	//	Triangle* leafTriangle = childR->triangle;
	//	if (checkTriangleContactHelper(queryTriangle, leafTriangle, vertexes)) {
	//		layer++;
	//		//atomicAdd(count, 1);
	//		//colcount++;
	//		//if (layer == 20) atomicAdd(count, 1);
	//		//pushPairCollision(colList, queryTriangle->ID, leaf_triangle->ID);
	//	}
	//}

	atomicAdd(count, 1);

	if (overlapL && !childL->isLeaf) _selfCheckCollisionRecursive(childL, queryBox, queryTriangle, vertexes, count, layer + 1);
	if (overlapR && !childR->isLeaf) _selfCheckCollisionRecursive(childR, queryBox, queryTriangle, vertexes, count, layer + 1);
}

__device__ void testRecursiveCap(Node* node, Node* leaves, unsigned int queryIdx, Box* queryBox, Triangle* queryTriangle, vec3f* vertexes, unsigned int* count, int layer)
{
	//if (layer < 0 || node == NULL) {
	//	atomicAdd(count, 1);
	//	return;
	//}

	//atomicAdd(count, 1);

	//int overlapL = node->childA->box.overlap(queryBox);
	//int overlapR = node->childB->box.overlap(queryBox);
	//int overlapR = queryBox->overlapTest(&node->childB->box);

	if (node->childA->box.overlap(queryBox) > 0) {
		if (!node->childA->isLeaf) testRecursiveCap(node->childA, leaves, queryIdx, queryBox, queryTriangle, vertexes, count, layer);
		else {
			//atomicAdd(count, 1);
			//if (checkTriangleContactHelper(node->childA->triangle, queryTriangle, vertexes) > 0) atomicAdd(count, 1);
			//atomicAdd(count, queryTriangle->neighborCount(node->childA->triangle));
			layer += leaves[queryIdx].triangle->neighborCount(leaves[node->childA->idx].triangle);
			//if (queryTriangle->neighborCount(node->childA->triangle) < 1) atomicAdd(count, 1);
		}
	}
	if (node->childB->box.overlap(queryBox) > 0) {
		//atomicAdd(count, 1);
		if (!node->childB->isLeaf) testRecursiveCap(node->childB, leaves, queryIdx, queryBox, queryTriangle, vertexes, count, layer);
		else {
			//atomicAdd(count, 1);
			//if (checkTriangleContactHelper(node->childB->triangle, queryTriangle, vertexes) > 0) atomicAdd(count, 1);
		}
	}

	//if (overlapL == 2) atomicAdd(count, 1);
	//if (overlapR == 2) atomicAdd(count, 1);

	//if (!node->childA->isLeaf) testRecursiveCap(node->childA, queryBox, queryTriangle, vertexes, count, layer);
	//if (!node->childB->isLeaf) testRecursiveCap(node->childB, queryBox, queryTriangle, vertexes, count, layer);
}

__device__ void neighborCountCheck(Triangle* queryTriangle, Node* leaves, unsigned int* count, unsigned int num)
{
	for (int i = 0; i < num; i++) {
		atomicAdd(count, queryTriangle->neighborCount(leaves[i].triangle));
	}
}

__global__ void selfCheckCollisionRecursive(Node* node, Node* leaves, vec3f* vertexes, unsigned int numObjects, unsigned int* count)
{
	// NOTE: 此处假设kernel以<BLOCK,THREAD>的1D方式启动
	const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // 当前线程在所有线程下的坐标
	const int total_thread = blockDim.x * gridDim.x;                // 一个grid内的所有线程数量，作为递增量

	for (int i = t_idx; i < 1; i += total_thread) {
		//_selfCheckCollisionRecursive(node, &leaves[i].box, leaves[i].triangle, vertexes, count, 0);       // FIXME
		testRecursiveCap(node, leaves, leaves[i].idx, &leaves[i].box, leaves[i].triangle, vertexes, count, i);
		//neighborCountCheck(leaves[i].triangle, leaves, count, numObjects);
	}
}

__global__ void checkDirectComp(Node* leaves, vec3f* vertexes, unsigned int numObjects, unsigned int* count)
{
	// NOTE: 此处假设kernel以<BLOCK,THREAD>的1D方式启动
	//const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // 当前线程在所有线程下的坐标
	//const int total_thread = blockDim.x * gridDim.x;                // 一个grid内的所有线程数量，作为递增量

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
