/*
    CPU code, still some bugs.
*/

#ifndef CPU_H
#define CPU_H

#include <cuda_runtime.h>
#include <vector>

#include "triangle.cuh"
#include "vec3f.cuh"
#include "box.cuh"
#include "bvh.cuh"
#include "tri_contact.cuh"
#include "cpu_math.h"

/* 通过寻找排序morton的最大前置0数量，找到对应的分割点。
    本算法需要与bvh建立过程相耦合。
*/

__device__ __host__ int findSplitCpu(unsigned long long int* mortons,
    int           first,
    int           last)
{
    // Identical Morton codes => split the range in the middle.

    unsigned long long int firstCode = mortons[first];
    unsigned long long int lastCode = mortons[last];

    if (firstCode == lastCode)
        return (first + last) >> 1;

    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.
    // 计算所有元素的最大相同位数（从最低位开始算起），这些LSB不会成为split点的待选点，将会被剔除
    int commonPrefix = clzll(firstCode ^ lastCode, 0);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.

    int split = first; // initial guess
    int step = last - first;

    do
    {
        step = (step + 1) >> 1; // exponential decrease
        int newSplit = split + step; // proposed new position

        if (newSplit < last)    // 检查划分点是否超过边界
        {
            unsigned long long int splitCode = mortons[newSplit];
            int splitPrefix = clzll(firstCode ^ splitCode, 0);

            // 只有在[first, split]内的相同位数多于所有元素的相同位数时，才会将split的位置向右移动
            if (splitPrefix > commonPrefix)
                split = newSplit; // accept proposal
        }
    } while (step > 1);

    return split;
}

__device__ __host__ Range determineRangeCpu(unsigned long long int* keys,
    int numObjects,
    int i)
{
    int d = (deltaCpu(i, i + 1, keys, numObjects) - deltaCpu(i, i - 1, keys, numObjects)) >= 0 ? 1 : -1;
    int delta_min = deltaCpu(i, i - d, keys, numObjects);
    int mlen = 2;
    //return Range(100, mlen);
    while (deltaCpu(i, i + mlen * d, keys, numObjects) > delta_min) {
        mlen <<= 1;
    }

    int l = 0;
    for (int t = mlen >> 1; t >= 1; t >>= 1) {
        if (deltaCpu(i, i + (l + t) * d, keys, numObjects) > delta_min) {
            l += t;
        }
    }
    int j = i + l * d;

    // 返回的范围中，前者一定是较小者，后者一定是较大者
    return Range(min(i, j), max(i, j));
}

__host__ void fillLeafNodesCpu(
    Triangle* sortedTriangles,       // 三角形排序后下标
    int numObjects,
    Node* leafNodes)
{
    // NOTE: 此处假设kernel以<BLOCK,THREAD>的1D方式启动
    //const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // 当前线程在所有线程下的坐标
    //const int total_thread = blockDim.x * gridDim.x;                // 一个grid内的所有线程数量，作为递增量

    // Construct leaf nodes.
    // Note: This step can be avoided by storing
    // the tree in a slightly different way.
    // TODO: 可并行化(√)
    // 对每一个object使用一个单独的数组来存储
    for (int idx = 0; idx < numObjects; idx++) { // in parallel
        leafNodes[idx].triangle = &sortedTriangles[idx];         // 填充叶节点时，叶节点的值本身就是GPU上的三角形的指针，因此拷贝到Host端也不可读
        leafNodes[idx].isLeaf = true;
        leafNodes[idx].idx = idx;
    }
}

__host__ void generateHierarchyParallelCpu(
    unsigned long long int* sortedMortonCodes, // 三角形排序后morton值
    int numObjects,
    Node* leafNodes,
    Node* internalNodes,
    unsigned int* parentWrongNum)
{

    // NOTE: 此处假设kernel以<BLOCK,THREAD>的1D方式启动
    //const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // 当前线程在所有线程下的坐标
    //const int total_thread = blockDim.x * gridDim.x;                // 一个grid内的所有线程数量，作为递增量

    // Construct internal nodes.
    // TODO: 可并行化(√)
    // 生成内部节点
    for (int idx = 0; idx < numObjects - 1; idx++) // in parallel
    {
        // Find out which range of objects the node corresponds to.
        // (This is where the magic happens!)

        Range range = determineRangeCpu(sortedMortonCodes, numObjects, idx);
        int first = range.x;
        int last = range.y;

        // Determine where to split the range
        int split = findSplitCpu(sortedMortonCodes, first, last);
        //int split = findSplitBinary(sortedMortonCodes, first, last);

        // Select childA.
        Node* childA;
        if (split == first)
            childA = &leafNodes[split];
        else
            childA = &internalNodes[split];

        // Select childB.
        Node* childB;
        if (split + 1 == last)
            childB = &leafNodes[split + 1];
        else
            childB = &internalNodes[split + 1];

        // Record parent-child relationships.
        internalNodes[idx].childA = childA;
        internalNodes[idx].childB = childB;

        if (childA->parent != NULL) parentWrongNum[0]++;
        childA->parent = &internalNodes[idx];    //  TODO: 某些child节点没有被访问到，导致自己的parent没有被赋值
        if (childB->parent != NULL) parentWrongNum[0]++;
        childB->parent = &internalNodes[idx];
    }

    // Node 0 is the root.

    //return &internalNodes[0];
}

__host__ void calBoundingBoxCpu(Node* leaves, vec3f* vertexes, unsigned int numOfLeaves) {
    //const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // 当前线程在所有线程下的坐标
    //const int total_thread = blockDim.x * gridDim.x;                // 一个grid内的所有线程数量，作为递增量

    for (int i = 0; i < numOfLeaves; i++) {
        // 首先为叶节点计算box
        leaves[i].box.set(leaves[i].triangle, vertexes);
        leaves[i].childCount = 1;
        Node* curNode = leaves[i].parent;

        // 从底向上计算box，直到根节点
        while (curNode != NULL) {
            if (curNode->bounded++ == 0) {       // 利用原子操作刷新flag
                break;                                      // 如果是第一个到达该节点的路径，则立刻退出
            }
            // 二叉树，只允许第二个来到节点的路径通过
            else {
                //Node* inode = (InternalNode*)curNode;
                Box newbox;
                newbox.merge(&(curNode->childA->box), &(curNode->childB->box));
                curNode->box = newbox;
                curNode->childCount = 1 + curNode->childA->childCount + curNode->childB->childCount;
                //curNode->depth = max(curNode->childA->depth, curNode->childB->depth) + 1;
                curNode = curNode->parent;
            }
        }
    }
}

__host__ void findCollisionIterativeCpu(Node* root, Triangle* qt, Box* qb, vec3f* vs, unsigned int* count, unsigned int* colTris)
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
                    if (checkTriangleContactHelper(qt, childL->triangle, vs) > 0) {
                        int curIdx = (*count)++;
                        colTris[2 * curIdx] = qt->ID;
                        colTris[2 * curIdx + 1] = childL->triangle->ID;
                    }
                }
            }
            else {
                stack[sptr++] = childL;
            }
        }

        if (overlapR > 0) {
            if (childR->isLeaf) {
                childR->visited = 1;
                if (qt->neighborCount(childR->triangle) < 1) {
                    if (checkTriangleContactHelper(qt, childR->triangle, vs) > 0) {
                        int curIdx = (*count)++;
                        colTris[2 * curIdx] = qt->ID;
                        colTris[2 * curIdx + 1] = childR->triangle->ID;
                    }
                }
            }
            else {
                stack[sptr++] = childR;
            }
        }

        node = stack[--sptr];
    }
}

__host__ void findCollisionsCpu(Node* root, Node* leaves, vec3f* vs, unsigned int numObjects, unsigned int* count, unsigned int* colTris)
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

    for (int i = 0; i < numObjects; i++) {
        findCollisionIterativeCpu(root, leaves[i].triangle, &leaves[i].box, vs, count, colTris);
    }
}

__host__ void checkInternalNodesCpu(Node* nodes, unsigned int numObjects, unsigned int* nullParentNum, unsigned int* wrongBoundNum, unsigned int* nullChildNum, unsigned int* notInternalCount, unsigned int* uninitBoxCount)
{
    //// NOTE: 此处假设kernel以<BLOCK,THREAD>的1D方式启动
    //const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // 当前线程在所有线程下的坐标
    //const int total_thread = blockDim.x * gridDim.x;                // 一个grid内的所有线程数量，作为递增量

    for (int i = 0; i < numObjects; i++) {
        //if (nodes[i].isLeaf()) atomicAdd(notInternalCount, 1);
        if (nodes[i].bounded != 2) wrongBoundNum++;
        if (nodes[i].parent == NULL) nullParentNum++;
        if (nodes[i].childA == NULL) nullChildNum++;
        if (nodes[i].childB == NULL) nullChildNum++;
        if (nodes[i].box.selfCheck() == 0) uninitBoxCount++;
    }
}

__host__ void checkLeafNodesCpu(Node* nodes, unsigned int numObjects, unsigned int* nullParentNum, unsigned int* nullTriangleNum, unsigned int* notLeafCount, unsigned int* illegalboxCount)
{
    //// NOTE: 此处假设kernel以<BLOCK,THREAD>的1D方式启动
    //const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // 当前线程在所有线程下的坐标
    //const int total_thread = blockDim.x * gridDim.x;                // 一个grid内的所有线程数量，作为递增量

    for (int i = 0; i < numObjects; i++) {
        if (nodes[i].parent == NULL) nullParentNum++;
        if (nodes[i].triangle == NULL || nodes[i].triangle->selfCheck() > 0) nullTriangleNum++;
        if (nodes[i].box.selfCheck() == 0)
            illegalboxCount++;;
    }
}

__host__ void checkTriangleIdxCpu(Node* ns, vec3f* vs, unsigned int num, unsigned int maxv, unsigned int* count)
{
    //// NOTE: 此处假设kernel以<BLOCK,THREAD>的1D方式启动
    //const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // 当前线程在所有线程下的坐标
    //const int total_thread = blockDim.x * gridDim.x;                // 一个grid内的所有线程数量，作为递增量

    for (int i = 0; i < num; i++) {
        if (ns[i].triangle == NULL) {
            count++;
            continue;
        }
        for (int j = 0; j < 3; j++) {
            unsigned int vidx = ns[i].triangle->vIdx[j];
            if (vidx >= maxv) count++;
            else {
                vec3f vec = vs[vidx];
                double p = vec.x;
            }
        }
    }
}

#endif // CPU_H