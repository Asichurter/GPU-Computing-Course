/*
* This code mainly references an Nvidia blog: 
* https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
*/

#ifndef BVH_H
#define BVH_H

#include <cuda_runtime.h>

#include "triangle.cuh"
#include "vec3f.cuh"
#include "box.cuh"

typedef struct range{
	int x, y;
	__device__ __host__ range() {}
    __host__ range(int a) { x = y = a; }    // for test
	__device__ __host__ range(int a, int b) {
		x = a;
		y = b;
	}
}Range;

class Node {
public:
    unsigned int idx;               // index of node
    unsigned int bounded = 0;       // calculated bounding box count of child nodes
    unsigned int childCount;        // node count under subtree
    unsigned int visited;           // if visited while traversing (for debug)

    bool isLeaf;                    // if leaf node
	Node *parent;                   // parent node
    Node *childA, *childB;          // left/right child node, only internal nodes have
    Box box;                        // AABB bounding box
    Triangle *triangle;             // corresponding triangle, only leaf nodes have

    __host__ __device__ Node() { 
        parent = NULL; 
        childA = childB = NULL;
        triangle = NULL;
    }
};	


#include <vector>

#define delta(i,j,keys,n) ((j >= 0 && j < n) ? __clzll(keys[i] ^ keys[j]) : -1) 

/* 
    Finding split point of the range by inspecting maximum leading zeros.
    This algorithm should be coupled with BVH generation.

    通过寻找排序morton的最大前置0数量，找到对应的分割点。
    本算法需要与bvh建立过程相耦合。
*/
__device__ int findSplit(unsigned long long int* mortons,
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
    int commonPrefix = __clzll(firstCode ^ lastCode);

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
            int splitPrefix = __clzll(firstCode ^ splitCode);

            // 只有在[first, split]内的相同位数多于所有元素的相同位数时，才会将split的位置向右移动
            if (splitPrefix > commonPrefix)
                split = newSplit; // accept proposal
        }
    } while (step > 1);

    return split;
}

__device__ Range determineRange(unsigned long long int* keys,
    int numObjects,
    int i)
{
    int d = (delta(i, i + 1, keys, numObjects) - delta(i, i - 1, keys, numObjects)) >= 0 ? 1 : -1;
    int delta_min = delta(i, i - d, keys, numObjects);
    int mlen = 2;
    //return Range(100, mlen);
    while (delta(i, i + mlen * d, keys, numObjects) > delta_min) {
        mlen <<= 1;
    }

    int l = 0;
    for (int t = mlen >> 1; t >= 1; t >>= 1) {
        if (delta(i, i + (l + t) * d, keys, numObjects) > delta_min) {
            l += t;
        }
    }
    int j = i + l * d;

    // 返回的范围中，前者一定是较小者，后者一定是较大者
    return Range(min(i, j), max(i, j));
    //return Range(i, j);
}

__global__ void fillLeafNodes(
    Triangle* sortedTriangles,       // 三角形排序后下标
    int numObjects,
    Node* leafNodes) 
{
    // NOTE: 此处假设kernel以<BLOCK,THREAD>的1D方式启动
    const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // 当前线程在所有线程下的坐标
    const int total_thread = blockDim.x * gridDim.x;                // 一个grid内的所有线程数量，作为递增量

    // Construct leaf nodes.
    // Note: This step can be avoided by storing
    // the tree in a slightly different way.
    // TODO: 可并行化(√)
    // 对每一个object使用一个单独的数组来存储
    for (int idx = t_idx; idx < numObjects; idx += total_thread) { // in parallel
        leafNodes[idx].triangle = &sortedTriangles[idx];         // 填充叶节点时，叶节点的值本身就是GPU上的三角形的指针，因此拷贝到Host端也不可读
        leafNodes[idx].isLeaf = true;
        leafNodes[idx].idx = idx;
    }
}

__global__ void generateHierarchyParallel(
    unsigned long long int* sortedMortonCodes, // 三角形排序后morton值
    int numObjects,
    Node* leafNodes,
    Node* internalNodes,
    unsigned int* parentWrongNum)
{

    // NOTE: 此处假设kernel以<BLOCK,THREAD>的1D方式启动
    const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // 当前线程在所有线程下的坐标
    const int total_thread = blockDim.x * gridDim.x;                // 一个grid内的所有线程数量，作为递增量

    // Construct internal nodes.
    // TODO: 可并行化(√)
    // 生成内部节点
    for (int idx = t_idx; idx < numObjects - 1; idx += total_thread) // in parallel
    {
        // Find out which range of objects the node corresponds to.
        // (This is where the magic happens!)

        Range range = determineRange(sortedMortonCodes, numObjects, idx);
        int first = range.x;
        int last = range.y;

        // Determine where to split the range
        int split = findSplit(sortedMortonCodes, first, last);
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

        if (childA->parent != NULL) atomicAdd(parentWrongNum, 1);
        childA->parent = &internalNodes[idx];    //  TODO: 某些child节点没有被访问到，导致自己的parent没有被赋值
        if (childB->parent != NULL) atomicAdd(parentWrongNum, 1);
        childB->parent = &internalNodes[idx];

        //atomicAdd(parentSetNum, 2);
    }
}

__global__ void newGenerateHierarchyParallel(
    unsigned int* keys, // 三角形排序后morton值
    int numObjects,
    Node* leafNodes,
    Node* internalNodes,
    unsigned int* parentWrongNum)
{

    // NOTE: 此处假设kernel以<BLOCK,THREAD>的1D方式启动
    const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // 当前线程在所有线程下的坐标
    const int total_thread = blockDim.x * gridDim.x;                // 一个grid内的所有线程数量，作为递增量

    for (unsigned int i = t_idx; i < numObjects - 1; i += total_thread) // in parallel
    {
        int d = (delta(i, i + 1, keys, numObjects) - delta(i, i - 1, keys, numObjects)) >= 0 ? 1 : -1;
        unsigned int delta_min = delta(i, i - d, keys, numObjects);
        unsigned int mlen = 2;
        //return Range(100, mlen);
        while (delta(i, i + mlen * d, keys, numObjects) > delta_min) {
            mlen *= 2;
        }

        unsigned int l = 0;
        for (unsigned int t = mlen/2; t >= 1; t /= 2) {
            if (delta(i, i + (l + t) * d, keys, numObjects) > delta_min) {
                l += t;
            }
        }
        unsigned int j = i + l * d;

        int delta_node = delta(i, j, keys, numObjects);
        int s = 0;
        for (unsigned int t = l/2; t >= 1; t /= 2) {
            if (delta(i, i + (s + t) * d, keys, numObjects) > delta_node) {
                s += t;
            }
        }
        int split = i + s * d + min(d, 0);

        Node *childA, *childB;
        if (min(i, j) == split) childA = ((Node*)(&leafNodes[split]));
        else childA = ((Node*)(&internalNodes[split]));

        if (max(i, j) == split + 1) childB = ((Node*)(&leafNodes[split + 1]));
        else childB = ((Node*)(&internalNodes[split + 1]));

        // Record parent-child relationships.
        internalNodes[i].childA = childA;
        internalNodes[i].childB = childB;

        if (childA->parent != NULL) atomicAdd(parentWrongNum, 1);
        childA->parent = ((Node*)&internalNodes[i]);    //  TODO: 某些child节点没有被访问到，导致自己的parent没有被赋值
        if (childB->parent != NULL) atomicAdd(parentWrongNum, 1);
        childB->parent = ((Node*)&internalNodes[i]);      
    }
}

__global__ void calBoundingBox(Node* leaves, vec3f* vertexes, unsigned int numOfLeaves) {
    const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // 当前线程在所有线程下的坐标
    const int total_thread = blockDim.x * gridDim.x;                // 一个grid内的所有线程数量，作为递增量

    for (int i = t_idx; i < numOfLeaves; i += total_thread) {
        // 首先为叶节点计算box
        leaves[i].box.set(leaves[i].triangle, vertexes);    
        leaves[i].childCount = 1;
        Node* curNode = leaves[i].parent;

        // 从底向上计算box，直到根节点
        while (curNode != NULL) {
            if (atomicAdd(&(curNode->bounded), 1) == 0) {       // 利用原子操作刷新flag
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

#endif // BVH_H