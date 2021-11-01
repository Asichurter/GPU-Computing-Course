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

    ͨ��Ѱ������morton�����ǰ��0�������ҵ���Ӧ�ķָ�㡣
    ���㷨��Ҫ��bvh������������ϡ�
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
    // ��������Ԫ�ص������ͬλ���������λ��ʼ���𣩣���ЩLSB�����Ϊsplit��Ĵ�ѡ�㣬���ᱻ�޳�
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

        if (newSplit < last)    // ��黮�ֵ��Ƿ񳬹��߽�
        {
            unsigned long long int splitCode = mortons[newSplit];
            int splitPrefix = __clzll(firstCode ^ splitCode);

            // ֻ����[first, split]�ڵ���ͬλ����������Ԫ�ص���ͬλ��ʱ���ŻὫsplit��λ�������ƶ�
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

    // ���صķ�Χ�У�ǰ��һ���ǽ�С�ߣ�����һ���ǽϴ���
    return Range(min(i, j), max(i, j));
    //return Range(i, j);
}

__global__ void fillLeafNodes(
    Triangle* sortedTriangles,       // ������������±�
    int numObjects,
    Node* leafNodes) 
{
    // NOTE: �˴�����kernel��<BLOCK,THREAD>��1D��ʽ����
    const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // ��ǰ�߳��������߳��µ�����
    const int total_thread = blockDim.x * gridDim.x;                // һ��grid�ڵ������߳���������Ϊ������

    // Construct leaf nodes.
    // Note: This step can be avoided by storing
    // the tree in a slightly different way.
    // TODO: �ɲ��л�(��)
    // ��ÿһ��objectʹ��һ���������������洢
    for (int idx = t_idx; idx < numObjects; idx += total_thread) { // in parallel
        leafNodes[idx].triangle = &sortedTriangles[idx];         // ���Ҷ�ڵ�ʱ��Ҷ�ڵ��ֵ�������GPU�ϵ������ε�ָ�룬��˿�����Host��Ҳ���ɶ�
        leafNodes[idx].isLeaf = true;
        leafNodes[idx].idx = idx;
    }
}

__global__ void generateHierarchyParallel(
    unsigned long long int* sortedMortonCodes, // �����������mortonֵ
    int numObjects,
    Node* leafNodes,
    Node* internalNodes,
    unsigned int* parentWrongNum)
{

    // NOTE: �˴�����kernel��<BLOCK,THREAD>��1D��ʽ����
    const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // ��ǰ�߳��������߳��µ�����
    const int total_thread = blockDim.x * gridDim.x;                // һ��grid�ڵ������߳���������Ϊ������

    // Construct internal nodes.
    // TODO: �ɲ��л�(��)
    // �����ڲ��ڵ�
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
        childA->parent = &internalNodes[idx];    //  TODO: ĳЩchild�ڵ�û�б����ʵ��������Լ���parentû�б���ֵ
        if (childB->parent != NULL) atomicAdd(parentWrongNum, 1);
        childB->parent = &internalNodes[idx];

        //atomicAdd(parentSetNum, 2);
    }
}

__global__ void newGenerateHierarchyParallel(
    unsigned int* keys, // �����������mortonֵ
    int numObjects,
    Node* leafNodes,
    Node* internalNodes,
    unsigned int* parentWrongNum)
{

    // NOTE: �˴�����kernel��<BLOCK,THREAD>��1D��ʽ����
    const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // ��ǰ�߳��������߳��µ�����
    const int total_thread = blockDim.x * gridDim.x;                // һ��grid�ڵ������߳���������Ϊ������

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
        childA->parent = ((Node*)&internalNodes[i]);    //  TODO: ĳЩchild�ڵ�û�б����ʵ��������Լ���parentû�б���ֵ
        if (childB->parent != NULL) atomicAdd(parentWrongNum, 1);
        childB->parent = ((Node*)&internalNodes[i]);      
    }
}

__global__ void calBoundingBox(Node* leaves, vec3f* vertexes, unsigned int numOfLeaves) {
    const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // ��ǰ�߳��������߳��µ�����
    const int total_thread = blockDim.x * gridDim.x;                // һ��grid�ڵ������߳���������Ϊ������

    for (int i = t_idx; i < numOfLeaves; i += total_thread) {
        // ����ΪҶ�ڵ����box
        leaves[i].box.set(leaves[i].triangle, vertexes);    
        leaves[i].childCount = 1;
        Node* curNode = leaves[i].parent;

        // �ӵ����ϼ���box��ֱ�����ڵ�
        while (curNode != NULL) {
            if (atomicAdd(&(curNode->bounded), 1) == 0) {       // ����ԭ�Ӳ���ˢ��flag
                break;                                      // ����ǵ�һ������ýڵ��·�����������˳�
            }
            // ��������ֻ����ڶ��������ڵ��·��ͨ��
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