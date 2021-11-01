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

/* ͨ��Ѱ������morton�����ǰ��0�������ҵ���Ӧ�ķָ�㡣
    ���㷨��Ҫ��bvh������������ϡ�
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
    // ��������Ԫ�ص������ͬλ���������λ��ʼ���𣩣���ЩLSB�����Ϊsplit��Ĵ�ѡ�㣬���ᱻ�޳�
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

        if (newSplit < last)    // ��黮�ֵ��Ƿ񳬹��߽�
        {
            unsigned long long int splitCode = mortons[newSplit];
            int splitPrefix = clzll(firstCode ^ splitCode, 0);

            // ֻ����[first, split]�ڵ���ͬλ����������Ԫ�ص���ͬλ��ʱ���ŻὫsplit��λ�������ƶ�
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

    // ���صķ�Χ�У�ǰ��һ���ǽ�С�ߣ�����һ���ǽϴ���
    return Range(min(i, j), max(i, j));
}

__host__ void fillLeafNodesCpu(
    Triangle* sortedTriangles,       // ������������±�
    int numObjects,
    Node* leafNodes)
{
    // NOTE: �˴�����kernel��<BLOCK,THREAD>��1D��ʽ����
    //const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // ��ǰ�߳��������߳��µ�����
    //const int total_thread = blockDim.x * gridDim.x;                // һ��grid�ڵ������߳���������Ϊ������

    // Construct leaf nodes.
    // Note: This step can be avoided by storing
    // the tree in a slightly different way.
    // TODO: �ɲ��л�(��)
    // ��ÿһ��objectʹ��һ���������������洢
    for (int idx = 0; idx < numObjects; idx++) { // in parallel
        leafNodes[idx].triangle = &sortedTriangles[idx];         // ���Ҷ�ڵ�ʱ��Ҷ�ڵ��ֵ�������GPU�ϵ������ε�ָ�룬��˿�����Host��Ҳ���ɶ�
        leafNodes[idx].isLeaf = true;
        leafNodes[idx].idx = idx;
    }
}

__host__ void generateHierarchyParallelCpu(
    unsigned long long int* sortedMortonCodes, // �����������mortonֵ
    int numObjects,
    Node* leafNodes,
    Node* internalNodes,
    unsigned int* parentWrongNum)
{

    // NOTE: �˴�����kernel��<BLOCK,THREAD>��1D��ʽ����
    //const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // ��ǰ�߳��������߳��µ�����
    //const int total_thread = blockDim.x * gridDim.x;                // һ��grid�ڵ������߳���������Ϊ������

    // Construct internal nodes.
    // TODO: �ɲ��л�(��)
    // �����ڲ��ڵ�
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
        childA->parent = &internalNodes[idx];    //  TODO: ĳЩchild�ڵ�û�б����ʵ��������Լ���parentû�б���ֵ
        if (childB->parent != NULL) parentWrongNum[0]++;
        childB->parent = &internalNodes[idx];
    }

    // Node 0 is the root.

    //return &internalNodes[0];
}

__host__ void calBoundingBoxCpu(Node* leaves, vec3f* vertexes, unsigned int numOfLeaves) {
    //const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // ��ǰ�߳��������߳��µ�����
    //const int total_thread = blockDim.x * gridDim.x;                // һ��grid�ڵ������߳���������Ϊ������

    for (int i = 0; i < numOfLeaves; i++) {
        // ����ΪҶ�ڵ����box
        leaves[i].box.set(leaves[i].triangle, vertexes);
        leaves[i].childCount = 1;
        Node* curNode = leaves[i].parent;

        // �ӵ����ϼ���box��ֱ�����ڵ�
        while (curNode != NULL) {
            if (curNode->bounded++ == 0) {       // ����ԭ�Ӳ���ˢ��flag
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
    // NOTE: �˴�����kernel��<BLOCK_GRID,THREAD_GRID>��2D��ʽ����
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int t_idx = x + y * blockDim.x * gridDim.x;
    unsigned int total_thread = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
    unsigned int thread_per_block = blockDim.x * blockDim.y;
    //unsigned int t_idx_of_block = thread_idx
    unsigned int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
    unsigned int i = t_idx;

    //__shared__ Node* queryNodes[];

    // TODO: 1. ��root���߲����ڲ��ڵ�д��constant���� 2. ��query��Ҷ�ڵ����shared�м���

    // NOTE: �˴�����kernel��<BLOCK,THREAD>��1D��ʽ����
    //const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // ��ǰ�߳��������߳��µ�����
    //const int total_thread = blockDim.x * gridDim.x;                // һ��grid�ڵ������߳���������Ϊ������
    //unsigned int i = t_idx;

    for (int i = 0; i < numObjects; i++) {
        findCollisionIterativeCpu(root, leaves[i].triangle, &leaves[i].box, vs, count, colTris);
    }
}

__host__ void checkInternalNodesCpu(Node* nodes, unsigned int numObjects, unsigned int* nullParentNum, unsigned int* wrongBoundNum, unsigned int* nullChildNum, unsigned int* notInternalCount, unsigned int* uninitBoxCount)
{
    //// NOTE: �˴�����kernel��<BLOCK,THREAD>��1D��ʽ����
    //const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // ��ǰ�߳��������߳��µ�����
    //const int total_thread = blockDim.x * gridDim.x;                // һ��grid�ڵ������߳���������Ϊ������

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
    //// NOTE: �˴�����kernel��<BLOCK,THREAD>��1D��ʽ����
    //const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // ��ǰ�߳��������߳��µ�����
    //const int total_thread = blockDim.x * gridDim.x;                // һ��grid�ڵ������߳���������Ϊ������

    for (int i = 0; i < numObjects; i++) {
        if (nodes[i].parent == NULL) nullParentNum++;
        if (nodes[i].triangle == NULL || nodes[i].triangle->selfCheck() > 0) nullTriangleNum++;
        if (nodes[i].box.selfCheck() == 0)
            illegalboxCount++;;
    }
}

__host__ void checkTriangleIdxCpu(Node* ns, vec3f* vs, unsigned int num, unsigned int maxv, unsigned int* count)
{
    //// NOTE: �˴�����kernel��<BLOCK,THREAD>��1D��ʽ����
    //const int t_idx = threadIdx.x + blockIdx.x * blockDim.x;        // ��ǰ�߳��������߳��µ�����
    //const int total_thread = blockDim.x * gridDim.x;                // һ��grid�ڵ������߳���������Ϊ������

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