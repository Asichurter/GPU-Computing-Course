#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <set>

#include "load_obj.h"
#include "collision.cuh"
#include "check.cuh"
#include "./common/book.h"

#define COL_MAX_LEN 1000000

void printElapsedTime(cudaEvent_t* start, cudaEvent_t* stop, const char* opname) {
	printf("\nTime of %s:  ", opname);
	float   elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, *start, *stop));
	printf("%3.1f ms\n", elapsedTime);
}

void printTriangleVertex(Triangle* t, vector<vec3f>& vs)
{
	vec3f *v1 = &vs[t->vIdx[0]], *v2 = &vs[t->vIdx[1]], *v3 = &vs[t->vIdx[2]];
	printf("# v1(%u)=(%.4f, %.4f, %.4f), v2(%u)=(%.4f, %.4f, %.4f), v3(%u)=(%.4f, %.4f, %.4f)\n", 
		t->vIdx[0], v1->x, v1->y, v1->z, t->vIdx[1], v2->x, v2->y, v2->z, t->vIdx[2], v3->x, v3->y, v3->z);
}

void makeAndPrintSet(unsigned int* data, unsigned int num, const char* title)
{
	set<unsigned int> dset;
	for (int i = 0; i < num; i++) {
		dset.insert(data[i]);
	}

	printf("\n\n%s（%u points in total）:\n", title, dset.size());
	set<unsigned int>::iterator it;
	for (it = dset.begin(); it != dset.end(); it++) {
		printf("%u\n", *it);
	}
}

int main()
{
	cudaEvent_t start, stop, m_start, m_stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventCreate(&m_start));
	HANDLE_ERROR(cudaEventCreate(&m_stop));

	HANDLE_ERROR(cudaEventRecord(m_start, 0));

	const std::string file_path = "F:/坚果云文件/我的坚果云/研一上/cuda/projects/CollisionDetection/flag-2000-changed.obj";

	std::vector<vec3f> vertexes;
	std::vector<Triangle> triangles;
	std::vector<unsigned long long int> mortons;

	loadObj(file_path, vertexes, triangles, mortons);

	vec3f* v_ptr;
	Triangle* t_ptr;
	unsigned long long int* m_ptr;
	Node* leaf_nodes;
	Node* internal_nodes;
	unsigned int* collision_list;
	unsigned int* test_val;
	unsigned int temp_nums[100];
	unsigned int h_collision_list[1000];
	Triangle* colTris;
	//cudaDeviceProp prop;

	/* 分配和拷贝GPU内存 */
	HANDLE_ERROR(cudaMalloc((void**)&v_ptr, vertexes.size() * sizeof(vec3f)));
	HANDLE_ERROR(cudaMalloc((void**)&t_ptr, triangles.size() * sizeof(Triangle)));
	HANDLE_ERROR(cudaMalloc((void**)&m_ptr, mortons.size() * sizeof(unsigned long long int)));
	HANDLE_ERROR(cudaMalloc((void**)&collision_list, 1000 * sizeof(unsigned int)));
	HANDLE_ERROR(cudaMalloc((void**)&test_val, sizeof(unsigned int)));
	HANDLE_ERROR(cudaMalloc((void**)&colTris, 100 * sizeof(Triangle)));
	HANDLE_ERROR(cudaMalloc((void**)&leaf_nodes, mortons.size() * sizeof(Node)));
	HANDLE_ERROR(cudaMalloc((void**)&internal_nodes, (mortons.size() - 1) * sizeof(Node)));
	HANDLE_ERROR(cudaMemcpy(v_ptr, &vertexes[0], vertexes.size() * sizeof(vec3f), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(t_ptr, &triangles[0], triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(m_ptr, &mortons[0], mortons.size() * sizeof(unsigned long long int), cudaMemcpyHostToDevice));

	/* 填充叶节点 */
	HANDLE_ERROR(cudaEventRecord(start, 0));
	fillLeafNodes <<< 128, 128 >>> (t_ptr, mortons.size(), leaf_nodes);
	HANDLE_ERROR(cudaEventRecord(stop, 0)); HANDLE_ERROR(cudaEventSynchronize(stop));
	printElapsedTime(&start, &stop, "fillLeafNode");
	//std::cout << "- fillLeafNodes returned" << std::endl << std::endl;
	//HANDLE_ERROR(cudaMemcpy(test_leaves, leaf_nodes, sizeof(LeafNode)*10, cudaMemcpyDeviceToHost));

	/* 生成BVH */
	HANDLE_ERROR(cudaMemset(collision_list, 0, sizeof(unsigned int) * 5));
	HANDLE_ERROR(cudaEventRecord(start, 0));
	generateHierarchyParallel <<< 128, 128 >>> (m_ptr, mortons.size(), leaf_nodes, internal_nodes, &collision_list[0]);
	HANDLE_ERROR(cudaEventRecord(stop, 0)); HANDLE_ERROR(cudaEventSynchronize(stop));
	printElapsedTime(&start, &stop, "generateHierarchyParallel");
	HANDLE_ERROR(cudaMemcpy(temp_nums, collision_list, 5 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	printf("\n- generateHierarchyParallel check result: wrongParentNum = %u, with total nodes=%u\n\n", temp_nums[0], mortons.size() - 1);

	/* 计算包围盒 */
	HANDLE_ERROR(cudaEventRecord(start, 0));
	calBoundingBox <<< 128, 128 >>> (leaf_nodes, v_ptr, mortons.size());
	//std::cout << "- calBoundingBox returned" << std::endl << std::endl;
	HANDLE_ERROR(cudaEventRecord(stop, 0)); HANDLE_ERROR(cudaEventSynchronize(stop));
	printElapsedTime(&start, &stop, "calBoundingBox");

	/* 内部节点和叶节点自检 */
	HANDLE_ERROR(cudaMemset(collision_list, 0, sizeof(unsigned int) * 5));
	HANDLE_ERROR(cudaEventRecord(start, 0));
	checkInternalNodes <<< 128, 128 >>> (internal_nodes, mortons.size()-1, &collision_list[0], &collision_list[1], &collision_list[2], &collision_list[3], &collision_list[4]);
	HANDLE_ERROR(cudaEventRecord(stop, 0)); HANDLE_ERROR(cudaEventSynchronize(stop));
	printElapsedTime(&start, &stop, "checkInternalNodes");
	HANDLE_ERROR(cudaMemcpy(temp_nums, collision_list, 5 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	printf("\n- Internal node check result: nullParentnum = %u, wrongBoundCount=%u, nullChildCount=%u, notInternalCount=%u, uninitBoxCount=%u, with total nodes=%u\n\n", temp_nums[0], temp_nums[1], temp_nums[2], temp_nums[3], temp_nums[4], mortons.size()-1);

	HANDLE_ERROR(cudaMemset(collision_list, 0, sizeof(unsigned int) * 5));
	HANDLE_ERROR(cudaEventRecord(start, 0));
	checkLeafNodes <<< 128, 128 >>> (leaf_nodes, mortons.size(), &collision_list[0], &collision_list[1], &collision_list[2], &collision_list[3]);
	HANDLE_ERROR(cudaEventRecord(stop, 0)); HANDLE_ERROR(cudaEventSynchronize(stop));
	printElapsedTime(&start, &stop, "checkLeafNodes");
	HANDLE_ERROR(cudaMemcpy(temp_nums, collision_list, 5 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	printf("\n- Leaf node check result: nullParentnum = %u, nullTriangle=%u, notLeafCount=%u, illegalBoxCount=%u, with total nodes=%u\n\n", temp_nums[0], temp_nums[1], temp_nums[2], temp_nums[3], mortons.size());

	HANDLE_ERROR(cudaMemset(collision_list, 0, sizeof(unsigned int) * 5));
	HANDLE_ERROR(cudaEventRecord(start, 0));
	checkTriangleIdx <<< 128, 128 >>> (leaf_nodes, v_ptr, mortons.size(), 632674, &collision_list[0]);
	HANDLE_ERROR(cudaEventRecord(stop, 0)); HANDLE_ERROR(cudaEventSynchronize(stop));
	printElapsedTime(&start, &stop, "checkTriangleIdx");
	HANDLE_ERROR(cudaMemcpy(temp_nums, collision_list, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	printf("\n- Triangle check result: illegal triangle vidx num = %u, with total triangles=%u\n\n", temp_nums[0], mortons.size());
	printf("\n$ triangle num = %u, mortons num = %u, vertex num = %u\n\n", triangles.size(), mortons.size(), vertexes.size());

	/* 测试根节点的两个子节点的box */
	//Node testLeaf, testInternals[3];
	//unsigned int overlap[2];
	//HANDLE_ERROR(cudaMemset(collision_list, 0, sizeof(unsigned int) * 5));
	//HANDLE_ERROR(cudaMemcpy(testInternals, internal_nodes, sizeof(Node), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaMemcpy(testInternals + 1, testInternals[0].childA, sizeof(Node), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaMemcpy(testInternals + 2, testInternals[0].childB, sizeof(Node), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaMemcpy(&testLeaf, leaf_nodes, sizeof(Node), cudaMemcpyDeviceToHost));
	//checkBoxOverlapFunc <<< 1, 1 >>> (testInternals[0].childA, leaf_nodes, collision_list);
	//checkBoxOverlapFunc <<< 1, 1 >>> (testInternals[0].childB, leaf_nodes, collision_list+1);
	//HANDLE_ERROR(cudaMemcpy(&overlap, collision_list, 2*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	//printf("$ overlap result: L=%u, R=%u\n", overlap[0], overlap[1]);
	//printf("$ leaf box: [(%f, %f), (%f, %f), (%f, %f)]\n", testLeaf.box.x1, testLeaf.box.x2, testLeaf.box.y1, testLeaf.box.y2, testLeaf.box.z1, testLeaf.box.z2);
	//printf("$ root box: [(%f, %f), (%f, %f), (%f, %f)]\n", testInternals[0].box.x1, testInternals[0].box.x2, testInternals[0].box.y1, testInternals[0].box.y2, testInternals[0].box.z1, testInternals[0].box.z2);
	//printf("$ root's childA box: [(%f, %f), (%f, %f), (%f, %f)]\n", testInternals[1].box.x1, testInternals[1].box.x2, testInternals[1].box.y1, testInternals[1].box.y2, testInternals[1].box.z1, testInternals[1].box.z2);
	//printf("$ root's childB box: [(%f, %f), (%f, %f), (%f, %f)]\n", testInternals[2].box.x1, testInternals[2].box.x2, testInternals[2].box.y1, testInternals[2].box.y2, testInternals[2].box.z1, testInternals[2].box.z2);

	/* 测试前两个叶节点的box的merge */
	// **********************************************************************************************************
	//Node ls[10];
	//Box merged;
	//HANDLE_ERROR(cudaMemcpy(ls, leaf_nodes, 10*sizeof(Node), cudaMemcpyDeviceToHost));
	//printf("\nmorton codes:\n");
	//for (int i = 0; i < 10; i++) {
	//	printf("%llu (ax=%f, ay=%f, az=%f)\n", triangles[i].morton, triangles[i].ax, triangles[i].ay, triangles[i].az);
	//}
	//printf("\n\n");

	//printf("\nmorton xyz:\n");
	//// x
	//printf("x = [");
	//for (int j = 0; j < 10; j++) {
	//	printf("%f,", triangles[j].ax);
	//}
	//printf("]\n");
	//// y
	//printf("y = [");
	//for (int j = 0; j < 10; j++) {
	//	printf("%f,", triangles[j].ay);
	//}
	//printf("]\n");
	//// z
	//printf("z = [");
	//for (int j = 0; j < 10; j++) {
	//	printf("%f,", triangles[j].az);
	//}
	//printf("]\n");
	//printf("\n\n");

	//for (int i = 0; i < 10; i++) {
	//	printBox(&ls[i].box, "box");
	//}
	//merged.merge(&ls[0].box, &ls[1].box);
	//printBox(&merged, "merged of 1-st and 2-nd box: ");
	//printf("\n");
	// **********************************************************************************************************

	/* BOX测试 */
	//HANDLE_ERROR(cudaMemset(collision_list, 0, sizeof(unsigned int) * 5));
	//HANDLE_ERROR(cudaMemcpy(temp_nums, collision_list, sizeof(unsigned int) * 5, cudaMemcpyDeviceToHost));
	//printf("\n- collision list after init: %u, %u, %u, %u, %u\n\n", temp_nums[0], temp_nums[1], temp_nums[2], temp_nums[3], temp_nums[4]);
	//----------------------------------------------------------------------------------------------------------
	//Node* h_internals = new Node[3];
	//HANDLE_ERROR(cudaMemcpy(h_internals, internal_nodes, sizeof(Node), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaMemcpy(h_internals + 1, h_internals[0].childA, sizeof(Node), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaMemcpy(h_internals + 2, h_internals[0].childB, sizeof(Node), cudaMemcpyDeviceToHost));
	//
	//printf("$ childA & childB box check result: %u\n$ childA box:[(%f, %f), (%f, %f), (%f, %f)]\n$ childB box: [(%f, %f), (%f, %f), (%f, %f)]\n\n",
	//	checkBoxOverlap(&h_internals[1].box, &h_internals[2].box),
	//	h_internals[1].box.x1, h_internals[1].box.x2, h_internals[1].box.y1, h_internals[1].box.y2, h_internals[1].box.z1, h_internals[1].box.z2,
	//	h_internals[2].box.x1, h_internals[2].box.x2, h_internals[2].box.y1, h_internals[2].box.y2, h_internals[2].box.z1, h_internals[2].box.z2);
	//printf("$ root info: childA has %u nodes, childB has %u nodes\n\n", h_internals[1].childCount, h_internals[2].childCount);

	/* 寻找碰撞点对 */
	HANDLE_ERROR(cudaEventRecord(start, 0));
	dim3 blocks(128, 128);
	dim3 threads(128);
	findCollisions <<< blocks, threads>>> (&internal_nodes[0], leaf_nodes, v_ptr, mortons.size(), test_val, collision_list);
	HANDLE_ERROR(cudaEventRecord(stop, 0)); HANDLE_ERROR(cudaEventSynchronize(stop));
	printElapsedTime(&start, &stop, "findCollisions");
	HANDLE_ERROR(cudaMemcpy(temp_nums, test_val, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(h_collision_list, collision_list, 1000 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	printf("\n\n- contact val = %u\n", temp_nums[0]);
	
	printf("\nCollision pair (%u triangle pairs in total):\n", temp_nums[0]);
	for (int i = 0; i < temp_nums[0]; i++) {
		printf("%07u - %07u\n", h_collision_list[2*i], h_collision_list[2*i+1]);
	}

	makeAndPrintSet(h_collision_list, 2 * temp_nums[0], "Collision Triangles:");

	HANDLE_ERROR(cudaFree(v_ptr));
	HANDLE_ERROR(cudaFree(t_ptr));
	HANDLE_ERROR(cudaFree(m_ptr));
	HANDLE_ERROR(cudaFree(leaf_nodes));
	HANDLE_ERROR(cudaFree(internal_nodes));
	HANDLE_ERROR(cudaFree(collision_list));
	HANDLE_ERROR(cudaFree(test_val));
	HANDLE_ERROR(cudaFree(colTris));

	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));

	std::cout << "- Successfully Return" << std::endl;

	HANDLE_ERROR(cudaEventRecord(m_stop, 0)); HANDLE_ERROR(cudaEventSynchronize(m_stop));
	printElapsedTime(&m_start, &m_stop, "Total Time");

	return 0;
}