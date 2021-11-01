/*
	Collision detection using binary-tree based BVH.
	This code uses Nvidia's cuda tutorial as an utility, which refers to the book.h below.

	Author: Asichurter
	Date: 2021-11-01
*/

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <set>

#include "load_obj.h"
#include "collision.cuh"
#include "check.cuh"
#include "./common/book.h"

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

	printf("\n\n%s£¨%u points in total£©:\n", title, dset.size());
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

	// Set your OBJ file path here
	const std::string file_path = "./resources/flag-2000-changed.obj";

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

	/* Allocate and copy GPU memory */
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

	/* Fill leaf nodes with triangles */
	HANDLE_ERROR(cudaEventRecord(start, 0));
	fillLeafNodes <<< 128, 128 >>> (t_ptr, mortons.size(), leaf_nodes);
	HANDLE_ERROR(cudaEventRecord(stop, 0)); HANDLE_ERROR(cudaEventSynchronize(stop));
	printElapsedTime(&start, &stop, "fillLeafNode");

	/* Generate BVH parallel */
	HANDLE_ERROR(cudaMemset(collision_list, 0, sizeof(unsigned int) * 5));
	HANDLE_ERROR(cudaEventRecord(start, 0));
	generateHierarchyParallel <<< 128, 128 >>> (m_ptr, mortons.size(), leaf_nodes, internal_nodes, &collision_list[0]);
	HANDLE_ERROR(cudaEventRecord(stop, 0)); HANDLE_ERROR(cudaEventSynchronize(stop));
	printElapsedTime(&start, &stop, "generateHierarchyParallel");
	HANDLE_ERROR(cudaMemcpy(temp_nums, collision_list, 5 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	printf("\n- generateHierarchyParallel check result: wrongParentNum = %u, with total nodes=%u\n\n", temp_nums[0], mortons.size() - 1);

	/* Calculate bounding box bottom-up */
	HANDLE_ERROR(cudaEventRecord(start, 0));
	calBoundingBox <<< 128, 128 >>> (leaf_nodes, v_ptr, mortons.size());
	//std::cout << "- calBoundingBox returned" << std::endl << std::endl;
	HANDLE_ERROR(cudaEventRecord(stop, 0)); HANDLE_ERROR(cudaEventSynchronize(stop));
	printElapsedTime(&start, &stop, "calBoundingBox");

	/* Self-check internal nodes and leaf nodes */
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

	/* Find collision pairs */
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