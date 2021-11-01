/*
    CPU code, still some bugs.
*/

//#include <cuda_runtime.h>
//#include <vector>
//#include <string>
//#include <set>
//
//#include "load_obj.h"
//#include "collision.cuh"
//#include "check.cuh"
//#include "./common/book.h"
//#include "cpu.cuh"
//
//#define COL_MAX_LEN 1000000
//
//void printElapsedTime(cudaEvent_t* start, cudaEvent_t* stop, const char* opname) {
//	printf("\nTime of %s:  ", opname);
//	float   elapsedTime;
//	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, *start, *stop));
//	printf("%3.1f ms\n", elapsedTime);
//}
//
//void makeAndPrintSet(unsigned int* data, unsigned int num, const char* title)
//{
//	set<unsigned int> dset;
//	for (int i = 0; i < num; i++) {
//		dset.insert(data[i]);
//	}
//
//	printf("\n\n%s（%u points in total）:\n", title, dset.size());
//	set<unsigned int>::iterator it;
//	for (it = dset.begin(); it != dset.end(); it++) {
//		printf("%u\n", *it);
//	}
//}
//
//int main()
//{
//	cudaEvent_t start, stop, m_start, m_stop;
//	HANDLE_ERROR(cudaEventCreate(&start));
//	HANDLE_ERROR(cudaEventCreate(&stop));
//	HANDLE_ERROR(cudaEventCreate(&m_start));
//	HANDLE_ERROR(cudaEventCreate(&m_stop));
//
//	HANDLE_ERROR(cudaEventRecord(m_start, 0));
//
//	const std::string file_path = "F:/坚果云文件/我的坚果云/研一上/cuda/projects/CollisionDetection/flag-2000-changed.obj";
//
//	std::vector<vec3f> vertexes;
//	std::vector<Triangle> triangles;
//	std::vector<unsigned long long int> mortons;
//
//	loadObj(file_path, vertexes, triangles, mortons);
//	const unsigned int m_size = mortons.size();
//	const unsigned int v_size = vertexes.size();
//
//	vec3f* v_ptr = &vertexes[0]; // new vec3f[v_size];
//	Triangle* t_ptr = &triangles[0]; // new Triangle[m_size];
//	unsigned long long int* m_ptr = &mortons[0]; // new unsigned long long int[m_size];
//	Node* leaf_nodes = new Node[m_size];
//	Node* internal_nodes = new Node[m_size-1];
//	unsigned int* collision_list = new unsigned int[10000];
//	unsigned int test_val;
//
//	memset(collision_list, 0, 10000 * sizeof(unsigned int));
//
//	/* 填充叶节点 */
//	fillLeafNodesCpu(t_ptr, m_size, leaf_nodes);
//
//	/* 生成BVH */
//	printf("\n- before generateHierarchyParallel, wrongParentNum = %u\n", collision_list[0]);
//	generateHierarchyParallelCpu(m_ptr, m_size, leaf_nodes, internal_nodes, collision_list);
//	printf("\n- generateHierarchyParallel check result: wrongParentNum = %u, with total nodes=%u\n\n", collision_list[0], m_size-1);
//
//	/* 计算包围盒 */
//	calBoundingBoxCpu(leaf_nodes, v_ptr, m_size);
//
//	/* 内部节点和叶节点自检 */
//	memset(collision_list, 0, sizeof(unsigned int) * 5);
//	checkInternalNodesCpu(internal_nodes, m_size - 1, &collision_list[0], &collision_list[1], &collision_list[2], &collision_list[3], &collision_list[4]);
//	printf("\n- Internal node check result: nullParentnum = %u, wrongBoundCount=%u, nullChildCount=%u, notInternalCount=%u, uninitBoxCount=%u, with total nodes=%u\n\n", collision_list[0], collision_list[1], collision_list[2], collision_list[3], collision_list[4], m_size-1);
//
//	memset(collision_list, 0, sizeof(unsigned int) * 5);
//	checkLeafNodesCpu(leaf_nodes, m_size, &collision_list[0], &collision_list[1], &collision_list[2], &collision_list[3]);
//	printf("\n- Leaf node check result: nullParentnum = %u, nullTriangle=%u, notLeafCount=%u, illegalBoxCount=%u, with total nodes=%u\n\n", collision_list[0], collision_list[1], collision_list[2], collision_list[3], m_size);
//
//	memset(collision_list, 0, sizeof(unsigned int) * 5);
//	checkTriangleIdxCpu(leaf_nodes, v_ptr, m_size, 632674, &collision_list[0]);
//	printf("\n- Triangle check result: illegal triangle vidx num = %u, with total triangles=%u\n\n", collision_list[0], mortons.size());
//	printf("\n$ triangle num = %u, mortons num = %u, vertex num = %u\n\n", triangles.size(), mortons.size(), vertexes.size());
//
//	/* 寻找碰撞点对 */
//	//findCollisionsCpu(internal_nodes, leaf_nodes, v_ptr, m_size, &test_val, collision_list);
//	////HANDLE_ERROR(cudaMemcpy(temp_nums, test_val, sizeof(unsigned int), cudaMemcpyDeviceToHost));
//	////HANDLE_ERROR(cudaMemcpy(h_collision_list, collision_list, 1000 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
//	//printf("\n\n- contact val = %u\n", test_val);
//
//	//printf("\nCollision pair (%u triangle pairs in total):\n", test_val);
//	//for (int i = 0; i < test_val; i++) {
//	//	printf("%07u - %07u\n", collision_list[2 * i], collision_list[2 * i + 1]);
//	//}
//
//	//makeAndPrintSet(collision_list, 2 * test_val, "Collision Triangles:");
//
//	std::cout << "- Successfully Return" << std::endl;
//
//	HANDLE_ERROR(cudaEventRecord(m_stop, 0)); HANDLE_ERROR(cudaEventSynchronize(m_stop));
//	printElapsedTime(&m_start, &m_stop, "Total Time");
//
//	printf("\n test for clzll: %u\n", clzll(4567, 1));
//
//	//printf("\n test for __builtin_clz: %u\n", __builtin_clz(1278));
//
//	return 0;
//}