//#include <iostream>
//#include <stdio.h>
//#include "bvh.cuh"
//#include "check.cuh"
//
//using namespace std;
//
//int main()
//{
//	unsigned int keys[] = { 1,2,4,5,19, 24, 25, 30 };
//	//Range range = determineRange(keys, 8, 0);
//	//cout << range.x << " " << range.y << endl;
//
//	Range* d_range;
//	Range h_range = Range(4);
//	int* d_nums;
//	int h_nums[3];
//
//	cudaMalloc((void**)&d_nums, sizeof(int)*3);
//	
//	testFunc <<< 1,1 >>> (d_nums);
//
//	cudaMemcpy(h_nums, d_nums, sizeof(int)*3, cudaMemcpyDeviceToHost);
//
//	printf("range.x=%d, range.y=%d, split=%d\n", h_nums[0], h_nums[1], h_nums[2]);
//
//	cudaFree(d_nums);
//
//	return 0;
//}