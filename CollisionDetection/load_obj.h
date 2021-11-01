#pragma once
#ifndef LOAD_OBJ_H
#define LOAD_OBJ_H

#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <thrust/sort.h>
#include <thrust/functional.h>

#include "triangle.cuh"
#include "vec3f.cuh"
#include "morton.h"

using namespace std;

bool cmp(long long int a, long long int b)
{
	return a <= b;
}


void loadObj(const string filename, 
	vector<vec3f>& vertexes, 
	vector<Triangle>& triangles,
	vector<unsigned long long int>& mortons)
{
	std::ifstream in(filename.c_str());

	if (!in.good())
	{
		cout << "* ERROR: loading obj:(" << filename << ") file is not good" << "\n";
		exit(0);
	}

	char buffer[256], str[255];
	float f1, f2, f3;
	float xmin = 1000, ymin = 1000, zmin = 1000;

	while (!in.getline(buffer, 255).eof())
	{
		buffer[255] = '\0';

		sscanf_s(buffer, "%s", str, 255);

		// reading a vertex
		if (buffer[0] == 'v' && (buffer[1] == ' ' || buffer[1] == 32))
		{
			if (sscanf(buffer, "v %f %f %f", &f1, &f2, &f3) == 3)
			{
				vertexes.push_back(vec3f(f1, f2, f3));
				if (f1 < xmin) xmin = f1;
				if (f2 < ymin) ymin = f2;
				if (f3 < zmin) zmin = f3;
			}
			else
			{
				cout << "* ERROR: vertex not in wanted format in OBJLoader" << "\n";
				exit(-1);
			}
		}
		// reading FaceMtls 
		else if (buffer[0] == 'f' && (buffer[1] == ' ' || buffer[1] == 32))
		{
			Triangle f, nv;
			int v1, v2, v3;
			int nt = sscanf(buffer, "f %d/%d %d/%d %d/%d", &v1, &nv.vIdx[0], &v2, &nv.vIdx[1], &v3, &nv.vIdx[2]);
			if (nt != 6)
			{
				printf("* ERROR: I don't know the format of that FaceMtl (while only read %d vertex of face)\n", nt);
				//cout << "ERROR: I don't know the format of that FaceMtl" << "\n";
				exit(-1);
			}

			const int v_size = vertexes.size() + 1;
			if (v1 >= v_size || v2 >= v_size || v3 >= v_size) {
				printf("* ERROR: Vertex of face out of bound, v_size: %d, v_idx of face: %d,%d,%d\n", v_size, v1, v2, v3);
			}

			f.vIdx[0] = v1 - 1;
			f.vIdx[1] = v2 - 1;
			f.vIdx[2] = v3 - 1;

			// Use the center of the triangle vertexes to represent the triangle while calculating Morton code
			// NOTE: Here assumes all vertexes have been read into vector when reading faces, unless memory asccess error will occur
			// 使用三角形的中心（三个顶点的平均值）作为三角形的表示点，计算morton值
			// NOTE: 此处假设了在读取face时，所有的vertex都已经被读入，否则将会出现访问超界
			vec3f* p1 = &vertexes[v1 - 1], *p2 = &vertexes[v2 - 1], *p3 = &vertexes[v3 - 1];
			double xAvg = (p1->x + p2->x + p3->x) / 3, yAvg = (p1->y + p2->y + p3->y) / 3, zAvg = (p1->z + p2->z + p3->z) / 3;
			unsigned long long int tri_morton_code = morton3D(xAvg, yAvg, zAvg);

			// 按照相同顺序将三角形和对应的morton值放入，方便后续将morton作为key进行排序
			f.ID = triangles.size();
			f.morton = tri_morton_code;
			f.ax = xAvg;
			f.ay = yAvg;
			f.az = zAvg;

			triangles.push_back(f);
			mortons.push_back(tri_morton_code);
		}
	}

	// Sort triangles according to Morton codes
	// 根据morton值来对三角形进行排序
	thrust::sort_by_key(mortons.begin(), mortons.end(), triangles.begin());
	
	unsigned int wrongSortCount = 0;
	for (int i = 0; i < mortons.size() - 1; i++) {
		if (mortons[i] >= mortons[i + 1]) {
			wrongSortCount++;
			printf("* ERROR: not subject to sort! %d-th and %d-th morton value: %u, %u\n", i, i + 1, mortons[i], mortons[i + 1]);
		}
	}

	printf("\nObj File Loaded:\n");
	printf("- %u vertexes loaded\n", vertexes.size());
	printf("- %u triangles loaded\n", triangles.size());
	printf("- wrong morton sort count: %u\n", wrongSortCount);
	printf("- First morton code: %llu, last morton code: %llu\n\n", mortons[0], mortons[mortons.size() - 1]);
	printf("- xmin=%f, ymin=%f, z=%f\n", xmin, ymin, zmin);
}

#endif // !LOAD_OBJ_H
