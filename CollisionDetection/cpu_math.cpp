#include "cpu_math.h"
#include <stdio.h>

//int clzll(unsigned long long int n) {
//	return __builtin_clzll(n);
//}

int clzll(unsigned long long int x, int flag=0)
{
    // Keep shifting x by one until leftmost bit
    // does not become 1.
    int total_bits = sizeof(x) * 8;
    int res = 0;
    while (!(x & (1 << (total_bits - 1))))
    {
        x = (x << 1);
        res++;
    }

    if (flag) printf("\n$ total bits: %d\n", total_bits);

    return res;
}