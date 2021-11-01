/*
    CPU code, still some bugs.
*/

#ifndef CPU_MATH_H
#define CPU_MATH_H

int clzll(unsigned long long int n, int);

#define deltaCpu(i,j,keys,n) ((j >= 0 && j < n) ? clzll(keys[i] ^ keys[j], 0) : -1)

#endif // !CPU_MATH_H

