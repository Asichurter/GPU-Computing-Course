#pragma once
#ifndef MORTON_H
#define MORTON_H

# include <assert.h>

unsigned long long int expand64Bits(unsigned long long int v)
{
    //v = (v * 0x00010001u) & 0xFF0000FFu;
    //v = (v * 0x00000101u) & 0x0F00F00Fu;
    //v = (v * 0x00000011u) & 0xC30C30C3u;
    //v = (v * 0x00000005u) & 0x49249249u;
    //return v;

    v &= 0x1fffff;
    v = (v | v << 32) & 0x1f00000000ffff;
    v = (v | v << 16) & 0x1f0000ff0000ff;
    v = (v | v << 8) & 0x100f00f00f00f00f;
    v = (v | v << 4) & 0x10c30c30c30c30c3;
    v = (v | v << 2) & 0x1249249249249249;

    //v &= 0x3ff;
    //v = (v | v << 16) & 0x30000ff;   //  << < THIS IS THE MASK for shifting 16 (for bit 8 and 9)
    //v = (v | v << 8) & 0x300f00f;
    //v = (v | v << 4) & 0x30c30c3;
    //v = (v | v << 2) & 0x9249249;
    
    return v;
}

unsigned int expand32Bits(unsigned int v)
{
    v &= 0x3ff;
    v = (v | v << 16) & 0x30000ff;   //  << < THIS IS THE MASK for shifting 16 (for bit 8 and 9)
    v = (v | v << 8) & 0x300f00f;
    v = (v | v << 4) & 0x30c30c3;
    v = (v | v << 2) & 0x9249249;

    return v;
}

// Norm coordinate to [0,1]
double normX(double x)
{
    return (x - 0.004501) / 3.08;
}

// Norm coordinate to [0,1]
double normY(double y)
{
    return (y + 0.476622) / 0.76;
}

// Norm coordinate to [0,1]
double normZ(double z)
{
    return (z + 0.381965) / 2.36;
}

unsigned long int zyxAxisCoding(unsigned long long int x, unsigned long long int y, unsigned long long int z)
{
    return (z << 42) + (y << 21) + x;
}

unsigned long int xzyAxisCoding(unsigned long long int x, unsigned long long int y, unsigned long long int z)
{
    return (x << 42) + (z << 21) + y;
}

unsigned long long int morton3D(double x, double y, double z)

{
    const unsigned int scale = 1048576;
    double ex = normX(x) * scale;
    double ey = normY(y) * scale;
    double ez = normZ(z) * scale;

    assert(ex > 0 && ey > 0 && ez > 0);

    unsigned long long int xx = expand64Bits(ex);
    unsigned long long int yy = expand64Bits(ey);
    unsigned long long int zz = expand64Bits(ez);

    // interleaving merge 
    // ´íÎ»merge£¬x×óÒÆ2Î»£¬y×óÒÆ1Î»
    return (xx << 2) | (yy << 1) | zz;
    //return zyxAxisCoding(ex, ey, ez);

}

#endif // !MORTON_H