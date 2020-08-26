#include <stdio.h>
#include <stdint.h>

#define A_CONSTANT 1132489760
#define M_CONSTANT ((unsigned int)(1<<31)-1)

/*  Written in 2019 by David Blackman and Sebastiano Vigna (vigna@acm.org)

http://vigna.di.unimi.it/

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */


/* This is xoshiro128++ 1.0, one of our 32-bit all-purpose, rock-solid
   generators. It has excellent speed, a state size (128 bits) that is
   large enough for mild parallelism, and it passes all tests we are aware
   of.

   For generating just single-precision (i.e., 32-bit) floating-point
   numbers, xoshiro128+ is even faster.

   The state must be seeded so that it is not everywhere zero. */


uint32_t rotl(const uint32_t x, int k) {
	return (x << k) | (x >> (32 - k));
}


uint32_t next(uint32_t s[4]) {
	const uint32_t result = rotl(s[0] + s[3], 7) + s[0];

	const uint32_t t = s[1] << 9;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;

	s[3] = rotl(s[3], 11);

	return result;
}

/**
* ベルヌーイ分布関数
*
* 乱数を生成し、ベルヌーイ分布を作成
* @param[out] r ベルヌーイ分布の分散ランダム値
* @param[in] METHOD 0:Xorshift, 1:MCG31
* @param[in] seed 乱数のseed
* @param[in] N 要素数
* @param[in] p 確率
* @return エラーコード
*/
int RngBernoulli_fujitsu(
    int METHOD,
    int seed,
    int N,
    int* r,
    double p
){
    int32_t rand, i, output;
    uint32_t tmp;
    uint32_t s[4];
    float u;

    // Xorshift random generator
    if(METHOD == 0/* xoshiro128++*/){
        // seed setting
        s[0]=seed;
        s[1]=362436069;
        s[2]=521288629;
        s[3]=88675123;

        // random number generated
        for(i=0; i<N; i++){
            tmp = next(s);
            u = (float)(tmp)/UINT32_MAX;
            output=0;
            if(u<=p)output=1;
            r[i]=output;
        }
    }
    // MGC31 random generator -> bad root
    else if(METHOD == 1/* MGC31 */){
        rand = seed;
        for(i=0; i<N; i++){
            rand = (A_CONSTANT*rand)%M_CONSTANT;
            u = (float)(rand)/M_CONSTANT;
            output=0;
            if(u<=p)output=1;
            r[i]=output;
        }
    }
    return 0;
}
