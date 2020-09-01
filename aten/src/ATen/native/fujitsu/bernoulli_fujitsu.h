#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
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


struct StreamStatePtr_fujitsu{
    int METHOD;                /* METHOD 0:Xorshift, 1:MCG31 */
    uint32_t *s;               /* seed */
};


inline uint32_t rotl(const uint32_t x, int k) {
	return (x << k) | (x >> (32 - k));
}


inline svuint32_t rotlsve(svuint32_t x, int k, svbool_t pg) {
	svuint32_t rotl_1 = svlsl_z(pg, x, k);
	svuint32_t rotl_2 = svlsr_z(pg, x, (32 - k));
	svuint32_t rotl_3 = svorr_z(pg, rotl_1, rotl_2);
	return rotl_3;
}

/**
* Step forward one sequence for random number generation.
*/

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
* sve version function of next.
*/

svuint32_t nextsve(uint32_t s[64]) {
	int64_t j = 0;
	svbool_t pg_next = svwhilelt_b32_s64(j, 64);
		svuint32_t s_0_sve = svld1(pg_next, &s[j]);
		svuint32_t s_1_sve = svld1(pg_next, &s[j+16]);
		svuint32_t s_2_sve = svld1(pg_next, &s[j+32]);
		svuint32_t s_3_sve = svld1(pg_next, &s[j+48]);
		svuint32_t add_0_3 = svadd_u32_z(pg_next, s_0_sve, s_3_sve);
		svuint32_t rotl_sve = rotlsve(add_0_3, 7, pg_next);
		svuint32_t result_sve = svadd_z(pg_next, rotl_sve, s_0_sve);
		svuint32_t t_sve = svlsl_z(pg_next, s_1_sve, 9);
		s_2_sve = sveor_z(pg_next, s_2_sve, s_0_sve);
		s_3_sve = sveor_z(pg_next, s_3_sve, s_1_sve);
		s_1_sve = sveor_z(pg_next, s_1_sve, s_2_sve);
		s_0_sve = sveor_z(pg_next, s_0_sve, s_3_sve);
		s_2_sve = sveor_z(pg_next, s_2_sve, t_sve);
		s_3_sve = rotlsve(s_3_sve, 11, pg_next);
		svst1(pg_next, &s[j], s_0_sve);
		svst1(pg_next, &s[j+16], s_1_sve);
		svst1(pg_next, &s[j+32], s_2_sve);
		svst1(pg_next, &s[j+48], s_3_sve);
		return result_sve;
}

/**
* Preparation for sve calculation.
*
* @param[out] stream
* @param[in] METHOD 0:Xorshift, 1:MCG31
* @param[in] seed The initial state for random number generation.
*/

void NewStream_fujitsu(
    struct StreamStatePtr_fujitsu *stream, 
    int METHOD, 
    int seed
){
    stream->METHOD = METHOD;
    stream->s=(uint32_t*)malloc(sizeof(uint32_t)*64);

    uint32_t s_sub[4];
    stream->s[0] = s_sub[0]=seed;
    stream->s[16] = s_sub[1]=362436069;
    stream->s[32] = s_sub[2]=521288629;
    stream->s[48] = s_sub[3]=88675123;

	uint32_t JUMP[] = { 0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b };
	for(int i = 0; i < 15 ; i++){
		uint32_t s0 = 0;
		uint32_t s1 = 0;
		uint32_t s2 = 0;
		uint32_t s3 = 0;
		for(int j = 0; j < sizeof JUMP / sizeof *JUMP; j++){
			for(int b = 0; b < 32; b++) {
				if (JUMP[j] & UINT32_C(1) << b) {
					s0 ^= s_sub[0];
					s1 ^= s_sub[1];
					s2 ^= s_sub[2];
					s3 ^= s_sub[3];
				}
				next(s_sub);
			}
 		}
        stream->s[i+1] = s_sub[0] = s0;
        stream->s[i+17] = s_sub[1] = s1;
        stream->s[i+33] = s_sub[2] = s2;
        stream->s[i+49] = s_sub[3] = s3;
	}
}


/**
* To support multi-threaded computation.
*
* @param[out] stream.s 
* @param[in] tmp, tmp_sub[64], s_sub[4] A temporary place to save data. 
*/

void SkipAheadStream_fujitsu(
    struct StreamStatePtr_fujitsu stream, 
    int begin
){
    int64_t i = 0;
    svuint32_t tmp;
    uint32_t tmp_sub[64] = {0};
    uint32_t s_sub[4] = {0};
    int64_t N = begin/16;
    int suffix_1 = begin%16;
    int suffix_2 = (16-(begin%16))%16;

    for(int i=0; i < N; i++){
	    tmp = nextsve(stream.s);
    }

//
//Adjust if begin is not a multiple of 16;
//Move the start position by suffix_1;
//
    for(int i = 0; i < 16; i++){	
	    tmp_sub[i] = stream.s[(suffix_1+i)%16];
	    tmp_sub[i+16] = stream.s[((suffix_1+i)%16)+16];
	    tmp_sub[i+32] = stream.s[((suffix_1+i)%16)+32];
	    tmp_sub[i+48] = stream.s[((suffix_1+i)%16)+48];
    }

//
//Adjust if begin is not a multiple of 16;
//Only a part of the array is advanced by next function.
//
    for(int i = 0; i < suffix_1; i++){
        s_sub[0] = tmp_sub[i+suffix_2];
        s_sub[1] = tmp_sub[i+16+suffix_2];
        s_sub[2] = tmp_sub[i+32+suffix_2];
        s_sub[3] = tmp_sub[i+48+suffix_2];
        next(s_sub);
        tmp_sub[i+suffix_2] = s_sub[0];
        tmp_sub[i+16+suffix_2] = s_sub[1];
        tmp_sub[i+32+suffix_2] = s_sub[2];
        tmp_sub[i+48+suffix_2] = s_sub[3];   
    }
   
    for(int i = 0; i < 64; i++){
	stream.s[i] = tmp_sub[i];
    }
}


/**
* Generation of random numbers by Bernoulli distribution.
*  
* @param[out] r Value (0 or 1) genereted by the Bernoulli function.
* @param[in] METHOD 0:Xorshift, 1:MCG31
* @param[in] StreamStatePtr_fujitsu stream 
* @param[in] N Number of random numbers to generate.
* @param[in] r Array of random numbers.
* @param[in] p Parameters of the Bernoulli distribution.
* @return 0
*/
int RngBernoulli_fujitsu(
    int METHOD, 
    struct StreamStatePtr_fujitsu stream, 
    int N, 
    int* r, 
    double p
){
    int64_t i = 0;
    svuint32_t tmp;

    // Xorshift random generator
    if(METHOD == 0/* xoshiro128++*/){
        i = 0;
        float32_t UINT32_MAX_float_rec = (float32_t)(1.0/UINT32_MAX); 
        svbool_t pg = svwhilelt_b32_s64(i, N);
        do{
    	    tmp = nextsve(stream.s);
            svfloat32_t tmp_f = svcvt_f32_z(pg, tmp);
            svfloat32_t u_sve = svmulx_z(pg, tmp_f, UINT32_MAX_float_rec);
            svbool_t mask = svcmple(pg, u_sve, p);
            svint32_t r_sve = svdup_s32_z(mask, 1);
            svst1(pg, &r[i], r_sve);
            i += svcntw();
            pg = svwhilelt_b32_s64(i, N);
        }while(svptest_any(svptrue_b32(), pg));	 
    }
    return 0;
}

/**
* Delete stream.
*
* @param[in] StreamStatePtr_fujitsu stream
*/
void DeleteStream_fujitsu(
    struct StreamStatePtr_fujitsu stream
){
    free(stream.s);
}

