#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
#define A_CONSTANT_1  48271
#define A_CONSTANT_16 1357852417
#define M_CONSTANT    0x7FFFFFFF


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

/*  
Used multiplicative congruential generator (MCG): 
Park, Stephen K.; Miller, Keith W.; Stockmeyer, Paul K. (1988). 
"Technical Correspondence: Response" (PDF). Communications of the ACM. 36 (7): 108?110. 
doi:10.1145/159544.376068.
*/


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
* Step forward one sequence for random number generation with xoshiro128++.
*/

uint32_t next_xorshift(uint32_t s[4]) {
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
* sve version function of next with xoshiro128++.
*/

svuint32_t nextsve_xorshift(uint32_t s[64]) {
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
* Step forward one sequence for random number generation with MCG.
*/

uint32_t next_MCG(uint32_t s[1], uint32_t A_CONSTANT) {
	s[0] = (A_CONSTANT*s[0])&M_CONSTANT;
	return s[0];
}

/**
* sve version function of next with MCG.
*/
svuint32_t  nextsve_MCG(uint32_t s[16], uint32_t A_CONSTANT) {
    int64_t j = 0;
    svbool_t pg = svwhilelt_b32_s64(j, 16);
    svuint32_t s0_sve = svld1(pg, &s[j]);
    svuint32_t u0_sve = svmul_n_u32_z(pg, s0_sve, A_CONSTANT);
    svuint32_t result0_sve = svand_z(pg, u0_sve, (uint32_t)M_CONSTANT);
    svst1(pg, &s[j], result0_sve);
    return result0_sve;
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

    // Initialization 
    if(METHOD == 0/* xoshiro128++*/){
        // memory allocation
        // It is recommended to have more elements than L1 cache.
        // Cache coherency degrades performance when less than L1 cache
        stream->s=(uint32_t*)malloc(sizeof(uint32_t)*4096); // The actual number of required elements is 64.

        // Set seed
        uint32_t s_sub[4];
        stream->s[0] = s_sub[0]=(seed==0)?1:seed;;
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
                    next_xorshift(s_sub);
                }
            }
            stream->s[i+1] = s_sub[0] = s0;
            stream->s[i+17] = s_sub[1] = s1;
            stream->s[i+33] = s_sub[2] = s2;
            stream->s[i+49] = s_sub[3] = s3;
        }
    }else if(METHOD == 1/* MCG*/){
        // It is recommended to have more elements than L1 cache.
        // Cache coherency degrades performance when less than L1 cache
        stream->s=(uint32_t*)malloc(sizeof(uint32_t)*4096); // The actual number of required elements is 16.

        // Set seed
        stream->s[0] = (seed==0)?1:seed;
        for(int i=0; i < 15; i++){
          stream->s[i+1] = next_MCG(stream->s, (uint32_t)A_CONSTANT_1);
        }
        stream->s[0] = (seed==0)?1:seed;
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

    // SkipAheadStream for xoshiro128++ 
    if(stream.METHOD == 0/* xoshiro128++*/){
        uint32_t tmp_sub[64] = {0};
        uint32_t s_sub[4] = {0};
        int64_t N = begin/16;
        int suffix_1 = begin%16;
        int suffix_2 = (16-(begin%16))%16;

        for(int i=0; i < N; i++){
            svuint32_t tmp = nextsve_xorshift(stream.s);
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
            next_xorshift(s_sub);
            tmp_sub[i+suffix_2] = s_sub[0];
            tmp_sub[i+16+suffix_2] = s_sub[1];
            tmp_sub[i+32+suffix_2] = s_sub[2];
            tmp_sub[i+48+suffix_2] = s_sub[3];   
        }
    
        for(int i = 0; i < 64; i++){
            stream.s[i] = tmp_sub[i];
        }

    // SkipAheadStream for MCG 
    }else if(stream.METHOD == 1/* MCG*/){
 
        // Set jump variables
        uint32_t total_element = begin;
        uint32_t jump_num = 21;
        uint32_t jump_constant[21] ={A_CONSTANT_1, 182605793, 1533981633, 773027713, A_CONSTANT_16, 1820286465,
                                     1065532417, 2031450113, 1516957697, 1440079873, 799784961, 1868005377, 
                                     514785281, 1029570561, 2059141121, 1970798593, 1794113537, 1440743425, 
                                     734003201, 1468006401, 788529153/*=A_CONSTANT_1048576*/};
        // Compute
        for(int i_jump=jump_num-1; i_jump>=0; i_jump--){
            int nloop = total_element>>i_jump;
            if(nloop==0)continue;
            for(int i = 0; i < nloop; i++){
                svuint32_t tmp = nextsve_MCG(stream.s, jump_constant[i_jump]);
            }
            total_element -= (nloop<<i_jump);
            if(total_element==0)break;
        }
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
    if(stream.METHOD == 0/* xoshiro128++*/){
        i = 0;
        float32_t UINT32_MAX_float_rec = (float32_t)(1.0/UINT32_MAX); 
        svbool_t pg = svwhilelt_b32_s64(i, N);
        do{
    	    tmp = nextsve_xorshift(stream.s);
            svfloat32_t tmp_f = svcvt_f32_z(pg, tmp);
            svfloat32_t u_sve = svmulx_z(pg, tmp_f, UINT32_MAX_float_rec);
            svbool_t mask = svcmple(pg, u_sve, p);
            svint32_t r_sve = svdup_s32_z(mask, 1);
            svst1(pg, &r[i], r_sve);
            i += svcntw();
            pg = svwhilelt_b32_s64(i, N);
        }while(svptest_any(svptrue_b32(), pg));	 
    }else if(stream.METHOD == 1/* MCG*/){
        i = 0;
        float32_t INT32_MAX_float_rec = (float32_t)(1.0/M_CONSTANT); 
        svbool_t pg = svwhilelt_b32_s64(i, N);
        do{
    	    tmp = nextsve_MCG(stream.s, (uint32_t)A_CONSTANT_16);
            svfloat32_t tmp_f = svcvt_f32_z(pg, tmp);
            svfloat32_t u_sve = svmulx_z(pg, tmp_f, INT32_MAX_float_rec);
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
