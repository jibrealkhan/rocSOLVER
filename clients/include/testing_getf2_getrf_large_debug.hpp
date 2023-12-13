/* **************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "client_util.hpp"
#include "clientcommon.hpp"
#include "lapack_host_reference.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

#define GETF2_SPKER_MAX_M 1024 //always <= 1024
#define GETF2_SPKER_MAX_N 256 //always <= 256
#define GETF2_SSKER_MAX_M 512 //always <= 512 and <= GETF2_SPKER_MAX_M
#define GETF2_SSKER_MAX_N 64 //always <= wavefront and <= GETF2_SPKER_MAX_N
#define GETF2_OPTIM_NGRP \
    16, 15, 8, 8, 8, 8, 8, 8, 6, 6, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
#define GETRF_NUM_INTERVALS_REAL 4
#define GETRF_INTERVALS_REAL 64, 512, 1856, 2944
#define GETRF_BLKSIZES_REAL 0, 1, 32, 256, 512
#define GETRF_BATCH_NUM_INTERVALS_REAL 9
#define GETRF_BATCH_INTERVALS_REAL 40, 42, 46, 49, 52, 58, 112, 800, 1024
#define GETRF_BATCH_BLKSIZES_REAL 0, 32, 0, 16, 0, 32, 1, 32, 64, 160
#define GETRF_NPVT_NUM_INTERVALS_REAL 2
#define GETRF_NPVT_INTERVALS_REAL 64, 512
#define GETRF_NPVT_BLKSIZES_REAL 0, -1, 512
#define GETRF_NPVT_BATCH_NUM_INTERVALS_REAL 6
#define GETRF_NPVT_BATCH_INTERVALS_REAL 40, 168, 448, 512, 896, 1408
#define GETRF_NPVT_BATCH_BLKSIZES_REAL 0, -24, -32, -64, 32, 96, 512

#define GETRF_NUM_INTERVALS_COMPLEX 4
#define GETRF_INTERVALS_COMPLEX 64, 512, 1024, 2944
#define GETRF_BLKSIZES_COMPLEX 0, 1, 32, 96, 512
#define GETRF_BATCH_NUM_INTERVALS_COMPLEX 10
#define GETRF_BATCH_INTERVALS_COMPLEX 23, 28, 30, 32, 40, 48, 56, 64, 768, 1024
#define GETRF_BATCH_BLKSIZES_COMPLEX 0, 16, 0, 1, 24, 16, 24, 16, 48, 64, 160
#define GETRF_NPVT_NUM_INTERVALS_COMPLEX 2
#define GETRF_NPVT_INTERVALS_COMPLEX 64, 512
#define GETRF_NPVT_BLKSIZES_COMPLEX 0, -1, 512
#define GETRF_NPVT_BATCH_NUM_INTERVALS_COMPLEX 5
#define GETRF_NPVT_BATCH_INTERVALS_COMPLEX 20, 32, 42, 512, 1408
#define GETRF_NPVT_BATCH_BLKSIZES_COMPLEX 0, -16, -32, -48, 64, 128

inline rocblas_int get_index(rocblas_int* intervals, rocblas_int max, rocblas_int dim)
{
    rocblas_int i;

    for(i = 0; i < max; ++i)
    {
        if(dim <= intervals[i])
            break;
    }

    return i;
}

/** This function returns the outer block size based on defined variables
    tunable by the user (defined in ideal_sizes.hpp) **/
template <bool ISBATCHED, typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
rocblas_int getrf_get_blksize(rocblas_int dim, const bool pivot)
{
    rocblas_int blk;

    if(ISBATCHED)
    {
        if(pivot)
        {
            rocblas_int size[] = {GETRF_BATCH_BLKSIZES_REAL};
            rocblas_int intervals[] = {GETRF_BATCH_INTERVALS_REAL};
            rocblas_int max = GETRF_BATCH_NUM_INTERVALS_REAL;
            blk = size[get_index(intervals, max, dim)];
        }
        else
        {
            rocblas_int size[] = {GETRF_NPVT_BATCH_BLKSIZES_REAL};
            rocblas_int intervals[] = {GETRF_NPVT_BATCH_INTERVALS_REAL};
            rocblas_int max = GETRF_NPVT_BATCH_NUM_INTERVALS_REAL;
            blk = size[get_index(intervals, max, dim)];
        }
    }
    else
    {
        if(pivot)
        {
            rocblas_int size[] = {GETRF_BLKSIZES_REAL};
            rocblas_int intervals[] = {GETRF_INTERVALS_REAL};
            rocblas_int max = GETRF_NUM_INTERVALS_REAL;
            blk = size[get_index(intervals, max, dim)];
        }
        else
        {
            rocblas_int size[] = {GETRF_NPVT_BLKSIZES_REAL};
            rocblas_int intervals[] = {GETRF_NPVT_INTERVALS_REAL};
            rocblas_int max = GETRF_NPVT_NUM_INTERVALS_REAL;
            blk = size[get_index(intervals, max, dim)];
        }
    }

    if(blk == 1 || blk == -1)
        blk *= dim;

    return blk;
}

/** Complex type version **/
template <bool ISBATCHED, typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
rocblas_int getrf_get_blksize(rocblas_int dim, const bool pivot)
{
    rocblas_int blk;

    if(ISBATCHED)
    {
        if(pivot)
        {
            rocblas_int size[] = {GETRF_BATCH_BLKSIZES_COMPLEX};
            rocblas_int intervals[] = {GETRF_BATCH_INTERVALS_COMPLEX};
            rocblas_int max = GETRF_BATCH_NUM_INTERVALS_COMPLEX;
            blk = size[get_index(intervals, max, dim)];
        }
        else
        {
            rocblas_int size[] = {GETRF_NPVT_BATCH_BLKSIZES_COMPLEX};
            rocblas_int intervals[] = {GETRF_NPVT_BATCH_INTERVALS_COMPLEX};
            rocblas_int max = GETRF_NPVT_BATCH_NUM_INTERVALS_COMPLEX;
            blk = size[get_index(intervals, max, dim)];
        }
    }
    else
    {
        if(pivot)
        {
            rocblas_int size[] = {GETRF_BLKSIZES_COMPLEX};
            rocblas_int intervals[] = {GETRF_INTERVALS_COMPLEX};
            rocblas_int max = GETRF_NUM_INTERVALS_COMPLEX;
            blk = size[get_index(intervals, max, dim)];
        }
        else
        {
            rocblas_int size[] = {GETRF_NPVT_BLKSIZES_COMPLEX};
            rocblas_int intervals[] = {GETRF_NPVT_INTERVALS_COMPLEX};
            rocblas_int max = GETRF_NPVT_NUM_INTERVALS_COMPLEX;
            blk = size[get_index(intervals, max, dim)];
        }
    }

    if(blk == 1 || blk == -1)
        blk *= dim;

    return blk;
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void getf2_getrf_large_debug_initData(const rocblas_handle handle,
                          const rocblas_int m,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Ud& dIpiv,
                          const rocblas_stride stP,
                          Ud& dInfo,
                          const rocblas_int bc,
                          Th& hA,
                          Uh& hIpiv,
                          Uh& hInfo,
                          const bool singular)
{
    if(CPU)
    {
        T tmp;
        rocblas_init<T>(hA, true);

        for(rocblas_int b = 0; b < bc; ++b)
        {
            // scale A to avoid singularities
            for(rocblas_int i = 0; i < m; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] += 400;
                    else
                        hA[b][i + j * lda] -= 4;
                }
            }

            // shuffle rows to test pivoting
            // always the same permuation for debugging purposes
            for(rocblas_int i = 0; i < m / 2; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    tmp = hA[b][i + j * lda];
                    hA[b][i + j * lda] = hA[b][m - 1 - i + j * lda];
                    hA[b][m - 1 - i + j * lda] = tmp;
                }
            }

            if(singular && (b == bc / 4 || b == bc / 2 || b == bc - 1))
            {
                // When required, add some singularities
                // (always the same elements for debugging purposes).
                // The algorithm must detect the first zero pivot in those
                // matrices in the batch that are singular
                rocblas_int j = n / 4 + b;
                j -= (j / n) * n;
                for(rocblas_int i = 0; i < m; i++)
                    hA[b][i + j * lda] = 0;
                j = n / 2 + b;
                j -= (j / n) * n;
                for(rocblas_int i = 0; i < m; i++)
                    hA[b][i + j * lda] = 0;
                j = n - 1 + b;
                j -= (j / n) * n;
                for(rocblas_int i = 0; i < m; i++)
                    hA[b][i + j * lda] = 0;
            }
        }
    }

    if(GPU)
    {
        // now copy data to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <bool STRIDED, bool GETRF, typename T, typename Td, typename Ud, typename Th, typename Uh>
void getf2_getrf_large_debug_getError(const rocblas_handle handle,
                          const rocblas_int m,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Ud& dIpiv,
                          const rocblas_stride stP,
                          Ud& dInfo,
                          const rocblas_int bc,
                          Th& hA,
                          Th& hARes,
                          Uh& hIpiv,
                          Uh& hIpivRes,
                          Uh& hInfo,
                          Uh& hInfoRes,
                          double* max_err,
                          const bool singular)
{
    // input data initialization
    getf2_getrf_large_debug_initData<true, true, T>(handle, m, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA,
                                        hIpiv, hInfo, singular);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_getf2_getrf(STRIDED, GETRF, handle, m, n, dA.data(), lda, stA,
                                              dIpiv.data(), stP, dInfo.data(), bc));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hIpivRes.transfer_from(dIpiv));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));



    // CPU lapack
    for(rocblas_int b = 0; b < bc; ++b)  // Probably here too
    {
        GETRF ? cpu_getrf(m, n, hA[b], lda, hIpiv[b], hInfo[b])
              : cpu_getf2(m, n, hA[b], lda, hIpiv[b], hInfo[b]);
    }

    // expecting original matrix to be non-singular
    // error is ||hA - hARes|| / ||hA|| (ideally ||LU - Lres Ures|| / ||LU||)
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    double err;
    *max_err = 0;


    // size of outer blocks
    rocblas_int dim = min(m, n);
    rocblas_int blk = getrf_get_blksize<false, T>(dim, true);
    rocblas_int jb;
    // A loop that iterates over the blocks
    // Need to make changes to the loop with the entry point of the blocks
    for(rocblas_int j = 0; j < dim; j += blk) // Probably here.
    {   
        jb = min(dim - j, blk);
        // j + j * lda because the elements are at the diagnol
        err = norm_error('F', jb, jb, lda, hA[0] + j + j * lda, hARes[0] + j + j * lda);
        std::cout<<"The error inside the block is "<<err<<std::endl;
        *max_err = err > *max_err ? err : *max_err;
    }
}

// Function for performance data.
template <bool STRIDED, bool GETRF, typename T, typename Td, typename Ud, typename Th, typename Uh>
void getf2_getrf_large_debug_getPerfData(const rocblas_handle handle,
                             const rocblas_int m,
                             const rocblas_int n,
                             Td& dA,
                             const rocblas_int lda,
                             const rocblas_stride stA,
                             Ud& dIpiv,
                             const rocblas_stride stP,
                             Ud& dInfo,
                             const rocblas_int bc,
                             Th& hA,
                             Uh& hIpiv,
                             Uh& hInfo,
                             double* gpu_time_used,
                             double* cpu_time_used,
                             const rocblas_int hot_calls,
                             const int profile,
                             const bool profile_kernels,
                             const bool perf,
                             const bool singular)
{
    *cpu_time_used = nan(""); // no timing on cpu-lapack execution
    *gpu_time_used = nan(""); // no timing on gpu-lapack execution
}

template <bool BATCHED, bool STRIDED, bool GETRF, typename T>
void testing_getf2_getrf_large_debug(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int m = argus.get<rocblas_int>("m");
    rocblas_int n = argus.get<rocblas_int>("n", m);
    rocblas_int lda = argus.get<rocblas_int>("lda", m);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stP = argus.get<rocblas_stride>("strideP", min(m, n));

    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stARes = (argus.unit_check || argus.norm_check) ? stA : 0;
    rocblas_stride stPRes = (argus.unit_check || argus.norm_check) ? stP : 0;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_P = size_t(min(m, n));
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_PRes = (argus.unit_check || argus.norm_check) ? size_P : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || lda < m || bc < 0);
    if(invalid_size)
    {
        // if(BATCHED)
        //     EXPECT_ROCBLAS_STATUS(
        //         rocsolver_getf2_getrf(STRIDED, GETRF, handle, m, n, (T* const*)nullptr, lda, stA,
        //                               (rocblas_int*)nullptr, stP, (rocblas_int*)nullptr, bc),
        //         rocblas_status_invalid_size);
        // else
        //     EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED, GETRF, handle, m, n, (T*)nullptr,
        //                                                 lda, stA, (rocblas_int*)nullptr, stP,
        //                                                 (rocblas_int*)nullptr, bc),
        //                           rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query || !USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        if(BATCHED)
            CHECK_ALLOC_QUERY(rocsolver_getf2_getrf(STRIDED, GETRF, handle, m, n, (T* const*)nullptr,
                                                    lda, stA, (rocblas_int*)nullptr, stP,
                                                    (rocblas_int*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_getf2_getrf(STRIDED, GETRF, handle, m, n, (T*)nullptr, lda,
                                                    stA, (rocblas_int*)nullptr, stP,
                                                    (rocblas_int*)nullptr, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));
        if(argus.mem_query)
        {
            rocsolver_bench_inform(inform_mem_query, size);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_device_memory_size(handle, size));
    }

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hARes(size_ARes, 1, bc);
        host_strided_batch_vector<rocblas_int> hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<rocblas_int> hIpivRes(size_PRes, 1, stPRes, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_strided_batch_vector<rocblas_int> dIpiv(size_P, 1, stP, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());

        // check quick return
        if(m == 0 || n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED, GETRF, handle, m, n, dA.data(), lda,
                                                        stA, dIpiv.data(), stP, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            getf2_getrf_large_debug_getError<STRIDED, GETRF, T>(handle, m, n, dA, lda, stA, dIpiv, stP, dInfo,
                                                    bc, hA, hARes, hIpiv, hIpivRes, hInfo, hInfoRes,
                                                    &max_error, argus.singular);

        // collect performance data
        if(argus.timing)
            getf2_getrf_large_debug_getPerfData<STRIDED, GETRF, T>(
                handle, m, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA, hIpiv, hInfo, &gpu_time_used,
                &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels, argus.perf,
                argus.singular);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<rocblas_int> hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<rocblas_int> hIpivRes(size_PRes, 1, stPRes, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<rocblas_int> dIpiv(size_P, 1, stP, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());

        // check quick return
        if(m == 0 || n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED, GETRF, handle, m, n, dA.data(), lda,
                                                        stA, dIpiv.data(), stP, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            getf2_getrf_large_debug_getError<STRIDED, GETRF, T>(handle, m, n, dA, lda, stA, dIpiv, stP, dInfo,
                                                    bc, hA, hARes, hIpiv, hIpivRes, hInfo, hInfoRes,
                                                    &max_error, argus.singular);

        // collect performance data
        if(argus.timing)
            getf2_getrf_large_debug_getPerfData<STRIDED, GETRF, T>(
                handle, m, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA, hIpiv, hInfo, &gpu_time_used,
                &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels, argus.perf,
                argus.singular);
    }

    // validate results for rocsolver-test
    // using min(m,n) * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, min(m, n));

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("m", "n", "lda", "strideP", "batch_c");
                rocsolver_bench_output(m, n, lda, stP, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("m", "n", "lda", "strideA", "strideP", "batch_c");
                rocsolver_bench_output(m, n, lda, stA, stP, bc);
            }
            else
            {
                rocsolver_bench_output("m", "n", "lda");
                rocsolver_bench_output(m, n, lda);
            }
            rocsolver_bench_header("Results:");
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us", "error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, max_error);
            }
            else
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us");
                rocsolver_bench_output(cpu_time_used, gpu_time_used);
            }
            rocsolver_bench_endl();
        }
        else
        {
            if(argus.norm_check)
                rocsolver_bench_output(gpu_time_used, max_error);
            else
                rocsolver_bench_output(gpu_time_used);
        }
    }

    // ensure all arguments were consumed
    argus.validate_consumed();
}

