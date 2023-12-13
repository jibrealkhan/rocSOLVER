/* **************************************************************************
 * Copyright (C) 2020-2021 Advanced Micro Devices, Inc. All rights reserved.
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

#include "testing_getf2_getrf_large_debug.hpp"
#include "testing_getf2_getrf_npvt.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, int> getrf_large_debug_tuple;

// each matrix_size_range vector is a {m, lda, singular}
// if singular = 1, then the used matrix for the tests is singular

// case when m = n = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> very_large_debug_matrix_size_range = {
    // normal (valid) samples
    {15000, 15000},              //0
    {15001, 15001},              //1
    {15250, 15250},              //2
};

const vector<int> very_large_debug_n_size_range = {15000, 15001, 15250};


Arguments getrf_large_debug_setup_arguments(getrf_large_debug_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    int n_size = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("m", matrix_size[0]);
    arg.set<rocblas_int>("n", n_size);
    arg.set<rocblas_int>("lda", matrix_size[1]);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

class GETF2_GETRF_LARGE_DEBUG : public ::TestWithParam<getrf_large_debug_tuple>
{
protected:
    GETF2_GETRF_LARGE_DEBUG() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = getrf_large_debug_setup_arguments(GetParam());


        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_getf2_getrf_large_debug<BATCHED, STRIDED, true, T>(arg);
    }
};

class GETF2_GETRF_LARGE_DEBUG_NPVT : public ::TestWithParam<getrf_large_debug_tuple>
{
protected:
    GETF2_GETRF_LARGE_DEBUG_NPVT() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = getrf_large_debug_setup_arguments(GetParam());

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_getf2_getrf_npvt<BATCHED, STRIDED, true, T>(arg);
    }
};

// Non Batch Tests
TEST_P(GETF2_GETRF_LARGE_DEBUG, __double_complex){
    run_tests<false, false, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(weekly_lapack,
                         GETF2_GETRF_LARGE_DEBUG,
                         Combine(ValuesIn(very_large_debug_matrix_size_range), ValuesIn(very_large_debug_n_size_range)));