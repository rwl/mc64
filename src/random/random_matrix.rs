//! Copyright (C) 2014 Science and Technology Facilities Council (STFC)
//!
//! This is a reimplementation of similar functionality to YM11 based on the
//! description in the "Method" section of the YM11 documentation.
//!
//! FIXME: I don't think the positive definite case is implemented as per doc yet!

use std::cmp::min;

use crate::random::RandomState;

#[derive(Copy, Clone)]
pub enum MatrixType {
    UNSPECIFIED = 0,
    RealRect = 1,
    // RealUnsym = 2,
    RealSymPsdef = 3,
    RealSymIndef = 4,
    RealSkew = 6,
}

// Error constants
// pub const ERROR_ALLOCATION: i32 = -1; // Allocation failed
pub const ERROR_ARG: i32 = -3; // m, n or nnz < 1
pub const ERROR_NONSQUARE: i32 = -4; // m!=n contradicts matrix_type
pub const ERROR_SINGULAR: i32 = -5;

/// Generate a random m x n matrix with nnz non-zeroes.
/// User can additionally specify a symmetric matrix (requires m==n), forced
/// non-singularity, and the sorting of entries within columns.
///
/// FIXME: This routine will be slow if we're asked for a (near) dense matrix
/// In this case, we might be better served by finding holes or using the first
/// part of random permutations
pub fn random_matrix_generate(
    state: &mut RandomState,
    matrix_type: MatrixType,
    m: usize,
    n: usize,
    nnz: usize,
    ptr: &mut [usize],       // length n+1
    row: &mut [usize],       // length nnz
    val: Option<&mut [f64]>, // length nnz if provided
    nonsingular: Option<bool>,
    sort: Option<bool>,
) -> i32 {
    // Generate local logical flags
    let lnonsingular = nonsingular.unwrap_or(false);
    let lsort = sort.unwrap_or(false);

    // Handle matrix type
    let lsymmetric = match matrix_type {
        MatrixType::UNSPECIFIED | MatrixType::RealRect => false,
        // MatrixType::RealUnsym => {
        //     if m != n {
        //         // Matrix is not square - did user mean SPRAL_MATRIX_REAL_RECT?
        //         return ERROR_NONSQUARE;
        //     }
        //     false
        // }
        MatrixType::RealSymPsdef | MatrixType::RealSymIndef | MatrixType::RealSkew => {
            if m != n {
                // Matrix is not square - did user mean SPRAL_MATRIX_REAL_RECT?
                return ERROR_NONSQUARE;
            }
            true
        }
    };

    // Check args
    if m < 1 || n < 1 || nnz < 1 {
        // Args out of range
        return ERROR_ARG;
    }

    if (lsymmetric && (n * (n + 1) / 2 < nnz)) || (!lsymmetric && ((m * n) < nnz)) {
        // Too many non-zeroes for matrix
        return ERROR_ARG;
    }

    if lnonsingular && (nnz < usize::min(m, n)) {
        // Requested a non-singular matrix, but not enough non-zeroes
        return ERROR_SINGULAR;
    }

    // Allocate non-zeroes to columns
    let mut cnt = vec![0; n];
    let mut rperm = Vec::new();
    let mut cperm = Vec::new();

    if lsymmetric {
        // In symmetric case, structural non-singularity is guaranteed by adding
        // the diagonal
        if lnonsingular {
            rperm = vec![0; m];
            cperm = vec![0; n];

            // To be consistent with unsymmetric case, we satisfy the following
            // through identity permutations:
            // If cperm(i)<=min(m,n) then that column has a structural non-zero
            // in position rperm(cperm(i))
            for i in 0..n {
                rperm[i] = i + 1;
                cperm[i] = i + 1;
            }

            // Note: In Rust we're using 0-indexed arrays but keeping 1-indexed values
            // to stay consistent with the original Fortran
            for i in 0..n {
                cnt[cperm[i] - 1] += 1;
            }
        }

        // Generate column assignments of remaining entries
        let mut ii = nnz;
        if lnonsingular {
            ii = nnz - usize::min(m, n);
        }

        for _ in 0..ii {
            let mut j = random_sym_wt_integer(state, n) - 1;
            while cnt[j] >= (m - j) {
                j = random_sym_wt_integer(state, n) - 1;
            }
            cnt[j] += 1;
        }
    } else {
        // If we force (structural) non-singularity, generate locations and
        // add to column counts
        if lnonsingular {
            rperm = vec![0; m];
            cperm = vec![0; n];

            // We generate random permutations of rows and columns
            // We use the first min(m,n) of each permutation
            // If cperm(i)<=min(m,n) then that column has a structural non-zero
            // in position rperm(cperm(i))
            random_perm(state, m, &mut rperm);
            random_perm(state, n, &mut cperm);

            for i in 0..n {
                if cperm[i] <= min(m, n) {
                    cnt[i] = 1;
                }
            }
        }

        // Generate column assignments of remaining entries
        let mut ii = nnz;
        if lnonsingular {
            ii = nnz - min(m, n);
        }

        for _ in 0..ii {
            let mut j = state.random_integer(n);
            while cnt[j] >= m {
                j = state.random_integer(n);
            }
            cnt[j] += 1;
        }

        // Debug check in Fortran
        let sum_cnt: usize = cnt.iter().sum();
        assert_eq!(sum_cnt, nnz);
    }

    // Determine row values
    let mut rused = vec![false; m];
    ptr[0] = 0;

    for i in 0..n {
        // Determine end of col
        ptr[i + 1] = ptr[i] + cnt[i];
        let mut jj = ptr[i];

        // Add non-singular entry if required
        if lnonsingular {
            if cperm[i] <= min(m, n) {
                let k = rperm[cperm[i] - 1] - 1;
                row[jj] = k;
                rused[k] = true;
                jj += 1;
            }
        }

        // Add normal entries
        let mut minidx = 1;
        if lsymmetric {
            minidx = i + 1;
        }

        while jj < (ptr[i + 1] - 1) {
            let mut k = random_integer_in_range(state, minidx, m) - 1;
            while rused[k] {
                k = random_integer_in_range(state, minidx, m) - 1;
            }
            row[jj] = k + 1; // Store as 1-indexed
            rused[k] = true;
            jj += 1;
        }

        // Reset rused(:)
        for jj in ptr[i]..ptr[i + 1] {
            rused[row[jj]] = false;
        }
    }

    // Optionally, sort
    if lsort {
        dbl_tr_sort(m, n, ptr, row)
    }

    // Determine values
    if let Some(val_slice) = val {
        for jj in 0..ptr[n] {
            val_slice[jj] = state.random_real(None);
        }
    }

    0
}

/// Returns a random number in range [0,n-1] weighted by number of entries in
/// lower half triangle
///
/// Do this by only accepting randomly generated column with frequency
/// proportional to number of entries in column
fn random_sym_wt_integer(state: &mut RandomState, n: usize) -> usize {
    let mut r1 = state.random_integer(n);
    let mut r2 = state.random_integer(n);

    while r2 < r1 {
        r1 = state.random_integer(n);
        r2 = state.random_integer(n);
    }

    r1
}

/// Returns a random integer in range [minv,maxv] inclusive
fn random_integer_in_range(state: &mut RandomState, minv: usize, maxv: usize) -> usize {
    minv + state.random_integer(maxv - minv + 1)
}

/// Returns a random permutation of length n in perm using Knuth shuffles
fn random_perm(state: &mut RandomState, n: usize, perm: &mut [usize]) {
    // Initialize with identity
    for i in 0..n {
        perm[i] = i;
    }

    // Go through positions i=1:n-1
    for i in 0..n {
        // Swap perm(i) with perm(j), where j is random in [i:n]
        let j = random_integer_in_range(state, i, n - 1);
        let temp = perm[i];
        perm[i] = perm[j];
        perm[j] = temp;
    }
}

/// Sort a matrix's columns to increase row order
///
/// Uses a double transpose algorithm to do so
/// Based on a modified version of subroutine from HSL_MC78
fn dbl_tr_sort(m: usize, n: usize, ptr: &mut [usize], row: &mut [usize]) {
    let mut ptr2 = vec![0; m + 2];

    // Count number of entries in each row. ptr2[i+2] = #entries in row i
    for node in 0..n {
        for ii in ptr[node]..ptr[node + 1] {
            let j = row[ii];
            ptr2[j + 2] += 1;
        }
    }

    // Determine row starts. ptr2[i+1] = start of row i
    ptr2[0] = 1;
    ptr2[1] = 1;
    for i in 0..m {
        ptr2[i + 2] = ptr2[i + 1] + ptr2[i + 2];
    }

    let total_entries = ptr2[m + 1] - 1;
    let mut col = vec![0; total_entries];

    // Now fill in col array
    for node in 0..n {
        for ii in ptr[node]..ptr[node + 1] {
            let j = row[ii]; // row entry
            col[ptr2[j + 1] - 1] = node + 1;
            ptr2[j + 1] += 1;
        }
    }

    // Finally transpose back into nodes
    let mut nptr = vec![0; n];
    for i in 0..n {
        nptr[i] = ptr[i];
    }

    for i in 0..m {
        for jj in (ptr2[i] - 1)..(ptr2[i + 1] - 1) {
            let node = col[jj] - 1;
            row[nptr[node]] = i + 1;
            nptr[node] += 1;
        }
    }
}
