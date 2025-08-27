//! Copyright (C) 2014 Science and Technology Facilities Council (STFC)
//!
//! This is a reimplementation of similar functionality to YM11 based on the
//! description in the "Method" section of the YM11 documentation.
//!
//! FIXME: I don't think the positive definite case is implemented as per doc yet!

// Using our previously translated random module
use crate::tests::random::{random_integer32 as random_integer, random_real, RandomState};

// Matrix type constants
pub const UNSPECIFIED: i32 = 0;
pub const REAL_RECT: i32 = 1;
pub const REAL_UNSYM: i32 = 2;
pub const REAL_SYM_PSDEF: i32 = 3;
pub const REAL_SYM_INDEF: i32 = 4;
pub const REAL_SKEW: i32 = 6;

// Error constants
pub const ERROR_ALLOCATION: i32 = -1; // Allocation failed
pub const ERROR_MATRIX_TYPE: i32 = -2; // Bad matrix type
pub const ERROR_ARG: i32 = -3; // m, n or nnz < 1
pub const ERROR_NONSQUARE: i32 = -4; // m!=n contradicts matrix_type
pub const ERROR_SINGULAR: i32 = -5; // request non-singular but nnz<min(m,n)

/// Generate a random m x n matrix with nnz non-zeroes.
/// User can additionally specify a symmetric matrix (requires m==n), forced
/// non-singularity, and the sorting of entries within columns.
pub fn random_matrix_generate(
    state: &mut RandomState,
    matrix_type: i32,
    m: usize,
    n: usize,
    nnz: usize,
    ptr: &mut [usize],       // length n+1
    row: &mut [usize],       // length nnz
    val: Option<&mut [f64]>, // length nnz if provided
    nonsingular: Option<bool>,
    sort: Option<bool>,
) -> (usize, Option<usize>) {
    // (flag, stat)
    // Create temporary 64-bit version of ptr
    let mut ptr64 = Vec::with_capacity(n as usize + 1);
    for _ in 0..=n {
        ptr64.push(0i64);
    }

    // Create temporary 64-bit version of row
    let mut row64 = Vec::with_capacity(nnz);
    for _ in 0..nnz {
        ptr64.push(0i64);
    }

    // Call 64-bit version
    let result = random_matrix_generate64(
        state,
        matrix_type,
        m as i32,
        n as i32,
        nnz as i64,
        &mut ptr64,
        &mut row64,
        val,
        nonsingular,
        sort,
    );

    // Copy back to 32-bit ptr
    for i in 0..=n {
        ptr[i] = ptr64[i] as usize;
    }
    // Copy back to 32-bit row
    for i in 0..nnz {
        row[i] = row64[i] as usize;
    }

    (result.0 as usize, result.1.map(|r| r as usize))
}

// /// Generate a random m x n matrix with nnz non-zeroes.
// /// User can additionally specify a symmetric matrix (requires m==n), forced
// /// non-singularity, and the sorting of entries within columns.
// pub fn random_matrix_generate32(
//     state: &mut RandomState,
//     matrix_type: i32,
//     m: i32,
//     n: i32,
//     nnz: i32,
//     ptr: &mut [i32],         // length n+1
//     row: &mut [i32],         // length nnz
//     val: Option<&mut [f64]>, // length nnz if provided
//     nonsingular: Option<bool>,
//     sort: Option<bool>,
// ) -> (i32, Option<i32>) {
//     // (flag, stat)
//     // Create temporary 64-bit version of ptr
//     let mut ptr64 = Vec::with_capacity(n as usize + 1);
//     for _ in 0..=n {
//         ptr64.push(0i64);
//     }
//
//     // Call 64-bit version
//     let result = random_matrix_generate64(
//         state,
//         matrix_type,
//         m,
//         n,
//         nnz as i64,
//         &mut ptr64,
//         row,
//         val,
//         nonsingular,
//         sort,
//     );
//
//     // Copy back to 32-bit ptr
//     for i in 0..=n as usize {
//         ptr[i] = ptr64[i] as i32;
//     }
//
//     result
// }

/// Generate a random m x n matrix with nnz non-zeroes.
/// User can additionally specify a symmetric matrix (requires m==n), forced
/// non-singularity, and the sorting of entries within columns.
///
/// FIXME: This routine will be slow if we're asked for a (near) dense matrix
/// In this case, we might be better served by finding holes or using the first
/// part of random permutations
pub fn random_matrix_generate64(
    state: &mut RandomState,
    matrix_type: i32,
    m: i32,
    n: i32,
    nnz: i64,
    ptr: &mut [i64],         // length n+1
    row: &mut [i32],         // length nnz
    val: Option<&mut [f64]>, // length nnz if provided
    nonsingular: Option<bool>,
    sort: Option<bool>,
) -> (i32, Option<i32>) {
    // (flag, stat)
    // Initialize return codes
    let mut flag = 0;
    let mut stat = 0;

    // Generate local logical flags
    let lnonsingular = nonsingular.unwrap_or(false);
    let lsort = sort.unwrap_or(false);

    // Handle matrix type
    let lsymmetric = match matrix_type {
        UNSPECIFIED | REAL_RECT => false,
        REAL_UNSYM => {
            if m != n {
                // Matrix is not square - did user mean SPRAL_MATRIX_REAL_RECT?
                return (ERROR_NONSQUARE, Some(0));
            }
            false
        }
        REAL_SYM_PSDEF | REAL_SYM_INDEF | REAL_SKEW => {
            if m != n {
                // Matrix is not square - did user mean SPRAL_MATRIX_REAL_RECT?
                return (ERROR_NONSQUARE, Some(0));
            }
            true
        }
        _ => {
            // COMPLEX or unknown matrix type
            return (ERROR_MATRIX_TYPE, Some(0));
        }
    };

    // Check args
    if m < 1 || n < 1 || nnz < 1 {
        // Args out of range
        return (ERROR_ARG, Some(0));
    }

    if (lsymmetric && (n as i64 * (n as i64 + 1) / 2 < nnz))
        || (!lsymmetric && ((m as i64 * n as i64) < nnz))
    {
        // Too many non-zeroes for matrix
        return (ERROR_ARG, Some(0));
    }

    if lnonsingular && (nnz < std::cmp::min(m, n) as i64) {
        // Requested a non-singular matrix, but not enough non-zeroes
        return (ERROR_SINGULAR, Some(0));
    }

    // Allocate non-zeroes to columns
    let mut cnt = vec![0; n as usize];
    let mut rperm = Vec::new();
    let mut cperm = Vec::new();

    if lsymmetric {
        // In symmetric case, structural non-singularity is guaranteed by adding
        // the diagonal
        if lnonsingular {
            rperm = vec![0; m as usize];
            cperm = vec![0; n as usize];

            // To be consistent with unsymmetric case, we satisfy the following
            // through identity permutations:
            // If cperm(i)<=min(m,n) then that column has a structural non-zero
            // in position rperm(cperm(i))
            for i in 0..n as usize {
                rperm[i] = i as i32 + 1;
                cperm[i] = i as i32 + 1;
            }

            // Note: In Rust we're using 0-indexed arrays but keeping 1-indexed values
            // to stay consistent with the original Fortran
            for i in 0..n as usize {
                cnt[cperm[i] as usize - 1] += 1;
            }
        }

        // Generate column assignments of remaining entries
        let mut ii = nnz;
        if lnonsingular {
            ii = nnz - std::cmp::min(m, n) as i64;
        }

        for _ in 0..ii {
            let mut j = random_sym_wt_integer(state, n) as usize - 1;
            while cnt[j] >= (m - j as i32) {
                j = random_sym_wt_integer(state, n) as usize - 1;
            }
            cnt[j] += 1;
        }
    } else {
        // If we force (structural) non-singularity, generate locations and
        // add to column counts
        if lnonsingular {
            rperm = vec![0; m as usize];
            cperm = vec![0; n as usize];

            // We generate random permutations of rows and columns
            // We use the first min(m,n) of each permutation
            // If cperm(i)<=min(m,n) then that column has a structural non-zero
            // in position rperm(cperm(i))
            random_perm(state, m, &mut rperm);
            random_perm(state, n, &mut cperm);

            for i in 0..n as usize {
                if cperm[i] <= std::cmp::min(m, n) {
                    cnt[i] = 1;
                }
            }
        }

        // Generate column assignments of remaining entries
        let mut ii = nnz;
        if lnonsingular {
            ii = nnz - std::cmp::min(m, n) as i64;
        }

        for _ in 0..ii {
            let mut j = random_integer(state, n) as usize - 1;
            while cnt[j] >= m {
                j = random_integer(state, n) as usize - 1;
            }
            cnt[j] += 1;
        }

        // Debug check in Fortran
        let sum_cnt: i32 = cnt.iter().sum();
        assert_eq!(sum_cnt as i64, nnz);
    }

    // Determine row values
    let mut rused = vec![false; m as usize];
    ptr[0] = 1; // 1-indexed like Fortran

    for i in 0..n as usize {
        // Determine end of col
        ptr[i + 1] = ptr[i] + cnt[i] as i64;
        let mut jj = ptr[i] as usize - 1; // Convert to 0-indexed

        // Add non-singular entry if required
        if lnonsingular {
            if cperm[i] <= std::cmp::min(m, n) {
                let k = rperm[cperm[i] as usize - 1] as usize - 1;
                row[jj] = k as i32 + 1; // Store as 1-indexed
                rused[k] = true;
                jj += 1;
            }
        }

        // Add normal entries
        let mut minidx = 1;
        if lsymmetric {
            minidx = i as i32 + 1;
        }

        while jj < (ptr[i + 1] - 1) as usize {
            let mut k = random_integer_in_range(state, minidx, m) as usize - 1;
            while rused[k] {
                k = random_integer_in_range(state, minidx, m) as usize - 1;
            }
            row[jj] = k as i32 + 1; // Store as 1-indexed
            rused[k] = true;
            jj += 1;
        }

        // Reset rused(:)
        for jj in (ptr[i] - 1) as usize..(ptr[i + 1] - 1) as usize {
            rused[row[jj] as usize - 1] = false;
        }
    }

    // Optionally, sort
    if lsort {
        match dbl_tr_sort(m, n, ptr, row) {
            Ok(_) => {}
            Err(e) => {
                return (ERROR_ALLOCATION, Some(e));
            }
        }
    }

    // Determine values
    if let Some(val_slice) = val {
        for jj in 0..(ptr[n as usize] - 1) as usize {
            val_slice[jj] = random_real(state, None);
        }
    }

    (flag, Some(stat))
}

/// Returns a random number in range [1,n] weighted by number of entries in
/// lower half triangle
///
/// Do this by only accepting randomly generated column with frequency
/// proportional to number of entries in column
fn random_sym_wt_integer(state: &mut RandomState, n: i32) -> i32 {
    let mut r1 = random_integer(state, n);
    let mut r2 = random_integer(state, n);

    while r2 < r1 {
        r1 = random_integer(state, n);
        r2 = random_integer(state, n);
    }

    r1
}

/// Returns a random integer in range [minv,maxv] inclusive
fn random_integer_in_range(state: &mut RandomState, minv: i32, maxv: i32) -> i32 {
    minv + random_integer(state, maxv - minv + 1) - 1
}

/// Returns a random permutation of length n in perm using Knuth shuffles
fn random_perm(state: &mut RandomState, n: i32, perm: &mut [i32]) {
    // Initialize with identity
    for i in 0..n as usize {
        perm[i] = i as i32 + 1;
    }

    // Go through positions i=1:n-1
    for i in 0..(n - 1) as usize {
        // Swap perm(i) with perm(j), where j is random in [i:n]
        let j = random_integer_in_range(state, i as i32 + 1, n) as usize - 1;
        let temp = perm[i];
        perm[i] = perm[j];
        perm[j] = temp;
    }
}

/// Sort a matrix's columns to increase row order
///
/// Uses a double transpose algorithm to do so
/// Based on a modified version of subroutine from HSL_MC78
fn dbl_tr_sort(m: i32, n: i32, ptr: &mut [i64], row: &mut [i32]) -> Result<(), i32> {
    let mut ptr2 = vec![0i64; m as usize + 2];

    // Count number of entries in each row. ptr2[i+2] = #entries in row i
    for node in 0..n as usize {
        for ii in (ptr[node] - 1) as usize..(ptr[node + 1] - 1) as usize {
            let j = row[ii] as usize - 1;
            ptr2[j + 2] += 1;
        }
    }

    // Determine row starts. ptr2[i+1] = start of row i
    ptr2[0] = 1;
    ptr2[1] = 1;
    for i in 0..m as usize {
        ptr2[i + 2] = ptr2[i + 1] + ptr2[i + 2];
    }

    let total_entries = (ptr2[m as usize + 1] - 1) as usize;
    let mut col = vec![0i32; total_entries];

    // Now fill in col array
    for node in 0..n as usize {
        for ii in (ptr[node] - 1) as usize..(ptr[node + 1] - 1) as usize {
            let j = row[ii] as usize - 1; // row entry
            col[ptr2[j + 1] as usize - 1] = node as i32 + 1;
            ptr2[j + 1] += 1;
        }
    }

    // Finally transpose back into nodes
    let mut nptr = vec![0i64; n as usize];
    for i in 0..n as usize {
        nptr[i] = ptr[i];
    }

    for i in 0..m as usize {
        for jj in (ptr2[i] - 1) as usize..(ptr2[i + 1] - 1) as usize {
            let node = col[jj] as usize - 1;
            row[nptr[node] as usize - 1] = i as i32 + 1;
            nptr[node] += 1;
        }
    }

    Ok(())
}
