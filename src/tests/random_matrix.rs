use crate::random::{
    random_matrix_generate, MatrixType, RandomState, ERROR_ARG, ERROR_NONSQUARE, ERROR_SINGULAR,
};
use std::cmp::min;

// const ERROR_MATRIX_TYPE: i32 = -2;
// const ERROR_ARG: i32 = -3;
// const ERROR_NONSQUARE: i32 = -4;
// const ERROR_SINGULAR: i32 = -5;
//
// // In spral C library, SPRAL_MATRIX_UNSPECIFIED is -1
// const UNSPECIFIED: i32 = -1;

#[test]
fn test_errors() {
    let mut state = RandomState::default();
    let mut ptr: Vec<usize> = vec![0; 101];
    let mut row: Vec<usize> = vec![0; 1000];
    let mut m; // = 100;
    let mut n = 100;
    let mut nnz = 1000;

    // Test bad args
    println!(" * Testing m < 1.............................");
    m = 0;
    let flag = random_matrix_generate(
        &mut state,
        MatrixType::UNSPECIFIED,
        m,
        n,
        nnz,
        &mut ptr,
        &mut row,
        None,
        None,
        None,
    );
    assert_eq!(flag, ERROR_ARG);
    m = 100; // restore

    println!(" * Testing n < 1.............................");
    n = 0;
    let flag = random_matrix_generate(
        &mut state,
        MatrixType::UNSPECIFIED,
        m,
        n,
        nnz,
        &mut ptr,
        &mut row,
        None,
        None,
        None,
    );
    assert_eq!(flag, ERROR_ARG);
    n = 100; // restore

    println!(" * Testing nnz < 1...........................");
    nnz = 0;
    let flag = random_matrix_generate(
        &mut state,
        MatrixType::UNSPECIFIED,
        m,
        n,
        nnz,
        &mut ptr,
        &mut row,
        None,
        None,
        None,
    );
    assert_eq!(flag, ERROR_ARG);
    //nnz = 1000; // restore

    println!(" * Testing nnz > m*n (unsym).................");
    nnz = m * n + 1;
    let flag = random_matrix_generate(
        &mut state,
        MatrixType::UNSPECIFIED,
        m,
        n,
        nnz,
        &mut ptr,
        &mut row,
        None,
        None,
        None,
    );
    assert_eq!(flag, ERROR_ARG);
    // nnz = 1000; // restore

    println!(" * Testing nnz > n*(n+1)/2 (sym).............");
    let matrix_type = MatrixType::RealSymIndef;
    nnz = n * (n + 1) / 2 + 1;
    let flag = random_matrix_generate(
        &mut state,
        matrix_type,
        m,
        n,
        nnz,
        &mut ptr,
        &mut row,
        None,
        None,
        None,
    );
    assert_eq!(flag, ERROR_ARG);
    nnz = 1000; // restore

    // Test non-square
    let matrix_type = MatrixType::RealSymPsdef;
    m = n + 1;
    println!(" * Testing non-square + SYM_PSDEF............");
    let flag = random_matrix_generate(
        &mut state,
        matrix_type,
        m,
        n,
        nnz,
        &mut ptr,
        &mut row,
        None,
        None,
        None,
    );
    assert_eq!(flag, ERROR_NONSQUARE);

    let matrix_type = MatrixType::RealSymIndef;
    println!(" * Testing non-square + SYM_INDEF............");
    let flag = random_matrix_generate(
        &mut state,
        matrix_type,
        m,
        n,
        nnz,
        &mut ptr,
        &mut row,
        None,
        None,
        None,
    );
    assert_eq!(flag, ERROR_NONSQUARE);

    let matrix_type = MatrixType::RealSkew;
    println!(" * Testing non-square + SKEW.................");
    let flag = random_matrix_generate(
        &mut state,
        matrix_type,
        m,
        n,
        nnz,
        &mut ptr,
        &mut row,
        None,
        None,
        None,
    );
    assert_eq!(flag, ERROR_NONSQUARE);
    m = 100; // restore

    // Test singular but insufficient nnz
    nnz = n - 1;
    println!(" * Testing non-singular but nnz too small....");
    let flag = random_matrix_generate(
        &mut state,
        MatrixType::UNSPECIFIED,
        m,
        n,
        nnz,
        &mut ptr,
        &mut row,
        None,
        Some(true),
        None,
    );
    assert_eq!(flag, ERROR_SINGULAR);
}

#[test]
fn test_random_symmetric() {
    const NPROB: usize = 100;
    const MAXN: usize = 10000;
    const MAXNNZ_FACTOR: usize = 10;

    let mut state = RandomState::default();
    let mut ptr = vec![0; MAXN + 1];
    let mut row = vec![0; MAXNNZ_FACTOR * MAXN];
    let mut val = vec![0.0; MAXNNZ_FACTOR * MAXN];

    let matrix_type = MatrixType::RealSymIndef;
    for prblm in 1..=NPROB {
        let n = if prblm < 10 {
            prblm
        } else {
            state.random_integer(MAXN)
        };
        if n == 0 {
            continue;
        }
        let nnz = state.random_integer(min((n + 1) / 2, MAXNNZ_FACTOR) * n);
        if nnz == 0 {
            continue;
        }
        let nonsingular = state.random_integer(2) != 0;
        let sort = state.random_integer(2) != 0;

        println!(
            " * no. {}, n = {}, nnz = {}, flags = {}{}",
            prblm,
            n,
            nnz,
            if nonsingular { 'T' } else { 'F' },
            if sort { 'T' } else { 'F' }
        );

        let flag = random_matrix_generate(
            &mut state,
            matrix_type,
            n,
            n,
            nnz,
            &mut ptr,
            &mut row,
            Some(&mut val),
            Some(nonsingular),
            Some(sort),
        );

        if nonsingular && nnz < n {
            assert_eq!(flag, ERROR_SINGULAR);
        } else {
            assert_eq!(flag, 0);
            chk_random_symmetric(n, nnz, &ptr, &row, &val, nonsingular, sort);
        }
    }
}

fn chk_random_symmetric(
    n: usize,
    nnz: usize,
    ptr: &[usize],
    row: &[usize],
    val: &[f64],
    nonsingular: bool,
    sort: bool,
) {
    assert_eq!(ptr[n], nnz + 1);

    for i in 0..n {
        assert!(ptr[i + 1] >= ptr[i]);
        let mut dpresent = false;
        for j in ptr[i]..ptr[i + 1] {
            if row[j] == i {
                dpresent = true;
            }
            assert!(row[j] >= i && row[j] < n);
            if sort && j > ptr[i] {
                assert!(row[j] > row[j - 1]);
            }
            assert!(val[j].abs() <= 1.0);
        }
        if nonsingular {
            assert!(
                dpresent,
                "nonsingular requested but diagonal not present in column {}",
                i
            );
        }
    }
}

#[test]
fn test_random_unsymmetric() {
    const NPROB: usize = 100;
    const MAXN: usize = 10000;
    const MAXNNZ_FACTOR: usize = 10;

    let mut state = RandomState::default();
    let mut ptr = vec![0; MAXN + 1];
    let mut row = vec![0; MAXNNZ_FACTOR * MAXN];
    let mut val = vec![0.0; MAXNNZ_FACTOR * MAXN];

    let matrix_type = MatrixType::UNSPECIFIED;
    for prblm in 1..=NPROB {
        let (m, n) = if prblm < 10 {
            (prblm, prblm + 1)
        } else {
            (
                state.random_integer(MAXN) as usize,
                state.random_integer(MAXN) as usize,
            )
        };
        if m == 0 || n == 0 {
            continue;
        }
        let nnz = state.random_integer(min((n + 1) / 2, MAXNNZ_FACTOR) * n);
        if nnz == 0 {
            continue;
        }
        let nonsingular = state.random_integer(2) != 0;
        let sort = state.random_integer(2) != 0;

        println!(
            " * no. {}, m = {}, n = {}, nnz = {}, flags = {}{}",
            prblm,
            m,
            n,
            nnz,
            if nonsingular { 'T' } else { 'F' },
            if sort { 'T' } else { 'F' }
        );

        let flag = random_matrix_generate(
            &mut state,
            matrix_type,
            m,
            n,
            nnz,
            &mut ptr,
            &mut row,
            Some(&mut val),
            Some(nonsingular),
            Some(sort),
        );

        if nonsingular && nnz < min(m, n) {
            assert_eq!(flag, ERROR_SINGULAR);
        } else {
            assert_eq!(flag, 0);
            chk_random_unsymmetric(m, n, nnz, &ptr, &row, &val, nonsingular, sort);
        }
    }
}

fn chk_random_unsymmetric(
    m: usize,
    n: usize,
    nnz: usize,
    ptr: &[usize],
    row: &[usize],
    val: &[f64],
    nonsingular: bool,
    sort: bool,
) {
    assert_eq!(ptr[n], nnz);

    for i in 0..n {
        assert!(ptr[i + 1] >= ptr[i]);
        for j in ptr[i]..ptr[i + 1] {
            assert!(row[j] < m);
            if sort && j > ptr[i] {
                assert!(row[j] > row[j - 1]);
            }
            assert!(val[j].abs() <= 1.0);
        }
    }

    if nonsingular {
        let mut rcnt = vec![0; m];
        for i in 0..nnz {
            rcnt[row[i]] += 1;
        }
        let non_empty_rows = rcnt.iter().filter(|&&c| c > 0).count();
        assert!(non_empty_rows >= min(m, n));

        let mut non_empty_cols = 0;
        for i in 0..n {
            if ptr[i + 1] > ptr[i] {
                non_empty_cols += 1;
            }
        }
        assert!(non_empty_cols >= min(m, n));
    }
}
