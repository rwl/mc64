// use crate::tests::spral::{random_integer, INITIAL_SEED};
use crate::tests::random::RandomState;
use crate::tests::{gen_random_sym, gen_random_unsym, MatrixType};
use crate::{hungarian_scale_sym, hungarian_scale_unsym, HungarianInform, HungarianOptions};
// use spral::random::{random_integer, INITIAL_SEED};

/// Test [hungarian_scale_sym] with singular matrix.
#[test]
fn test_hungarian_sym_singular() {
    let m: usize = 3;
    let n: usize = 3;
    let nz: usize = 2;
    let ising: usize = 3;

    let mut a = MatrixType {
        n,
        m,
        ptr: Vec::new(),
        row: Vec::new(),
        val: Vec::new(),
    };
    let mut options = HungarianOptions::default();
    let mut inform = HungarianInform::default();
    let mut match_result: Vec<i32> = vec![0; m];
    let mut scaling: Vec<f64> = vec![0.0; n];

    // Produce warning rather than error
    options.scale_if_singular = true;

    a.n = n;
    a.m = m;
    a.ptr = vec![1, 2, 3, 3];
    a.row = vec![1, 2];
    a.val = vec![2.0, 1.0];

    hungarian_scale_sym(
        a.n,
        &a.ptr,
        &a.row,
        &a.val,
        &mut scaling,
        &options,
        &mut inform,
        Some(&mut match_result),
    );

    assert!(inform.flag == 1, "Returned inform.flag = {}", inform.flag);

    assert!(
        match_result[ising - 1] == 0,
        "Singular column {} has value {}",
        ising,
        match_result[ising - 1]
    );
}

/// Test [hungarian_scale_unsym] with singular matrix.
#[test]
fn test_hungarian_unsym_singular() {
    let m: usize = 3;
    let n: usize = 5;
    let nz: usize = 6;
    let ising: usize = 3;

    let mut a = MatrixType {
        n,
        m,
        ptr: Vec::new(),
        row: Vec::new(),
        val: Vec::new(),
    };
    let mut options = HungarianOptions::default();
    let mut inform = HungarianInform::default();
    let mut match_result: Vec<i32> = vec![0; m];
    let mut rscaling: Vec<f64> = vec![0.0; m];
    let mut cscaling: Vec<f64> = vec![0.0; n];

    // Produce warning rather than error
    options.scale_if_singular = true;

    a.n = n;
    a.m = m;
    a.ptr = vec![1, 3, 5, 6, 6, 7];
    a.row = vec![1, 2, 1, 2, 2, 2];
    a.val = vec![2.0, 1.0, 1.0, 4.0, 1.0, 1.0];

    hungarian_scale_unsym(
        a.m,
        a.n,
        &a.ptr,
        &a.row,
        &a.val,
        &mut rscaling,
        &mut cscaling,
        Some(&mut match_result),
        &options,
        &mut inform,
    );

    assert!(inform.flag == 1, "Returned inform.flag = {}", inform.flag);

    assert!(
        match_result[ising - 1] == 0,
        "Singular row {} matched to {}",
        ising,
        match_result[ising - 1]
    );
}

/// Test [hungarian_scaling_sym] with random matrices.
#[test]
fn test_hungarian_sym_random() {
    const MAX_N: usize = 1000;
    const MAX_NZ: usize = 1_000_000;
    const N_PROB: usize = 100;
    const ERR_TOL: f64 = 1e-10;

    let mut a = MatrixType {
        n: 0,
        m: 0,
        ptr: Vec::new(),
        row: Vec::new(),
        val: Vec::new(),
    };
    let mut scaling = Vec::new();
    let mut match_result = Vec::new();
    let mut rmax = Vec::new();
    let mut cnt = Vec::new();

    let options = HungarianOptions::default();
    let mut inform = HungarianInform::default();

    // let mut state = INITIAL_SEED;
    let mut state = RandomState::default();

    for prblm in 1..=N_PROB {
        // Generate parameters
        a.n = state.random_integer(MAX_N);
        if prblm < 21 {
            a.n = prblm;
        } // check very small problems
        let i = (a.n.pow(2) / 10).checked_sub(a.n).unwrap_or(0);
        let nza = a.n + state.random_integer(i);

        println!(" - no. {} n = {} nza = {}...", prblm, a.n, nza);

        assert!(
            a.n <= MAX_N,
            "bad random matrix: n = {} > max_n = {}",
            a.n,
            MAX_N
        );
        assert!(
            nza <= MAX_NZ,
            "bad random matrix: nza = {} > max_nz = {}",
            nza,
            MAX_NZ
        );

        a.ptr = vec![0; a.n + 1];
        a.row = vec![0; nza];
        a.val = vec![0.0; nza];

        scaling = vec![0.0; a.n];
        match_result = vec![0; a.n];
        rmax = vec![0.0; a.n];
        cnt = vec![0; a.n];

        gen_random_sym(&mut a, nza, &mut state, None);

        // Call scaling
        hungarian_scale_sym(
            a.n,
            &a.ptr,
            &a.row,
            &a.val,
            &mut scaling,
            &options,
            &mut inform,
            Some(&mut match_result),
        );
        assert!(inform.flag >= 0, "returned inform.flag = {}", inform.flag);

        // Ensure each row and column are matched
        cnt.fill(0);
        for i in 0..a.n {
            let j = match_result[i] as usize;
            // assert!(j != 0 && j <= a.n, "match({}) = {}", i, j); // TODO: C
            assert!(j < a.n, "match({}) = {}", i, j);
            cnt[j] += 1;
            // TODO: check a.row[]
            if !a.row[a.ptr[j]..a.ptr[j + 1]].iter().any(|&k| k == i)
                && !a.row[a.ptr[i]..a.ptr[i + 1]].iter().any(|&k| k == j)
            {
                panic!("matched on ({},{}) but no such entry", i, j);
            }
        }
        assert!(!cnt[..a.n].iter().any(|&c| c != 1), "mismatched row");

        // Ensure all scaled entries are <= 1.0 and each row/col has an entry at 1
        rmax.fill(0.0);
        for i in 0..a.n {
            let mut cmax = 0.0;
            for j in a.ptr[i]..a.ptr[i + 1] {
                let v = (scaling[i] * a.val[j] * scaling[a.row[j]]).abs();
                if v >= 1.0 + ERR_TOL {
                    panic!("scaled entry = {:.4e}", v);
                }
                cmax = f64::max(cmax, v);
                rmax[a.row[j]] = f64::max(rmax[a.row[j]], v);
            }
            rmax[i] = f64::max(rmax[i], cmax);
        }

        for (i, &r) in rmax.iter().enumerate().take(a.n) {
            if r < 1.0 - ERR_TOL {
                panic!("rmax({}) = {:.4e}", i + 1, r);
            }
        }
    }
}

/// Test [hungarian_scaling_unsym] with random matrices.
#[test]
fn test_hungarian_unsym_random() {
    const MAX_N: usize = 1000;
    const MAX_NZ: usize = 1_000_000;
    const N_PROB: usize = 100;
    const ERR_TOL: f64 = 1e-10;

    let mut a = MatrixType {
        n: 0,
        m: 0,
        ptr: Vec::new(),
        row: Vec::new(),
        val: Vec::new(),
    };
    let mut rscaling = Vec::new();
    let mut cscaling = Vec::new();
    let mut match_result = Vec::new();
    let mut rmax = Vec::new();
    let mut cnt = Vec::new();

    let options = HungarianOptions::default();
    let mut inform = HungarianInform::default();

    // let mut state = INITIAL_SEED;
    let mut state = RandomState::default();

    for prblm in 1..=N_PROB {
        // Generate parameters
        a.n = state.random_integer(MAX_N);
        a.m = state.random_integer(MAX_N);
        if state.random_integer(2) == 1 {
            a.m = a.n; // 50% chance of unsym vs rect
        }
        if prblm < 21 {
            a.n = prblm; // check very small problems
            a.m = prblm;
        }
        let i = (a.m * a.n / 2)
            .checked_sub(usize::max(a.m, a.n))
            .unwrap_or(0);
        let nza = usize::max(a.m, a.n) + state.random_integer(i);

        print!(" - no. {} m = {} n = {} nza = {}...", prblm, a.m, a.n, nza);

        assert!(
            a.n <= MAX_N,
            "bad random matrix: n = {} > max_n = {}",
            a.n,
            MAX_N
        );
        assert!(
            a.m <= MAX_N,
            "bad random matrix: m = {} > max_n = {}",
            a.m,
            MAX_N
        );
        assert!(
            nza <= MAX_NZ,
            "bad random matrix: nza = {} > max_nz = {}",
            nza,
            MAX_NZ
        );

        a.ptr = vec![0; a.n + 1];
        a.row = vec![0; nza];
        a.val = vec![0.0; nza];

        rscaling = vec![0.0; a.m];
        cscaling = vec![0.0; a.n];
        match_result = vec![0; a.m];
        rmax = vec![0.0; a.m];
        cnt = vec![0; a.n];

        gen_random_unsym(&mut a, nza, &mut state);

        // Call scaling
        hungarian_scale_unsym(
            a.m,
            a.n,
            &a.ptr,
            &a.row,
            &a.val,
            &mut rscaling,
            &mut cscaling,
            Some(&mut match_result),
            &options,
            &mut inform,
        );
        assert!(inform.flag >= 0, "returned inform.flag = {}", inform.flag);

        // Ensure each row and column are matched [and on an entry that exists]
        let mut nmatch = 0;
        cnt.fill(0);
        for i in 0..a.m {
            let j = match_result[i];
            assert!((j as usize) < a.n, "match({}) = {}", i, j);
            if j != -1 {
                cnt[j as usize] += 1;
                nmatch += 1;
                // TODO: check range
                if !a.row[a.ptr[j as usize]..a.ptr[j as usize + 1]]
                    .iter()
                    .any(|&k| k == i)
                {
                    panic!("matched on ({},{}) but no such entry", i, j);
                }
            }
        }
        if nmatch != usize::min(a.m, a.n) {
            panic!("Only matched {} in {}x{} matrix", nmatch, a.m, a.n);
        }
        if cnt.iter().take(a.n).any(|&c| c > 1) {
            panic!("mismatched row");
        }

        // Ensure all scaled entries are <= 1.0 and each row/col has an entry at 1
        rmax.fill(0.0);
        for i in 0..a.n {
            let mut cmax = 0.0;
            for j in a.ptr[i]..a.ptr[i + 1] {
                let v = (cscaling[i] * a.val[j] * rscaling[a.row[j]]).abs();
                if v >= 1.0 + ERR_TOL {
                    panic!("scaled entry = {:.4e}", v);
                }
                cmax = f64::max(cmax, v);
                rmax[a.row[j]] = f64::max(rmax[a.row[j]], v);
            }
            if cmax < 1.0 - ERR_TOL && a.ptr[i] != a.ptr[i + 1] {
                panic!("cmax({}) = {:.4e}", i + 1, cmax);
            }
        }

        for i in 0..a.m {
            if rmax[i] < 1.0 - ERR_TOL {
                // Check non-empty row before we complain
                let rcnt = (0..a.n)
                    .map(|j| {
                        a.row[a.ptr[j]..a.ptr[j + 1]]
                            .iter()
                            .filter(|&&k| a.row[k] == i)
                            .count()
                    })
                    .sum::<usize>();
                if rcnt > 0 {
                    panic!("rmax({}) = {:.4e}", i + 1, rmax[i]);
                }
            }
        }
    }
}
