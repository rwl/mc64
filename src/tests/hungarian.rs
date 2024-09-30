use crate::tests::{gen_random_sym, gen_random_unsym, MatrixType};
use crate::{hungarian_scale_sym, hungarian_scale_unsym, HungarianInform, HungarianOptions};
use spral::random::{random_integer, INITIAL_SEED};

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
        ptr: vec![0; MAX_N + 1],
        row: vec![0; 2 * MAX_NZ],
        val: vec![0.0; 2 * MAX_NZ],
    };
    let mut scaling = vec![0.0; MAX_N];
    let mut match_result = vec![0; MAX_N];
    let mut rmax = vec![0.0; MAX_N];
    let mut cnt = vec![0; MAX_N];

    let options = HungarianOptions::default();
    let mut inform = HungarianInform::default();

    let mut state = INITIAL_SEED;

    for prblm in 1..=N_PROB {
        // Generate parameters
        a.n = random_integer(&mut state, MAX_N);
        if prblm < 21 {
            a.n = prblm;
        } // check very small problems
        let mut i = a.n.pow(2) / 10 - a.n;
        i = usize::max(0, i);
        let nza = a.n + random_integer(&mut state, i);

        print!(" - no. {} n = {} nza = {}...", prblm, a.n, nza);

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
            assert!(j != 0 && j <= a.n, "match({}) = {}", i, j); // TODO: C
            cnt[j - 1] += 1;
            // TODO: check a.row[]
            if !a.row[a.ptr[j - 1]..a.ptr[j]].iter().any(|&k| k == i + 1)
                && !a.row[a.ptr[i]..a.ptr[i + 1]].iter().any(|&k| k == j)
            {
                panic!("matched on ({},{}) but no such entry", i + 1, j);
            }
        }
        assert!(!cnt.iter().any(|&c| c != 1), "mismatched row");

        // Ensure all scaled entries are <= 1.0 and each row/col has an entry at 1
        rmax.fill(0.0);
        for i in 0..a.n {
            let mut cmax = 0.0;
            for j in a.ptr[i]..a.ptr[i + 1] {
                let v = (scaling[i] * a.val[j] * scaling[a.row[j] - 1]).abs();
                if v >= 1.0 + ERR_TOL {
                    panic!("scaled entry = {:.4e}", v);
                }
                cmax = f64::max(cmax, v);
                rmax[a.row[j] - 1] = f64::max(rmax[a.row[j] - 1], v);
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
        ptr: vec![0; MAX_N + 1],
        row: vec![0; 2 * MAX_NZ],
        val: vec![0.0; 2 * MAX_NZ],
    };
    let mut rscaling = vec![0.0; MAX_N];
    let mut cscaling = vec![0.0; MAX_N];
    let mut match_result = vec![0; MAX_N];
    let mut rmax = vec![0.0; MAX_N];
    let mut cnt = vec![0; MAX_N];

    let options = HungarianOptions::default();
    let mut inform = HungarianInform::default();

    let mut state = INITIAL_SEED;

    for prblm in 1..=N_PROB {
        // Generate parameters
        a.n = random_integer(&mut state, MAX_N);
        a.m = random_integer(&mut state, MAX_N);
        if random_integer(&mut state, 2) == 1 {
            a.m = a.n; // 50% chance of unsym vs rect
        }
        if prblm < 21 {
            a.n = prblm; // check very small problems
            a.m = prblm;
        }
        let mut i = a.m * a.n / 2 - usize::max(a.m, a.n);
        i = usize::max(0, i);
        let nza = usize::max(a.m, a.n) + random_integer(&mut state, i);

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
            &options,
            &mut inform,
            Some(&mut match_result),
        );
        assert!(inform.flag >= 0, "returned inform.flag = {}", inform.flag);

        // Ensure each row and column are matched [and on an entry that exists]
        let mut nmatch = 0;
        cnt.fill(0);
        for i in 0..a.m {
            let j = match_result[i] as usize;
            assert!(j != 0 && j <= a.n, "match({}) = {}", i, j); // TODO: C
            if j != 0 {
                cnt[j - 1] += 1;
                nmatch += 1;
                // TODO: check range
                if !a.row[a.ptr[j - 1]..a.ptr[j]].iter().any(|&k| k == i + 1) {
                    panic!("matched on ({},{}) but no such entry", i + 1, j);
                }
            }
        }
        if nmatch != usize::min(a.m, a.n) {
            panic!("Only matched {} in {}x{} matrix", nmatch, a.m, a.n);
        }
        if cnt.iter().take(a.m).any(|&c| c > 1) {
            panic!("mismatched row");
        }

        // Ensure all scaled entries are <= 1.0 and each row/col has an entry at 1
        rmax.fill(0.0);
        for i in 0..a.n {
            let mut cmax = 0.0;
            for j in a.ptr[i]..a.ptr[i + 1] {
                let v = (cscaling[i] * a.val[j] * rscaling[a.row[j] - 1]).abs();
                if v >= 1.0 + ERR_TOL {
                    panic!("scaled entry = {:.4e}", v);
                }
                cmax = f64::max(cmax, v);
                rmax[a.row[j] - 1] = f64::max(rmax[a.row[j] - 1], v);
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
                            .filter(|&&k| a.row[k] == i + 1)
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
