use crate::tests::{gen_random_sym, gen_random_unsym, MatrixType};
use crate::{auction_scale_sym, auction_scale_unsym, AuctionInform, AuctionOptions};
use spral::random::{random_integer, INITIAL_SEED};

/// Testing `auction_scaling_sym` with random matrices.
#[test]
fn test_auction_sym_random() {
    let max_n = 1000;
    let max_nz = 1000000;
    let n_prob = 100;

    // let mut rng = rand::thread_rng();
    let mut state = INITIAL_SEED;

    let mut a = MatrixType {
        n: 0,
        m: 0,
        ptr: vec![0; max_n + 1],
        row: vec![0; 2 * max_nz],
        val: vec![0.0; 2 * max_nz],
    };

    let mut scaling = vec![0.0; max_n];
    let mut match_result = vec![0; max_n];
    let mut rmax = vec![0.0; max_n];
    let mut cnt = vec![0; max_n];

    let options = AuctionOptions::default();
    let mut inform = AuctionInform::default();

    for prblm in 1..=n_prob {
        // Generate parameters
        a.n = if prblm < 21 {
            prblm
        } else {
            random_integer(&mut state, max_n)
        };
        let i = usize::max(0, a.n.pow(2) / 2 - a.n);
        let nza = a.n + random_integer(&mut state, i);

        print!(" - no. {} n = {} nza = {}...", prblm, a.n, nza);

        assert!(
            a.n <= max_n,
            "bad random matrix: n = {} > max_n = {}",
            a.n,
            max_n
        );
        assert!(
            nza <= max_nz,
            "bad random matrix: nza = {} > max_nz = {}",
            nza,
            max_nz
        );

        gen_random_sym(&mut a, nza, &mut state, None);

        // Call scaling
        auction_scale_sym(
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

        // Ensure most rows and columns are matched, and none more than once
        let mut nmatch = 0;
        cnt.fill(0);

        for i in 0..a.n {
            let j = match_result[i] as usize;
            assert!(j < a.n, "match[{}] = {}", i, j);
            if j != 0 {
                cnt[j] += 1;
                nmatch += 1;
                assert!(
                    a.row[a.ptr[j]..a.ptr[j + 1]].contains(&i)
                        && a.row[a.ptr[i]..a.ptr[i + 1]].contains(&j),
                    "matched on ({},{}) but no such entry",
                    i,
                    j
                );
            }
        }

        assert!(
            nmatch >= (0.9 * a.n as f64) as usize,
            "Only matched {} pivots ({:.1}%)",
            nmatch,
            (100.0 * nmatch as f64) / a.n as f64
        );

        assert!(!cnt.iter().any(|&x| x > 1), "mismatched row");

        // Ensure all scaled entries are <= 1.0 and each row/col has an entry at 1
        rmax.fill(0.0);
        for i in 0..a.n {
            let mut cmax = 0.0;
            for j in a.ptr[i]..a.ptr[i + 1] {
                let v = (scaling[i] * a.val[j] * scaling[a.row[j]]).abs();
                assert!(v < 1.0 + 1.0, "Scaled entry = {:.4e}", v); // TODO: check
                cmax = f64::max(cmax, v);
                rmax[a.row[j]] = f64::max(rmax[a.row[j]], v);
            }
            rmax[i] = f64::max(rmax[i], cmax);
        }

        for (i, &r) in rmax.iter().enumerate().take(a.n) {
            assert!(r >= 1.0 - 0.25, "rmax[{}] = {:.4e}", i, r);
        }
    }
}

/// Test [auction_scaling_unsym] with random matrices.
#[test]
fn test_auction_unsym_random() {
    const MAX_N: usize = 1000;
    const MAX_NZ: usize = 1_000_000;
    const N_PROB: usize = 100;

    const SCALE_TOL: f64 = 2.0; // How much above 1.0 can scaled value be?
    const MAX_TOL: f64 = 0.01; // How much less than 1.0 can col max be?
    const MAX_EXCEPT: f64 = 0.05; // proportion of bad entries allowed

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
        let i = (a.m * a.n) / 2 - usize::max(a.m, a.n);
        let i = usize::max(0, i);
        let nza = usize::max(a.m, a.n) + random_integer(&mut state, i);

        print!(" - no. {} m = {} n = {} nza = {}...", prblm, a.m, a.n, nza);

        gen_random_unsym(&mut a, nza, &mut state);

        // Call scaling
        let options = AuctionOptions::default();
        let mut inform = AuctionInform::default();
        auction_scale_unsym(
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

        // Ensure most rows and columns are matched, and none more than once
        let mut nmatch = 0;
        cnt.fill(0);

        for i in 0..a.m {
            let j = match_result[i] as usize;
            assert!(j < a.n, "match({}) = {}", i, j);
            if j != 0 {
                cnt[j] += 1;
                nmatch += 1;
                assert!(
                    a.row[a.ptr[j]..a.ptr[j + 1]].contains(&i),
                    "matched on ({},{}) but no such entry",
                    i,
                    j
                );
            }
        }

        assert!(
            nmatch >= (0.9 * usize::min(a.m, a.n) as f64) as usize,
            "Only matched {} pivots ({:.1}%)",
            nmatch,
            (100.0 * nmatch as f64) / usize::min(a.m, a.n) as f64
        );

        assert!(!cnt.iter().any(|&x| x > 1), "mismatched row");

        // Ensure all scaled entries are <= 1.0 and each row/col has an entry at 1
        let mut nexcept = 0;
        rmax.fill(0.0);

        for i in 0..a.n {
            let mut cmax = 0.0;
            for j in a.ptr[i]..a.ptr[i + 1] {
                let v = (cscaling[i] * a.val[j] * rscaling[a.row[j]]).abs();
                if v >= 1.0 + SCALE_TOL {
                    nexcept += 1;
                }
                cmax = f64::max(cmax, v);
                rmax[a.row[j]] = f64::max(rmax[a.row[j]], v);
            }
            if a.ptr[i] != a.ptr[i + 1] && cmax < 1.0 - MAX_TOL {
                nexcept += 1;
            } else if a.ptr[i] == a.ptr[i + 1] && cscaling[i] != 1.0 {
                panic!(
                    "cscaling({}) for empty col = {} match = {}",
                    i, cscaling[i], match_result[i]
                );
            }
        }

        for i in 0..a.m {
            if rmax[i] < 1.0 - MAX_TOL {
                let rcnt = a
                    .ptr
                    .windows(2)
                    .map(|w| a.row[w[0]..w[1]].iter().filter(|&&x| x == i).count())
                    .sum::<usize>();
                if rcnt != 0 {
                    nexcept += 1;
                }
            }
        }

        if nexcept as f64 > MAX_EXCEPT * (a.ptr[a.n] - 1) as f64 {
            panic!(
                "Too many exceptional entries = {:.2}",
                nexcept as f64 / (a.ptr[a.n] - 1) as f64
            );
        }
    }
}
