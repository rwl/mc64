use crate::tests::{gen_random_sym, gen_random_unsym, MatrixType};
use crate::{equilib_scale_sym, equilib_scale_unsym, EquilibInform, EquilibOptions};
use spral::random::{random_integer, INITIAL_SEED};

/// Testing [equilib_scaling_sym] with random matrices.
#[test]
pub fn test_equilib_sym_random() {
    let max_n = 1000;
    let max_nz = 1000000;
    let n_prob = 100;

    let mut a = MatrixType {
        n: 0,
        m: 0,
        ptr: vec![0; max_n + 1],
        row: vec![0; 2 * max_nz],
        val: vec![0.0; 2 * max_nz],
    };

    let mut scaling = vec![0.0; max_n];
    let mut rinf = vec![0.0; max_n];

    let options = EquilibOptions::default();
    let mut inform = EquilibInform::default();

    let mut state = INITIAL_SEED;

    for prblm in 1..=n_prob {
        // Generate parameters
        a.n = if prblm < 21 {
            prblm
        } else {
            random_integer(&mut state, max_n)
        };
        let i = usize::max(0, (a.n.pow(2) / 2) - a.n);
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
        equilib_scale_sym(
            a.n,
            &a.ptr,
            &a.row,
            &a.val,
            &mut scaling,
            &options,
            &mut inform,
        );
        assert!(inform.flag >= 0, "returned inform.flag = {}", inform.flag);

        // Ensure inf norm of all scaled rows is close to 1.0
        rinf.fill(0.0);
        for i in 0..a.n {
            let mut cmax = 0.0;
            for j in a.ptr[i]..a.ptr[i + 1] {
                let v = (scaling[i] * a.val[j] * scaling[a.row[j]]).abs();
                cmax = f64::max(cmax, v);
                rinf[a.row[j]] = f64::max(rinf[a.row[j]], v);
            }
            rinf[i] = f64::max(rinf[i], cmax);
        }

        for i in 0..a.n {
            assert!((1.0 - rinf[i]) <= 0.05, "rinf({}) = {:.4e}", i, rinf[i]);
        }
    }
}

/// Testing [equilib_scaling_unsym] with random matrices.
#[test]
fn test_equilib_unsym_random() {
    let max_n = 1000;
    let max_nz = 1000000;
    let n_prob = 100;

    let mut state = INITIAL_SEED;

    let mut a = MatrixType {
        n: 0,
        m: 0,
        ptr: vec![0; max_n + 1],
        row: vec![0; 2 * max_nz],
        val: vec![0.0; 2 * max_nz],
    };

    let mut rscaling = vec![0.0; max_n];
    let mut cscaling = vec![0.0; max_n];
    let mut rinf = vec![0.0; max_n];

    let options = EquilibOptions::default();
    let mut inform = EquilibInform::default();

    for prblm in 1..=n_prob {
        // Generate parameters
        a.n = random_integer(&mut state, max_n);
        a.m = random_integer(&mut state, max_n);
        if random_integer(&mut state, 2) == 1 {
            a.m = a.n; // 50% chance of unsym vs rect
        }
        if prblm < 21 {
            a.n = prblm; // check very small problems
            a.m = prblm;
        }
        let i = usize::max(0, a.m * a.n / 2 - usize::max(a.m, a.n));
        let nza = usize::max(a.m, a.n) + random_integer(&mut state, i);

        print!(" - no. {} m = {} n = {} nza = {}...", prblm, a.m, a.n, nza);

        assert!(
            a.n <= max_n,
            "bad random matrix: n = {} > max_n = {}",
            a.n,
            max_n
        );
        assert!(
            a.m <= max_n,
            "bad random matrix: m = {} > max_n = {}",
            a.m,
            max_n
        );
        assert!(
            nza <= max_nz,
            "bad random matrix: nza = {} > max_nz = {}",
            nza,
            max_nz
        );

        gen_random_unsym(&mut a, nza, &mut state);

        // Call scaling
        equilib_scale_unsym(
            a.m,
            a.n,
            &a.ptr,
            &a.row,
            &a.val,
            &mut rscaling,
            &mut cscaling,
            &options,
            &mut inform,
        );
        assert!(inform.flag >= 0, "returned inform.flag = {}", inform.flag);

        // Ensure inf norm of all scaled rows is close to 1.0
        rinf.fill(0.0);
        for i in 0..a.n {
            if a.ptr[i] == a.ptr[i + 1] {
                continue; // Empty column
            }
            let mut c_max = 0.0;
            for j in a.ptr[i]..a.ptr[i + 1] {
                let v = (cscaling[i] * a.val[j] * rscaling[a.row[j]]).abs();
                c_max = f64::max(c_max, v);
                rinf[a.row[j]] = f64::max(rinf[a.row[j]], v);
            }
            assert!(1.0 - c_max <= 0.05, "cinf({}) = {:.4e}", i + 1, c_max);
        }

        for i in 0..a.m {
            if 1.0 - rinf[i] > 0.05 {
                // Check non-empty row before we complain
                let r_cnt: usize = (0..a.n)
                    .map(|j| {
                        a.row[a.ptr[j]..a.ptr[j + 1]]
                            .iter()
                            .filter(|&&k| k == i)
                            .count()
                    })
                    .sum();

                assert!(r_cnt <= 0, "rinf({}) = {:.4e}", i + 1, rinf[i]);
            }
        }
    }
}
