use spral::matrix_util::{REAL_RECT, REAL_SYM_INDEF};
use spral::random::random_integer;
use spral::random_matrix::{random_matrix_generate, NONSINGULAR, SORT};

mod auction;
mod equilib;
mod hungarian;

struct MatrixType {
    n: usize,
    m: usize,
    ptr: Vec<usize>,
    row: Vec<usize>,
    val: Vec<f64>,
}

/// Generate a random symmetric matrix
fn gen_random_sym(a: &mut MatrixType, nza: usize, state: &mut u32, zr: Option<usize>) {
    // Generate a
    // let mut flag = 0;
    random_matrix_generate(
        state,
        REAL_SYM_INDEF,
        a.n,
        a.n,
        nza,
        &mut a.ptr,
        &mut a.row,
        // &mut flag,
        Some(&mut a.val),
        NONSINGULAR | SORT,
    );
    // if flag != 0 {
    //     println!("Bad flag from random_matrix_generate()");
    // }

    if let Some(zr) = zr {
        // Scan along row
        for i in 0..a.ptr[zr - 1] {
            if a.row[i] == zr - 1 {
                a.val[i] = 0.0;
            }
        }
        // Scan along column
        for i in a.ptr[zr - 1]..a.ptr[zr] {
            a.val[i] = 0.0;
        }
    } else if a.n > 3 {
        // Put some zeros on diagonal, observing first entry in column
        // is always the diagonal after sorting
        // but don't have all zeros in the col.
        let l = random_integer(state, a.n / 2);
        for k in (0..a.n).step_by(l.max(1)) {
            if a.ptr[k + 1] > a.ptr[k] + 1 {
                let i = a.ptr[k];
                a.val[i] = 0.0;
            }
        }
        // also make sure we have some large off diagonals
        for k in 0..a.n {
            let i = a.ptr[k + 1] - 1;
            a.val[i] *= 1000.0;
        }
    }
}

/// Generate a random unsymmetric matrix
fn gen_random_unsym(a: &mut MatrixType, nza: usize, state: &mut u32) {
    // Generate a
    // let mut flag = 0;
    random_matrix_generate(
        state,
        REAL_RECT,
        a.m,
        a.n,
        nza,
        &mut a.ptr,
        &mut a.row,
        // &mut flag,
        Some(&mut a.val),
        NONSINGULAR | SORT,
    );
    // if flag != 0 {
    //     println!("Bad flag from random_matrix_generate()");
    // }

    // make sure we have some large entries
    let mut k = 0;
    while k < a.ptr[a.n] - 1 {
        a.val[k] *= 1000.0;
        k += random_integer(state, 5);
    }
}
