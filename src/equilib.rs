// Inf-norm Equilibriation Algorithm Implementation

use std::iter::zip;

/// Used to specify options to the routines [equilib_scale_sym] and
/// [equilib_scale_unsym].
#[derive(Debug, Clone, Copy)]
pub struct EquilibOptions {
    pub array_base: usize, // Not in Fortran type
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tol: f64,
}

impl Default for EquilibOptions {
    fn default() -> Self {
        Self {
            array_base: 0, // C
            max_iterations: 10,
            tol: 1e-8,
        }
    }
}

/// Used to return information about the execution of the algorithm.
#[derive(Debug, Clone, Copy)]
pub struct EquilibInform {
    /// Gives the exit status of the algorithm.
    pub flag: i32,
    pub stat: i32,
    /// Number of iteration performed.
    pub iterations: usize,
}

/// Find a matching-based symmetric scaling using the norm-equilibration algorithm.
///
/// The scaled matrix is such that the infinity norm of each row and column are
/// equal to `1.0`.
///
/// We implement Algorithm 1 of:
///
///     "A Symmetry Preserving Algorithm for Matrix Scaling"
///     Philip Knight, Daniel Ruiz and Bora Ucar
///     INRIA Research Report 7552 (Novemeber 2012)
///
/// (This is similar to the algorithm used in MC77, but is a complete
/// reimplementation from the above paper to ensure it is 100% STFC
/// copyright and can be released as open source)
pub(crate) fn inf_norm_equilib_sym(
    n: usize,
    ptr: &[usize],
    row: &[usize],
    val: &[f64],
    scaling: &mut [f64],
    options: &EquilibOptions,
    inform: &mut EquilibInform,
) {
    let mut maxentry = vec![0.0; n];

    scaling.fill(1.0);

    for itr in 1..=options.max_iterations {
        // Find maximum entry in each row and col
        // Recall: matrix is symmetric, but we only have half
        maxentry.fill(0.0);

        for c in 0..n {
            for j in ptr[c]..ptr[c + 1] {
                let r = row[j];
                let v = (scaling[r] * val[j] * scaling[c]).abs();
                maxentry[r] = f64::max(maxentry[r], v);
                maxentry[c] = f64::max(maxentry[c], v);
            }
        }

        // Update scaling (but beware empty cols)
        for (s, &m) in zip(scaling, &maxentry) {
            if m > 0.0 {
                *s /= m.sqrt();
            }
        }

        // Test convergence
        if maxentry
            .iter()
            .map(|&m| (1.0 - m).abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            < options.tol
        {
            break;
        }

        inform.iterations = itr;
    }
}

/// Find a matching-based unsymmetric scaling using the norm-equilibration algorithm.
///
/// The scaled matrix is such that the infinity norm of each row and column are
/// equal to `1.0`.
///
/// We implement Algorithm 1 of:
///
///     "A Symmetry Preserving Algorithm for Matrix Scaling"
///     Philip Knight, Daniel Ruiz and Bora Ucar
///     INRIA Research Report 7552 (November 2012)
///
/// (This is similar to the algorithm used in MC77, but is a complete
/// reimplementation from the above paper to ensure it is 100% STFC
/// copyright and can be released as open source)
fn inf_norm_equilib_unsym(
    m: usize,
    n: usize,
    ptr: &[usize],
    row: &[usize],
    val: &[f64],
    rscaling: &mut [f64],
    cscaling: &mut [f64],
    options: &EquilibOptions,
    inform: &mut EquilibInform,
) {
    let mut rmaxentry = vec![0.0; m];
    let mut cmaxentry = vec![0.0; n];

    rscaling.fill(1.0);
    cscaling.fill(1.0);

    for itr in 1..options.max_iterations {
        // Find maximum entry in each row and col
        rmaxentry.fill(0.0);
        cmaxentry.fill(0.0);

        for c in 0..n {
            for j in ptr[c]..ptr[c + 1] {
                let r = row[j];
                let v = (rscaling[r] * val[j] * cscaling[c]).abs();
                rmaxentry[r] = f64::max(rmaxentry[r], v);
                cmaxentry[c] = f64::max(cmaxentry[c], v);
            }
        }

        // Update scaling (but beware empty rows/cols)
        for (r, &m) in zip(rscaling, &rmaxentry) {
            if m > 0.0 {
                *r /= m.sqrt();
            }
        }
        for (c, &m) in zip(cscaling, &cmaxentry) {
            if m > 0.0 {
                *c /= m.sqrt();
            }
        }

        // Test convergence
        let rmax_diff = rmaxentry
            .iter()
            .map(|&m| (1.0 - m).abs())
            .fold(0.0, f64::max);
        let cmax_diff = cmaxentry
            .iter()
            .map(|&m| (1.0 - m).abs())
            .fold(0.0, f64::max);

        if rmax_diff < options.tol && cmax_diff < options.tol {
            break;
        }

        inform.iterations = itr;
    }
}
