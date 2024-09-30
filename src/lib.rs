//! This package generates various scalings (and matchings) of real sparse matrices.
//!
//! Given a **symmetric** matrix `A`, it finds a diagonal matrix `D`
//! such that the scaled matrix
//!
//!     Â = DAD
//!
//! has specific numerical properties.
//!
//! Given an **unsymmetric** or **rectangular** matrix `A`, it finds
//! diagonal matrices `D`<sub>`r`</sub> and `D`<sub>`c`</sub> such that the scaled matrix
//!
//!     Â = D_r A D_c
//!
//! has specific numerical properties.
//!
//! The specific numerical properties delivered depends on the algorithm used:
//!
//! **Matching-based**
//!    algorithms scale `A` such that the maximum (absolute) value in each row
//!    and column of `Â` is exactly `1.0`, where the entries of
//!    maximum value form a maximum cardinality matching. The
//!    Hungarian algorithm delivers an optimal matching
//!    slowly, whereas the auction algorithm delivers an
//!    approximate matching quickly.
//!
//! **Norm-equilibration**
//!    algorithms scale `A` such that the infinity norm of each row and
//!    column of `Â` is `1.0 ± τ` (for some user specified
//!    tolerance `τ`).

mod auction;
mod equilib;
mod errcode;
mod hungarian;
mod matrix_util;
mod postproc;

#[cfg(test)]
mod tests;

pub use auction::*;
pub use equilib::*;
pub use errcode::*;
pub use hungarian::*;
