use crate::matrix_util::half_to_full;
use crate::postproc::match_postproc;

#[derive(Debug, Clone, Copy)]
pub struct AuctionOptions {
    pub array_base: usize, // Not in Fortran type
    pub max_iterations: usize,
    pub max_unchanged: [usize; 3],
    pub min_proportion: [f64; 3],
    pub eps_initial: f64,
}

impl Default for AuctionOptions {
    fn default() -> Self {
        Self {
            array_base: 0, // C
            max_iterations: 30000,
            max_unchanged: [10, 100, 100],
            min_proportion: [0.90, 0.0, 0.0],
            eps_initial: 0.01,
        }
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct AuctionInform {
    pub flag: i32,
    pub stat: i32,
    pub matched: usize,
    pub iterations: usize,
    pub unmatchable: usize,
}

/// Auction algorithm for symmetric matrix scaling
///
/// This function implements the auction algorithm for symmetric matrix scaling.
/// It uses the unsymmetric implementation internally and averages the row and column
/// scalings to obtain a symmetric scaling.
pub fn auction_scale_sym(
    n: usize,
    ptr: &[usize],
    row: &[usize],
    val: &[f64],
    scaling: &mut [f64],
    options: &AuctionOptions,
    inform: &mut AuctionInform,
    match_result: Option<&mut [i32]>,
) {
    inform.flag = 0; // Initialize to success

    // Allocate memory
    let mut rscaling = vec![0.0; n];
    let mut cscaling = vec![0.0; n];

    // Call unsymmetric implementation with flag to expand half matrix
    match match_result {
        Some(match_vec) => {
            auction_match(
                true,
                n,
                n,
                ptr,
                row,
                val,
                match_vec,
                &mut rscaling,
                &mut cscaling,
                options,
                inform,
            );
        }
        None => {
            let mut perm = vec![0; n];
            auction_match(
                true,
                n,
                n,
                ptr,
                row,
                val,
                &mut perm,
                &mut rscaling,
                &mut cscaling,
                options,
                inform,
            );
        }
    }

    // Average rscaling and cscaling to get symmetric scaling
    for (s, (&r, &c)) in scaling.iter_mut().zip(rscaling.iter().zip(cscaling.iter())) {
        *s = f64::exp((r + c) / 2.0);
    }
}

/// Auction-based scaling for unsymmetric matrices.
///
/// # Arguments
/// * `m` - number of rows
/// * `n` - number of columns
/// * `ptr` - column pointers of A
/// * `row` - row indices of A (lower triangle)
/// * `val` - entries of A (in same order as in row)
/// * `rscaling` - row scaling factors (output)
/// * `cscaling` - column scaling factors (output)
/// * `options` - auction algorithm options
/// * `inform` - information about the execution
/// * `match` - optional matching result
pub fn auction_scale_unsym(
    m: usize,
    n: usize,
    ptr: &[usize],
    row: &[usize],
    val: &[f64],
    rscaling: &mut [f64],
    cscaling: &mut [f64],
    options: &AuctionOptions,
    inform: &mut AuctionInform,
    match_result: Option<&mut [i32]>,
) {
    inform.flag = 0; // Initialize to success

    match match_result {
        Some(match_vec) => {
            auction_match(
                false, m, n, ptr, row, val, match_vec, rscaling, cscaling, options, inform,
            );
        }
        None => {
            let mut perm = vec![0; m];
            auction_match(
                false, m, n, ptr, row, val, &mut perm, rscaling, cscaling, options, inform,
            );
        }
    }

    // Apply exponential to scaling factors
    for r in rscaling.iter_mut() {
        *r = r.exp();
    }
    for c in cscaling.iter_mut() {
        *c = c.exp();
    }
}

// An implementation of the auction algorithm to solve the assignment problem
// i.e. max_M sum_{(i,j)\in M} a_{ij}    where M is a matching.
// The dual variables u_i for row i and v_j for col j can be used to find
// a good scaling after postprocessing.
// We're aiming for:
// a_ij - u_i - v_j == 0    if (i,j) in M
// a_ij - u_i - v_j <= 0    otherwise
//
// Motivation of algorithm:
// Each unmatched column bids for its preferred row. Best bid wins.
// Prices (dual variables) are updated to reflect cost of using 2nd best instead
// for future auction rounds.
// i.e. Price of using entry (i,j) is a_ij - u_i
//
// In this implementation, only one column is bidding in each phase. This is
// largely equivalent to the motivation above but allows for faster progression
// as the same row can move through multiple partners during a single pass
// through the matrix.
fn auction_match_core(
    m: usize,
    n: usize,
    ptr: &[usize],
    row: &[usize],
    val: &[f64],
    match_result: &mut [i32],
    dualu: &mut [f64],
    dualv: &mut [f64],
    options: &AuctionOptions,
    inform: &mut AuctionInform,
) {
    inform.flag = 0;
    inform.unmatchable = 0;

    let mut owner = vec![0; m];
    let mut next = vec![0; n];

    let minmn = m.min(n);
    let mut unmatched = minmn;
    match_result.fill(0); // 0 = unmatched, -1 = unmatched+ineligible
    owner.fill(0);
    dualu.fill(0.0);
    // dualv is set for each column as it becomes matched, otherwise we use
    // the value supplied on input (calculated as something sensible during
    // preprocessing)

    let mut prev = usize::MAX;
    let mut nunchanged = 0;

    // Initially all columns are unmatched
    let mut tail = n;
    for i in 0..n {
        next[i] = i;
    }

    // Iterate until we run out of unmatched buyers
    let mut eps = options.eps_initial;
    for itr in 0..options.max_iterations {
        if unmatched == 0 {
            break; // nothing left to match
        }
        // Bookkeeping to determine number of iterations with no change
        if unmatched != prev {
            nunchanged = 0;
        }
        prev = unmatched;
        nunchanged += 1;
        // Test if we satisfy termination conditions
        for i in 0..3 {
            if nunchanged >= options.max_unchanged[i]
                && (minmn - unmatched) as f64 / minmn as f64 >= options.min_proportion[i]
            {
                inform.iterations = itr;
                return;
            }
        }
        // Update epsilon scaling
        eps = f64::min(1.0, eps + 1.0 / (n as f64 + 1.0));
        // Now iterate over all unmatched entries listed in next[0..tail]
        // As we progress, build list for next iteration in next[0..insert]
        let mut insert = 0;
        for cptr in 0..tail {
            let col = next[cptr];
            if match_result[col] != 0 {
                continue; // already matched or ineligible
            }
            if ptr[col] == ptr[col + 1] {
                continue; // empty col (only ever fails on first iteration - not put in next[0..insert] thereafter)
            }
            // Find best value of a_ij - u_i for current column
            // This occurs for i=bestr with value bestu
            // second best value stored as bestv
            let j = ptr[col];
            let mut bestr = row[j];
            let mut bestu = val[j] - dualu[bestr];
            let mut bestv = f64::NEG_INFINITY;
            for j in ptr[col] + 1..ptr[col + 1] {
                let u = val[j] - dualu[row[j]];
                if u > bestu {
                    bestv = bestu;
                    bestr = row[j];
                    bestu = u;
                } else if u > bestv {
                    bestv = u;
                }
            }
            if bestv == f64::NEG_INFINITY {
                bestv = 0.0; // No second best
            }
            // Check if matching this column gives us a net benefit
            if bestu > 0.0 {
                // There is a net benefit, match column col to row bestr
                // if bestr was previously matched to col k, unmatch it
                dualu[bestr] += bestu - bestv + eps;
                dualv[col] = bestv - eps; // satisfy a_ij - u_i - v_j = 0
                match_result[col] = bestr as i32;
                unmatched -= 1;
                let k = owner[bestr];
                owner[bestr] = col;
                if k != 0 {
                    // Mark column k as unmatched
                    match_result[k] = 0; // unmatched
                    unmatched += 1;
                    next[insert] = k;
                    insert += 1;
                }
            } else {
                // No net benefit, mark col as ineligible for future consideration
                match_result[col] = -1; // ineligible
                unmatched -= 1;
                inform.unmatchable += 1;
            }
        }
        tail = insert;
    }
    inform.iterations = options.max_iterations;

    // We expect unmatched columns to have match_result[col] = 0
    for m in match_result.iter_mut() {
        if *m == -1 {
            *m = 0;
        }
    }
}

/// Find a scaling through a matching-based approach using the auction algorithm
/// This function actually performs pre/post-processing around the call to
/// auction_match_core() to actually use the auction algorithm
///
/// This consists of finding a2_ij = 2*maxentry - cmax + log(|a_ij|)
/// where cmax is the log of the absolute maximum in the column
/// and maxentry is the maximum value of cmax-log(|a_ij|) across entire matrix
/// The cmax-log(|a_ij|) term converts from max product to max sum problem and
/// normalises scaling across matrix. The 2*maxentry term biases the result
/// towards a high cardinality solution.
///
/// Lower triangle only as input (log(half)+half->full faster than log(full))
fn auction_match(
    expand: bool,
    m: usize,
    n: usize,
    ptr: &[usize],
    row: &[usize],
    val: &[f64],
    match_result: &mut [i32],
    rscaling: &mut [f64],
    cscaling: &mut [f64],
    options: &AuctionOptions,
    inform: &mut AuctionInform,
) {
    inform.flag = 0;

    // Reset ne for the expanded symmetric matrix
    let mut ne = ptr[n] - 1;
    ne = 2 * ne;

    // Expand matrix, drop explicit zeroes and take log absolute values
    let mut ptr2 = vec![0; n + 1];
    let mut row2 = vec![0; ne];
    let mut val2 = vec![0.0; ne];
    let mut cmax = vec![0.0; n];
    let mut cmatch = vec![0; n];

    let mut klong = 0;
    for i in 0..n {
        ptr2[i] = klong;
        for jlong in ptr[i]..ptr[i + 1] {
            if val[jlong] == 0.0 {
                continue;
            }
            row2[klong] = row[jlong];
            val2[klong] = val[jlong].abs().ln();
            klong += 1;
        }
    }
    ptr2[n] = klong;

    if expand {
        if m != n {
            // Should never get this far with a non-square matrix
            inform.flag = -99;
            return;
        }
        let mut iw = vec![0; 5 * n];
        half_to_full(n, &mut row2, &mut ptr2, &mut iw, Some(&mut val2), true);
    }

    // Compute column maximums
    for i in 0..n {
        if ptr2[i + 1] <= ptr2[i] {
            // Empty col
            cmax[i] = 0.0;
            continue;
        }
        let colmax = val2[ptr2[i]..ptr2[i + 1]]
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        cmax[i] = colmax;
        for v in &mut val2[ptr2[i]..ptr2[i + 1]] {
            *v = colmax - *v;
        }
    }

    let maxentry = val2[..ptr2[n]]
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    // Use 2*maxentry+1 to prefer high cardinality matchings (+1 avoids 0 cols)
    let maxentry = 2.0 * maxentry + 1.0;
    for v in &mut val2[..ptr2[n]] {
        *v = maxentry - *v;
    }
    for (c, &max) in cscaling.iter_mut().zip(cmax.iter()) {
        *c = -max; // equivalent to scale=1.0 for unmatched cols that core algorithm doesn't visit
    }

    auction_match_core(
        m,
        n,
        &ptr2,
        &row2,
        &val2,
        &mut cmatch,
        rscaling,
        cscaling,
        options,
        inform,
    );
    inform.matched = cmatch.iter().filter(|&&x| x != 0).count();

    // Calculate an adjustment so row and col scaling similar orders of magnitude
    // and undo pre processing
    for r in rscaling.iter_mut() {
        *r = -(*r) + maxentry;
    }
    for (c, &max) in cscaling.iter_mut().zip(cmax.iter()) {
        *c = -(*c) - max;
    }

    // Convert row->col matching into col->row one
    match_result.fill(0);
    for (i, &m) in cmatch.iter().enumerate() {
        if m == 0 {
            continue; // unmatched row
        }
        match_result[m as usize] = i as i32;
    }
    match_postproc(
        m,
        n,
        ptr,
        row,
        val,
        rscaling,
        cscaling,
        inform.matched,
        match_result,
        &mut inform.flag,
    );
}
