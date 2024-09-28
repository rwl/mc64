// Hungarian Algorithm implementation (MC64)

use crate::errcode::{ERROR_SINGULAR, WARNING_SINGULAR};
use crate::matrix_util::half_to_full;
use crate::postproc::match_postproc;
use std::iter::zip;

#[derive(Default, Debug, Clone, Copy)]
pub struct HungarianOptions {
    pub array_base: usize, // Not in Fortran type
    pub scale_if_singular: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct HungarianInform {
    pub flag: i32,
    pub stat: i32,
    pub matched: usize,
}

/// Find a matching-based symmetric scaling using the Hungarian algorithm.
///
/// The scaled matrix is such that the entry of maximum absolute value in each
/// row and column is `1.0`.
pub fn hungarian_scale_sym(
    n: usize,
    ptr: &[usize],
    row: &[usize],
    val: &[f64],
    scaling: &mut [f64],
    options: &HungarianOptions,
    inform: &mut HungarianInform,
    match_result: Option<&mut [i32]>,
) {
    inform.flag = 0; // Initialize to success

    let mut rscaling = vec![0.0; n];
    let mut cscaling = vec![0.0; n];

    match match_result {
        Some(match_vec) => {
            hungarian_wrapper(
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
            hungarian_wrapper(
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

    for i in 0..n {
        scaling[i] = ((rscaling[i] + cscaling[i]) / 2.0).exp();
    }
}

/// Scales an unsymmetric matrix using the Hungarian algorithm.
///
/// # Arguments
/// * `m` - number of rows
/// * `n` - number of columns
/// * `ptr` - column pointers of A
/// * `row` - row indices of A (lower triangle)
/// * `val` - entries of A (in same order as in row)
/// * `rscaling` - output row scaling factors
/// * `cscaling` - output column scaling factors
/// * `options` - options for the Hungarian algorithm
/// * `inform` - output information about the execution
/// * `match` - optional output matching
pub fn hungarian_scale_unsym(
    m: usize,
    n: usize,
    ptr: &[usize],
    row: &[usize],
    val: &[f64],
    rscaling: &mut [f64],
    cscaling: &mut [f64],
    options: &HungarianOptions,
    inform: &mut HungarianInform,
    match_out: Option<&mut [i32]>,
) {
    inform.flag = 0; // Initialize to success

    // Call main routine
    match match_out {
        Some(match_result) => {
            hungarian_wrapper(
                false,
                m,
                n,
                ptr,
                row,
                val,
                match_result,
                rscaling,
                cscaling,
                options,
                inform,
            );
        }
        None => {
            let mut perm = vec![0; m];
            hungarian_wrapper(
                false, m, n, ptr, row, val, &mut perm, rscaling, cscaling, options, inform,
            );
        }
    }

    // Apply post processing
    for r in rscaling.iter_mut() {
        *r = r.exp();
    }
    for c in cscaling.iter_mut() {
        *c = c.exp();
    }
}

// This function wraps the core algorithm of `hungarian_match()`. It provides
// pre- and post-processing to transform a maximum product assignment to a
// minimum sum assignment problem (and back again). It also has post-processing
// to handle the case of a structurally singular matrix as per Duff and Pralet
// (though the efficacy of such an approach is disputed!).
//
// This code is adapted from HSL_MC64 v2.3.1
fn hungarian_wrapper(
    sym: bool,
    m: usize,
    n: usize,
    ptr: &[usize],
    row: &[usize],
    val: &[f64],
    match_result: &mut [i32],
    rscaling: &mut [f64],
    cscaling: &mut [f64],
    options: &HungarianOptions,
    inform: &mut HungarianInform,
) {
    assert_eq!(ptr.len(), n + 1);

    inform.flag = 0;
    inform.stat = 0;
    let mut ne = ptr[n] - 1;

    // Reset ne for the expanded symmetric matrix
    ne = 2 * ne;

    // Expand matrix, drop explicit zeroes and take log absolute values
    let mut ptr2 = vec![0; n + 1];
    let mut row2 = vec![0; ne];
    let mut val2 = vec![0.0; ne];
    let mut iw = vec![0; 5 * n];
    let mut dualu = vec![0.0; m];
    let mut dualv = vec![0.0; n];
    let mut cmax = vec![0.0; n];

    let mut klong = 0;
    for i in 0..n {
        ptr2[i] = klong;
        for jlong in ptr[i]..ptr[i + 1] {
            if val[jlong] == 0.0 {
                continue;
            }
            row2[klong] = row[jlong];
            val2[klong] = val[jlong].abs();
            klong += 1;
        }
        // Following log is separated from above loop to expose expensive
        // log operation to vectorization.
        for v in &mut val2[ptr2[i]..klong] {
            *v = v.ln();
        }
    }
    ptr2[n] = klong;

    if sym {
        half_to_full(n, &mut row2, &mut ptr2, &mut iw, Some(&mut val2), true);
    }

    // Compute column maximums
    for i in 0..n {
        let colmax = val2[ptr2[i]..ptr2[i + 1]]
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| f64::max(a, b));
        cmax[i] = colmax;
        for v in &mut val2[ptr2[i]..ptr2[i + 1]] {
            *v = colmax - *v;
        }
    }

    hungarian_match(
        m,
        n,
        &ptr2,
        &row2,
        &val2,
        match_result,
        &mut inform.matched,
        &mut dualu,
        &mut dualv,
    );

    if inform.matched != usize::min(m, n) {
        // Singular matrix
        if options.scale_if_singular {
            // Just issue warning then continue
            inform.flag = WARNING_SINGULAR;
        } else {
            // Issue error and return identity scaling
            inform.flag = ERROR_SINGULAR;
            rscaling.fill(0.0);
            cscaling.fill(0.0);
            return;
        }
    }

    if !sym || inform.matched == n {
        // Unsymmetric or symmetric and full rank
        // Note that in this case m=n
        rscaling.copy_from_slice(&dualu[..m]);
        for (c, (dv, &max)) in zip(cscaling.iter_mut(), zip(dualv, &cmax)) {
            *c = dv - max;
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
        return;
    }

    // If we reach this point then structurally rank deficient.
    // As matching may not involve full set of rows and columns, but we need
    // a symmetric matching/scaling, we can't just return the current matching.
    // Instead, we build a full rank submatrix and call matching on it.

    let mut old_to_new = vec![0; n];
    let mut new_to_old = vec![0; n];
    let mut cperm = vec![0; n];

    // let mut j = inform.matched + 1;
    let mut j = inform.matched; // TODO: double check
    let mut k = 0;
    for i in 0..m {
        if match_result[i] < 0 {
            // row i is not part of the matching
            old_to_new[i] = -(j as i32);
            j += 1;
        } else {
            // k += 1;
            // old_to_new[i] holds the new index for variable i after
            // removal of singular part and new_to_old[k] is the
            // original index for k
            old_to_new[i] = k as i32;
            new_to_old[k] = i;

            k += 1; // TODO: double check
        }
    }

    // Overwrite ptr2, row2 and val2
    ne = 0;
    k = 0;
    ptr2[0] = 0;
    let mut j2 = 0;
    for i in 0..n {
        let j1 = j2;
        j2 = ptr2[i + 1];
        // skip over unmatched entries
        if match_result[i] < 0 {
            continue;
        }
        k += 1;
        for jlong in j1..j2 {
            let jj = row2[jlong];
            if match_result[jj] < 0 {
                continue;
            }
            row2[ne] = old_to_new[jj] as usize;
            val2[ne] = val2[jlong];
            ne += 1;
        }
        ptr2[k] = ne; // TODO: check k
    }
    // nn is order of non-singular part.
    let nn = k;
    hungarian_match(
        nn,
        nn,
        &ptr2,
        &row2,
        &val2,
        &mut cperm,
        &mut inform.matched,
        &mut dualu,
        &mut dualv,
    );

    for i in 0..n {
        let j = old_to_new[i];
        if j < 0 {
            rscaling[i] = f64::NEG_INFINITY;
        } else {
            // Note: we need to subtract col max using old matrix numbering
            rscaling[i] = (dualu[j as usize] + dualv[j as usize] - cmax[i]) / 2.0;
        }
    }

    match_result.fill(-1);
    for i in 0..nn {
        let j = cperm[i];
        match_result[new_to_old[i]] = j;
    }

    for i in 0..n {
        if match_result[i] == -1 {
            match_result[i] = old_to_new[i];
        }
    }

    // Apply Duff and Pralet correction to unmatched row scalings
    let mut cscale = vec![0.0; n];
    // For columns i not in the matched set I, set
    //     s_i = 1 / (max_{k in I} | a_ik s_k |)
    // with convention that 1/0 = 1
    cscale.copy_from_slice(&rscaling[..n]);
    for i in 0..n {
        for jlong in ptr[i]..ptr[i + 1] {
            let k = row[jlong];
            if cscale[i] == f64::NEG_INFINITY && cscale[k] != f64::NEG_INFINITY {
                // i not in I, k in I
                rscaling[i] = f64::max(rscaling[i], val[jlong].abs().ln() + rscaling[k]);
            }
            if cscale[k] == f64::NEG_INFINITY && cscale[i] != f64::NEG_INFINITY {
                // k not in I, i in I
                rscaling[k] = f64::max(rscaling[k], val[jlong].abs().ln() + rscaling[i]);
            }
        }
    }
    for i in 0..n {
        if cscale[i] != f64::NEG_INFINITY {
            continue; // matched part
        }
        if rscaling[i] == f64::NEG_INFINITY {
            rscaling[i] = 0.0;
        } else {
            rscaling[i] = -rscaling[i];
        }
    }
    // As symmetric, scaling is averaged on return, but rscaling[:] is correct,
    // so just copy to cscaling to fix this
    cscaling.copy_from_slice(&rscaling[..n]);
}

// Subroutine that initialize matching and (row) dual variable into a suitable
// state for main Hungarian algorithm.
//
// The heuristic guarantees that the generated partial matching is optimal
// on the restriction of the graph to the matched rows and columns.
fn hungarian_init_heuristic(
    m: usize,
    n: usize,
    ptr: &[usize],
    row: &[usize],
    val: &[f64],
    num: &mut usize,
    iperm: &mut [i32],
    jperm: &mut [usize],
    dualu: &mut [f64],
    // d[j] current improvement from matching in col j
    d: &mut [f64],
    // position of smallest entry of row
    l: &mut [usize],
    // position we have reached in current search
    search_from: &mut [usize],
) {
    assert_eq!(ptr.len(), n + 1);
    assert_eq!(row.len(), n);
    assert_eq!(val.len(), n);
    assert_eq!(dualu.len(), m);
    assert_eq!(d.len(), n);
    assert_eq!(l.len(), m);
    assert_eq!(search_from.len(), n);

    // Set up initial matching on smallest entry in each row (as far as possible)
    //
    // Find smallest entry in each col, and record it
    dualu.fill(f64::INFINITY);
    l.fill(0);
    for j in 0..n {
        for k in ptr[j]..ptr[j + 1] {
            let i = row[k];
            if val[k] > dualu[i] {
                continue;
            }
            dualu[i] = val[k]; // Initialize dual variables
            iperm[i] = j as i32; // Record col
            l[i] = k; // Record posn in row
        }
    }

    // Loop over rows in turn. If we can match on smallest entry in row (i.e.
    // column not already matched) then do so. Avoid matching on dense columns
    // as this makes Hungarian algorithm take longer.
    for i in 0..m {
        let j = iperm[i] as usize; // Smallest entry in row i is (i,j)
        if j == 0 {
            continue; // skip empty rows
        }
        iperm[i] = 0;
        if jperm[j] != 0 {
            continue; // If we've already matched column j, skip this row
        }
        // Don't choose cheap assignment from dense columns
        if (ptr[j + 1] - ptr[j] > m / 10) && (m > 50) {
            continue;
        }
        // Assignment of column j to row i
        *num += 1;
        iperm[i] = j as i32;
        jperm[j] = l[i];
    }

    // If we already have a complete matching, we're already done!
    if *num == usize::min(m, n) {
        return;
    }

    // Scan unassigned columns; improve assignment
    d.fill(0.0);
    search_from.copy_from_slice(&ptr[..n]);

    'improve_assign: for j in 0..n {
        if jperm[j] != 0 {
            continue; // column j already matched
        }
        if ptr[j] >= ptr[j + 1] {
            continue; // column j is empty
        }

        // Find smallest value of di = a_ij - u_i in column j
        // In case of a tie, prefer first unmatched, then first matched row.
        let mut i0 = row[ptr[j]];
        let mut vj = val[ptr[j]] - dualu[i0];
        let mut k0 = ptr[j];

        for k in (ptr[j] + 1)..ptr[j + 1] {
            let i = row[k];
            let di = val[k] - dualu[i];
            if di > vj {
                continue;
            }
            if (di == vj) && (di != f64::INFINITY) {
                if (iperm[i] != 0) || (iperm[i0] == 0) {
                    continue;
                }
            }
            vj = di;
            i0 = i;
            k0 = k;
        }

        // Record value of matching on (i0,j)
        d[j] = vj;

        // If row i is unmatched, then match on (i0,j) immediately
        if iperm[i0] == 0 {
            *num += 1;
            jperm[j] = k0;
            iperm[i0] = j as i32;
            search_from[j] = k0 + 1;
            continue;
        }

        // Otherwise, row i is matched. Consider all rows i in column j that tie
        // for this vj value. Such a row currently matches on (i,jj). Scan column
        // jj looking for an unmatched row ii that improves value of matching. If
        // one exists, then augment along length 2 path (i,j)->(ii,jj)
        for k in k0..ptr[j + 1] {
            let i = row[k];
            if (val[k] - dualu[i]) > vj {
                continue; // Not a tie for vj value
            }
            let jj = iperm[i] as usize;
            // Scan remaining part of assigned column jj
            for kk in search_from[jj]..ptr[jj + 1] {
                let ii = row[kk];
                if iperm[ii] > 0 {
                    continue; // row ii already matched
                }
                if (val[kk] - dualu[ii]) <= d[jj] {
                    // By matching on (i,j) and (ii,jj) we do better than existing
                    // matching on (i,jj) alone.
                    jperm[jj] = kk;
                    iperm[ii] = jj as i32;
                    search_from[jj] = kk + 1;
                    *num += 1;
                    jperm[j] = k;
                    iperm[i] = j as i32;
                    search_from[j] = k + 1;
                    continue 'improve_assign;
                }
            }
            search_from[jj] = ptr[jj + 1];
        }
    }
}

// Provides the core Hungarian Algorithm implementation for solving the
// minimum sum assignment problem as per Duff and Koster.
//
// This code is adapted from MC64 v 1.6.0
fn hungarian_match(
    m: usize,          // number of rows
    n: usize,          // number of cols
    ptr: &[usize],     // column pointers
    row: &[usize],     // row pointers
    val: &[f64], // value of the entry that corresponds to row[k]. All values val[k] must be non-negative.
    iperm: &mut [i32], // matching itself: row i is matched to column iperm[i]
    num: &mut usize, // cardinality of the matching
    dualu: &mut [f64], // dualu[i] is the reduced weight for row[i]
    dualv: &mut [f64], // dualv[j] is the reduced weight for col[j]
) {
    // ) -> Result<(Vec<i32>, usize, Vec<f64>, Vec<f64>), Box<dyn std::error::Error>> {
    // let mut iperm = vec![-1; m]; // matching itself: row i is matched to column iperm[i]
    // let mut num = 0; // cardinality of the matching
    // let mut dualu = vec![0.0; m]; // dualu[i] is the reduced weight for row[i]
    // let mut dualv = vec![0.0; n]; // dualv[j] is the reduced weight for col[j]

    // a[jperm[j]] is entry of A for matching in column j.
    let mut jperm = vec![0; n];
    // a[out[i]] is the new entry in a on which we match going along the short path back to original col.
    let mut out = vec![0; n];
    // pr[i] is a pointer to the next column along the shortest path back to the original column
    let mut pr = vec![0; n];
    // q[1:qlen] forms a binary heap
    // data structure sorted by d[q[i]] value. q[low:up] is a list of rows
    // with equal d[i] which is lower or equal to smallest in the heap.
    // q[up:n] is a list of already visited rows.
    let mut q = vec![0; m];
    let mut longwork = vec![0; m];
    // l(:) is an inverse of q(:)
    let mut l = vec![0; m];
    // d[i] is current shortest distance
    // to row i from current column (d_i from Fig 4.1 of Duff and Koster paper)
    let mut d = vec![f64::INFINITY; usize::max(m, n)];

    hungarian_init_heuristic(
        m,
        n,
        ptr,
        row,
        val,
        num,
        iperm,
        &mut jperm,
        dualu,
        &mut d,
        &mut longwork,
        &mut out,
    );

    if *num == usize::min(m, n) {
        // return Ok((iperm, num, dualu, dualv)); // If we got a complete matching, we're done  TODO: check go to 1000
        return; // If we got a complete matching, we're done  TODO: check go to 1000
    }

    // Repeatedly find augmenting paths until all columns are included in the
    // matching. At every step the current matching is optimal on the restriction
    // of the graph to currently matched rows and columns.

    // Main loop ... each pass round this loop is similar to Dijkstra's
    // algorithm for solving the single source shortest path problem.
    for jord in 0..n {
        if jperm[jord] != 0 {
            continue;
        }
        // jord is next unmatched column
        // dmin is the length of shortest path in the tree
        let mut dmin = f64::INFINITY;
        let mut qlen = 0;
        let mut low = m;
        let mut up = m;
        // csp is the cost of the shortest augmenting path to unassigned row
        // row[isp]. The corresponding column index is jsp.
        let mut csp = f64::INFINITY;
        // Build shortest path tree starting from unassigned column (root) jord
        let j = jord;
        let mut isp = 0;
        let mut jsp = 0;
        pr[j] = !0; // Using !0 (maximum value) as -1 for usize

        // Scan column j
        for klong in ptr[j]..ptr[j + 1] {
            let i = row[klong];
            let dnew = val[klong] - dualu[i];
            if dnew >= csp {
                continue;
            }
            if iperm[i] == -1 {
                csp = dnew;
                isp = klong;
                jsp = j;
            } else {
                if dnew < dmin {
                    dmin = dnew;
                }
                d[i] = dnew;
                qlen += 1;
                longwork[qlen - 1] = klong;
            }
        }

        // Initialize heap Q and Q2 with rows held in longwork[:qlen]
        let q0 = qlen;
        qlen = 0;
        for kk in 0..q0 {
            let klong = longwork[kk];
            let i = row[klong];
            if csp <= d[i] {
                d[i] = f64::INFINITY;
                continue;
            }
            if d[i] <= dmin {
                low -= 1;
                q[low] = i;
                l[i] = low;
            } else {
                qlen += 1;
                l[i] = qlen;
                heap_update(i, m, &mut q, &mut d, &mut l);
            }
            let jj = iperm[i] as usize;
            out[jj] = klong;
            pr[jj] = j;
        }

        // Main loop for finding augmenting paths
        for _ in 0..*num {
            // If Q2 is empty, extract rows from Q
            if low == up {
                if qlen == 0 {
                    break;
                }
                let i = q[0]; // Peek at top of heap
                if d[i] >= csp {
                    break;
                }
                dmin = d[i];
                // Extract all paths that have length dmin and store in q[low:up]
                while qlen > 0 {
                    let i = q[0]; // Peek at top of heap
                    if d[i] > dmin {
                        break;
                    }
                    let i = heap_pop(&mut qlen, m, &mut q, &mut d, &mut l);
                    low -= 1;
                    q[low] = i;
                    l[i] = low;
                }
            }

            // q0 is row whose distance d[q0] to the root is smallest
            let q0 = q[up - 1];
            let dq0 = d[q0];
            // Exit loop if path to q0 is longer than the shortest augmenting path
            if dq0 >= csp {
                break;
            }
            up -= 1;

            // Scan column that matches with row q0
            let j = iperm[q0] as usize;
            let vj = dq0 - val[jperm[j]] + dualu[q0];

            for klong in ptr[j]..ptr[j + 1] {
                let i = row[klong];
                if l[i] >= up {
                    continue;
                }
                // dnew is new cost
                let dnew = vj + val[klong] - dualu[i];
                // Do not update d[i] if dnew ge cost of shortest path
                if dnew >= csp {
                    continue;
                }
                if iperm[i] == -1 {
                    // Row i is unmatched; update shortest path info
                    csp = dnew;
                    isp = klong;
                    jsp = j;
                } else {
                    // Row i is matched; do not update d[i] if dnew is larger
                    let di = d[i];
                    if di <= dnew {
                        continue;
                    }
                    if l[i] >= low {
                        continue;
                    }
                    d[i] = dnew;
                    if dnew <= dmin {
                        let lpos = l[i];
                        if lpos != 0 {
                            heap_delete(lpos, &mut qlen, m, &mut q, &mut d, &mut l);
                        }
                        low -= 1;
                        q[low] = i;
                        l[i] = low;
                    } else {
                        if l[i] == 0 {
                            qlen += 1;
                            l[i] = qlen;
                        }
                        heap_update(i, m, &mut q, &mut d, &mut l);
                    }
                    let jj = iperm[i] as usize;
                    out[jj] = klong;
                    pr[jj] = j;
                }
            }
        }

        // If csp = RINF, no augmenting path is found
        if csp == f64::INFINITY {
            continue; // TODO: goto 190
        }

        // Find augmenting path by tracing backward in pr; update iperm,jperm
        *num += 1;
        let mut i = row[isp];
        iperm[i] = jsp as i32;
        jperm[jsp] = isp;
        let mut j = jsp;

        // loop {
        for _ in 0..*num {
            let jj = pr[j];
            if jj == !0 {
                break;
            }
            let klong = out[j];
            i = row[klong];
            iperm[i] = jj as i32;
            jperm[jj] = klong;
            j = jj;
        }

        // Update U for rows in q(up:m)  TODO: check range
        for kk in up..m {
            let i = q[kk];
            dualu[i] = dualu[i] + d[i] - csp;
        }

        // 190:
        for kk in low..m {
            let i = q[kk];
            d[i] = f64::INFINITY;
            l[i] = 0;
        }
        for kk in 0..qlen {
            let i = q[kk];
            d[i] = f64::INFINITY;
            l[i] = 0;
        }
    }

    // 1000:
    // Set dual column variables
    for j in 0..n {
        let klong = jperm[j];
        if klong != 0 {
            dualv[j] = val[klong] - dualu[row[klong]];
        } else {
            dualv[j] = 0.0;
        }
    }

    // Zero dual row variables for unmatched rows
    for i in 0..m {
        if iperm[i] == -1 {
            dualu[i] = 0.0;
        }
    }

    // Complete iperm for structurally singular matrix
    if *num != usize::min(m, n) {
        jperm.fill(0);
        let mut k = 0;
        for i in 0..m {
            if iperm[i] == -1 {
                k += 1;
                out[k - 1] = i;
            } else {
                let j = iperm[i] as usize;
                jperm[j] = i;
            }
        }
        k = 0;
        for j in 0..n {
            if jperm[j] != 0 {
                continue;
            }
            k += 1;
            let jdum = out[k - 1];
            iperm[jdum] = -(j as i32) - 1;
        }
    }

    // Ok((iperm, num, dualu, dualv))
}

// Update the position of an element in the heap when its value has decreased.
//
// This function is adapted from MC64 v 1.6.0
fn heap_update(idx: usize, _n: usize, q: &mut [usize], val: &[f64], l: &mut [usize]) {
    // Get current position of i in heap.
    let mut pos = l[idx];
    if pos <= 1 {
        // idx is already at root of heap, but set q as it may have only just been inserted.
        q[pos - 1] = idx;
        return;
    }

    // Keep trying to move i towards root of heap until it can't go any further
    let v = val[idx];
    while pos > 1 {
        let parent_pos = pos / 2;
        let parent_idx = q[parent_pos - 1];
        // If parent is better than idx, stop moving
        if v >= val[parent_idx] {
            break;
        }
        // Otherwise, swap idx and parent
        q[pos - 1] = parent_idx;
        l[parent_idx] = pos;
        pos = parent_pos;
    }
    // Finally set idx in the place it reached.
    q[pos - 1] = idx;
    l[idx] = pos;
}

// Remove and return the root node from the binary heap.
//
// This function is adapted from MC64 v 1.6.0
fn heap_pop(qlen: &mut usize, n: usize, q: &mut [usize], val: &[f64], l: &mut [usize]) -> usize {
    // Return value is the old root of the heap
    let root = q[0];

    // Delete the root
    heap_delete(1, qlen, n, q, val, l);

    root
}

// Delete an element at a given position from the heap.
//
// This function is adapted from MC64 v 1.6.0
fn heap_delete(
    pos0: usize,
    qlen: &mut usize,
    _n: usize,
    q: &mut [usize],
    d: &[f64],
    l: &mut [usize],
) {
    // If we're trying to remove the last item, just delete it.
    if *qlen == pos0 {
        *qlen -= 1;
        return;
    }

    // Replace index in position pos0 with last item and fix heap property
    let idx = q[*qlen - 1];
    let v = d[idx];
    *qlen -= 1; // shrink heap
    let mut pos = pos0; // pos is current position of node I in the tree

    // Move up if appropriate
    if pos > 1 {
        loop {
            let parent = pos / 2;
            let qk = q[parent - 1];
            if v >= d[qk] {
                break;
            }
            q[pos - 1] = qk;
            l[qk] = pos;
            pos = parent;
            if pos <= 1 {
                break;
            }
        }
    }
    q[pos - 1] = idx;
    l[idx] = pos;
    if pos != pos0 {
        return; // Item moved up, hence doesn't need to move down
    }

    // Otherwise, move item down
    loop {
        let mut child = 2 * pos;
        if child > *qlen {
            break;
        }
        let mut dk = d[q[child - 1]];
        if child < *qlen {
            let dr = d[q[child]];
            if dk > dr {
                child += 1;
                dk = dr;
            }
        }
        if v <= dk {
            break;
        }
        let qk = q[child - 1];
        q[pos - 1] = qk;
        l[qk] = pos;
        pos = child;
    }
    q[pos - 1] = idx;
    l[idx] = pos;
}
