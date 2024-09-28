// Performs post-processing on the results of the Hungarian algorithm.
//
// This function adjusts the scaling factors for rows and columns based on the matching
// results, and handles cases where the number of rows and columns are not equal.
pub(crate) fn match_postproc(
    m: usize,
    n: usize,
    ptr: &[usize],
    row: &[usize],
    val: &[f64],
    rscaling: &mut [f64],
    cscaling: &mut [f64],
    nmatch: usize,
    match_result: &[i32],
    _flag: &mut i32,
) {
    if m == n {
        // Square
        // Just perform post-processing and magnitude adjustment
        let ravg = rscaling.iter().sum::<f64>() / m as f64;
        let cavg = cscaling.iter().sum::<f64>() / n as f64;
        let adjust = (ravg - cavg) / 2.0;
        for r in rscaling {
            *r -= adjust;
        }
        for c in cscaling {
            *c += adjust;
        }
    } else if m < n {
        // More columns than rows
        let mut cmax = vec![0.0; n]; // Allocate some workspace

        // First perform post-processing and magnitude adjustment based on match
        let mut ravg = 0.0;
        let mut cavg = 0.0;
        for (i, &m) in match_result.iter().enumerate() {
            if m == 0 {
                continue;
            }
            ravg += rscaling[i];
            cavg += cscaling[m as usize];
        }
        ravg /= nmatch as f64;
        cavg /= nmatch as f64;
        let adjust = (ravg - cavg) / 2.0;
        for r in rscaling.iter_mut() {
            *r -= adjust;
        }
        for c in cscaling.iter_mut() {
            *c += adjust;
        }

        // For each unmatched col, scale max entry to 1.0
        for i in 0..n {
            let colmax = (ptr[i]..ptr[i + 1])
                .map(|j| (val[j] * rscaling[row[j]].exp()).abs())
                .fold(0.0, f64::max);
            cmax[i] = if colmax == 0.0 {
                0.0
            } else {
                (1.0 / colmax).ln()
            };
        }
        for &m in match_result {
            if m != 0 {
                cmax[m as usize] = cscaling[m as usize];
            }
        }
        cscaling.copy_from_slice(&cmax);
    } else if n < m {
        // More rows than columns
        let mut rmax = vec![0.0; m]; // Allocate some workspace

        // First perform post-processing and magnitude adjustment based on match
        let mut ravg = 0.0;
        let mut cavg = 0.0;
        for (i, &m) in match_result.iter().enumerate() {
            if m == 0 {
                continue;
            }
            ravg += rscaling[i];
            cavg += cscaling[m as usize];
        }
        ravg /= nmatch as f64;
        cavg /= nmatch as f64;
        let adjust = (ravg - cavg) / 2.0;
        for r in rscaling.iter_mut() {
            *r -= adjust;
        }
        for c in cscaling.iter_mut() {
            *c += adjust;
        }

        // Find max column-scaled value in each row from unmatched cols
        for i in 0..n {
            for j in ptr[i]..ptr[i + 1] {
                let v = (val[j] * cscaling[i].exp()).abs();
                rmax[row[j]] = f64::max(rmax[row[j]], v);
            }
        }

        // Calculate scaling for each row, but overwrite with correct values for
        // matched rows, then copy entire array over rscaling[:]
        for (i, &m) in match_result.iter().enumerate() {
            if m != 0 {
                continue;
            }
            rscaling[i] = if rmax[i] == 0.0 {
                0.0
            } else {
                (1.0 / rmax[i]).ln()
            };
        }
    }
}
