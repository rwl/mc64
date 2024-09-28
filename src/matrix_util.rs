// Generate the expanded structure for a matrix a with a symmetric sparsity pattern
// given the structure for the lower triangular part.
// Diagonal entries need not be present.
//
// Note: this is a modified version of mc34_expand from hsl_mc34
pub(crate) fn half_to_full(
    // Holds the order of a.
    n: usize,
    // must be set by the user to
    // hold the row indices of the lower triangular part of a.
    // the entries of a single column must be
    // contiguous. the entries of column j must precede those of column
    // j+1, and there must be no wasted space between
    // columns. row indices within a column may be in any order.  on
    // exit, it will have the same meaning but will be changed to hold
    // the row indices of the entries in the expanded structure.  diagonal
    // entries need not be present. the new row indices added in the
    // upper triangular part will be in order for each column and will
    // precede the row indices for the lower triangular part which will
    // remain in the input order.
    row: &mut [usize],
    // must be set by the user so that ptr(j) is the position in row
    // of the first entry in column j and
    // ptr(n+1) must be set to one more than the total number of
    // entries.  on exit, ptr(j) will have the same meaning but
    // will be changed to point to the position of the first entry of
    // column j in the expanded structure. the new value of
    // ptr(n+1) will be one greater than the number of entries in
    // the expanded structure.
    ptr: &mut [usize],
    // workspace
    iw: &mut [i32],
    // if present, a(1:ptr(n+1)-1) must be set by the user so that
    // a(k) holds the value of the entry in row(k).
    // on exit, a will hold the values of the entries in the expanded
    // structure corresponding to the output values of row.
    mut a: Option<&mut [f64]>,
    cbase: bool,
) {
    let rebase = if cbase { 1 } else { 0 };

    let oldtau = ptr[n] - 1 + rebase;
    iw[..n].fill(0);

    // iw[j] set to total number entries in col. j of expanded mx.
    let mut ndiag = 0;
    for j in 0..n {
        let i1 = ptr[j] + rebase;
        let i2 = ptr[j + 1] - 1 + rebase;
        iw[j] += (i2 - i1 + 1) as i32;
        for ii in i1..=i2 {
            let i = row[ii] + rebase;
            if i != j + rebase {
                iw[i - rebase] += 1; // TODO: check rebase
            } else {
                ndiag += 1;
            }
        }
    }

    let newtau = 2 * oldtau - ndiag;
    // ipkp1 points to position after end of column being currently processed
    let mut ipkp1 = oldtau + 1;
    // ckp1 points to position after end of same column in expanded structure
    let mut ckp1 = newtau + 1;

    // go through the array in the reverse order placing lower triangular
    // elements in appropriate slots.
    for j in (0..n).rev() {
        let i1 = ptr[j] + rebase;
        let i2 = ipkp1;
        let lenk = i2 - i1;
        // jstart is running pointer to position in new structure
        let mut jstart = ckp1;
        // set ikp1 for next column
        ipkp1 = i1;
        let i2 = i2 - 1;

        // run through columns in reverse order
        // lower triangular part of col. moved to end of same column in expanded form
        if let Some(a) = a.as_mut() {
            for ii in (i1..=i2).rev() {
                jstart -= 1;
                a[jstart] = a[ii];
                row[jstart] = row[ii]; // rebase cancels
            }
        } else {
            for ii in (i1..=i2).rev() {
                jstart -= 1;
                row[jstart] = row[ii]; // rebase cancels
            }
        }

        // ptr is set to position of first entry in lower triangular part of
        // column j in expanded form
        ptr[j] = jstart - rebase;
        // set ckp1 for next column
        ckp1 = ckp1 - iw[j] as usize;
        // reset iw[j] to number of entries in lower triangle of column.
        iw[j] = lenk as i32;
    }

    // again sweep through the columns in the reverse order, this
    // time when one is handling column j the upper triangular
    // elements a(j,i) are put in position.
    for j in (0..n).rev() {
        let i1 = ptr[j] + rebase;
        let i2 = ptr[j] + iw[j] as usize - 1 + rebase;
        // run down column in order
        // note that i is always greater than or equal to j
        if let Some(a) = a.as_mut() {
            for ii in i1..=i2 {
                let i = row[ii] + rebase;
                // TODO: check rebase
                if i == j + rebase {
                    continue;
                }
                ptr[i - rebase] -= 1; // rebase cancels
                let ipos = ptr[i - rebase];
                a[ipos] = a[ii];
                row[ipos] = j - rebase;
            }
        } else {
            for ii in i1..=i2 {
                let i = row[ii] as usize + rebase;
                if i == j + rebase {
                    continue;
                }
                ptr[i - rebase] -= 1; // rebase cancels  TODO: check rebase
                let ipos = ptr[i - rebase];
                row[ipos] = j - rebase;
            }
        }
    }
    ptr[n] = newtau + 1 - rebase;
}
