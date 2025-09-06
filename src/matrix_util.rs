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
    // cbase: bool,
) {
    // let rebase = if cbase { 1 } else { 0 };

    let oldtau = ptr[n]; // number of entries in symmetric storage
    iw[..n].fill(0);

    // iw[j] set to total number entries in col. j of expanded mx.
    let mut ndiag = 0; // number diagonal entries present
    for j in 0..n {
        let i1 = ptr[j];
        let i2 = ptr[j + 1];
        iw[j] += (i2 - i1) as i32; /* + 1;*/
        for ii in i1..i2 {
            let i = row[ii];
            if i != j {
                iw[i] += 1;
            } else {
                ndiag += 1;
            }
        }
    }
    // number of entries in expanded storage
    let newtau = 2 * oldtau - ndiag;
    // ipkp1 points to position after end of column being currently processed
    let mut ipkp1 = oldtau; /* + 1;*/
    // ckp1 points to position after end of same column in expanded structure
    let mut ckp1 = newtau; /* + 1;*/

    // go through the array in the reverse order placing lower triangular
    // elements in appropriate slots.
    for j in (0..n).rev() {
        let i1 = ptr[j];
        let i2 = ipkp1;
        // number of entries in col. j of original structure
        let lenk = i2 - i1;
        // jstart is running pointer to position in new structure
        let mut jstart = ckp1;
        // set ikp1 for next column
        ipkp1 = i1;
        //let i2 = i2 - 1;

        // run through columns in reverse order, from i2-1 down to i1.
        // lower triangular part of col. moved to end of same column in expanded form
        for ii in (i1..i2).rev() {
            jstart -= 1;
            if let Some(a) = a.as_mut() {
                a[jstart] = a[ii];
            }
            row[jstart] = row[ii];
        }

        // ptr is set to position of first entry in lower triangular part of
        // column j in expanded form
        ptr[j] = jstart;
        // set ckp1 for next column
        ckp1 -= iw[j] as usize;
        // reset iw[j] to number of entries in lower triangle of column.
        iw[j] = lenk as i32;
    }

    // again sweep through the columns in the reverse order, this
    // time when one is handling column j the upper triangular
    // elements a(j,i) are put in position.
    for j in (0..n).rev() {
        let i1 = ptr[j];
        let i2 = ptr[j] + iw[j] as usize;
        // run down column in order
        // note that i is always greater than or equal to j
        if let Some(a) = a.as_mut() {
            for ii in i1..i2 {
                let i = row[ii];
                if i == j {
                    continue;
                }
                ptr[i] -= 1;
                let ipos = ptr[i];
                a[ipos] = a[ii];
                row[ipos] = j;
            }
        } else {
            for ii in i1..i2 {
                let i = row[ii];
                if i == j {
                    continue;
                }
                ptr[i] -= 1;
                let ipos = ptr[i];
                row[ipos] = j;
            }
        }
    }
    ptr[n] = newtau;
}
