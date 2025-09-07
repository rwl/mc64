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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_half_to_full() {
        // This test checks the expansion of a 4x4 symmetric matrix
        // from its lower triangular representation.
        //
        // The matrix A is:
        // | 10  1  0  2 |
        // |  1 11  3  0 |
        // |  0  3 12  4 |
        // |  2  0  4 13 |
        //
        // The lower triangular part (L) in CSC format is used as input.
        // Non-zero entries of L:
        // Col 0: (0,0)=10, (1,0)=1, (3,0)=2
        // Col 1: (1,1)=11, (2,1)=3
        // Col 2: (2,2)=12, (3,2)=4
        // Col 3: (3,3)=13

        let n = 4;

        // --- Test case 1: With numerical values ---
        let mut row_with_a = vec![0, 1, 3, 1, 2, 2, 3, 3];
        let mut ptr_with_a = vec![0, 3, 5, 7, 8, 0]; // ptr has size n+2 in mc64, extra space for safety
        let mut a = vec![10.0, 1.0, 2.0, 11.0, 3.0, 12.0, 4.0, 13.0];
        let mut iw = vec![0; n];

        // The expanded structure will have 12 entries.
        // old_nnz = 8, ndiag = 4, new_nnz = 2*8 - 4 = 12.
        row_with_a.resize(12, 0);
        a.resize(12, 0.0);

        half_to_full(
            n,
            &mut row_with_a,
            &mut ptr_with_a[..n + 1],
            &mut iw,
            Some(&mut a),
        );

        // Expected full matrix in CSC format.
        // The function places upper triangular entries before lower triangular ones.
        // Col 0: (0,0)=10, (1,0)=1, (3,0)=2
        // Col 1: (0,1)=1, (1,1)=11, (2,1)=3
        // Col 2: (1,2)=3, (2,2)=12, (3,2)=4
        // Col 3: (0,3)=2, (2,3)=4, (3,3)=13
        let expected_ptr = vec![0, 3, 6, 9, 12];
        let expected_row = vec![0, 1, 3, 0, 1, 2, 1, 2, 3, 0, 2, 3];
        let expected_a: Vec<f64> = vec![
            10.0, 1.0, 2.0, 1.0, 11.0, 3.0, 3.0, 12.0, 4.0, 2.0, 4.0, 13.0,
        ];

        assert_eq!(&ptr_with_a[..n + 1], expected_ptr.as_slice());

        // The order of elements within a column can vary after expansion,
        // so we check column by column.
        for j in 0..n {
            let start = ptr_with_a[j];
            let end = ptr_with_a[j + 1];
            let mut col_entries: Vec<_> = row_with_a[start..end]
                .iter()
                .zip(&a[start..end])
                .map(|(&r, &v)| (r, v.to_bits())) // Use to_bits for float comparison
                .collect();
            col_entries.sort_unstable_by_key(|k| k.0);

            let expected_start = expected_ptr[j];
            let expected_end = expected_ptr[j + 1];
            let mut expected_col_entries: Vec<_> = expected_row[expected_start..expected_end]
                .iter()
                .zip(&expected_a[expected_start..expected_end])
                .map(|(&r, &v)| (r, v.to_bits()))
                .collect();
            expected_col_entries.sort_unstable_by_key(|k| k.0);

            assert_eq!(
                col_entries, expected_col_entries,
                "Mismatch in column {}",
                j
            );
        }

        // --- Test case 2: Without numerical values (structure only) ---
        let mut row_only = vec![0, 1, 3, 1, 2, 2, 3, 3];
        let mut ptr_only = vec![0, 3, 5, 7, 8, 0];
        let mut iw2 = vec![0; n];
        row_only.resize(12, 0);

        half_to_full(n, &mut row_only, &mut ptr_only[..n + 1], &mut iw2, None);

        assert_eq!(&ptr_only[..n + 1], expected_ptr.as_slice());

        for j in 0..n {
            let mut col_rows = row_only[ptr_only[j]..ptr_only[j + 1]].to_vec();
            col_rows.sort_unstable();
            let mut expected_col_rows = expected_row[expected_ptr[j]..expected_ptr[j + 1]].to_vec();
            expected_col_rows.sort_unstable();
            assert_eq!(
                col_rows, expected_col_rows,
                "Mismatch in column {} (structure only)",
                j
            );
        }
    }
}
