use std::process::exit;

use mc64::{
    equilib_scale_sym, print_matrix, scale_matrix_sym, EquilibInform, EquilibOptions, MatrixType,
};

fn main() {
    /* Other variables */
    let mut scaling = [0.0; 5];

    /* Data for symmetric matrix:
     * ( 2  1         )
     * ( 1  4  1    8 )
     * (    1  3  2   )
     * (       2      )
     * (    8       2 ) */
    let n = 5;
    let ptr = [0, 2, 5, 7, 7, 8];
    let row = [0, 1, 1, 2, 4, 2, 3, 4];
    let mut val = [2.0, 1.0, 4.0, 1.0, 8.0, 3.0, 2.0, 2.0];
    println!("Initial matrix:");
    print_matrix(-1, MatrixType::RealSymIndef, n, n, &ptr, &row, Some(&val));

    /* Perform symmetric scaling */
    let options = EquilibOptions::default();
    let mut inform = EquilibInform::default();
    equilib_scale_sym(n, &ptr, &row, &val, &mut scaling, &options, &mut inform);
    if inform.flag < 0 {
        print!("scaling::equilib_sym() returned with error {}", inform.flag);
        exit(1);
    }

    /* Print scaling */
    println!("Scaling: {:?}", scaling);

    /* Calculate scaled matrix and print it */
    scale_matrix_sym(n, &ptr, &row, &mut val, &scaling);
    println!("Scaled matrix:");
    print_matrix(-1, MatrixType::RealSymIndef, n, n, &ptr, &row, Some(&val));
}
