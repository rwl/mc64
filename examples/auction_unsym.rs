use std::process::exit;

use mc64::{
    auction_scale_unsym, print_matrix, scale_matrix_unsym, AuctionInform, AuctionOptions,
    MatrixType,
};

fn main() {
    /* Other variables */
    let mut matching = [0; 5];
    let mut rscaling = [0.0; 5];
    let mut cscaling = [0.0; 5];

    /* Data for unsymmetric matrix:
     * ( 2  5         )
     * ( 1  4       7 )
     * (    1     2   )
     * (       3      )
     * (    8       2 ) */
    let m = 5;
    let n = 5;
    let ptr = [0, 2, 6, 7, 8, 10];
    let row = [0, 1, 0, 1, 2, 4, 3, 2, 1, 4];
    let mut val = [2.0, 1.0, 5.0, 4.0, 1.0, 8.0, 3.0, 2.0, 7.0, 2.0];
    println!("Initial matrix:");
    print_matrix(-1, MatrixType::RealUnsym, m, n, &ptr, &row, Some(&val));

    /* Perform unsymmetric scaling */
    let options = AuctionOptions::default();
    let mut inform = AuctionInform::default();
    auction_scale_unsym(
        m,
        n,
        &ptr,
        &row,
        &val,
        &mut rscaling,
        &mut cscaling,
        &options,
        &mut inform,
        Some(&mut matching),
    );
    if inform.flag < 0 {
        print!(
            "scaling::auction_unsym() returned with error {}",
            inform.flag
        );
        exit(1);
    }

    /* Print scaling and matching */
    println!("Matching:    {:?}", matching);
    println!("Row Scaling: {:?}", rscaling);
    println!("Col Scaling: {:?}", cscaling);

    /* Calculate scaled matrix and print it */
    scale_matrix_unsym(m, n, &ptr, &row, &mut val, &rscaling, &cscaling);
    println!("Scaled matrix:");
    print_matrix(-1, MatrixType::RealUnsym, m, n, &ptr, &row, Some(&val));
}
