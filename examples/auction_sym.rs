use std::process::exit;

use mc64::{
    auction_scale_sym, print_matrix, scale_matrix_sym, AuctionInform, AuctionOptions, MatrixType,
};

fn main() {
    /* Derived types */
    // struct spral_scaling_auction_options options;
    // struct spral_scaling_auction_inform inform;

    /* Other variables */
    let mut matching = [0; 5];
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

    // Perform symmetric scaling
    // spral_scaling_auction_default_options(&options);
    let options = AuctionOptions::default();
    let mut inform = AuctionInform::default();
    auction_scale_sym(
        n,
        &ptr,
        &row,
        &val,
        &mut scaling,
        &options,
        &mut inform,
        Some(&mut matching),
    );
    if inform.flag < 0 {
        print!(
            "scaling::auction_sym() returned with error {:.5}",
            inform.flag,
        );
        exit(1);
    }

    // Print scaling and matching
    println!("Matching: {:?}", matching);
    println!("Scaling:  {:?}", scaling);

    // Calculate scaled matrix and print it
    scale_matrix_sym(n, &ptr, &row, &mut val, &scaling);
    println!("Scaled matrix:");
    print_matrix(-1, MatrixType::RealSymIndef, n, n, &ptr, &row, Some(&val));
}
