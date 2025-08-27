use std::process::exit;

use mc64::{hungarian_scale_unsym, HungarianInform, HungarianOptions};

fn main() {
    /* Derived types */
    // struct spral_scaling_hungarian_options options;
    // struct spral_scaling_hungarian_inform inform;

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
    let (m, n) = (5, 5);
    let ptr = [0, 2, 6, 7, 8, 10];
    let row = [0, 1, 0, 1, 2, 4, 3, 2, 1, 4];
    let mut val = [2.0, 1.0, 5.0, 4.0, 1.0, 8.0, 3.0, 2.0, 7.0, 2.0];
    println!("Initial matrix:");
    // spral::matrix::print_matrix(
    //     -1,
    //     spral::matrix::MatrixType::RealUnsym,
    //     m,
    //     n,
    //     &ptr,
    //     &row,
    //     &val,
    //     0,
    // ); TODO:

    // Perform symmetric scaling
    // spral_scaling_hungarian_default_options(&options);
    let options = HungarianOptions::default();
    let mut inform = HungarianInform::default();
    hungarian_scale_unsym(
        m,
        n,
        &ptr,
        &row,
        &val,
        &mut rscaling,
        &mut cscaling,
        Some(&mut matching),
        &options,
        &mut inform,
    );
    if inform.flag < 0 {
        print!(
            "scaling::hungarian_unsym() returned with error {:.5}",
            inform.flag,
        );
        exit(1);
    }

    // Print scaling and matching
    println!("Matching: {:?}", matching);
    println!("Row Scaling: {:?}", rscaling);
    println!("Col Scaling: {:?}", cscaling);

    // Calculate scaled matrix and print it
    for i in 0..n {
        for j in ptr[i]..ptr[i + 1] {
            val[j] = rscaling[row[j]] * val[j] * cscaling[i];
        }
    }
    println!("Scaled matrix:");
    // spral::matrix::print_matrix(
    //     -1,
    //     spral::matrix::MatrixType::RealUnsym,
    //     m,
    //     n,
    //     &ptr,
    //     &row,
    //     &val,
    //     0,
    // ); TODO:
}
