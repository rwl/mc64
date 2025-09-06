use crate::random::RandomState;

const NSAMPLES: usize = 50000;
const NBINS: usize = 100;
const REQUIRE_CONFIDENCE: f64 = 0.99;

#[test]
fn test_real_dist() {
    let mut state = RandomState::default();
    let mut bin = [0; NBINS];

    // Test (-1,1) distribution
    println!("Sampling Unif(-1,1)...... ");
    for _ in 0..NSAMPLES {
        let sample = state.random_real(None);
        let j = (NBINS as f64 * ((sample + 1.0) / 2.0)) as usize;
        if j < NBINS {
            bin[j] += 1;
        } else {
            bin[NBINS - 1] += 1;
        }
    }
    let chisq = (bin.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() * NBINS as f64)
        / NSAMPLES as f64
        - NSAMPLES as f64;
    let critical_value = chisq_pval(NBINS - 1, REQUIRE_CONFIDENCE);
    assert!(
        chisq < critical_value,
        "chisq statistic = {}, chisq required < {}",
        chisq,
        critical_value
    );

    // Test (0,1) distribution
    println!("Sampling Unif(0,1)....... ");
    bin.iter_mut().for_each(|m| *m = 0);
    for _ in 0..NSAMPLES {
        let sample = state.random_real(Some(true));
        let j = (NBINS as f64 * sample) as usize;
        if j < NBINS {
            bin[j] += 1;
        } else {
            bin[NBINS - 1] += 1;
        }
    }
    let chisq = (bin.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() * NBINS as f64)
        / NSAMPLES as f64
        - NSAMPLES as f64;
    let critical_value = chisq_pval(NBINS - 1, REQUIRE_CONFIDENCE);
    assert!(
        chisq < critical_value,
        "chisq statistic = {}, chisq required < {}",
        chisq,
        critical_value
    );
}

#[test]
fn test_integer_dist() {
    let mut state = RandomState::default();
    let mut bin = [0_usize; NBINS];

    println!("Sampling Unif(1,...,n)... ");
    for _ in 0..NSAMPLES {
        let sample = state.random_integer(NBINS);
        bin[sample] += 1;
    }
    let chisq = (bin.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() * NBINS as f64)
        / NSAMPLES as f64
        - NSAMPLES as f64;
    let critical_value = chisq_pval(NBINS - 1, REQUIRE_CONFIDENCE);
    assert!(
        chisq < critical_value,
        "chisq statistic = {}, chisq required < {}",
        chisq,
        critical_value
    );
}

fn chisq_pval(dof: usize, p: f64) -> f64 {
    let xp = if p == 0.99 {
        2.33
    } else {
        panic!("Uncoded pval for chisq_pval");
    };

    dof as f64 + (2.0 * dof as f64).sqrt() * xp + (2.0 * xp.powi(2)) / 3.0 - 2.0 / 3.0
}
