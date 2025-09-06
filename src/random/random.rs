// COPYRIGHT (c) 2014 Science and Technology Facilities Council
// Authors: Jonathan Hogg
//
// Implementation of simple LCG PRNG

/// Store random generator state
pub struct RandomState {
    x: usize,
}

impl Default for RandomState {
    fn default() -> Self {
        RandomState { x: 486502 }
    }
}

impl RandomState {
    // pub fn with_seed(seed: usize) -> Self {
    //     Self { x: seed }
    // }

    /// Real random number in the range
    /// [ 0, 1] (if positive is present and true); or
    /// [-1, 1] (otherwise)
    pub fn random_real(&mut self, positive: Option<bool>) -> f64 {
        let pos = positive.unwrap_or(false);

        // X_{n+1} = (aX_n + c) mod m
        let a: usize = 1103515245;
        let c: usize = 12345;
        let m: usize = 2_usize.pow(31);

        self.x = (a * self.x + c) % m;

        // Convert to a random real
        if pos {
            self.x as f64 / m as f64
        } else {
            1.0 - 2.0 * (self.x as f64) / (m as f64)
        }
    }

    /// Integer random number in the range `[0, n)` if n > 0.
    /// otherwise, the value n is returned
    pub fn random_integer(&mut self, n: usize) -> usize {
        if n == 0 {
            return n;
        }

        // X_{n+1} = (aX_n + c) mod m
        let a: usize = 1103515245;
        let c: usize = 12345;
        let m: usize = 2_usize.pow(31);

        self.x = (a * self.x + c) % m;

        // Take modulo n for return value
        ((self.x as f64) * ((n as f64) / (m as f64))) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::RandomState;

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
}
