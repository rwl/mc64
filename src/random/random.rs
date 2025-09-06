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

    /// Integer random number in the range [0, n-1] if n > 0.
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
        ((self.x as f64) * (n as f64) / (m as f64)) as usize
    }
}
