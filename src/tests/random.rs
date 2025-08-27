// COPYRIGHT (c) 2014 Science and Technology Facilities Council
// Authors: Jonathan Hogg
//
// Implementation of simple LCG PRNG
// Parameters as in glibc module spral_random

/// Store random generator state
pub struct RandomState {
    x: i32,
}

impl Default for RandomState {
    fn default() -> Self {
        RandomState { x: 486502 }
    }
}

impl RandomState {
    // /// Get random seed
    // pub fn get_seed(&self) -> i32 {
    //     self.x
    // }

    // /// Set random seed
    // pub fn set_seed(&mut self, seed: i32) {
    //     self.x = seed;
    // }

    /// Real random number in the range
    /// [ 0, 1] (if positive is present and true); or
    /// [-1, 1] (otherwise)
    pub fn random_real(&mut self, positive: Option<bool>) -> f64 {
        let pos = positive.unwrap_or(false);

        // X_{n+1} = (aX_n + c) mod m
        let a: i64 = 1103515245;
        let c: i64 = 12345;
        let m: i64 = 2_i64.pow(31);

        // Need to convert to i64 to avoid overflow
        let x_i64 = self.x as i64;
        self.x = ((a * x_i64 + c) % m) as i32;

        // Convert to a random real
        if pos {
            self.x as f64 / m as f64
        } else {
            1.0 - 2.0 * (self.x as f64) / (m as f64)
        }
    }

    /// Integer random number in the range [1,n] if n > 1.
    /// otherwise, the value n is returned
    pub fn random_integer64(&mut self, n: i64) -> i64 {
        if n <= 0 {
            return n;
        }

        // X_{n+1} = (aX_n + c) mod m
        let a: i64 = 1103515245;
        let c: i64 = 12345;
        let m: i64 = 2_i64.pow(31);

        // Need to convert to i64 to avoid overflow
        let x_i64 = self.x as i64;
        self.x = ((a * x_i64 + c) % m) as i32;

        // Take modulo n for return value
        ((self.x as f64) * (n as f64) / (m as f64)) as i64 + 1
    }

    /// Integer random number in the range [1,n] if n > 1.
    /// otherwise, the value n is returned
    pub fn random_integer(&mut self, n: usize) -> usize {
        if n <= 0 {
            return n;
        }

        // X_{n+1} = (aX_n + c) mod m
        let a: usize = 1103515245;
        let c: usize = 12345;
        let m: usize = 2_usize.pow(31);

        // Need to convert to i64 to avoid overflow
        let x_usize = self.x as usize;
        self.x = ((a * x_usize + c) % m) as i32;

        // Take modulo n for return value
        ((self.x as f64) * (n as f64) / (m as f64)) as usize + 1
    }

    /// Integer random number in the range [1,n] if n > 1.
    /// otherwise, the value n is returned
    pub fn random_integer32(&mut self, n: i32) -> i32 {
        // Just call 64-bit version with type casts
        self.random_integer64(n as i64) as i32
    }

    // /// Generate a random logical value
    // pub fn random_logical(&mut self) -> bool {
    //     let test = self.random_integer32(2);
    //     test == 1
    // }
}

// Public API functions that mirror the Fortran interface
pub fn random_real(state: &mut RandomState, positive: Option<bool>) -> f64 {
    state.random_real(positive)
}

pub fn random_integer32(state: &mut RandomState, n: i32) -> i32 {
    state.random_integer32(n)
}

// pub fn random_integer64(state: &mut RandomState, n: i64) -> i64 {
//     state.random_integer64(n)
// }
//
// pub fn random_logical(state: &mut RandomState) -> bool {
//     state.random_logical()
// }
//
// pub fn random_get_seed(state: &RandomState) -> i32 {
//     state.get_seed()
// }
//
// pub fn random_set_seed(state: &mut RandomState, seed: i32) {
//     state.set_seed(seed)
// }
