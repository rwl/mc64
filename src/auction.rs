#[derive(Debug, Clone, Copy)]
pub struct AuctionOptions {
    pub array_base: usize, // Not in Fortran type
    pub max_iterations: usize,
    pub max_unchanged: [usize; 3],
    pub min_proportion: [f64; 3],
    pub eps_initial: f64,
}

impl Default for AuctionOptions {
    fn default() -> Self {
        Self {
            array_base: 0, // C
            max_iterations: 30000,
            max_unchanged: [10, 100, 100],
            min_proportion: [0.90, 0.0, 0.0],
            eps_initial: 0.01,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AuctionInform {
    pub flag: i32,
    pub stat: i32,
    pub matched: usize,
    pub iterations: usize,
    pub unmatchable: usize,
}
