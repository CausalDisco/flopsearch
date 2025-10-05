use nalgebra::{Cholesky, DMatrix, Dyn};

use crate::bic::Bic;

pub fn init_score(data: &DMatrix<f64>, lambda: f64) -> Bic {
    Bic::new(data, lambda)
}

#[derive(Clone, Debug)]
pub struct GlobalScore {
    pub p: usize,
    pub local_scores: Vec<LocalScore>,
}

#[derive(Clone, Debug)]
pub struct LocalScore {
    pub bic: f64,
    pub chol: Cholesky<f64, Dyn>,
    pub parents: Vec<usize>,
}

impl GlobalScore {
    pub fn new(p: usize, score: &Bic) -> Self {
        Self {
            p,
            local_scores: (0..p)
                .map(|v| score.local_score_init(v, Vec::new()))
                .collect(),
        }
    }

    pub fn score(&self) -> f64 {
        self.local_scores.iter().map(|ls| ls.bic).sum()
    }
}
