use crate::{dynamic_cholesky::Cholesky, error::ScoreError};
use nalgebra::DMatrix;

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
    pub chol: Cholesky,
    pub parents: Vec<usize>,
}

impl GlobalScore {
    pub fn new(p: usize, score: &Bic) -> Result<Self, ScoreError> {
        let mut local_scores = Vec::new();
        for v in 0..p {
            local_scores.push(score.local_score_init(v, Vec::new())?);
        }
        Ok(Self { p, local_scores })
    }

    pub fn score(&self) -> f64 {
        self.local_scores.iter().map(|ls| ls.bic).sum()
    }
}
