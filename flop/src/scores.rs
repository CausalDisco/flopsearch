use crate::{dynamic_cholesky::Cholesky, error::ScoreError};

use crate::bic::Bic;

#[derive(Clone, Debug)]
pub struct GlobalScore {
    pub p: usize,
    pub local_scores: Vec<LocalScore>,
}

#[derive(Clone, Debug)]
pub struct LocalScore {
    pub bic: Option<f64>,
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

    pub fn get_bic(&self, score: &Bic) -> f64 {
        self.local_scores.iter().map(|ls| score.get_bic(ls)).sum()
    }
}

impl LocalScore {
    pub fn get_bic(&mut self, score: &Bic) -> f64 {
        if self.bic.is_none() {
            let new_value = score.get_bic(self);
            self.bic = Some(new_value);
        }
        self.bic.unwrap()
    }
}
