use nalgebra::DMatrix;

use crate::dynamic_cholesky::Cholesky;
use crate::scores::LocalScore;
use crate::utils;

/// BIC score for linear Gaussian models
#[derive(Debug)]
pub struct Bic {
    n: usize,
    logn: f64,
    lambda: f64,
    cov: DMatrix<f64>,
}

impl Bic {
    pub fn new(data: &DMatrix<f64>, lambda: f64) -> Self {
        Self {
            n: data.nrows(),
            logn: (data.nrows() as f64).ln(),
            lambda,
            cov: utils::corr_matrix(data),
        }
    }

    pub fn from_cov(n: usize, cov: DMatrix<f64>, lambda: f64) -> Self {
        Self {
            n,
            logn: (n as f64).ln(),
            lambda,
            cov,
        }
    }

    pub fn local_score_init(&self, v: usize, parents: Vec<usize>) -> LocalScore {
        let num_parents = parents.len();
        let mut parents_v = Vec::with_capacity(parents.len() + 1);
        parents_v.extend_from_slice(&parents);
        parents_v.push(v);
        let submat = utils::submatrix(&self.cov, &parents_v);
        let chol = Cholesky::for_matrix(submat);
        let std_var = chol.get_bottom_right();
        LocalScore {
            bic: self.compute_local_bic(num_parents, std_var),
            chol,
            parents,
        }
    }

    pub fn local_score_plus(&self, v: usize, old_local: &LocalScore, r: usize) -> LocalScore {
        let num_parents = old_local.parents.len() + 1;
        let mut new_parents_v = Vec::with_capacity(num_parents + 1);
        new_parents_v.extend_from_slice(&old_local.parents);
        new_parents_v.push(v);
        new_parents_v.push(r);
        let ins_col = utils::column_subvector(&self.cov, &new_parents_v, r);
        let new_chol = old_local.chol.insert_column_before_last(ins_col);
        let std_var = new_chol.get_bottom_right();

        let mut new_parents = new_parents_v;
        new_parents.pop();
        new_parents[num_parents - 1] = r;
        LocalScore {
            bic: self.compute_local_bic(num_parents, std_var),
            chol: new_chol,
            parents: new_parents,
        }
    }

    pub fn local_score_minus(&self, _v: usize, old_local: &LocalScore, r: usize) -> LocalScore {
        let num_parents = old_local.parents.len() - 1;
        let idx = old_local.parents.iter().position(|&u| u == r).unwrap();
        let mut new_parents = Vec::with_capacity(num_parents);
        new_parents.extend_from_slice(&old_local.parents[..idx]);
        new_parents.extend_from_slice(&old_local.parents[idx + 1..]);
        let new_chol = old_local.chol.remove_column(idx);
        let std_var = new_chol.get_bottom_right();

        LocalScore {
            bic: self.compute_local_bic(num_parents, std_var),
            chol: new_chol,
            parents: new_parents,
        }
    }

    #[inline(always)]
    fn compute_local_bic(&self, num_parents: usize, std_var: f64) -> f64 {
        2.0 * self.n as f64 * std_var.max(f64::MIN_POSITIVE).ln()
            + self.lambda * num_parents as f64 * self.logn
    }
}
