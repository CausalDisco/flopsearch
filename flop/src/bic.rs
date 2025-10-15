use nalgebra::DMatrix;

use crate::linalg::Cholesky;
use crate::matrix;
use crate::scores::LocalScore;

/// BIC score for linear Gaussian models
#[derive(Debug)]
pub struct Bic {
    n: usize,
    lambda: f64,
    cov: DMatrix<f64>,
}

impl Bic {
    pub fn new(data: &DMatrix<f64>, lambda: f64) -> Self {
        Self {
            n: data.nrows(),
            lambda,
            cov: matrix::corr_matrix(data),
        }
    }

    pub fn from_cov(n: usize, cov: DMatrix<f64>, lambda: f64) -> Self {
        Self { n, lambda, cov }
    }

    fn res_var_from_chol(var: f64, parent_chol: &Cholesky, mut covs: Vec<f64>) -> f64 {
        parent_chol.forward_solve(&mut covs);
        let mut sum = 0.0;
        for i in 0..covs.len() {
            sum += covs[i] * covs[i];
        }
        var - sum
    }

    //pub fn local_score_init(&self, v: usize, parents: Vec<usize>) -> LocalScore {
    //    let num_parents = parents.len();
    //    let submat = matrix::submatrix(&self.cov, &parents);
    //    let chol = submat.cholesky();
    //    let res_var = Self::res_var_from_chol(
    //        self.cov[(v, v)],
    //        &chol,
    //        matrix::column_subvector(&self.cov, &parents, v),
    //    );
    //    LocalScore {
    //        bic: self.compute_local_bic(num_parents, res_var),
    //        chol,
    //        parents,
    //    }
    //}

    pub fn local_score_init(&self, v: usize, parents: Vec<usize>) -> LocalScore {
        let num_parents = parents.len();
        let mut parents_v = Vec::with_capacity(parents.len() + 1);
        parents_v.extend_from_slice(&parents);
        parents_v.push(v);
        let submat = matrix::submatrix(&self.cov, &parents_v);
        let chol = submat.cholesky();
        let std_var = *chol.data.last().unwrap();
        LocalScore {
            bic: self.compute_local_bic(num_parents, std_var),
            chol,
            parents,
        }
    }

    //pub fn local_score_plus(&self, v: usize, old_local: &LocalScore, r: usize) -> LocalScore {
    //    let num_parents = old_local.parents.len() + 1;
    //    let mut new_parents = Vec::with_capacity(num_parents);
    //    new_parents.extend_from_slice(&old_local.parents);
    //    new_parents.push(r);
    //    let ins_col = matrix::column_subvector(&self.cov, &new_parents, r);
    //    let new_chol = old_local.chol.append_column(ins_col);
    //    let res_var = Self::res_var_from_chol(
    //        self.cov[(v, v)],
    //        &new_chol,
    //        matrix::column_subvector(&self.cov, &new_parents, v),
    //    );
    //    LocalScore {
    //        bic: self.compute_local_bic(num_parents, res_var),
    //        chol: new_chol,
    //        parents: new_parents,
    //    }
    //}

    pub fn local_score_plus(&self, v: usize, old_local: &LocalScore, r: usize) -> LocalScore {
        let num_parents = old_local.parents.len() + 1;
        let mut new_parents_v = Vec::with_capacity(num_parents + 1);
        new_parents_v.extend_from_slice(&old_local.parents);
        new_parents_v.push(v);
        new_parents_v.push(r);
        let ins_col = matrix::column_subvector(&self.cov, &new_parents_v, r);
        let new_chol = old_local.chol.insert_column_before_last(ins_col);
        let std_var = *new_chol.data.last().unwrap();
        let mut new_parents = new_parents_v;
        new_parents.pop();
        new_parents.pop();
        new_parents.push(r);
        LocalScore {
            bic: self.compute_local_bic(num_parents, std_var),
            chol: new_chol,
            parents: new_parents,
        }
    }

    pub fn local_score_minus(&self, _v: usize, old_local: &LocalScore, r: usize) -> LocalScore {
        let num_parents = old_local.parents.len() - 1;
        let idx = old_local.parents.iter().position(|&u| u == r).unwrap();
        let mut new_parents = old_local.parents.clone();
        new_parents.remove(idx);
        let new_chol = old_local.chol.remove_column(idx);
        let std_var = *new_chol.data.last().unwrap();
        LocalScore {
            bic: self.compute_local_bic(num_parents, std_var),
            chol: new_chol,
            parents: new_parents,
        }
    }

    fn compute_local_bic(&self, num_parents: usize, std_var: f64) -> f64 {
        2.0 * self.n as f64 * std_var.max(f64::MIN_POSITIVE).ln()
            + self.lambda * num_parents as f64 * (self.n as f64).ln()
    }
}
