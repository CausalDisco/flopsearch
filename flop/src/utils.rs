use nalgebra::{DMatrix, RowDVector};

pub fn rem_first(vec: &mut Vec<usize>, x: usize) {
    if let Some(pos) = vec.iter().position(|&u| u == x) {
        vec.remove(pos);
    }
}

// REMINDER: as we don't need matmul anymore we could drop the nalgebra dependency
pub fn cov_matrix(data: &DMatrix<f64>) -> DMatrix<f64> {
    // Welford
    let nrows = data.nrows();
    let ncols = data.ncols();

    let mut mean = RowDVector::zeros(ncols);
    let mut cov = DMatrix::zeros(ncols, ncols);

    for (k, row) in data.row_iter().enumerate() {
        let delta = row - &mean;
        mean += &delta / (k as f64 + 1.0);
        let delta2 = row - &mean;
        cov += delta.transpose() * &delta2;
    }

    &cov / nrows as f64
}

pub fn corr_matrix(data: &DMatrix<f64>) -> DMatrix<f64> {
    let mut cov = cov_matrix(data);
    let std_devs = cov.diagonal().map(|x| x.sqrt());

    // cols, then rows, so inner loop walks through contiguous memory
    for j in 0..cov.ncols() {
        for i in 0..cov.nrows() {
            cov[(i, j)] /= std_devs[i] * std_devs[j];
        }
    }
    cov
}

pub fn submatrix(matrix: &DMatrix<f64>, idxs: &[usize]) -> DMatrix<f64> {
    DMatrix::from_fn(idxs.len(), idxs.len(), |i, j| matrix[(idxs[i], idxs[j])])
}

pub fn column_subvector(matrix: &DMatrix<f64>, rows: &[usize], col: usize) -> Vec<f64> {
    rows.iter().map(|&row| matrix[(row, col)]).collect()
}
