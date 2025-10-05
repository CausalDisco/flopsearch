use nalgebra::{DMatrix, DVector};

pub(crate) fn cov_matrix(data: &DMatrix<f64>) -> DMatrix<f64> {
    let n = data.nrows();
    let mean_vector = data.row_mean();
    let mut centered_data = data.clone();
    for mut row in centered_data.row_iter_mut() {
        row -= mean_vector.clone();
    }
    (centered_data.transpose() * centered_data) / n as f64
}

#[allow(dead_code)]
pub(crate) fn corr_matrix(data: &DMatrix<f64>) -> DMatrix<f64> {
    let mut cov = cov_matrix(data);
    let std_devs = cov.diagonal().map(|x| x.sqrt());

    for i in 0..cov.nrows() {
        for j in 0..cov.ncols() {
            if std_devs[i] > 0.0 && std_devs[j] > 0.0 {
                cov[(i, j)] /= std_devs[i] * std_devs[j];
            }
        }
    }
    cov
}

pub(crate) fn submatrix(matrix: &DMatrix<f64>, rows: &[usize], cols: &[usize]) -> DMatrix<f64> {
    DMatrix::from_fn(rows.len(), cols.len(), |i, j| matrix[(rows[i], cols[j])])
}

pub(crate) fn column_subvector(matrix: &DMatrix<f64>, rows: &[usize], col: usize) -> DVector<f64> {
    DVector::from_fn(rows.len(), |i, _| matrix[(rows[i], col)])
}
