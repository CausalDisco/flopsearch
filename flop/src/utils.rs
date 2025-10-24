use nalgebra::DMatrix;
use rand::{rngs::ThreadRng, seq::SliceRandom};

pub fn rem_first(vec: &mut Vec<usize>, x: usize) {
    if let Some(pos) = vec.iter().position(|&u| u == x) {
        vec.remove(pos);
    }
}

pub fn rand_perm(p: usize, rng: &mut ThreadRng) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..p).collect();
    perm.shuffle(rng);
    perm
}

pub fn cov_matrix(data: &DMatrix<f64>) -> DMatrix<f64> {
    let n = data.nrows();
    let mean_vector = data.row_mean();
    let mut centered_data = data.clone();
    for mut row in centered_data.row_iter_mut() {
        row -= mean_vector.clone();
    }
    (centered_data.transpose() * centered_data) / n as f64
}

pub fn corr_matrix(data: &DMatrix<f64>) -> DMatrix<f64> {
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

pub fn submatrix(matrix: &DMatrix<f64>, idxs: &[usize]) -> DMatrix<f64> {
    DMatrix::from_fn(idxs.len(), idxs.len(), |i, j| matrix[(idxs[i], idxs[j])])
}

pub fn column_subvector(matrix: &DMatrix<f64>, rows: &[usize], col: usize) -> Vec<f64> {
    rows.iter().map(|&row| matrix[(row, col)]).collect()
}
