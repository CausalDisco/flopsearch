use nalgebra::{DMatrix, DVector};
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
    // TODO
    // let n = data.nrows();
    // let mean_vector = data.row_mean();
    // let mut centered_data = data.clone();
    // for mut row in centered_data.row_iter_mut() {
    //     row -= mean_vector.clone();
    // }
    // (&centered_data.transpose() * &centered_data) / n as f64

    // Welford
    let nrows = data.nrows();
    let ncols = data.ncols();

    // Transpose once: now each observation is a column
    let data_t = data.transpose();

    let mut mean = DVector::zeros(ncols);
    let mut cov = DMatrix::zeros(ncols, ncols);

    for (k, obs) in data_t.column_iter().enumerate() {
        let x = obs.clone_owned();
        let delta = &x - &mean;
        mean += &delta / (k as f64 + 1.0);
        let delta2 = &x - &mean;
        cov += &delta * delta2.transpose();
    }

    &cov / nrows as f64
}

pub fn corr_matrix(data: &DMatrix<f64>) -> DMatrix<f64> {
    let mut cov = cov_matrix(data);
    let std_devs = cov.diagonal().map(|x| x.sqrt());

    // cols, then rows, so inner loop walks through contiguous memory
    for j in 0..cov.ncols() {
        for i in 0..cov.nrows() {
            // TODO: needed as we fail on non-full-rank covmats anyways?
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
