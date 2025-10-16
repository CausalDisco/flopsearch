// TODO: for now work without explicit buffers
// -> rename buffer to data
// -> later check how much allocs cost

pub struct SquareMatrix {
    data: Vec<f64>,
    dim: usize,
}

impl SquareMatrix {
    pub fn new(data: Vec<f64>, dim: usize) -> Self {
        Self { data, dim }
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.dim + j]
    }

    pub fn set(&mut self, i: usize, j: usize, x: f64) {
        self.data[i * self.dim + j] = x;
    }

    // TODO: add error handling for degenerate matrices
    pub fn cholesky(&self) -> Cholesky {
        // create Cholesky row-by-row in square matrix
        let mut tmp_chol = SquareMatrix::new(vec![0.0; self.dim * self.dim], self.dim);
        for i in 0..self.dim {
            for j in 0..i + 1 {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += tmp_chol.get(i, k) * tmp_chol.get(j, k);
                }
                if i == j {
                    tmp_chol.set(i, j, (self.get(i, i) - sum).sqrt());
                } else {
                    tmp_chol.set(i, j, (self.get(i, j) - sum) / tmp_chol.get(j, j));
                }
            }
        }

        // convert to packed format
        let mut chol_data = Vec::with_capacity(Cholesky::chol_size(self.dim));
        for i in 0..self.dim {
            for j in i..self.dim {
                chol_data.push(tmp_chol.get(j, i));
            }
        }
        Cholesky::new(chol_data, self.dim)
    }
}

fn abs_diff_sum(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += (a[i] - b[i]).abs();
    }
    sum
}

#[test]
fn test_cholesky() {
    let input = SquareMatrix::new(
        vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0],
        3,
    );
    let chol = Cholesky::new(vec![2.0, 6.0, -8.0, 1.0, 5.0, 3.0], 3);
    let output = input.cholesky();
    assert!(abs_diff_sum(&chol.data, &output.data) < 1e-9);
}

// packed format column-major
#[derive(Clone, Debug)]
pub struct Cholesky {
    pub data: Vec<f64>,
    dim: usize,
}

impl Cholesky {
    fn new(data: Vec<f64>, dim: usize) -> Self {
        Self { data, dim }
    }

    pub fn forward_solve(&self, x: &mut [f64]) {
        let mut idx = 0;
        for i in 0..self.dim {
            x[i] /= self.data[idx];
            idx += 1;
            for j in i + 1..self.dim {
                x[j] -= self.data[idx + j - i - 1] * x[i];
            }
            idx += self.dim - i - 1;
        }
    }

    pub fn append_column(&self, mut x: Vec<f64>) -> Self {
        let mut new_chol = Self::new(
            Vec::with_capacity(Self::chol_size(self.dim + 1)),
            self.dim + 1,
        );
        self.forward_solve(&mut x);
        let mut idx = 0;
        for i in 0..self.dim {
            let stride = self.dim - i;
            new_chol
                .data
                .extend_from_slice(&self.data[idx..idx + stride]);
            new_chol.data.push(x[i]);
            idx += stride;
        }
        let mut sum = 0.0;
        for i in 0..self.dim {
            sum += x[i] * x[i];
        }
        new_chol.data.push((x[self.dim] - sum).sqrt());
        new_chol
    }

    pub fn insert_column_before_last(&self, mut x: Vec<f64>) -> Self {
        let mut new_chol = Self::new(
            Vec::with_capacity(Self::chol_size(self.dim + 1)),
            self.dim + 1,
        );
        self.forward_solve(&mut x);
        let mut sum = 0.0;
        for i in 0..self.dim {
            sum += x[i] * x[i];
        }
        x[self.dim] = (x[self.dim] - sum).sqrt();
        // Givens params
        let n = x.len();
        let a = x[n - 2];
        let b = x[n - 1];

        let mut c = 1.0;
        let mut s = 0.0;
        if b != 0.0 {
            if b.abs() > a.abs() {
                let tau = -a / b;
                s = -1.0 / (1.0 + tau * tau).sqrt();
                c = s * tau;
            } else {
                let tau = -b / a;
                c = 1.0 / (1.0 + tau * tau).sqrt();
                s = c * tau;
            }
        }

        // ensure positivity of r (new diagonal entry)
        let mut r = c * a - s * b;
        if r < 0.0 {
            c = -c;
            s = -s;
            r = -r;
        }

        x[n - 2] = r;
        x[n - 1] = 0.0;

        let mut idx = 0;
        for i in 0..self.dim + 1 {
            let mut stride = self.dim - i;
            if stride > 0 {
                stride -= 1;
            }
            new_chol
                .data
                .extend_from_slice(&self.data[idx..idx + stride]);
            idx += stride;
            if i < self.dim {
                new_chol.data.push(x[i]);
                new_chol.data.push(self.data[idx]);
            } else {
                new_chol.data.push(0.0);
            }
            idx += 1;
        }

        let m = self.data.len();
        let tau1 = self.data[m - 1];
        let tau2 = 0.0;
        let new_tau1 = c * tau1 - s * tau2;
        let new_tau2 = s * tau1 + c * tau2;

        let new_m = new_chol.data.len();
        new_chol.data[new_m - 2] = new_tau1;
        new_chol.data[new_m - 1] = new_tau2.abs();
        new_chol
    }

    pub fn remove_column(&self, k: usize) -> Self {
        let mut new_chol = Self::new(
            Vec::with_capacity(Self::chol_size(self.dim - 1)),
            self.dim - 1,
        );
        let mut x = Vec::with_capacity(self.dim - k);

        let mut idx = 0;
        for i in 0..self.dim {
            if i < k {
                let stride = k - i;
                new_chol
                    .data
                    .extend_from_slice(&self.data[idx..idx + stride]);
                idx += stride + 1;
                let stride = self.dim - k - 1;
                new_chol
                    .data
                    .extend_from_slice(&self.data[idx..idx + stride]);
                idx += stride;
            } else if i == k {
                // skip (k, k) element
                idx += 1;
                for _ in i + 1..self.dim {
                    x.push(self.data[idx]);
                    idx += 1;
                }
            } else {
                // Givens rotation
                // compute c and s
                let a = self.data[idx];
                let b = x[i - k - 1];
                let mut c = 1.0;
                let mut s = 0.0;
                if b != 0.0 {
                    if b.abs() > a.abs() {
                        let tau = -a / b;
                        s = -1.0 / (1.0 + tau * tau).sqrt();
                        c = s * tau;
                    } else {
                        let tau = -b / a;
                        c = 1.0 / (1.0 + tau * tau).sqrt();
                        s = c * tau;
                    }
                }

                // ensure positivity of r (new diagonal entry)
                let mut r = c * a - s * b;
                if r < 0.0 {
                    c = -c;
                    s = -s;
                    r = -r;
                }
                new_chol.data.push(r);
                x[i - k - 1] = 0.0;
                idx += 1;

                // apply Givens rotation
                for j in i + 1..self.dim {
                    let tau1 = self.data[idx];
                    let tau2 = x[j - k - 1];
                    new_chol.data.push(c * tau1 - s * tau2);
                    x[j - k - 1] = s * tau1 + c * tau2;
                    idx += 1;
                }
            }
        }
        new_chol
    }

    // put into utils
    fn chol_size(dim: usize) -> usize {
        dim * (dim + 1) / 2
    }
}

#[test]
fn test_cholesky_updates() {
    let a = SquareMatrix::new(vec![4.0, 12.0, 12.0, 37.0], 2);
    let chol_a = Cholesky::new(vec![2.0, 6.0, 1.0], 2);
    let out_a = a.cholesky();
    assert!(abs_diff_sum(&chol_a.data, &out_a.data) < 1e-9);

    let new_col = vec![-16.0, -43.0, 98.0];
    let out_b = chol_a.append_column(new_col);
    let chol_b = Cholesky::new(vec![2.0, 6.0, -8.0, 1.0, 5.0, 3.0], 3);
    assert!(abs_diff_sum(&chol_b.data, &out_b.data) < 1e-9);

    let out_c = chol_b.remove_column(2);
    assert!(abs_diff_sum(&chol_a.data, &out_c.data) < 1e-9);

    let out_d = chol_b.remove_column(1);
    let chol_d = Cholesky::new(vec![2.0, -8.0, 34.0f64.sqrt()], 2);
    assert!(abs_diff_sum(&chol_d.data, &out_d.data) < 1e-9);

    let out_e = chol_b.remove_column(0);
    let chol_e = Cholesky::new(
        vec![
            37.0f64.sqrt(),
            -43.0 * 37.0f64.sqrt() / 37.0,
            65749.0f64.sqrt() / 37.0,
        ],
        2,
    );
    assert!(abs_diff_sum(&chol_e.data, &out_e.data) < 1e-9);
}

#[test]
fn test_cholesky_insert_before_last() {
    let a = SquareMatrix::new(vec![4.0, -16.0, -16.0, 98.0], 2);
    let out_a = a.cholesky();
    let new_chol = out_a.insert_column_before_last(vec![12.0, -43.0, 37.0]);
    let chol = Cholesky::new(vec![2.0, 6.0, -8.0, 1.0, 5.0, 3.0], 3);
    assert!(abs_diff_sum(&chol.data, &new_chol.data) < 1e-9);
}

#[test]
fn test_cholesky_insert_before_last2() {
    let a = SquareMatrix::new(vec![37.0], 1);
    let out_a = a.cholesky();
    let new_chol = out_a.insert_column_before_last(vec![12.0, 4.0]);
    let chol = Cholesky::new(vec![2.0, 6.0, 1.0], 2);
    assert!(abs_diff_sum(&chol.data, &new_chol.data) < 1e-9);
}
