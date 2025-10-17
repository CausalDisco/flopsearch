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

    // this is one of the hot loops
    #[inline(always)]
    fn forward_solve(&self, x: &mut [f64]) {
        // this unsafe code is very slightly faster than the safe version below
        unsafe {
            let mut data_ptr = self.data.as_ptr();
            let x_ptr = x.as_mut_ptr();

            for i in 0..self.dim {
                let diag = *data_ptr;
                let xi = *x_ptr.add(i) / diag;
                *x_ptr.add(i) = xi;
                data_ptr = data_ptr.add(1);

                let mut xj_ptr = x_ptr.add(i + 1);

                for _ in i + 1..self.dim {
                    let aj = *data_ptr;
                    let xj = *xj_ptr - aj * xi;
                    *xj_ptr = xj;

                    data_ptr = data_ptr.add(1);
                    xj_ptr = xj_ptr.add(1);
                }
            }
        }

        // safe version of the code above
        // let mut idx = 0;
        // for i in 0..self.dim {
        //     x[i] /= self.data[idx];
        //     idx += 1;
        //     let mult = x[i];
        //     let lhs = &mut x[i + 1..self.dim];
        //     let rhs = &self.data[idx..idx - i - 1 + self.dim];
        //     for j in 0..lhs.len() {
        //         lhs[j] -= rhs[j] * mult;
        //     }
        //     idx += self.dim - i - 1;
        // }
    }

    #[inline(always)]
    fn make_givens(a: f64, b: f64) -> (f64, f64, f64) {
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
        (c, s, r)
    }

    pub fn insert_column_before_last(&self, mut x: Vec<f64>) -> Self {
        // allocate new Cholesky
        let new_size = Self::chol_size(self.dim + 1);
        let mut new_chol = Self::new(Vec::with_capacity(new_size), self.dim + 1);
        unsafe {
            new_chol.data.set_len(new_size);
        }

        // solve for x (added row if it would be appended)
        self.forward_solve(&mut x);
        let mut sum = 0.0;
        for &el in x[0..self.dim].iter() {
            sum += el * el;
        }
        x[self.dim] = (x[self.dim] - sum).sqrt();

        // Givens rotation necessary for triangular shape when inserting x before the last row
        // this part is cheap, it just manipulates four distinct values
        let n = x.len();
        let (c, s, r) = Self::make_givens(x[n - 2], x[n - 1]);
        x[n - 2] = r;
        x[n - 1] = 0.0;
        let prev_corner = *self.data.last().unwrap();
        let new_left_of_corner = c * prev_corner;
        let new_corner = s * prev_corner;

        // fill Cholesky with new values, also one of the hot loops
        // in particular copying over the old values
        let mut idx = 0;
        for i in 0..self.dim {
            let stride = self.dim - i - 1;
            new_chol.data[idx + i..idx + i + stride].copy_from_slice(&self.data[idx..idx + stride]);
            idx += stride;
            new_chol.data[idx + i] = x[i];
            new_chol.data[idx + i + 1] = self.data[idx];
            idx += 1;
        }
        new_chol.data[new_size - 2] = new_left_of_corner;
        new_chol.data[new_size - 1] = new_corner.abs();
        new_chol
    }

    pub fn remove_column(&self, k: usize) -> Self {
        // allocate new Cholesky
        let new_size = Self::chol_size(self.dim - 1);
        let mut new_chol = Self::new(Vec::with_capacity(new_size), self.dim - 1);
        unsafe {
            new_chol.data.set_len(new_size);
        }

        // column vector that needs to be zeroed with Givens rotations
        let mut x = Vec::with_capacity(self.dim - k);

        let mut idx = 0;
        for i in 0..self.dim {
            if i < k {
                let stride = k - i;
                new_chol.data[idx - i..idx - i + stride]
                    .copy_from_slice(&self.data[idx..idx + stride]);
                idx += stride + 1;
                let stride = self.dim - k - 1;
                new_chol.data[idx - i - 1..idx - i - 1 + stride]
                    .copy_from_slice(&self.data[idx..idx + stride]);
                idx += stride;
            } else if i == k {
                // skip (k, k) element
                idx += 1;
                let stride = self.dim - i - 1;
                x.extend_from_slice(&self.data[idx..idx + stride]);
                idx += stride;
            } else {
                let (c, s, r) = Self::make_givens(self.data[idx], x[i - k - 1]);
                new_chol.data[idx - self.dim] = r;
                x[i - k - 1] = 0.0;
                idx += 1;

                // apply Givens rotation to column
                for j in i + 1..self.dim {
                    let tau1 = self.data[idx];
                    let tau2 = x[j - k - 1];
                    new_chol.data[idx - self.dim] = c * tau1 - s * tau2;
                    x[j - k - 1] = s * tau1 + c * tau2;
                    idx += 1;
                }
            }
        }
        new_chol
    }

    // put into utils or something -> or just put directly in the functions tbh
    #[inline(always)]
    fn chol_size(dim: usize) -> usize {
        dim * (dim + 1) / 2
    }
}

#[test]
fn test_cholesky_updates() {
    let input = SquareMatrix::new(
        vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0],
        3,
    );
    let chol = Cholesky::new(vec![2.0, 6.0, -8.0, 1.0, 5.0, 3.0], 3);
    let output = input.cholesky();
    assert!(abs_diff_sum(&chol.data, &output.data) < 1e-9);

    let out_c = output.remove_column(2);
    let chol_c = Cholesky::new(vec![2.0, 6.0, 1.0], 3);
    assert!(abs_diff_sum(&chol_c.data, &out_c.data) < 1e-9);

    let out_d = output.remove_column(1);
    let chol_d = Cholesky::new(vec![2.0, -8.0, 34.0f64.sqrt()], 2);
    assert!(abs_diff_sum(&chol_d.data, &out_d.data) < 1e-9);

    let out_e = output.remove_column(0);
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
