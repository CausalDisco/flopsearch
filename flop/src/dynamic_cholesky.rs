use nalgebra::DMatrix;
use std::mem::MaybeUninit;

// Implement our own Cholesky updates for better performance and error-handling compared to what nalgebra offers
#[derive(Clone, Debug)]
pub struct Cholesky {
    // packed column-major format
    data: Vec<f64>,
    dim: usize,
}

impl Cholesky {
    fn new(data: Vec<f64>, dim: usize) -> Self {
        Self { data, dim }
    }

    pub fn for_matrix(mat: DMatrix<f64>) -> Option<Self> {
        let dim = mat.nrows();
        let nalgebra_chol = mat.cholesky()?;
        let mut packed_chol = Vec::with_capacity(dim * (dim + 1) / 2);
        for i in 0..dim {
            for j in i..dim {
                packed_chol.push(nalgebra_chol.l_dirty()[(j, i)]);
            }
        }
        Some(Cholesky::new(packed_chol, dim))
    }

    pub fn get_bottom_right(&self) -> f64 {
        *self.data.last().unwrap()
    }

    // this is one of the hot loops
    #[inline(always)]
    fn forward_solve(&self, x: &mut [f64]) -> Result<(), ()> {
        let mut diag_idx = 0;
        for i in 0..self.dim {
            unsafe {
                let diag = *self.data.get_unchecked(diag_idx);
                if diag <= 0.0 {
                    return Err(());
                }
                let xi = *x.get_unchecked(i) / diag;
                *x.get_unchecked_mut(i) = xi;

                let x_slice = &mut x[i + 1..];
                let chol_slice = &self.data[diag_idx + 1..];
                let len = self.dim - i - 1;

                for j in 0..len {
                    let xj = *x_slice.get_unchecked(j) - *chol_slice.get_unchecked(j) * xi;
                    *x_slice.get_unchecked_mut(j) = xj;
                }
                diag_idx += self.dim - i;
            }
        }
        Ok(())
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

    pub fn insert_column_before_last(&self, mut x: Vec<f64>) -> Option<Self> {
        let new_size = (self.dim + 1) * (self.dim + 2) / 2;
        let mut new_data: Vec<MaybeUninit<f64>> = Vec::with_capacity(new_size);
        unsafe {
            new_data.set_len(new_size);
        }

        // solve for x (added row if it would be appended)
        self.forward_solve(&mut x).ok()?;
        let mut sum = 0.0;
        for &el in x[0..self.dim].iter() {
            sum += el * el;
        }
        let new_diag_squared = x[self.dim] - sum;
        if new_diag_squared <= 0.0 {
            return None;
        }
        x[self.dim] = new_diag_squared.sqrt();

        // Givens rotation necessary for triangular shape when inserting x before the last row
        // this part is cheap, it just manipulates four distinct values
        let n = x.len();
        let (c, s, r) = Self::make_givens(x[n - 2], x[n - 1]);
        x[n - 2] = r;
        x[n - 1] = 0.0;
        let prev_corner = *self.data.last().unwrap();
        let new_left_of_corner = c * prev_corner;
        let new_corner = (s * prev_corner).abs();

        // fill Cholesky with new values, also one of the hot loops
        // in particular copying over the old values
        let mut idx = 0;
        for i in 0..self.dim {
            let stride = self.dim - i - 1;

            // SAFETY: write to valid, unitialized elements
            // only last vec element remains unitialized after this loop
            unsafe {
                let dst = new_data.as_mut_ptr().add(idx + i);
                let src = self.data.as_ptr().add(idx);

                std::ptr::copy_nonoverlapping(src, dst as *mut f64, stride);
                *dst.add(stride) = MaybeUninit::new(x[i]);
                *dst.add(stride + 1) = MaybeUninit::new(self.data[idx + stride]);
            }
            idx += stride + 1;
        }
        // SAFETY: overwrite at new_size - 2 and write new_size - 1 (last element)
        unsafe {
            *new_data.get_unchecked_mut(new_size - 2) = MaybeUninit::new(new_left_of_corner);
            *new_data.get_unchecked_mut(new_size - 1) = MaybeUninit::new(new_corner);
        }

        // SAFETY: all elements are initialized
        Some(Self::new(
            unsafe { std::mem::transmute::<Vec<MaybeUninit<f64>>, Vec<f64>>(new_data) },
            self.dim + 1,
        ))
    }

    pub fn remove_column(&self, k: usize) -> Self {
        let new_size = (self.dim - 1) * self.dim / 2;
        let mut new_data: Vec<MaybeUninit<f64>> = Vec::with_capacity(new_size);
        unsafe {
            new_data.set_len(new_size);
        }

        // column vector that needs to be zeroed with Givens rotations
        let mut x = Vec::with_capacity(self.dim - k);

        let mut idx = 0;
        for i in 0..self.dim {
            if i < k {
                // SAFETY: write to valid, unitialized elements (before removed col k)
                unsafe {
                    let stride = k - i;
                    let dst = new_data.as_mut_ptr().add(idx - i);
                    let src = self.data.as_ptr().add(idx);
                    std::ptr::copy_nonoverlapping(src, dst as *mut f64, stride);
                    let dst = dst.add(stride);
                    let src = src.add(stride + 1);
                    std::ptr::copy_nonoverlapping(src, dst as *mut f64, self.dim - k - 1);
                }
                idx += self.dim - i;
            } else if i == k {
                // skip (k, k) element
                idx += 1;
                let stride = self.dim - i - 1;
                x.extend_from_slice(&self.data[idx..idx + stride]);
                idx += stride;
            } else {
                let (c, s, r) = Self::make_givens(self.data[idx], x[i - k - 1]);
                // SAFETY: write diagonal element for columns after removed column k
                unsafe {
                    *new_data.get_unchecked_mut(idx - self.dim) = MaybeUninit::new(r);
                }
                x[i - k - 1] = 0.0;
                idx += 1;

                // apply Givens rotation to column
                for j in i + 1..self.dim {
                    let tau1 = self.data[idx];
                    let tau2 = x[j - k - 1];
                    // SAFETY: write below diagonal after removed column k
                    unsafe {
                        *new_data.get_unchecked_mut(idx - self.dim) =
                            MaybeUninit::new(c * tau1 - s * tau2);
                    }
                    x[j - k - 1] = s * tau1 + c * tau2;
                    idx += 1;
                }
            }
        }

        // SAFETY: all elements initialized
        Self::new(
            unsafe { std::mem::transmute::<Vec<MaybeUninit<f64>>, Vec<f64>>(new_data) },
            self.dim - 1,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    fn abs_diff_sum(a: &[f64], b: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..a.len() {
            sum += (a[i] - b[i]).abs();
        }
        sum
    }

    #[test]
    fn test_cholesky_removals() {
        let input = DMatrix::from_column_slice(
            3,
            3,
            &vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0],
        );
        let chol = Cholesky::new(vec![2.0, 6.0, -8.0, 1.0, 5.0, 3.0], 3);
        let output = Cholesky::for_matrix(input).unwrap();
        assert!(abs_diff_sum(&chol.data, &output.data) < 1e-9);

        let out_rem2 = output.remove_column(2);
        let chol_rem2 = Cholesky::new(vec![2.0, 6.0, 1.0], 3);
        assert!(abs_diff_sum(&chol_rem2.data, &out_rem2.data) < 1e-9);

        let out_rem1 = output.remove_column(1);
        let chol_rem1 = Cholesky::new(vec![2.0, -8.0, 34.0f64.sqrt()], 2);
        assert!(abs_diff_sum(&chol_rem1.data, &out_rem1.data) < 1e-9);

        let out_rem0 = output.remove_column(0);
        let chol_rem0 = Cholesky::new(
            vec![
                37.0f64.sqrt(),
                -43.0 * 37.0f64.sqrt() / 37.0,
                65749.0f64.sqrt() / 37.0,
            ],
            2,
        );
        assert!(abs_diff_sum(&chol_rem0.data, &out_rem0.data) < 1e-9);
    }

    #[test]
    fn test_cholesky_insert_before_last_2by2() {
        let input = DMatrix::from_column_slice(2, 2, &vec![4.0, -16.0, -16.0, 98.0]);
        let output = Cholesky::for_matrix(input).unwrap();
        let new_chol = output
            .insert_column_before_last(vec![12.0, -43.0, 37.0])
            .unwrap();
        let true_chol = Cholesky::new(vec![2.0, 6.0, -8.0, 1.0, 5.0, 3.0], 3);
        assert!(abs_diff_sum(&true_chol.data, &new_chol.data) < 1e-9);
    }

    #[test]
    fn test_cholesky_insert_before_last_1by1() {
        let input = DMatrix::from_column_slice(1, 1, &vec![37.0]);
        let output = Cholesky::for_matrix(input).unwrap();
        let new_chol = output.insert_column_before_last(vec![12.0, 4.0]).unwrap();
        let true_chol = Cholesky::new(vec![2.0, 6.0, 1.0], 2);
        assert!(abs_diff_sum(&true_chol.data, &new_chol.data) < 1e-9);
    }
}
