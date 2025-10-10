struct SquareMatrix {
    buffer: Vec<f64>,
    dim: usize,
}

// TODO: have separate Cholesky type
impl SquareMatrix {
    fn new(data: Vec<f64>, dim: usize) -> Self {
        Self { buffer: data, dim }
    }

    fn get(&self, i: usize, j: usize) -> f64 {
        self.buffer[j * self.dim + i]
    }

    fn set(&mut self, i: usize, j: usize, x: f64) {
        self.buffer[j * self.dim + i] = x;
    }

    // TODO: add error handling for degenerate matrices
    fn cholesky(&self) -> Cholesky {
        let mut chol = Cholesky::new(SquareMatrix::new(vec![0.0; self.dim * self.dim], self.dim));
        self.cholesky_buffered(&mut chol);
        chol
    }

    fn cholesky_buffered(&self, chol: &mut Cholesky) {
        chol.0.dim = self.dim;
        chol.0.buffer.resize(self.dim * self.dim, 0.0);
        for j in 0..self.dim {
            let mut sum = 0.0;
            for k in 0..j {
                sum += chol.0.get(j, k) * chol.0.get(j, k);
            }
            chol.0.set(j, j, (self.get(j, j) - sum).sqrt());
            for i in j + 1..self.dim {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += chol.0.get(i, k) * chol.0.get(j, k);
                }
                chol.0.set(i, j, (self.get(i, j) - sum) / chol.0.get(j, j));
            }
        }
    }
}

#[test]
fn test_cholesky() {
    let input = SquareMatrix::new(
        vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0],
        3,
    );
    let chol = Cholesky::new(SquareMatrix::new(
        vec![2.0, 6.0, -8.0, 0.0, 1.0, 5.0, 0.0, 0.0, 3.0],
        3,
    ));
    let output = input.cholesky();
    let mut abs_diff = 0.0;
    for i in 0..input.dim {
        for j in 0..input.dim {
            abs_diff += (output.0.get(i, j) - chol.0.get(i, j)).abs();
        }
    }
    assert!(abs_diff < 1e-9);
}

struct Cholesky(SquareMatrix);

impl Cholesky {
    fn new(mat: SquareMatrix) -> Self {
        Self(mat)
    }

    // TODO: inplace or using a buffer?
    fn insert_column(&mut self) {
        todo!()
    }

    fn remove_column(&mut self) {
        todo!()
    }
}
