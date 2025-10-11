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
        self.buffer[i * self.dim + j]
    }

    fn set(&mut self, i: usize, j: usize, x: f64) {
        self.buffer[i * self.dim + j] = x;
    }

    // TODO: add error handling for degenerate matrices
    fn cholesky(&self) -> Cholesky {
        let mut chol = Cholesky::new(SquareMatrix::new(vec![0.0; self.dim * self.dim], self.dim));
        for i in 0..self.dim {
            for j in 0..i + 1 {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += chol.0.get(i, k) * chol.0.get(j, k);
                }
                if i == j {
                    chol.0.set(i, j, (self.get(i, i) - sum).sqrt());
                } else {
                    chol.0.set(i, j, (self.get(i, j) - sum) / chol.0.get(j, j));
                }
            }
        }
        chol
    }
}

#[test]
fn test_cholesky() {
    let input = SquareMatrix::new(
        vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0],
        3,
    );
    let chol = Cholesky::new(SquareMatrix::new(
        vec![2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0],
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

struct Vector {
    buffer: Vec<f64>,
    dim: usize,
}

struct Cholesky(SquareMatrix);

impl Cholesky {
    fn new(mat: SquareMatrix) -> Self {
        Self(mat)
    }

    // TODO: inplace or using a buffer?
    // always insert before last row/col
    fn insert_column(&mut self) {
        // rewrite L11 into top left
        //
        todo!()
    }

    fn remove_column(&mut self) {
        todo!()
    }

    fn rank_update(&mut self, x: &mut Vector) {
        todo!()
    }
}
