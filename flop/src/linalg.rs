struct SquareMatrix {
    buffer: Vec<f64>,
    dim: usize,
}

// TODO: have separate Cholesky type
impl SquareMatrix {
    fn new(data: Vec<f64>, dim: usize) -> Self {
        Self { buffer: data, dim }
    }

    // TODO: add error handling for degenerate matrices
    fn cholesky(&self) -> Cholesky {
        let mut chol = Cholesky::new(SquareMatrix::new(vec![0.0; self.dim * self.dim], self.dim));
        self.cholesky_buffered(&mut chol);
        chol
    }

    fn cholesky_buffered(&self, chol: &mut Cholesky) {
        todo!()
    }
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
