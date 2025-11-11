#[derive(Debug)]
pub struct TokenBuffer {
    tokens: Vec<usize>,
    depth: usize,
}

impl TokenBuffer {
    pub fn new(n: usize) -> Self {
        Self {
            tokens: vec![0; n],
            depth: 1, // initialize cleared
        }
    }

    pub fn set(&mut self, i: usize) {
        self.tokens[i] = self.depth;
    }

    pub fn check(&self, i: usize) -> bool {
        self.tokens[i] == self.depth
    }

    pub fn clear(&mut self) {
        self.depth += 1;
    }
}
