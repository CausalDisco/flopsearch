#[derive(Debug)]
pub struct TokenBuffer {
    tokens: Vec<usize>,
    depth: usize,
}

impl TokenBuffer {
    pub fn new(n: usize) -> Self {
        Self {
            tokens: vec![0; n],
            depth: 0,
        }
    }

    pub fn set(&mut self, i: usize) {
        if self.tokens[i] == self.depth {
            return;
        }
        self.tokens[i] = self.depth;
    }

    pub fn check(&self, i: usize) -> bool {
        self.tokens[i] == self.depth
    }

    pub fn clear(&mut self) {
        self.depth += 1;
    }
}
