#[derive(Debug)]
pub enum FlopError {
    InvalidConfig(String),
    ScoreError(ScoreError),
}

impl std::fmt::Display for FlopError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FlopError::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            FlopError::ScoreError(err) => write!(f, "Score error: {}", err),
        }
    }
}

impl std::error::Error for FlopError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FlopError::ScoreError(err) => Some(err),
            _ => None,
        }
    }
}

impl From<ScoreError> for FlopError {
    fn from(err: ScoreError) -> Self {
        FlopError::ScoreError(err)
    }
}

#[derive(Debug)]
pub struct ScoreError {
    node: usize,
    parents: Vec<usize>,
    added_node: Option<usize>,
}

impl std::fmt::Display for ScoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(r) = self.added_node {
            write!(
                f,
                "could not compute local score for node {} with parents {:?} when adding parent {} (likely caused by rank deficiency or numerical issues)",
                self.node, self.parents, r
            )
        } else {
            write!(
                f,
                "could not compute local score for node {} with parents {:?} (likely caused by rank deficiency or numerical issues)",
                self.node, self.parents
            )
        }
    }
}

impl std::error::Error for ScoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

impl ScoreError {
    pub fn new_local(u: usize, parents: Vec<usize>) -> Self {
        ScoreError {
            node: u,
            parents,
            added_node: None,
        }
    }

    pub fn new_grow(u: usize, parents: Vec<usize>, r: usize) -> Self {
        ScoreError {
            node: u,
            parents,
            added_node: Some(r),
        }
    }
}
