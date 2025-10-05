use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;

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
