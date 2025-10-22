use crate::bic::Bic;
use crate::error::ScoreError;
use crate::scores::{GlobalScore, LocalScore};
use crate::token_buffer::TokenBuffer;
use crate::utils;

fn grow(
    v: usize,
    v_local: &mut LocalScore,
    non_parents: &mut Vec<usize>,
    score: &Bic,
) -> Result<(), ScoreError> {
    loop {
        let mut done = true;
        for &w in non_parents.clone().iter().rev() {
            let v_local_new = score.local_score_plus(v, v_local, w)?;
            if v_local_new.bic <= v_local.bic {
                *v_local = v_local_new;
                utils::rem_first(non_parents, w);
                done = false;
            }
        }
        if done {
            break;
        }
    }
    Ok(())
}

fn shrink(
    v: usize,
    v_local: &mut LocalScore,
    non_parents: &mut Vec<usize>,
    score: &Bic,
) -> Result<(), ScoreError> {
    loop {
        let mut done = true;
        for &w in v_local.parents.clone().iter().rev() {
            let v_local_new = score.local_score_minus(v, v_local, w)?;
            if v_local_new.bic <= v_local.bic {
                *v_local = v_local_new;
                non_parents.push(w);
                done = false;
            }
        }
        if done {
            break;
        }
    }
    Ok(())
}

// fit parents from scratch
pub fn fit_parents(v: usize, prefix: &[usize], score: &Bic) -> Result<LocalScore, ScoreError> {
    let parents = Vec::new();
    let mut non_parents = prefix.to_vec();
    let mut v_local = score.local_score_init(v, parents)?;
    grow(v, &mut v_local, &mut non_parents, score)?;
    shrink(v, &mut v_local, &mut non_parents, score)?;
    Ok(v_local)
}

fn set_diff(tokens: &mut TokenBuffer, s1: &[usize], s2: &[usize]) -> Vec<usize> {
    tokens.clear();
    for &x in s2.iter() {
        tokens.set(x);
    }
    let res = s1.iter().copied().filter(|&x| !tokens.check(x)).collect();
    res
}

pub fn fit_parents_minus(
    v: usize,
    v_local: &LocalScore,
    prefix: &[usize],
    r: usize,
    score: &Bic,
    tokens: &mut TokenBuffer,
) -> Result<LocalScore, ScoreError> {
    if !v_local.parents.contains(&r) {
        return Ok(v_local.clone());
    }

    let mut v_local_new = score.local_score_minus(v, v_local, r)?;
    let mut non_parents = set_diff(tokens, prefix, &v_local_new.parents);

    grow(v, &mut v_local_new, &mut non_parents, score)?;
    shrink(v, &mut v_local_new, &mut non_parents, score)?;

    Ok(v_local_new)
}

pub fn fit_parents_plus(
    v: usize,
    v_local: &LocalScore,
    prefix: &[usize],
    r: usize,
    score: &Bic,
    tokens: &mut TokenBuffer,
) -> Result<LocalScore, ScoreError> {
    // check if adding r is an improvement
    let mut v_local_new = score.local_score_plus(v, v_local, r)?;

    if v_local_new.bic > v_local.bic {
        return Ok(v_local.clone());
    }
    let mut non_parents = set_diff(tokens, prefix, &v_local_new.parents);

    grow(v, &mut v_local_new, &mut non_parents, score)?;
    shrink(v, &mut v_local_new, &mut non_parents, score)?;

    Ok(v_local_new)
}

// fit permutation from scratch
pub fn perm_to_dag(perm: &[usize], score: &Bic) -> Result<GlobalScore, ScoreError> {
    let mut g = GlobalScore::new(perm.len(), score)?;
    for (i, &v) in perm.iter().enumerate() {
        g.local_scores[v] = fit_parents(v, &perm[0..i], score)?;
    }
    Ok(g)
}
