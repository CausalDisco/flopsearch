use std::thread;
use std::time::Duration;

use nalgebra::DMatrix;
use rand::{thread_rng, Rng};
use std::sync::atomic::Ordering;

use crate::bic::Bic;
use crate::error::{FlopError, ScoreError};
use crate::global_abort::GLOBAL_ABORT;
use crate::graph::Dag;
use crate::scores::{GlobalScore, LocalScore};
use crate::token_buffer::TokenBuffer;
use crate::{fit_parents, pivoted_cholesky, utils};

static EPS: f64 = 1e-9;

pub struct FlopConfig {
    lambda: f64,
    restarts: Option<usize>,
    timeout: Option<f64>,
    manual_termination: bool,
}

impl FlopConfig {
    pub fn new(
        lambda: f64,
        restarts: Option<usize>,
        timeout: Option<f64>,
        manual_termination: bool,
    ) -> Self {
        Self {
            lambda,
            restarts,
            timeout,
            manual_termination,
        }
    }
}

pub fn run(data: &DMatrix<f64>, config: FlopConfig) -> Result<Dag, FlopError> {
    if !(config.restarts.is_some() || config.timeout.is_some() || config.manual_termination) {
        return Err(FlopError::InvalidConfig(
            "config is missing number of restarts or timeout or manual termination flag".to_owned(),
        ));
    }

    if config.manual_termination {
        ctrlc::set_handler(|| {
            GLOBAL_ABORT.store(true, Ordering::SeqCst);
        })
        .expect("Error setting Ctrl-C handler");
    }

    if let Some(timeout) = config.timeout {
        thread::spawn(move || {
            thread::sleep(Duration::from_secs_f64(timeout));
            GLOBAL_ABORT.store(true, Ordering::SeqCst);
        });
    }

    let p = data.ncols();
    let n = data.nrows();

    let mut rng = thread_rng();
    let num_perturbations = (p as f64).ln().round() as usize;

    let corr = utils::corr_matrix(data);

    let mut best_perm = match pivoted_cholesky::cholesky_left_min_diag(&corr) {
        None => utils::rand_perm(p, &mut rng),
        Some((_, order)) => order,
    };

    let score = Bic::from_cov(n, corr, config.lambda);

    let mut best_bic = f64::MAX;
    let mut best_g = None;

    let mut iter = 0;

    loop {
        if let Some(restarts) = config.restarts {
            if iter > restarts {
                break;
            }
        }
        let mut perm = best_perm.clone();
        if iter > 0 {
            for _ in 0..num_perturbations {
                let u = rng.gen_range(0..perm.len());
                let v = rng.gen_range(0..perm.len());
                perm.swap(u, v);
            }
        }

        let mut g = fit_parents::perm_to_dag(&perm, &score)?;
        let mut bic = g.score();

        loop {
            let last_bic = bic;

            for x in perm.clone() {
                reinsert(&mut perm, &mut g, &score, &mut bic, x)?;
                if iter > 0 && GLOBAL_ABORT.load(Ordering::SeqCst) {
                    break;
                }
            }

            // break if no improvement during full iteration
            if last_bic - EPS <= bic {
                break;
            }
            if iter > 0 && GLOBAL_ABORT.load(Ordering::SeqCst) {
                break;
            }
        }

        // need to be at least EPS better than previous optimum
        if bic < best_bic - EPS {
            best_bic = bic;
            best_perm = perm;
            best_g = Some(g);
        }
        if GLOBAL_ABORT.load(Ordering::SeqCst) {
            break;
        }
        iter += 1;
    }

    Ok(Dag::from_global_score(&best_g.unwrap()))
}

fn reinsert(
    perm: &mut Vec<usize>,
    g: &mut GlobalScore,
    score: &Bic,
    score_value: &mut f64,
    v: usize,
) -> Result<bool, ScoreError> {
    let v_index = perm.iter().position(|&x| x == v).unwrap();
    let mut v_curr_local = g.local_scores[v].clone();

    let mut best_diff = EPS; // allow small worsening in single moves
    let mut best_ins_pos = v_index;
    let mut curr_diff = 0.0;

    let mut v_best_local: Vec<Option<LocalScore>> = vec![None; perm.len()];
    let mut z_best_local: Vec<Option<LocalScore>> = vec![None; perm.len()];
    let mut tokens = TokenBuffer::new(g.p);

    // look at positions preceding v
    for pos in (0..v_index).rev() {
        // try to reinsert BEFORE element at pos, which we term z
        let z = perm[pos];
        let mut prefix = perm[0..pos].to_vec();

        let v_new_local =
            fit_parents::fit_parents_minus(v, &v_curr_local, &prefix, z, score, &mut tokens)?;
        let v_score_diff = v_new_local.bic - v_curr_local.bic;
        v_curr_local = v_new_local.clone();

        // parents of z are updated based on addition of v
        prefix.push(v);
        let z_curr_local = &g.local_scores[z];
        let z_new_local =
            fit_parents::fit_parents_plus(z, z_curr_local, &prefix, v, score, &mut tokens)?;
        let z_score_diff = z_new_local.bic - z_curr_local.bic;

        curr_diff += v_score_diff + z_score_diff;
        if curr_diff < best_diff {
            best_diff = curr_diff;
            best_ins_pos = pos;
            // this will only be needed if v is put at pos
            v_best_local[pos] = Some(v_new_local);
        }
        z_best_local[pos] = Some(z_new_local);
    }
    // look at positions succeeding v
    // start with some resets
    curr_diff = 0.0;
    v_curr_local = g.local_scores[v].clone();

    for pos in v_index + 1..perm.len() {
        // try to reinsert AFTER element at pos, which we again term z
        let z = perm[pos];
        let mut prefix = perm[0..pos + 1].to_vec();
        // remove v from prefix
        utils::rem_first(&mut prefix, v);
        // parents of v are updated based on addition of z
        let v_new_local =
            fit_parents::fit_parents_plus(v, &v_curr_local, &prefix, z, score, &mut tokens)?;
        let v_score_diff = v_new_local.bic - v_curr_local.bic;
        v_curr_local = v_new_local.clone();

        // remove z from prefix
        utils::rem_first(&mut prefix, z);
        let z_curr_local = &g.local_scores[z];
        // parents of z are updated based on removal of v
        let z_new_local =
            fit_parents::fit_parents_minus(z, z_curr_local, &prefix, v, score, &mut tokens)?;
        let z_score_diff = z_new_local.bic - z_curr_local.bic;

        curr_diff += v_score_diff + z_score_diff;
        if curr_diff < best_diff {
            best_diff = curr_diff;
            best_ins_pos = pos;
            // this will only be needed if v is put at pos
            v_best_local[pos] = Some(v_new_local);
        }
        z_best_local[pos] = Some(z_new_local);
    }

    if best_ins_pos == v_index {
        return Ok(false);
    }

    *score_value += best_diff;

    g.local_scores[v] = v_best_local[best_ins_pos].clone().unwrap();
    if best_ins_pos < v_index {
        for (i, &z) in perm[best_ins_pos..v_index].iter().enumerate() {
            g.local_scores[z] = z_best_local[best_ins_pos + i].clone().unwrap();
        }
    } else {
        for (i, &z) in perm[v_index + 1..best_ins_pos + 1].iter().enumerate() {
            g.local_scores[z] = z_best_local[v_index + i + 1].clone().unwrap();
        }
    }
    perm.remove(v_index);
    perm.insert(best_ins_pos, v);
    Ok(true)
}
