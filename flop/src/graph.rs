use crate::scores::GlobalScore;

#[derive(Debug)]
pub struct Dag {
    pub p: usize,
    pub parents: Vec<Vec<usize>>,
}

impl Dag {
    pub fn new(p: usize) -> Self {
        Self {
            p,
            parents: vec![Vec::new(); p],
        }
    }

    pub fn from_global_score(g: &GlobalScore) -> Self {
        Self {
            p: g.p,
            parents: g.local_scores.iter().map(|ls| ls.parents.clone()).collect(),
        }
    }

    pub fn from_edge_list(p: usize, edges: Vec<(usize, usize)>) -> Self {
        let mut parents = vec![Vec::new(); p];
        edges.iter().for_each(|&(u, v)| parents[v].push(u));
        Self { p, parents }
    }

    pub fn to_cpdag(&self) -> Cpdag {
        let mut dir_edges = Vec::new();
        let mut undir_edges = Vec::new();

        let to = self.topological_ordering();
        let mut inv_to = vec![0; to.len()];
        to.iter().enumerate().for_each(|(i, &x)| inv_to[x] = i);

        let mut is_compelled = vec![false; self.p];
        let mut compelled_ingoing: Vec<Vec<usize>> = vec![Vec::new(); self.p];

        for &y in to.iter() {
            let mut parents_y = self.parents[y].clone();
            if parents_y.is_empty() {
                continue;
            }

            parents_y.sort();
            parents_y.iter().for_each(|&u| is_compelled[u] = false);

            let x = *parents_y.iter().max_by_key(|&u| inv_to[*u]).unwrap();
            let mut parents_x = self.parents[x].clone();
            parents_x.sort();

            let mut done = false;
            for &w in compelled_ingoing[x].iter() {
                if parents_y.binary_search(&w).is_err() {
                    parents_y.iter().for_each(|&u| is_compelled[u] = true);
                    done = true;
                    break;
                } else {
                    is_compelled[w] = true;
                }
            }
            if !done {
                for &z in parents_y.iter() {
                    if z == x {
                        continue;
                    }
                    if parents_x.binary_search(&z).is_err() {
                        parents_y.iter().for_each(|&u| is_compelled[u] = true);
                        break;
                    }
                }
            }
            for &v in parents_y.iter() {
                if is_compelled[v] {
                    compelled_ingoing[y].push(v);
                    dir_edges.push((v, y));
                } else {
                    undir_edges.push((v, y));
                }
            }
        }
        Cpdag::new(self.p, dir_edges, undir_edges)
    }

    fn top_ordering_dfs(&self, vis: &mut Vec<bool>, ord: &mut Vec<usize>, u: usize) {
        if vis[u] {
            return;
        }
        vis[u] = true;
        for &v in self.parents[u].iter() {
            self.top_ordering_dfs(vis, ord, v);
        }
        ord.push(u);
    }

    pub fn topological_ordering(&self) -> Vec<usize> {
        let mut vis = vec![false; self.p];
        let mut ord: Vec<usize> = Vec::new();
        for u in 0..self.p {
            if !vis[u] {
                self.top_ordering_dfs(&mut vis, &mut ord, u);
            }
        }
        ord
    }

    pub fn add_edge(&mut self, u: usize, v: usize) {
        self.parents[v].push(u);
    }

    pub fn output(&self) {
        let p = self.p;
        let m = self.parents.iter().map(|x| x.len()).sum::<usize>();
        println!("{p} {m} dag");
        for (v, pa) in self.parents.iter().enumerate() {
            for &u in pa.iter() {
                println!("{} {} directed", u, v);
            }
        }
    }
}

#[derive(Debug)]
pub struct Cpdag {
    pub p: usize,
    pub out_neighbors: Vec<Vec<usize>>,
    pub undir_neighbors: Vec<Vec<usize>>,
}

impl Cpdag {
    pub fn new(p: usize, dir_edges: Vec<(usize, usize)>, undir_edges: Vec<(usize, usize)>) -> Self {
        let mut out_neighbors = vec![Vec::new(); p];
        let mut undir_neighbors = vec![Vec::new(); p];
        dir_edges.iter().for_each(|&(u, v)| {
            out_neighbors[u].push(v);
        });
        undir_edges.iter().for_each(|&(u, v)| {
            undir_neighbors[u].push(v);
            undir_neighbors[v].push(u);
        });
        Self {
            p,
            out_neighbors,
            undir_neighbors,
        }
    }

    pub fn output(&self) {
        let p = self.p;
        let m = self.out_neighbors.iter().map(|x| x.len()).sum::<usize>()
            + self.undir_neighbors.iter().map(|x| x.len()).sum::<usize>() / 2;
        println!("{p} {m} cpdag");
        for (u, out) in self.out_neighbors.iter().enumerate() {
            for &v in out.iter() {
                println!("{} {} directed", u, v);
            }
        }
        for (u, out) in self.undir_neighbors.iter().enumerate() {
            for &v in out.iter() {
                if u < v {
                    println!("{} {} undirected", u, v);
                }
            }
        }
    }
}
