use clap::{Args, Parser};
use csv::ReaderBuilder;
use flop::algo::FlopConfig;
use nalgebra::DMatrix;
use std::error::Error;
use std::fs::File;

/// A command line tool for running the FLOP causal discovery algorithm
#[derive(Parser, Debug)]
#[command(
    name = "flop",
    version,
    after_help = "\x1b[1m\x1b[4mOutput:\x1b[0m Prints the graph to stdout as edge list in csv format with columns 'from', 'to' and 'edge-type'\n\n \x1b[1m\x1b[4mExamples:\x1b[0m\n  Run with 50 restarts: flop path_to_data_file.csv 2.0 --restarts 50\n  Run for 5 seconds:     flop path_to_data_file.csv 1.0 --timeout 5.0\n  Run until kill signal: flop path_to_data_file.csv 1.0 --manual-termination"
)]
struct Cli {
    /// Path to the data csv file
    data_file: String,

    /// Penalty parameter lambda
    lambda: f64,

    #[command(flatten)]
    termination: TerminationArgs,

    /// Output a DAG instead of a CPDAG (the latter is default behaviour)
    #[arg(short, long)]
    output_dag: bool,

    /// Perform the backward phase of GES before termination (needed for asymptotic consistency, but typically leads to no further improvements)
    #[arg(short, long)]
    ges_backward: bool,
}

#[derive(Args, Debug)]
#[group(required = true, multiple = false)]
struct TerminationArgs {
    /// Number of restarts of the local search (a reasonable default value is '50')
    #[arg(short, long, group = "termination")]
    restarts: Option<usize>,

    /// Timeout in seconds (provide this instead of 'restarts' to let FLOP run for a specified amount of time)
    #[arg(short, long, group = "termination")]
    timeout: Option<f64>,

    /// Run FLOP until it is manually terminated (provide this instead of 'restarts' or 'timeout' to stop FLOP by sending SIGTERM)
    #[arg(short, long, group = "termination")]
    manual_termination: bool,
}

fn read_data(data_file: &String) -> Result<(Vec<String>, DMatrix<f64>), Box<dyn Error>> {
    let file = File::open(data_file)?;
    let mut rdr = ReaderBuilder::new().from_reader(file);
    let mut data_slice: Vec<f64> = Vec::new();
    let mut rows = 0;
    for result in rdr.records() {
        let record = result?;
        let mut row: Vec<f64> = record
            .iter()
            .map(|s| s.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()?;
        data_slice.append(&mut row);
        rows += 1;
    }
    Ok((
        rdr.headers()?.iter().map(|s| s.to_owned()).collect(),
        DMatrix::from_row_slice(rows, data_slice.len() / rows, &data_slice),
    ))
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let (_, data) = read_data(&cli.data_file)?;

    let flop_config = FlopConfig::new(
        cli.lambda,
        cli.termination.restarts,
        cli.termination.timeout,
        cli.termination.manual_termination,
    );
    let g = flop::algo::run(&data, flop_config);

    if cli.output_dag {
        g?.output();
    } else {
        g?.to_cpdag().output();
    }
    Ok(())
}
