# flop

The core Rust implementation of the FLOP causal discovery algorithm. Apart from usage through the Python and R packages, it can also be used as a CLI by compiling with
```
cargo build --release
```
and running
```
./target/release/flop data_file.csv 2.0 --restarts 20
```
