use std::sync::atomic::AtomicBool;

pub static GLOBAL_ABORT: AtomicBool = AtomicBool::new(false);
