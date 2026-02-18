// This file defines the library interface for the project.
// It exports functions, structs, and modules that can be used by other projects.

pub mod optimizer;
pub mod processing;
pub mod reporting;
pub mod cli;

pub use optimizer::*;
pub use processing::*;
pub use reporting::*;
pub use cli::{Args, parse_size_string};