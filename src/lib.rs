mod ast;
mod eval;
mod parser;
mod tokenizer;

pub use eval::{eval, eval_expr, Value};
