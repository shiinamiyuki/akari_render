use std::{fs::File, rc::Rc};

use crate::Graph;

pub struct ParserConfig {
    pub file_loader: Box<dyn Fn(&str) -> std::io::Result<File>>,
}
impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            file_loader: Box::new(|path| File::open(path)),
        }
    }
}
pub struct ParsingLog {
    pub warnings: Vec<String>,
}
#[derive(Debug, Clone)]
pub struct SourceLocation {
    pub file: Rc<String>,
    pub start: usize,
    pub end: usize,
}
#[derive(Debug)]
pub struct ParsingError {
    pub message: String,
    pub location: SourceLocation,
}

struct Parser {
    src: Vec<char>,
}

pub fn parse(input: &str, config: ParserConfig) -> Result<(Graph, ParsingLog), ParsingError> {
    unimplemented!()
}
