use crate::Scene;
use std::io::Read;
use std::{
    collections::{BTreeMap, HashSet},
    fs::File,
};

#[derive(Clone, Debug, Copy)]
pub enum Number {
    Integer(i64),
    Float(f64),
}
impl Number {
    pub fn to_float(&self) -> f64 {
        match self {
            Number::Integer(i) => *i as f64,
            Number::Float(f) => *f,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Token {
    String(String),
    Ident(String),
    Number(Number),
    Symbol(char),
    WhiteSpace,
}
impl Token {
    pub fn is_symbol(&self, c: char) -> bool {
        if let Token::Symbol(s) = self {
            *s == c
        } else {
            false
        }
    }
    pub fn as_number(&self) -> Number {
        if let Token::Number(n) = self {
            *n
        } else {
            panic!("Not a number: {:?}", self);
        }
    }
}
pub struct TokenStream {
    tokens: Vec<Token>,
    pos: usize,
}
pub fn tokenize(s: String) -> TokenStream {
    let chars = s.chars().collect::<Vec<_>>();
    let mut tokens = Vec::new();
    let mut i = 0;
    let symbols = [
        '(', ')', '[', ']', '{', '}', ',', ':', ';', '=', '+', '-', '*', '/', '@', '!', '?', '^',
    ]
    .into_iter()
    .collect::<HashSet<_>>();
    while i < chars.len() {
        if chars[i].is_whitespace() {
            tokens.push(Token::WhiteSpace);
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
        }
        if chars[i].is_alphabetic() || chars[i] == '_' {
            let mut ident = String::new();
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                ident.push(chars[i]);
                i += 1;
            }
            tokens.push(Token::Ident(ident));
        }
        if chars[i].is_numeric() || chars[i] == '-' {
            let mut number = String::new();
            while i < chars.len()
                && (chars[i].is_numeric()
                    || chars[i] == '-'
                    || chars[i] == '.'
                    || chars[i] == 'e'
                    || chars[i] == 'E')
            {
                number.push(chars[i]);
                i += 1;
            }
            let try_parse_int = number.parse::<i64>();
            if try_parse_int.is_ok() {
                tokens.push(Token::Number(Number::Integer(try_parse_int.unwrap())));
            } else {
                let try_parse_float = number.parse::<f64>();
                if try_parse_float.is_ok() {
                    tokens.push(Token::Number(Number::Float(try_parse_float.unwrap())));
                } else {
                    panic!("Failed to parse number: {}", number);
                }
            }
        }
        if symbols.contains(&chars[i]) {
            tokens.push(Token::Symbol(chars[i]));
            i += 1;
        }
        if chars[i] == '"' {
            let mut string = String::new();
            i += 1;
            while i < chars.len() && chars[i] != '"' {
                if chars[i] == '\\' {
                    i += 1;
                    if i >= chars.len() {
                        panic!("Unexpected end of string");
                    }
                    if chars[i] == 'n' {
                        string.push('\n');
                    } else if chars[i] == 't' {
                        string.push('\t');
                    } else if chars[i] == 'r' {
                        string.push('\r');
                    } else if chars[i] == '\\' {
                        string.push('\\');
                    } else if chars[i] == '"' {
                        string.push('"');
                    } else {
                        panic!("Unknown escape sequence: \\{}", chars[i]);
                    }
                    i += 1;
                } else {
                    string.push(chars[i]);
                    i += 1;
                }
            }
            tokens.push(Token::String(string));
            i += 1;
        }
    }
    TokenStream { tokens, pos: 0 }
}

impl TokenStream {
    pub fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }
    pub fn expect_symbol(&mut self, c: char) {
        if let Some(t) = self.peek() {
            if t.is_symbol(c) {
                self.pos += 1;
                return;
            }
        }
        panic!("Expected symbol: {}", c);
    }
    pub fn parse_reference(&mut self) -> String {
        self.expect_symbol('@');
        if let Some(t) = self.peek() {
            if let Token::Ident(s) = t.clone() {
                self.pos += 1;
                return s;
            }
        }
        panic!("Expected reference but got: {:?}", self.peek());
    }
    pub fn parse_key(&mut self) -> String {
        if let Some(t) = self.peek() {
            if let Token::Ident(s) = t.clone() {
                self.pos += 1;
                return s;
            }
        }
        panic!("Expected key but got: {:?}", self.peek());
    }
    pub fn parse_reference_array(&mut self) -> Vec<String> {
        let mut arr = Vec::new();
        self.expect_symbol('{');
        while let Some(t) = self.peek() {
            if t.is_symbol('}') {
                self.pos += 1;
                break;
            }
            arr.push(self.parse_reference());
            if let Some(t) = self.peek() {
                if t.is_symbol('}') {
                    self.pos += 1;
                    break;
                }
                self.expect_symbol(',');
            }
        }
        arr
    }
    pub fn parse_string_array(&mut self) -> Vec<String> {
        let mut arr = Vec::new();
        self.expect_symbol('{');
        while let Some(t) = self.peek() {
            if t.is_symbol('}') {
                self.pos += 1;
                break;
            }
            arr.push(self.parse_key());
            if let Some(t) = self.peek() {
                if t.is_symbol('}') {
                    self.pos += 1;
                    break;
                }
                self.expect_symbol(',');
            }
        }
        arr
    }
    pub fn parse_float_array(&mut self) -> Vec<f64> {
        let mut arr = Vec::new();
        self.expect_symbol('{');
        while let Some(t) = self.peek() {
            if t.is_symbol('}') {
                self.pos += 1;
                break;
            }
            arr.push(t.as_number().to_float());
            self.pos += 1;
            if let Some(t) = self.peek() {
                if t.is_symbol('}') {
                    self.pos += 1;
                    break;
                }
                self.expect_symbol(',');
            }
        }
        arr
    }
}
// struct ImportContext {
//     scene: Scene,
//     ts: TokenStream,
// }
// impl ImportContext {
//     fn import(&mut self) {

//     }
// }
// pub fn import_luisa_scene(path: &str, out: &str) -> Scene {
//     let mut file = File::open(path).unwrap_or_else(|e| panic!("Failed to open file: {}", e));
//     let mut contents = String::new();
//     file.read_to_string(&mut contents)
//         .unwrap_or_else(|e| panic!("Failed to read file: {}", e));
//     let tokens = tokenize(contents);
//     let mut ctx = ImportContext {
//         scene: Scene::new(),
//         ts: tokens,
//     };
//     ctx.import();
//     unsafe { ctx.scene.compact(out.into()).unwrap() };
//     ctx.scene
// }
