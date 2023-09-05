use std::collections::HashMap;

use crate::{
    Node, NodeDesc, NodeGraph, NodeGraphDesc, NodeId, NodeKind, NodeLink, SocketIn, SocketKind,
    SocketOut, SocketValue,
};

/*
 *  Syntax:
 *  statement = ident '=' expr
 *  expr = node_value ('.' ident)? | list
 *  node_value = float | int | bool | string | enum | ident | node_ctor
 *  enum = ident '::' ident
 *  node_ctor = ident '[' input_list ']'
 *  input = input_name '=' expr
 *  input_list = input | input ',' input_list
 *  list = '[' input_value_list ']'
 *
 */
#[derive(Debug)]
pub struct ParseError {
    msg: String,
    line: usize,
    col: usize,
}
#[derive(Clone, Debug)]
enum Value {
    Float(f64),
    Int(i64),
    Bool(bool),
    String(String),
    Enum(String, String), // name,variant
    NodeId(NodeId),
    NodeOutput(NodeId, String),
    List(Vec<Value>),
}
impl Value {
    fn as_node(&self) -> Option<&NodeId> {
        match self {
            Value::NodeId(id) => Some(id),
            _ => None,
        }
    }
}

struct Parser<'a> {
    chars: Vec<char>,
    env: HashMap<NodeId, Value>,
    graph: NodeGraph,
    pos: usize,
    line: usize,
    col: usize,
    desc: &'a NodeGraphDesc,
    node_desc_map: HashMap<String, &'a NodeDesc>,
    tmp_cnt: usize,
}
impl<'a> Parser<'a> {
    fn new(src: &str, desc: &'a NodeGraphDesc) -> Self {
        let mut parser = Self {
            chars: src.chars().collect(),
            env: HashMap::new(),
            graph: NodeGraph {
                nodes: HashMap::new(),
            },
            pos: 0,
            line: 1,
            col: 1,
            desc,
            node_desc_map: HashMap::new(),
            tmp_cnt: 0,
        };
        for node in &desc.nodes {
            parser.node_desc_map.insert(node.name.clone(), node);
        }
        parser
    }
    fn next_char(&mut self) -> Option<char> {
        // dbg!((self.line, self.col));
        let c = self.chars.get(self.pos).cloned();
        if c == Some('\n') {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        self.pos += 1;
        c
    }
    fn peek(&self, n: usize) -> Option<char> {
        self.chars.get(self.pos + n).cloned()
    }
    fn peek_s(&self, n: usize) -> Option<String> {
        let end = self.pos + n;
        if end > self.chars.len() {
            return None;
        }
        let s = &self.chars[self.pos..end];
        Some(s.iter().collect())
    }
    fn match_s(&self, s: &[&str]) -> bool {
        for (i, &s) in s.iter().enumerate() {
            let p = self.peek_s(i);
            if p.as_ref().map(|s| s.as_str()) != Some(s) {
                return false;
            }
        }
        true
    }
    fn is_valid_number_end(&self, c: char) -> bool {
        c.is_whitespace() || c == ',' || c == ']' || c == ')' || c == '#'
    }
    fn parse_comment(&mut self) {
        if self.peek(0) == Some('#') {
            while let Some(c) = self.next_char() {
                if c == '\n' {
                    break;
                }
            }
        }
    }
    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek(0) {
            if c.is_whitespace() {
                self.next_char();
            } else if c == '#' {
                self.parse_comment();
            } else {
                break;
            }
        }
    }
    fn parse_number(&mut self) -> Result<Value, ParseError> {
        let mut int_part = 0;
        let mut frac_part = 0f64;
        let mut frac_exp = 0;
        let mut is_float = false;
        let sign = if self.peek(0).map(|c| c == '-') == Some(true) {
            self.next_char();
            -1
        } else {
            1
        };
        if self.match_s(&["0x"]) || self.match_s(&["0X"]) {
            self.next_char();
            self.next_char();
            while let Some(c) = self.peek(0) {
                if c.is_digit(16) {
                    int_part = int_part * 16 + c.to_digit(16).unwrap() as i64;
                    self.next_char();
                } else {
                    if !self.is_valid_number_end(c) {
                        return Err(ParseError {
                            msg: format!("invalid character in hex number: {}", c),
                            line: self.line,
                            col: self.col,
                        });
                    }
                    break;
                }
            }
            return Ok(Value::Int(int_part * sign));
        } else {
            while let Some(c) = self.peek(0) {
                if !is_float {
                    if c.is_digit(10) {
                        int_part = int_part * 10 + c.to_digit(10).unwrap() as i64;
                        self.next_char();
                    } else if c == '.' {
                        self.next_char();
                        is_float = true;
                    } else {
                        if !self.is_valid_number_end(c) {
                            return Err(ParseError {
                                msg: format!("invalid character in number: {}", c),
                                line: self.line,
                                col: self.col,
                            });
                        }
                        break;
                    }
                } else {
                    if c.is_digit(10) {
                        frac_exp -= 1;
                        frac_part = frac_part * 10. + c.to_digit(10).unwrap() as f64;
                        self.next_char();
                    } else {
                        if !self.is_valid_number_end(c) {
                            return Err(ParseError {
                                msg: format!("invalid character in number: {}", c),
                                line: self.line,
                                col: self.col,
                            });
                        }
                        break;
                    }
                }
            }
            if is_float {
                return Ok(Value::Float(
                    sign as f64 * ((int_part as f64 + frac_part * 10f64.powi(frac_exp))),
                ));
            } else {
                return Ok(Value::Int(int_part * sign));
            }
        }
    }
    fn parse_identifier(&mut self) -> Result<String, ParseError> {
        let mut s = String::new();
        while let Some(c) = self.peek(0) {
            if c.is_alphanumeric() || c == '_' || c == '#' || c == '$' {
                s.push(c);
                self.next_char();
            } else {
                break;
            }
        }
        Ok(s)
    }
    fn parse_list(&mut self) -> Result<Value, ParseError> {
        if self.peek(0) != Some('[') {
            return Err(ParseError {
                msg: format!("expected '[' but found '{}'", self.peek(0).unwrap_or('\0')),
                line: self.line,
                col: self.col,
            });
        }
        self.next_char();
        let mut list = Vec::new();
        loop {
            self.skip_whitespace();
            if self.peek(0) == Some(']') {
                self.next_char();
                break;
            }
            let mut value = self.parse_value(None)?;
            if let Value::NodeId(id) = value {
                let node = &self.graph.nodes[&id];
                if node.outputs.len() != 1 {
                    return Err(ParseError {
                        msg: format!(
                            "node shorthand can only be used with single-output node only: {}",
                            id.0
                        ),
                        line: self.line,
                        col: self.col,
                    });
                }
                value = Value::NodeOutput(id, node.outputs.keys().next().unwrap().clone());
            }
            list.push(value);
            self.skip_whitespace();
            if self.peek(0) == Some(']') {
                self.next_char();
                break;
            } else if self.peek(0) == Some(',') {
                self.next_char();
            } else {
                return Err(ParseError {
                    msg: "expected ',' or ']' when parsing list".to_string(),
                    line: self.line,
                    col: self.col,
                });
            }
        }
        Ok(Value::List(list))
    }
    fn gen_id(&mut self) -> NodeId {
        loop {
            let id = NodeId(format!("tmp{}", self.tmp_cnt));
            self.tmp_cnt += 1;
            if !self.env.contains_key(&id) {
                return id;
            }
        }
    }
    fn parse_ctor(&mut self, id: Option<NodeId>, node: String) -> Result<Value, ParseError> {
        let id = id.unwrap_or_else(|| self.gen_id());
        if !self.node_desc_map.contains_key(&node) {
            return Err(ParseError {
                msg: format!("unknown node type: {}", node),
                line: self.line,
                col: self.col,
            });
        }
        self.skip_whitespace();
        if self.peek(0) != Some('[') {
            return Err(ParseError {
                msg: "expected '['".to_string(),
                line: self.line,
                col: self.col,
            });
        }
        self.next_char();
        let desc = self.node_desc_map[&node];
        let mut inputs = HashMap::new();
        loop {
            self.skip_whitespace();
            if self.peek(0) == Some(']') {
                self.next_char();
                break;
            }
            let ident = self.parse_identifier()?;
            self.skip_whitespace();
            if self.peek(0) == Some('=') {
                self.next_char();
                self.skip_whitespace();
                let mut v = self.parse_value(None)?;
                if let Value::NodeId(id) = v {
                    let node = &self.graph.nodes[&id];
                    if node.outputs.len() != 1 {
                        return Err(ParseError {
                            msg: format!(
                                "node shorthand can only be used with single-output node only: {}",
                                id.0
                            ),
                            line: self.line,
                            col: self.col,
                        });
                    }
                    v = Value::NodeOutput(id, node.outputs.keys().next().unwrap().clone());
                }
                let expected_kind = desc.inputs.iter().find(|sk| sk.name == ident).map_or_else(
                    || {
                        return Err(ParseError {
                            msg: format!("invalid key: {}", ident),
                            line: self.line,
                            col: self.col,
                        });
                    },
                    |i| Ok(&i.kind),
                )?;
                let get_kind = |v: &Value| -> Result<SocketKind, ParseError> {
                    Ok(match v {
                        Value::Float(_) => SocketKind::Float,
                        Value::Int(_) => SocketKind::Int,
                        Value::Bool(_) => SocketKind::Bool,
                        Value::String(_) => SocketKind::String,
                        Value::Enum(e, _) => SocketKind::Enum(e.clone()),
                        Value::NodeId(_) => unreachable!(),
                        Value::NodeOutput(from_id, s) => {
                            if !self.env.contains_key(from_id) {
                                return Err(ParseError {
                                    msg: format!("unknown node: {}", from_id.0),
                                    line: self.line,
                                    col: self.col,
                                });
                            }
                            let from = self.env[from_id].as_node();
                            if from.is_none() {
                                return Err(ParseError {
                                    msg: format!("invalid node output: {}.{}", from_id.0, s),
                                    line: self.line,
                                    col: self.col,
                                });
                            }
                            let from = from.unwrap();
                            let from = &self.graph.nodes[from];
                            let from_ty = match &from.kind {
                                NodeKind::Node { ty } => ty,
                                _ => unreachable!(),
                            };
                            let from_desc = self.node_desc_map[from_ty];
                            let socket = from_desc.outputs.iter().find(|sk| sk.name == *s);
                            if socket.is_none() {
                                return Err(ParseError {
                                    msg: format!("invalid node output: {}.{}", from_id.0, s),
                                    line: self.line,
                                    col: self.col,
                                });
                            }
                            let socket = socket.unwrap();
                            socket.kind.clone()
                        }
                        Value::List(_) => unreachable!(),
                    })
                };
                let actual_kind = match &v {
                    Value::List(vs) => {
                        if vs.is_empty() {
                            None
                        } else {
                            let first = vs.first().unwrap();
                            let kind = get_kind(first)?;
                            for i in 0..vs.len() {
                                if get_kind(&vs[i])? != kind {
                                    return Err(ParseError {
                                        msg: format!("invalid list: {:?}", vs),
                                        line: self.line,
                                        col: self.col,
                                    });
                                }
                            }
                            Some(SocketKind::List(Box::new(kind), Some(vs.len())))
                        }
                    }
                    v @ _ => Some(get_kind(v)?),
                };
                let macthed = match expected_kind {
                    SocketKind::List(expected_inner, expected_len) => {
                        actual_kind.is_none() || {
                            let actual_kind = actual_kind.as_ref().unwrap();
                            match actual_kind {
                                SocketKind::List(inner, len) => {
                                    if inner != expected_inner {
                                        false
                                    } else if let Some(expected_len) = expected_len {
                                        len.unwrap() == *expected_len
                                    } else {
                                        true
                                    }
                                }
                                _ => false,
                            }
                        }
                    }
                    _ => actual_kind.as_ref().unwrap() == expected_kind,
                };
                if !macthed {
                    return Err(ParseError {
                        msg: format!(
                            "invalid input type: expected {:?}, got {:?}",
                            expected_kind, actual_kind
                        ),
                        line: self.line,
                        col: self.col,
                    });
                }
                inputs.insert(ident, v);
            } else {
                return Err(ParseError {
                    msg: format!("expected '=' but found '{}'", self.peek(0).unwrap_or('\0')),
                    line: self.line,
                    col: self.col,
                });
            }
            self.skip_whitespace();
            if self.peek(0) == Some(']') {
                self.next_char();
                break;
            } else if self.peek(0) == Some(',') {
                self.next_char();
            } else {
                return Err(ParseError {
                    msg: format!(
                        "expected ',' or ']' when parsing node constructor but found '{}'",
                        self.peek(0).unwrap_or('\0')
                    ),
                    line: self.line,
                    col: self.col,
                });
            }
        }
        fn to_socket_value(
            parser: &mut Parser,
            to: NodeId,
            k: String,
            v: Value,
            kind: SocketKind,
        ) -> SocketValue {
            match v {
                Value::Float(v) => SocketValue::Float(v),
                Value::Int(v) => SocketValue::Int(v),
                Value::Bool(v) => SocketValue::Bool(v),
                Value::String(v) => SocketValue::String(v),
                Value::Enum(_, v) => SocketValue::Enum(v),
                Value::NodeId(v) => unreachable!(),
                Value::List(v) => {
                    // dbg!(&kind);
                    let inner = match kind {
                        SocketKind::List(inner, _) => *inner,
                        _ => unreachable!(),
                    };
                    SocketValue::List(
                        v.into_iter()
                            .map(|v| {
                                to_socket_value(parser, to.clone(), k.clone(), v, inner.clone())
                            })
                            .collect(),
                    )
                }
                Value::NodeOutput(node, socket) => {
                    // dbg!(&kind);
                    let link = NodeLink {
                        from: node.clone(),
                        from_socket: socket.clone(),
                        to: to,
                        to_socket: k,
                        ty: kind,
                    };

                    let node = &mut parser.graph.nodes.get_mut(&node).unwrap();
                    node.outputs
                        .get_mut(&socket)
                        .unwrap()
                        .links
                        .push(link.clone());
                    SocketValue::Node(Some(link))
                }
            }
        }
        if inputs.len() != desc.inputs.len() {
            return Err(ParseError {
                msg: format!(
                    "invalid number of inputs: expected {}, got {}",
                    desc.inputs.len(),
                    inputs.len()
                ),
                line: self.line,
                col: self.col,
            });
        }
        let node = Node {
            kind: NodeKind::Node { ty: node.clone() },
            inputs: inputs
                .into_iter()
                .map(|(k, v)| {
                    let kind = desc
                        .inputs
                        .iter()
                        .find(|sk| sk.name == k)
                        .unwrap()
                        .kind
                        .clone();
                    (
                        k.clone(),
                        SocketIn {
                            name: k.clone(),
                            value: to_socket_value(self, id.clone(), k.clone(), v, kind),
                        },
                    )
                })
                .collect(),
            outputs: desc
                .outputs
                .iter()
                .map(|sk| {
                    (
                        sk.name.clone(),
                        SocketOut {
                            name: sk.name.clone(),
                            links: vec![],
                        },
                    )
                })
                .collect(),
        };
        self.graph.nodes.insert(id.clone(), node);
        self.env.insert(id.clone(), Value::NodeId(id.clone()));
        Ok(Value::NodeId(id))
    }
    fn parse_string(&mut self, quote_ch: char) -> Result<Value, ParseError> {
        if self.peek(0) != Some(quote_ch) {
            return Err(ParseError {
                msg: format!("expected '{}'", quote_ch),
                line: self.line,
                col: self.col,
            });
        }
        self.next_char();
        let mut s = String::new();
        while let Some(c) = self.peek(0) {
            if c == quote_ch {
                self.next_char();
                return Ok(Value::String(s));
            } else if c == '\\' {
                self.next_char();
                if let Some(c) = self.peek(0) {
                    match c {
                        'n' => s.push('\n'),
                        'r' => s.push('\r'),
                        't' => s.push('\t'),
                        '\\' => s.push('\\'),
                        '"' => s.push('"'),
                        _ => {
                            return Err(ParseError {
                                msg: format!("invalid escape sequence: \\{}", c),
                                line: self.line,
                                col: self.col,
                            })
                        }
                    }
                } else {
                    return Err(ParseError {
                        msg: "unexpected end of input".to_string(),
                        line: self.line,
                        col: self.col,
                    });
                }
            } else {
                s.push(c);
                self.next_char();
            }
        }
        Err(ParseError {
            msg: "unexpected end of input".to_string(),
            line: self.line,
            col: self.col,
        })
    }
    fn parse_value(&mut self, var: Option<String>) -> Result<Value, ParseError> {
        if let Some(var) = &var {
            if self.env.contains_key(&NodeId(var.clone())) {
                return Err(ParseError {
                    msg: format!("identifier '{}' already defined", var),
                    line: self.line,
                    col: self.col,
                });
            }
        }

        let c = self.peek(0).ok_or(ParseError {
            msg: "unexpected end of input".to_string(),
            line: self.line,
            col: self.col,
        })?;
        if c.is_digit(10) || c == '+' || c == '-' {
            let v = self.parse_number()?;
            if let Some(var) = var {
                self.env.insert(NodeId(var), v.clone());
            }
            Ok(v)
        } else if c == '"' || c == '\'' {
            let v = self.parse_string(c)?;
            if let Some(var) = var {
                self.env.insert(NodeId(var), v.clone());
            }
            Ok(v)
        } else if c == '[' {
            let v = self.parse_list()?;
            if let Some(var) = var {
                self.env.insert(NodeId(var), v.clone());
            }
            Ok(v)
        } else if c.is_alphabetic() || c == '_' || c == '#' || c == '$' {
            let ident = self.parse_identifier()?;
            self.skip_whitespace();

            if self.peek(0) == Some('[') {
                if self.node_desc_map.contains_key(&ident) {
                    self.parse_ctor(var.map(|v| NodeId(v)), ident)
                } else {
                    return Err(ParseError {
                        msg: format!("unknown node type: {}", ident),
                        line: self.line,
                        col: self.col,
                    });
                }
            } else if self.peek(0) == Some('.') {
                self.next_char();
                let ident2 = self.parse_identifier()?;
                Ok(Value::NodeOutput(NodeId(ident), ident2))
            } else if self.peek(0) == Some(':') {
                self.next_char();
                if self.peek(0) != Some(':') {
                    return Err(ParseError {
                        msg: "expected '::'".to_string(),
                        line: self.line,
                        col: self.col,
                    });
                }
                self.next_char();
                let ident2 = self.parse_identifier()?;
                Ok(Value::Enum(ident, ident2))
            } else {
                Ok(Value::NodeId(NodeId(ident)))
            }
        } else {
            Err(ParseError {
                msg: format!("unexpected character: {}", c),
                line: self.line,
                col: self.col,
            })
        }
    }
    fn parse_statement(&mut self) -> Result<(), ParseError> {
        self.skip_whitespace();
        if self.peek(0).is_none() {
            return Ok(());
        }
        let ident = self.parse_identifier()?;
        // dbg!(&ident);
        self.skip_whitespace();
        if self.peek(0) != Some('=') {
            return Err(ParseError {
                msg: format!("expected '=' but found '{}'", self.peek(0).unwrap_or('\0')),
                line: self.line,
                col: self.col,
            });
        }
        self.next_char();
        self.skip_whitespace();
        self.parse_value(Some(ident))?;
        self.skip_whitespace();
        Ok(())
    }
    fn parse(mut self) -> Result<NodeGraph, ParseError> {
        loop {
            self.skip_whitespace();
            if self.peek(0).is_none() {
                break;
            }
            self.parse_statement()?;
        }
        Ok(self.graph)
    }
}
pub fn parse(src: &str, desc: &NodeGraphDesc) -> Result<NodeGraph, ParseError> {
    let parser = Parser::new(src, desc);
    parser.parse()
}
