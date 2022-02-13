use std::{collections::HashMap, rc::Rc};

use crate::var::Program;


pub(crate) struct CodeGen {
    program: Rc<Program>,
}
