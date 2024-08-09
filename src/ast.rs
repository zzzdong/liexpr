use std::collections::HashMap;
use std::fmt;

use crate::tokenizer::Token;

#[derive(Debug, Clone, Default)]
pub struct Program {
    pub(crate) statements: Vec<Statement>,
    pub(crate) functions: HashMap<String, Function>,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub(crate) name: String,
    pub(crate) parameters: Vec<String>,
    pub(crate) body: BlockStatement,
}

#[derive(Debug, Clone)]
pub enum Statement {
    Empty,
    Continue,
    Break,
    Return {
        value: Option<Box<Expression>>,
    },
    Let {
        name: String,
        value: Option<Box<Expression>>,
    },
    For {
        initializer: Box<Statement>,
        condition: Option<Box<Expression>>,
        increment: Option<Box<Expression>>,
        body: BlockStatement,
    },
    If {
        condition: Box<Expression>,
        then_branch: BlockStatement,
        else_branch: Option<Box<Statement>>,
    },
    Expression {
        expression: Box<Expression>,
    },
    Block(BlockStatement),
    Function(Function),
}

#[derive(Debug, Clone)]
pub struct BlockStatement(pub(crate) Vec<Statement>);

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Binary {
        left: Box<Expression>,
        operator: Operator,
        right: Box<Expression>,
    },
    Prefix {
        operator: Operator,
        expression: Box<Expression>,
    },
    Postfix {
        operator: Operator,
        expression: Box<Expression>,
    },
    Literal {
        value: Literal,
    },
    Grouping {
        expression: Box<Expression>,
    },
    Variable {
        name: String,
    },
    Call {
        callee: Box<Expression>,
        arguments: Vec<Expression>,
    },
    Index {
        callee: Box<Expression>,
        index: Box<Expression>,
    },
    Array {
        elements: Vec<Expression>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operator {
    /// +
    Plus,
    /// -
    Minus,
    /// *
    Multiply,
    /// /
    Divide,
    /// %
    Modulo,
    /// !
    Negate,
    /// =
    Assign,
    /// ==
    Equal,
    /// !=
    NotEqual,
    /// >
    Greater,
    /// >=
    GreaterEqual,
    /// <
    Less,
    /// <=
    LessEqual,
    /// &&
    LogicalAnd,
    /// ||
    LogicalOr,
    /// .
    Access,
    /// ++
    Increase,
    /// --
    Decrease,
    /// +=
    PlusAssign,
    /// -=
    MinusAssign,
    /// *=
    MultiplyAssign,
    /// /=
    DivideAssign,
    /// %=
    ModuloAssign,
    /// call
    Call,
    /// index
    Index,
}

impl Operator {
    pub fn binary_from_token(token: &Token) -> Result<Operator, String> {
        match token {
            Token::Plus => Ok(Operator::Plus),
            Token::Minus => Ok(Operator::Minus),
            Token::Star => Ok(Operator::Multiply),
            Token::Slash => Ok(Operator::Divide),
            Token::Percent => Ok(Operator::Modulo),
            Token::Bang => Ok(Operator::Negate),
            Token::Equal => Ok(Operator::Assign),
            Token::EqualEqual => Ok(Operator::Equal),
            Token::BangEqual => Ok(Operator::NotEqual),
            Token::Greater => Ok(Operator::Greater),
            Token::GreaterEqual => Ok(Operator::GreaterEqual),
            Token::Less => Ok(Operator::Less),
            Token::LessEqual => Ok(Operator::LessEqual),
            Token::AndAnd => Ok(Operator::LogicalAnd),
            Token::OrOr => Ok(Operator::LogicalOr),
            Token::Dot => Ok(Operator::Access),
            Token::PlusPlus => Ok(Operator::Increase),
            Token::MinusMinus => Ok(Operator::Decrease),
            Token::PlusEqual => Ok(Operator::PlusAssign),
            Token::MinusEqual => Ok(Operator::MinusAssign),
            Token::StarEqual => Ok(Operator::MultiplyAssign),
            Token::SlashEqual => Ok(Operator::DivideAssign),
            Token::PercentEqual => Ok(Operator::ModuloAssign),
            _ => Err(format!("Invalid token {token:?} for binary operator")),
        }
    }

    pub fn prefix_from_token(token: &Token) -> Result<Operator, String> {
        match token {
            Token::Minus => Ok(Operator::Minus),
            Token::Bang => Ok(Operator::Negate),
            _ => Err(format!("Invalid token {token:?} for prefix operator")),
        }
    }

    pub fn postfix_from_token(token: &Token) -> Result<Operator, String> {
        match token {
            Token::PlusPlus => Ok(Operator::Increase),
            Token::MinusMinus => Ok(Operator::Decrease),
            Token::LeftParen => Ok(Operator::Call),
            Token::LeftBracket => Ok(Operator::Index),
            _ => Err(format!("Invalid token {token:?} for postfix operator")),
        }
    }
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operator::Plus => write!(f, "+"),
            Operator::Minus => write!(f, "-"),
            Operator::Multiply => write!(f, "*"),
            Operator::Divide => write!(f, "/"),
            Operator::Modulo => write!(f, "%"),
            Operator::Negate => write!(f, "!"),
            Operator::Assign => write!(f, "="),
            Operator::Equal => write!(f, "=="),
            Operator::NotEqual => write!(f, "!="),
            Operator::Greater => write!(f, ">"),
            Operator::GreaterEqual => write!(f, ">="),
            Operator::Less => write!(f, "<"),
            Operator::LessEqual => write!(f, "<="),
            Operator::LogicalAnd => write!(f, "&&"),
            Operator::LogicalOr => write!(f, "||"),
            Operator::Access => write!(f, "."),
            Operator::Increase => write!(f, "++"),
            Operator::Decrease => write!(f, "--"),
            Operator::PlusAssign => write!(f, "+="),
            Operator::MinusAssign => write!(f, "-="),
            Operator::MultiplyAssign => write!(f, "*="),
            Operator::DivideAssign => write!(f, "/="),
            Operator::ModuloAssign => write!(f, "%="),
            Operator::Call => write!(f, "()"),
            Operator::Index => write!(f, "[]"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Null,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
}
