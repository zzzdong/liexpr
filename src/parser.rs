use std::collections::HashMap;
use std::rc::Rc;

use crate::ast::*;
use crate::tokenizer::{Token, Tokens};

pub struct Parser {
    tokens: Tokens,
    next: Option<Token>,
}

impl Parser {
    pub fn new(tokens: Tokens) -> Parser {
        let mut tokens = tokens;

        let next = tokens.next();
        Parser { tokens, next }
    }

    pub fn parse_program(script: &str) -> Result<Program, String> {
        let mut parser = Parser::new(Tokens::new(script)?);

        parser.parse()
    }

    pub fn parse_expression(script: &str) -> Result<Expression, String> {
        let mut parser = Parser::new(Tokens::new(script)?);

        parser.parse_expr()
    }

    fn next_token(&mut self) -> Option<Token> {
        std::mem::replace(&mut self.next, self.tokens.next())
    }

    fn peek_token(&self) -> Option<&Token> {
        self.next.as_ref()
    }

    fn must_next(&mut self, token: &Token) -> Result<(), String> {
        match self.next_token() {
            Some(ref next) => {
                if next != token {
                    return Err(format!("Expected {:?}, but got {:?}", token, next));
                }
                Ok(())
            }
            _ => Err(format!("Expected {:?}, but got Eof", token)),
        }
    }

    pub fn parse(&mut self) -> Result<Program, String> {
        let mut statements = Vec::new();
        let mut functions = HashMap::new();

        while let Some(_token) = self.peek_token() {
            match self.parse_statement()? {
                Statement::Function(func) => {
                    functions.insert(func.name.clone(), Rc::new(func));
                }
                stmt => {
                    statements.push(stmt);
                }
            }
        }

        Ok(Program {
            statements,
            functions,
        })
    }

    fn parse_statement(&mut self) -> Result<Statement, String> {
        let token = self.peek_token().unwrap();

        let stmt = match token {
            Token::Semicolon => {
                self.must_next(&Token::Semicolon)?;
                Statement::Empty
            }
            Token::Break => self.parse_break_statement()?,
            Token::Continue => self.parse_continue_statement()?,
            Token::Return => self.parse_return_statement()?,
            Token::Let => self.parse_let_statement()?,
            Token::For => self.parse_for_statement()?,
            Token::If => self.parse_if_statement()?,
            Token::LeftBrace => self.parse_block_statement().map(Statement::Block)?,
            Token::Fn => self.parse_function_statement().map(Statement::Function)?,
            _ => self.parse_expression_statement()?,
        };

        Ok(stmt)
    }

    fn parse_block_statement(&mut self) -> Result<BlockStatement, String> {
        self.must_next(&Token::LeftBrace)?;

        let mut statements = Vec::new();

        while let Some(token) = self.peek_token() {
            match token {
                Token::RightBrace => {
                    self.must_next(&Token::RightBrace)?;
                    break;
                }
                _ => {
                    statements.push(self.parse_statement()?);
                }
            }
        }

        Ok(BlockStatement(statements))
    }

    fn parse_function_statement(&mut self) -> Result<Function, String> {
        self.must_next(&Token::Fn)?;

        let name = match self.next_token() {
            Some(Token::Identifier(name)) => name,
            _ => return Err("Expected identifier".to_string()),
        };

        self.must_next(&Token::LeftParen)?;

        let parameters =
            self.separated_list(Token::Comma, Self::parse_identifier, Token::RightParen)?;

        self.must_next(&Token::RightParen)?;

        let body = self.parse_block_statement()?;

        Ok(Function {
            name,
            parameters,
            body,
        })
    }

    fn parse_break_statement(&mut self) -> Result<Statement, String> {
        self.must_next(&Token::Break)?;
        self.must_next(&Token::Semicolon)?;

        Ok(Statement::Break)
    }

    fn parse_continue_statement(&mut self) -> Result<Statement, String> {
        self.must_next(&Token::Continue)?;
        self.must_next(&Token::Semicolon)?;

        Ok(Statement::Continue)
    }

    fn parse_return_statement(&mut self) -> Result<Statement, String> {
        self.must_next(&Token::Return)?;

        let expr = match self.peek_token() {
            Some(Token::Semicolon) => None,
            _ => Some(self.parse_expr()?),
        };

        Ok(Statement::Return {
            value: expr.map(Box::new),
        })
    }

    fn parse_let_statement(&mut self) -> Result<Statement, String> {
        self.must_next(&Token::Let)?;

        let name = match self.next_token() {
            Some(Token::Identifier(name)) => name,
            _ => return Err("Expected identifier".to_string()),
        };

        let value = match self.peek_token() {
            Some(Token::Equal) => {
                self.must_next(&Token::Equal)?;

                Some(Box::new(self.parse_expr()?))
            }
            _ => None,
        };

        self.must_next(&Token::Semicolon)?;

        Ok(Statement::Let { name, value })
    }

    fn parse_for_statement(&mut self) -> Result<Statement, String> {
        self.must_next(&Token::For)?;
        self.must_next(&Token::LeftParen)?;

        let initializer = self.parse_statement()?;
        match &initializer {
            Statement::Empty => {}
            Statement::Let { name, value } => {}
            Statement::Expression { expression } => {}
            _ => {
                return Err(format!(
                    "Unexpected initializer statement: {:?}",
                    initializer
                ))
            }
        }

        let condition = self
            .terminated_with(Token::Semicolon, Self::parse_expr)?
            .map(Box::new);

        let increment = self
            .terminated_with(Token::RightParen, Self::parse_expr)?
            .map(Box::new);

        let body = self.parse_block_statement()?;

        Ok(Statement::For {
            initializer: Box::new(initializer),
            condition,
            increment,
            body,
        })
    }

    fn parse_if_statement(&mut self) -> Result<Statement, String> {
        self.must_next(&Token::If)?;

        self.must_next(&Token::LeftParen)?;

        let condition = self.parse_expr()?;

        self.must_next(&Token::RightParen)?;

        let then_branch = self.parse_block_statement()?;

        let else_branch = match self.peek_token() {
            Some(Token::Else) => {
                self.must_next(&Token::Else)?;
                Some(self.parse_statement()?)
            }
            _ => None,
        };

        Ok(Statement::If {
            condition: Box::new(condition),
            then_branch,
            else_branch: else_branch.map(Box::new),
        })
    }

    fn parse_expression_statement(&mut self) -> Result<Statement, String> {
        let expr = self.parse_expr()?;
        self.must_next(&Token::Semicolon)?;

        Ok(Statement::Expression {
            expression: Box::new(expr),
        })
    }

    fn parse_identifier(&mut self) -> Result<String, String> {
        match self.next_token() {
            Some(Token::Identifier(name)) => Ok(name),
            _ => Err("Expected identifier".to_string()),
        }
    }

    fn parse_expr(&mut self) -> Result<Expression, String> {
        self.parse_sub_expression(Precedence::Lowest)
    }

    fn parse_sub_expression(&mut self, precedence: Precedence) -> Result<Expression, String> {
        let mut expr = self.parse_prefix()?;

        while let Some(token) = self.peek_token() {
            // println!(
            //     "parse_sub_expression, token{:?}: {:?}, precedence: {:?}",
            //     token,
            //     Self::token_precedence(&token),
            //     precedence
            // );

            let next_precedence = Self::token_precedence(token);
            if next_precedence <= precedence {
                return Ok(expr);
            }

            expr = self.parse_infix(expr, next_precedence)?;
        }

        Ok(expr)
    }

    fn parse_infix(
        &mut self,
        expr: Expression,
        precedence: Precedence,
    ) -> Result<Expression, String> {
        let token = match self.peek_token() {
            Some(token) => token,
            None => return Ok(expr),
        };

        match token {
            token if Operator::postfix_from_token(token).is_ok() => {
                self.parse_postfix(&token.clone(), expr)
            }
            _ => {
                let token = self.next_token().unwrap();
                let operator = Operator::binary_from_token(&token)?;

                let rhs = self.parse_sub_expression(precedence)?;

                Ok(Expression::Binary {
                    left: Box::new(expr),
                    operator,
                    right: Box::new(rhs),
                })
            }
        }
    }

    fn parse_postfix(&mut self, token: &Token, expr: Expression) -> Result<Expression, String> {
        match token {
            // Index postfix, expr[index]
            Token::LeftBracket => {
                self.must_next(&Token::LeftBracket)?;
                let index = self.parse_expr()?;
                self.must_next(&Token::RightBracket)?;

                Ok(Expression::Index {
                    callee: Box::new(expr),
                    index: Box::new(index),
                })
            }
            // Call postfix, expr(arg1, arg2, ...)
            Token::LeftParen => {
                self.must_next(&Token::LeftParen)?;
                let arguments =
                    self.separated_list(Token::Comma, Self::parse_expr, Token::RightParen)?;
                self.must_next(&Token::RightParen)?;

                Ok(Expression::Call {
                    callee: Box::new(expr),
                    arguments,
                })
            }
            // Postfix, expr++
            Token::PlusPlus => {
                self.must_next(&Token::PlusPlus)?;
                Ok(Expression::Postfix {
                    operator: Operator::Increase,
                    expression: Box::new(expr),
                })
            }
            // Postfix, expr--
            Token::MinusMinus => {
                self.must_next(&Token::PlusPlus)?;
                Ok(Expression::Postfix {
                    operator: Operator::Increase,
                    expression: Box::new(expr),
                })
            }
            _ => Ok(expr),
        }
    }

    fn parse_prefix(&mut self) -> Result<Expression, String> {
        let token = self
            .peek_token()
            .ok_or("Expect token for primary".to_string())?;

        match Operator::prefix_from_token(token) {
            Ok(operator) => {
                let _ = self.next_token().unwrap();
                // let expr = self.parse_sub_expression(Precedence::Prefix)?;
                let expr = self.parse_expr()?;
                Ok(Expression::Prefix {
                    operator,
                    expression: Box::new(expr),
                })
            }
            Err(_) => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<Expression, String> {
        let token = self
            .next_token()
            .ok_or("Expect token for primary".to_string())?;

        match token {
            Token::Literal(literal) => Ok(Expression::Literal { value: literal }),
            Token::Identifier(identifier) => Ok(Expression::Variable { name: identifier }),
            // grouping
            Token::LeftParen => {
                let expr = self.parse_expr()?;

                self.must_next(&Token::RightParen)?;

                Ok(Expression::Grouping {
                    expression: Box::new(expr),
                })
            }
            // array
            Token::LeftBracket => {
                let elements =
                    self.separated_list(Token::Comma, Self::parse_expr, Token::RightBracket)?;

                self.must_next(&Token::RightBracket)?;

                Ok(Expression::Array { elements })
            }
            _ => Err(format!("Unexpected token for primary: {:?}", token)),
        }
    }

    fn separated_list<T>(
        &mut self,
        separator: Token,
        parse_fn: fn(&mut Parser) -> Result<T, String>,
        terminater: Token,
    ) -> Result<Vec<T>, String> {
        let mut list = vec![];

        while let Some(token) = self.peek_token() {
            if token == &terminater {
                return Ok(list);
            }

            list.push(parse_fn(self)?);

            if self.peek_token() == Some(&terminater) {
                return Ok(list);
            }

            self.must_next(&separator)?;
        }

        Err(format!(
            "Expect {:?}, but got {:?}",
            terminater,
            self.peek_token()
        ))
    }

    fn terminated_with<T>(
        &mut self,
        terminater: Token,
        parse_fn: fn(&mut Parser) -> Result<T, String>,
    ) -> Result<Option<T>, String> {
        if self.peek_token() == Some(&terminater) {
            self.must_next(&terminater)?;
            return Ok(None);
        }
        let x = parse_fn(self)?;

        self.must_next(&terminater)?;

        Ok(Some(x))
    }

    fn token_precedence(token: &Token) -> Precedence {
        match token {
            Token::Equal
            | Token::PlusEqual
            | Token::MinusEqual
            | Token::StarEqual
            | Token::SlashEqual
            | Token::PercentEqual => Precedence::Assign,

            Token::AndAnd => Precedence::LogicAnd,
            Token::OrOr => Precedence::LogicOr,
            Token::EqualEqual | Token::BangEqual => Precedence::Equality,
            Token::Greater | Token::GreaterEqual | Token::Less | Token::LessEqual => {
                Precedence::Comparison
            }
            Token::Plus | Token::Minus => Precedence::Term,
            Token::Star | Token::Slash | Token::Percent => Precedence::Factor,
            Token::PlusPlus | Token::MinusMinus => Precedence::Postfix,
            Token::LeftParen => Precedence::Call,
            Token::LeftBracket => Precedence::Index,
            Token::Dot => Precedence::Access,
            Token::Literal(_) | Token::Identifier(_) => Precedence::Primary,

            _ => Precedence::Lowest,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
enum Precedence {
    Lowest = 0,
    Assign,
    LogicOr,
    LogicAnd,
    Equality,
    Comparison,
    Term,
    Factor,
    Prefix,
    Postfix,
    Call,
    Index,
    Access,
    Primary,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_expression() {
        let inputs = vec![
            "1 + 2 * 3 - 4 / 5",
            "(1 + 2) * 3 - 4 / 5",
            "a = 1 + 2 * 3 - 4 / 5",
            "a = b(c,d) + 1",
            "[1,2,3].push(1)",
            "a < b",
            "i = i++",
            "a.c[d]",
        ];

        for input in inputs {
            let tokens = Tokens::new(input).unwrap();
            let mut parser = Parser::new(tokens);

            let expr = parser.parse_expr();

            println!("expr: {expr:?}");
        }
    }

    #[test]
    fn test_parse_prefix() {
        let input = "---1 + 2 * 3";

        let tokens = Tokens::new(input).unwrap();
        let mut parser = Parser::new(tokens);

        let expr = parser.parse_expr();

        println!("expr: {expr:?}");
    }

    #[test]
    fn test_parse_postfix() {
        let input = "a + b[0] * c(d,f,g)";

        let tokens = Tokens::new(input).unwrap();
        let mut parser = Parser::new(tokens);

        let expr = parser.parse_expr();

        println!("expr: {expr:?}");
    }

    #[test]
    fn test_parse_statement() {
        let input = "let a = 1;";

        let tokens = Tokens::new(input).unwrap();
        let mut parser = Parser::new(tokens);

        let statement = parser.parse_statement();

        println!("statement: {statement:?}");
    }

    #[test]
    fn test_parse_block_statement() {
        let input = "for (;;){}";

        let tokens = Tokens::new(input).unwrap();
        let mut parser = Parser::new(tokens);

        let statement = parser.parse_statement();

        println!("statement: {statement:?}");
    }

    #[test]
    fn test_parser() {
        let input = r#"
            ;;;
            return x + y
            let a = 1;
            let b = 2;
            if (a > b) {
                a = a + b;
            } else {
                a = a - b;
            }

            let sum = 0;
            for (i = 0; i < 10; i++) {
                sum += i;
            }

            fn fib(n) {
                if (n <= 0) {
                    return 0;
                } else if (n == 1) {
                    return 1;
                } else {
                    return fib(n - 1) + fib(n - 2);
                }
            }
        "#;

        let tokens = Tokens::new(input).unwrap();
        let mut parser = Parser::new(tokens);

        let program = parser.parse();

        println!("program: {program:?}");
    }
}
