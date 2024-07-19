use std::collections::HashMap;

use crate::{ast::*, parser::Parser};

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Array(Vec<Value>),
}

/// Eval program.
///
/// # Example
///
/// ```
/// # use liexpr::{eval, Value};
/// assert_eq!(eval("let x = 5; return x + 1;"), Ok(Value::Integer(6)));
/// ```
pub fn eval(expr: &str) -> Result<Value, String> {
    let program = Parser::parse_program(expr)?;

    let mut evaluator = Evaluator::new();

    evaluator.eval(&program)
}

/// Eval expression.
///
/// # Example
///
/// ```
/// # use liexpr::{eval_expr, Value};
/// assert_eq!(eval_expr("1 + 2 * 3 - 4"), Ok(Value::Integer(3)));
/// ```
pub fn eval_expr(expr: &str) -> Result<Value, String> {
    let expression = Parser::parse_expression(expr)?;

    let mut evaluator = Evaluator::new();

    evaluator.eval_expression(&expression)
}

#[derive(Debug)]
struct StackFrame {
    locals: HashMap<String, Value>,
}

impl StackFrame {
    fn new() -> Self {
        Self {
            locals: HashMap::new(),
        }
    }
}

enum ControlFlow {
    Return(Value),
    Break,
    Continue,
}

#[derive(Debug)]
struct Evaluator {
    stack: Vec<StackFrame>,
}

impl Evaluator {
    fn new() -> Self {
        Self {
            stack: vec![StackFrame::new()],
        }
    }

    fn eval(&mut self, program: &Program) -> Result<Value, String> {
        if program.statements.is_empty() {
            return Ok(Value::Null);
        }

        for statement in &program.statements {
            if let Some(ControlFlow::Return(ret)) = self.eval_statement(statement)? {
                return Ok(ret);
            }
        }

        Ok(Value::Null)
    }

    fn eval_statement(&mut self, statement: &Statement) -> Result<Option<ControlFlow>, String> {
        // println!("-> stmt: {statement:?}");
        match statement {
            Statement::Empty => {}
            Statement::Return { value } => {
                let ret = match value {
                    Some(expr) => self.eval_expression(expr)?,
                    None => Value::Null,
                };
                return Ok(Some(ControlFlow::Return(ret)));
            }
            Statement::Let { name, value } => match value {
                Some(value) => {
                    let value = self.eval_expression(value)?;
                    self.insert_variable(name, value);
                }
                None => {
                    let value = Value::Null;
                    self.insert_variable(name, value);
                }
            },
            Statement::Expression { expression } => {
                self.eval_expression(expression)?;
            }
            Statement::Block { statements } => {
                self.stack.push(StackFrame::new());
                for statement in statements {
                    if let Some(ctrl) = self.eval_statement(statement)? {
                        return Ok(Some(ctrl));
                    }
                }

                self.stack.pop();
            }
            Statement::For {
                initializer,
                condition,
                increment,
                body,
            } => {
                self.stack.push(StackFrame::new());
                self.eval_statement(initializer)?;

                loop {
                    if let Some(condition) = condition {
                        // when condition is false, finish loop
                        if !self.eval_boolean_expression(condition)? {
                            break;
                        }
                    }

                    if let Some(ctrl) = self.eval_statement(body)? {
                        match ctrl {
                            ControlFlow::Continue => {}
                            ControlFlow::Break => break,
                            ControlFlow::Return(_) => return Ok(Some(ctrl)),
                        }
                    }

                    if let Some(increment) = increment {
                        self.eval_expression(increment)?;
                    }
                }

                self.stack.pop();
            }
            Statement::If {
                condition,
                then_branch,
                else_branch,
            } => match self.eval_boolean_expression(condition)? {
                true => {
                    if let Some(ctrl) = self.eval_statement(then_branch)? {
                        return Ok(Some(ctrl));
                    }
                }
                false => {
                    if let Some(else_branch) = else_branch {
                        if let Some(ctrl) = self.eval_statement(else_branch)? {
                            return Ok(Some(ctrl));
                        }
                    }
                }
            },
            Statement::Break => return Ok(Some(ControlFlow::Break)),
            Statement::Continue => return Ok(Some(ControlFlow::Continue)),
            _ => unimplemented!("Not implemented statement: {:?}", statement),
        }

        Ok(None)
    }

    fn eval_expression(&mut self, expr: &Expression) -> Result<Value, String> {
        match expr {
            Expression::Literal { value } => match value {
                Literal::Boolean(b) => Ok(Value::Boolean(*b)),
                Literal::Integer(i) => Ok(Value::Integer(*i)),
                Literal::Float(f) => Ok(Value::Float(*f)),
                Literal::String(s) => Ok(Value::String(s.clone())),
            },
            Expression::Variable { name } => {
                if let Some(value) = self.get_variable(name) {
                    Ok(value.clone())
                } else {
                    Err(format!("Variable not found: {}", name))
                }
            }
            Expression::Array { elements } => self.eval_array_expression(elements),
            Expression::Grouping { expression } => self.eval_expression(expression),
            Expression::Binary {
                left,
                operator,
                right,
            } => self.eval_binary_expression(left, operator, right),
            Expression::Prefix {
                operator,
                expression,
            } => self.eval_prefix_expression(operator, expression),
            Expression::Postfix {
                operator,
                expression,
            } => self.eval_postfix_expression(operator, expression),
            Expression::Index { callee, index } => self.eval_index_expression(callee, index),
            Expression::Call { callee, arguments } => self.eval_call_expression(callee, arguments),
            Expression::Empty => Ok(Value::Null),
        }
    }

    fn eval_array_expression(&mut self, elements: &[Expression]) -> Result<Value, String> {
        let mut values = vec![];
        for element in elements {
            values.push(self.eval_expression(element)?);
        }
        Ok(Value::Array(values))
    }

    fn eval_boolean_expression(&mut self, expr: &Expression) -> Result<bool, String> {
        match self.eval_expression(expr)? {
            Value::Boolean(b) => Ok(b),
            v => Err(format!("Expected boolean value, but found {v:?}")),
        }
    }

    fn eval_binary_expression(
        &mut self,
        left: &Expression,
        operator: &Operator,
        right: &Expression,
    ) -> Result<Value, String> {
        let rhs = self.eval_expression(right)?;

        match operator {
            Operator::Plus => {
                let lhs = self.eval_expression(left)?;
                match (lhs, rhs) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l + r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l + r)),
                    (Value::Float(l), Value::Integer(r)) => Ok(Value::Float(l + r as f64)),
                    (Value::Integer(l), Value::Float(r)) => Ok(Value::Float(l as f64 + r)),
                    (Value::String(l), Value::String(r)) => Ok(Value::String(l + &r)),
                    _ => Err(format!(
                        "Invalid binary operation: {:?} {} {:?}",
                        left, operator, right
                    )),
                }
            }
            Operator::Minus => {
                let lhs = self.eval_expression(left)?;
                match (lhs, rhs) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l - r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l - r)),
                    (Value::Float(l), Value::Integer(r)) => Ok(Value::Float(l - r as f64)),
                    (Value::Integer(l), Value::Float(r)) => Ok(Value::Float(l as f64 - r)),
                    _ => Err(format!(
                        "Invalid binary operation: {:?} {} {:?}",
                        left, operator, right
                    )),
                }
            }
            Operator::Multiply => {
                let lhs = self.eval_expression(left)?;
                match (lhs, rhs) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l * r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l * r)),
                    (Value::Float(l), Value::Integer(r)) => Ok(Value::Float(l * r as f64)),
                    (Value::Integer(l), Value::Float(r)) => Ok(Value::Float(l as f64 * r)),
                    _ => Err(format!(
                        "Invalid binary operation: {:?} {} {:?}",
                        left, operator, right
                    )),
                }
            }
            Operator::Divide => {
                let lhs = self.eval_expression(left)?;
                match (lhs, rhs) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l / r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l / r)),
                    (Value::Float(l), Value::Integer(r)) => Ok(Value::Float(l / r as f64)),
                    (Value::Integer(l), Value::Float(r)) => Ok(Value::Float(l as f64 / r)),
                    _ => Err(format!(
                        "Invalid binary operation: {:?} {} {:?}",
                        left, operator, right
                    )),
                }
            }
            Operator::Modulo => {
                let lhs = self.eval_expression(left)?;
                match (lhs, rhs) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l % r)),
                    _ => Err(format!(
                        "Invalid binary operation: {:?} {} {:?}",
                        left, operator, right
                    )),
                }
            }
            Operator::Equal => {
                let lhs = self.eval_expression(left)?;
                match (lhs, rhs) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l == r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l == r)),
                    (Value::String(l), Value::String(r)) => Ok(Value::Boolean(l == r)),
                    _ => Err(format!(
                        "Invalid binary operation: {:?} {} {:?}",
                        left, operator, right
                    )),
                }
            }
            Operator::NotEqual => {
                let lhs = self.eval_expression(left)?;
                match (lhs, rhs) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l != r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l != r)),
                    (Value::String(l), Value::String(r)) => Ok(Value::Boolean(l != r)),
                    _ => Err(format!(
                        "Invalid binary operation: {:?} {} {:?}",
                        left, operator, right
                    )),
                }
            }
            Operator::Greater => {
                let lhs = self.eval_expression(left)?;
                match (lhs, rhs) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l > r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l > r)),
                    _ => Err(format!(
                        "Invalid binary operation: {:?} {} {:?}",
                        left, operator, right
                    )),
                }
            }
            Operator::GreaterEqual => {
                let lhs = self.eval_expression(left)?;
                match (lhs, rhs) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l >= r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l >= r)),
                    _ => Err(format!(
                        "Invalid binary operation: {:?} {} {:?}",
                        left, operator, right
                    )),
                }
            }
            Operator::Less => {
                let lhs = self.eval_expression(left)?;
                match (lhs, rhs) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l < r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l < r)),
                    _ => Err(format!(
                        "Invalid binary operation: {:?} {} {:?}",
                        left, operator, right
                    )),
                }
            }
            Operator::LessEqual => {
                let lhs = self.eval_expression(left)?;
                match (lhs, rhs) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l <= r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l <= r)),
                    _ => Err(format!(
                        "Invalid binary operation: {:?} {} {:?}",
                        left, operator, right
                    )),
                }
            }
            Operator::LogicalAnd => {
                let lhs = self.eval_expression(left)?;
                match (lhs, rhs) {
                    (Value::Boolean(l), Value::Boolean(r)) => Ok(Value::Boolean(l && r)),
                    _ => Err(format!(
                        "Invalid binary operation: {:?} {} {:?}",
                        left, operator, right
                    )),
                }
            }
            Operator::LogicalOr => {
                let lhs = self.eval_expression(left)?;
                match (lhs, rhs) {
                    (Value::Boolean(l), Value::Boolean(r)) => Ok(Value::Boolean(l || r)),
                    _ => Err(format!(
                        "Invalid binary operation: {:?} {} {:?}",
                        left, operator, right
                    )),
                }
            }
            Operator::Assign => match left {
                Expression::Variable { name } => {
                    self.set_variable(name, rhs);
                    Ok(Value::Null)
                }
                _ => Err(format!("Invalid assignment: {:?} = {:?}", left, right)),
            },

            Operator::PlusAssign => match left {
                Expression::Variable { name } => {
                    let value = self.get_variable(name).unwrap();
                    match (value, rhs) {
                        (Value::Integer(l), Value::Integer(r)) => {
                            self.set_variable(name, Value::Integer(l + r));
                            Ok(Value::Null)
                        }
                        (Value::Float(l), Value::Float(r)) => {
                            self.set_variable(name, Value::Float(l + r));
                            Ok(Value::Null)
                        }
                        (Value::Float(l), Value::Integer(r)) => {
                            self.set_variable(name, Value::Float(l + r as f64));
                            Ok(Value::Null)
                        }
                        (Value::String(l), Value::String(r)) => {
                            self.set_variable(name, Value::String(l.to_owned() + &r));
                            Ok(Value::Null)
                        }
                        _ => Err(format!(
                            "Invalid binary operation: {:?} += {:?}",
                            left, right
                        )),
                    }
                }
                _ => Err(format!("Invalid assignment: {:?} += {:?}", left, right)),
            },

            Operator::MinusAssign => match left {
                Expression::Variable { name } => {
                    let value = self.get_variable(name).unwrap();
                    match (value, rhs) {
                        (Value::Integer(l), Value::Integer(r)) => {
                            self.set_variable(name, Value::Integer(l - r));
                            Ok(Value::Null)
                        }
                        (Value::Float(l), Value::Float(r)) => {
                            self.set_variable(name, Value::Float(l - r));
                            Ok(Value::Null)
                        }
                        (Value::Float(l), Value::Integer(r)) => {
                            self.set_variable(name, Value::Float(l - r as f64));
                            Ok(Value::Null)
                        }
                        _ => Err(format!(
                            "Invalid binary operation: {:?} -= {:?}",
                            left, right
                        )),
                    }
                }
                _ => Err(format!("Invalid assignment: {:?} -= {:?}", left, right)),
            },
            Operator::MultiplyAssign => match left {
                Expression::Variable { name } => {
                    let value = self.get_variable(name).unwrap();
                    match (value, rhs) {
                        (Value::Integer(l), Value::Integer(r)) => {
                            self.set_variable(name, Value::Integer(l * r));
                            Ok(Value::Null)
                        }
                        (Value::Float(l), Value::Float(r)) => {
                            self.set_variable(name, Value::Float(l * r));
                            Ok(Value::Null)
                        }
                        (Value::Float(l), Value::Integer(r)) => {
                            self.set_variable(name, Value::Float(l * r as f64));
                            Ok(Value::Null)
                        }
                        _ => Err(format!(
                            "Invalid binary operation: {:?} *= {:?}",
                            left, right
                        )),
                    }
                }
                _ => Err(format!("Invalid assignment: {:?} *= {:?}", left, right)),
            },
            Operator::DivideAssign => match left {
                Expression::Variable { name } => {
                    let value = self.get_variable(name).unwrap();
                    match (value, rhs) {
                        (Value::Integer(l), Value::Integer(r)) => {
                            self.set_variable(name, Value::Integer(l / r));
                            Ok(Value::Null)
                        }
                        (Value::Float(l), Value::Float(r)) => {
                            self.set_variable(name, Value::Float(l / r));
                            Ok(Value::Null)
                        }
                        (Value::Float(l), Value::Integer(r)) => {
                            self.set_variable(name, Value::Float(l / r as f64));
                            Ok(Value::Null)
                        }
                        _ => Err(format!(
                            "Invalid binary operation: {:?} /= {:?}",
                            left, right
                        )),
                    }
                }
                _ => Err(format!("Invalid assignment: {:?} /= {:?}", left, right)),
            },
            Operator::ModuloAssign => match left {
                Expression::Variable { name } => {
                    let value = self.get_variable(name).unwrap();
                    match (value, rhs) {
                        (Value::Integer(l), Value::Integer(r)) => {
                            self.set_variable(name, Value::Integer(l % r));
                            Ok(Value::Null)
                        }
                        _ => Err(format!(
                            "Invalid binary operation: {:?} %= {:?}",
                            left, right
                        )),
                    }
                }
                _ => Err(format!("Invalid assignment: {:?} %= {:?}", left, right)),
            },

            _ => Err(format!(
                "Unsupported binary operation: {:?} {} {:?}",
                left, operator, right
            )),
        }
    }

    fn eval_prefix_expression(
        &mut self,
        operator: &Operator,
        expression: &Expression,
    ) -> Result<Value, String> {
        match operator {
            Operator::Negate => match expression {
                Expression::Literal {
                    value: Literal::Boolean(b),
                } => Ok(Value::Boolean(!b)),

                _ => Err(format!("Invalid prefix operation: !{:?}", expression)),
            },
            Operator::Minus => match expression {
                Expression::Literal { value } => match value {
                    Literal::Integer(i) => Ok(Value::Integer(-i)),
                    Literal::Float(f) => Ok(Value::Float(-f)),
                    _ => Err(format!("Invalid prefix operation: -{:?}", expression)),
                },
                _ => Err(format!("Invalid prefix operation: -{:?}", expression)),
            },
            _ => Err(format!(
                "Unsupported prefix operation: {} {:?}",
                operator, expression
            )),
        }
    }

    fn eval_postfix_expression(
        &mut self,
        operator: &Operator,
        expression: &Expression,
    ) -> Result<Value, String> {
        match operator {
            Operator::Increase => match expression {
                Expression::Variable { name } => {
                    let value = self.get_variable(name).unwrap();
                    match value {
                        Value::Integer(i) => {
                            self.set_variable(name, Value::Integer(i + 1));
                            Ok(Value::Null)
                        }
                        _ => Err(format!("Invalidpostfix operation: {:?}++", expression)),
                    }
                }
                _ => Err(format!("Invalid postfix operation: {:?}++", expression)),
            },
            Operator::Decrease => match expression {
                Expression::Variable { name } => {
                    let value = self.get_variable(name).unwrap();
                    match value {
                        Value::Integer(i) => {
                            self.set_variable(name, Value::Integer(i - 1));
                            Ok(Value::Null)
                        }
                        _ => Err(format!("Invalid postfixoperation: {:?}--", expression)),
                    }
                }
                _ => Err(format!("Invalid postfix operation: {:?}--", expression)),
            },
            _ => Err(format!(
                "Unsupported postfix operation: {} {:?}",
                operator, expression
            )),
        }
    }

    fn eval_index_expression(
        &mut self,
        left: &Expression,
        index: &Expression,
    ) -> Result<Value, String> {
        let value = self.eval_expression(left)?;
        let idx = self.eval_expression(index)?;

        match (value, idx) {
            (Value::Array(array), Value::Integer(i)) => {
                if i < 0 || i >= array.len() as i64 {
                    return Err(format!("Index out of bounds: {:?}[{:?}]", left, index));
                }
                Ok(array[i as usize].clone())
            }
            _ => Err(format!("Invalid index operation: {:?}[{:?}]", left, index)),
        }
    }

    fn eval_call_expression(
        &mut self,
        expression: &Expression,
        args: &[Expression],
    ) -> Result<Value, String> {
        unimplemented!("eval_call_expression not implemented");
    }

    fn set_variable(&mut self, name: &str, value: Value) {
        for frame in self.stack.iter_mut().rev() {
            if frame.locals.contains_key(name) {
                frame.locals.insert(name.to_string(), value);
                return;
            }
        }
    }

    fn insert_variable(&mut self, name: &str, value: Value) {
        self.stack
            .last_mut()
            .unwrap()
            .locals
            .insert(name.to_string(), value);
    }

    fn get_variable(&self, name: &str) -> Option<&Value> {
        for frame in self.stack.iter().rev() {
            if let Some(value) = frame.locals.get(name) {
                return Some(value);
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_eval_integer_expression() {
        let mut evaluator = Evaluator::new();
        let expr = Expression::Literal {
            value: Literal::Integer(42),
        };
        let result = evaluator.eval_expression(&expr).unwrap();
        assert_eq!(result, Value::Integer(42));
    }

    #[test]
    fn test_eval_boolean_expression() {
        let mut evaluator = Evaluator::new();
        let expr = Expression::Literal {
            value: Literal::Boolean(true),
        };
        let result = evaluator.eval_expression(&expr).unwrap();
        assert_eq!(result, Value::Boolean(true));
    }

    #[test]
    fn test_eval_binary_expression() {
        let mut evaluator = Evaluator::new();
        let expr = Expression::Binary {
            left: Box::new(Expression::Literal {
                value: Literal::Integer(2),
            }),
            operator: Operator::Plus,
            right: Box::new(Expression::Literal {
                value: Literal::Integer(3),
            }),
        };
        let result = evaluator.eval_expression(&expr).unwrap();
        assert_eq!(result, Value::Integer(5));
    }

    #[test]
    fn test_eval_prefix_expression() {
        let mut evaluator = Evaluator::new();
        let expr = Expression::Prefix {
            operator: Operator::Minus,
            expression: Box::new(Expression::Literal {
                value: Literal::Integer(5),
            }),
        };
        let result = evaluator.eval_expression(&expr).unwrap();
        assert_eq!(result, Value::Integer(-5));
    }

    #[test]
    fn test_eval_expression() {
        let expr = "1 + 2 * 3 - 4 / 2 + 5";

        let result = eval_expr(expr).unwrap();

        assert_eq!(result, Value::Integer(10));
    }

    #[test]
    fn test_eval() {
        let inputs = vec![
            ("return 1 + 2 * 3 - 4 / 2 + 5;", Value::Integer(10)),
            ("let x = 1; return x;", Value::Integer(1)),
            ("let x = 1; let y = 2; return x + y;", Value::Integer(3)),
            (
                "let sum = 0; for (let i = 0; i < 10; i++) { sum += i; } return sum;",
                Value::Integer(45),
            ),
            (
                "let sum = 0; for (let i = 0; i < 10; i++) { if (i % 2 == 1) { sum += i; } } return sum;",
                Value::Integer(25),
            ),
            (
                "let sum = 0; for (let i = 0; i < 10; i++) { if (i % 2 == 1) { sum += i; } if (i == 5) { break; } } return sum;",
                Value::Integer(9),
            ),
            (
                "let sum = 0; for (let i = 0; i < 10; i++) { if (i % 2 == 0) { continue; } sum += i; } return sum;",
                Value::Integer(25),
            ),
        ];

        for (script, ret) in inputs {
            let result = eval(script);

            assert_eq!(result, Ok(ret));
        }
    }
}
