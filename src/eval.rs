use std::{
    borrow::Borrow,
    collections::HashMap,
    ops::{Deref, DerefMut},
};

use crate::{ast::*, parser::Parser, value::Value, ValueRef};

/// Eval program.
///
/// # Example
///
/// ```
/// # use liexpr::{eval, Value};
/// assert_eq!(eval("let x = 5; return x + 1;"), Ok(Value::Integer(6).into()));
/// ```
pub fn eval(expr: &str) -> Result<ValueRef, String> {
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
/// assert_eq!(eval_expr("1 + 2 * 3 - 4"), Ok(Value::Integer(3).into()));
/// ```
pub fn eval_expr(expr: &str) -> Result<ValueRef, String> {
    let expression = Parser::parse_expression(expr)?;

    let mut evaluator = Evaluator::new();

    evaluator.eval_expression(&expression)
}

#[derive(Debug)]
struct StackFrame {
    locals: HashMap<String, ValueRef>,
}

impl StackFrame {
    fn new() -> Self {
        Self {
            locals: HashMap::new(),
        }
    }
}

enum ControlFlow {
    Return(ValueRef),
    Break,
    Continue,
}

#[derive(Debug)]
struct Evaluator {
    stack: Vec<StackFrame>,
    functions: HashMap<String, Function>,
}

impl Evaluator {
    fn new() -> Self {
        Self {
            stack: vec![StackFrame::new()],
            functions: HashMap::new(),
        }
    }

    fn eval(&mut self, program: &Program) -> Result<ValueRef, String> {
        if program.statements.is_empty() {
            return Ok(Value::Null.into());
        }

        self.functions.clone_from(&program.functions);
        for (name, _func) in &program.functions {
            self.insert_variable(name, Value::UserFunction(name.to_owned()));
        }

        for statement in &program.statements {
            if let Some(ControlFlow::Return(ret)) = self.eval_statement(statement)? {
                return Ok(ret);
            }
        }

        Ok(Value::Null.into())
    }

    fn eval_statement(&mut self, statement: &Statement) -> Result<Option<ControlFlow>, String> {
        // println!("-> stmt: {statement:?}");
        match statement {
            Statement::Empty => {}
            Statement::Return { value } => {
                let ret = match value {
                    Some(expr) => self.eval_expression(expr)?,
                    None => Value::Null.into(),
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
            Statement::Block(block) => {
                self.eval_block_statement(block)?;
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

                    if let Some(ctrl) = self.eval_block_statement(body)? {
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
                    if let Some(ctrl) = self.eval_block_statement(then_branch)? {
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

    fn eval_block_statement(
        &mut self,
        block: &BlockStatement,
    ) -> Result<Option<ControlFlow>, String> {
        self.stack.push(StackFrame::new());
        for statement in &block.0 {
            if let Some(ctrl) = self.eval_statement(statement)? {
                self.stack.pop();
                return Ok(Some(ctrl));
            }
        }

        self.stack.pop();

        Ok(None)
    }

    fn eval_expression(&mut self, expr: &Expression) -> Result<ValueRef, String> {
        match expr {
            Expression::Literal { value } => match value {
                Literal::Boolean(b) => Ok(Value::Boolean(*b).into()),
                Literal::Integer(i) => Ok(Value::Integer(*i).into()),
                Literal::Float(f) => Ok(Value::Float(*f).into()),
                Literal::String(s) => Ok(Value::String(s.clone()).into()),
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
            Expression::Empty => Ok(Value::Null.into()),
        }
    }

    fn eval_array_expression(&mut self, elements: &[Expression]) -> Result<ValueRef, String> {
        let mut values = vec![];
        for element in elements {
            values.push(self.eval_expression(element)?);
        }
        Ok(Value::Array(values).into())
    }

    fn eval_boolean_expression(&mut self, expr: &Expression) -> Result<bool, String> {
        let value = self.eval_expression(expr)?;
        let value = value.borrow();
        match value.deref() {
            Value::Boolean(b) => Ok(*b),
            v => Err(format!("Expected boolean value, but found {v:?}")),
        }
    }

    fn eval_binary_expression(
        &mut self,
        left: &Expression,
        operator: &Operator,
        right: &Expression,
    ) -> Result<ValueRef, String> {
        let rhs = self.eval_expression(right)?;

        match operator {
            Operator::Plus
            | Operator::Minus
            | Operator::Multiply
            | Operator::Divide
            | Operator::Modulo
            | Operator::Equal
            | Operator::NotEqual
            | Operator::Greater
            | Operator::GreaterEqual
            | Operator::Less
            | Operator::LessEqual
            | Operator::LogicalAnd
            | Operator::LogicalOr => {
                let lhs = self.eval_expression(left)?;
                let lhs = lhs.borrow();
                let rhs = rhs.borrow();
                self.binop(operator, lhs.deref(), rhs.deref())
                    .map(Into::into)
            }
            Operator::Assign => match left {
                Expression::Variable { name } => {
                    self.set_variable(name, rhs);
                    Ok(Value::Null.into())
                }
                _ => Err(format!("Invalid assignment: {:?} = {:?}", left, right)),
            },

            Operator::PlusAssign => match left {
                Expression::Variable { name } => {
                    let value = self.get_variable(name).unwrap();
                    let lhs = value.borrow();
                    let rhs = rhs.borrow();

                    let value = self.binop(&Operator::Plus, lhs.deref(), rhs.deref())?;

                    self.set_variable(name, value.into());
                    Ok(Value::Null.into())
                }
                _ => Err(format!("Invalid assignment: {:?} += {:?}", left, right)),
            },

            Operator::MinusAssign => match left {
                Expression::Variable { name } => {
                    let value = self.get_variable(name).unwrap();
                    let lhs = value.borrow();
                    let rhs = rhs.borrow();

                    let value = self.binop(&Operator::Minus, lhs.deref(), rhs.deref())?;

                    self.set_variable(name, value.into());
                    Ok(Value::Null.into())
                }
                _ => Err(format!("Invalid assignment: {:?} -= {:?}", left, right)),
            },
            Operator::MultiplyAssign => match left {
                Expression::Variable { name } => {
                    let value = self.get_variable(name).unwrap();
                    let lhs = value.borrow();
                    let rhs = rhs.borrow();

                    let value = self.binop(&Operator::Multiply, lhs.deref(), rhs.deref())?;

                    self.set_variable(name, value.into());
                    Ok(Value::Null.into())
                }
                _ => Err(format!("Invalid assignment: {:?} *= {:?}", left, right)),
            },
            Operator::DivideAssign => match left {
                Expression::Variable { name } => {
                    let value = self.get_variable(name).unwrap();
                    let lhs = value.borrow();
                    let rhs = rhs.borrow();

                    let value = self.binop(&Operator::Divide, lhs.deref(), rhs.deref())?;

                    self.set_variable(name, value.into());
                    Ok(Value::Null.into())
                }
                _ => Err(format!("Invalid assignment: {:?} /= {:?}", left, right)),
            },
            Operator::ModuloAssign => match left {
                Expression::Variable { name } => {
                    let value = self.get_variable(name).unwrap();
                    let lhs = value.borrow();
                    let rhs = rhs.borrow();

                    let value = self.binop(&Operator::Modulo, lhs.deref(), rhs.deref())?;

                    self.set_variable(name, value.into());
                    Ok(Value::Null.into())
                }
                _ => Err(format!("Invalid assignment: {:?} %= {:?}", left, right)),
            },

            _ => Err(format!(
                "Unsupported binary operation: {:?} {} {:?}",
                left, operator, right
            )),
        }
    }

    fn binop(&mut self, operator: &Operator, lhs: &Value, rhs: &Value) -> Result<Value, String> {
        match operator {
            Operator::Plus => match (lhs, rhs) {
                (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l + r)),
                (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l + r)),
                (Value::Float(l), Value::Integer(r)) => Ok(Value::Float(l + *r as f64)),
                (Value::Integer(l), Value::Float(r)) => Ok(Value::Float(*l as f64 + r)),
                (Value::String(l), Value::String(r)) => Ok(Value::String(l.clone() + r)),
                _ => Err(format!(
                    "Invalid binary operation: {:?} {} {:?}",
                    lhs, operator, rhs
                )),
            },
            Operator::Minus => match (lhs, rhs) {
                (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l - r)),
                (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l - r)),
                (Value::Float(l), Value::Integer(r)) => Ok(Value::Float(l - *r as f64)),
                (Value::Integer(l), Value::Float(r)) => Ok(Value::Float(*l as f64 - r)),
                _ => Err(format!(
                    "Invalid binary operation: {:?} {} {:?}",
                    lhs, operator, rhs
                )),
            },
            Operator::Multiply => match (lhs, rhs) {
                (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l * r)),
                (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l * r)),
                (Value::Float(l), Value::Integer(r)) => Ok(Value::Float(l * *r as f64)),
                (Value::Integer(l), Value::Float(r)) => Ok(Value::Float(*l as f64 * r)),
                _ => Err(format!(
                    "Invalid binary operation: {:?} {} {:?}",
                    lhs, operator, rhs
                )),
            },
            Operator::Divide => match (lhs, rhs) {
                (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l / r)),
                (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l / r)),
                (Value::Float(l), Value::Integer(r)) => Ok(Value::Float(l / *r as f64)),
                (Value::Integer(l), Value::Float(r)) => Ok(Value::Float(*l as f64 / r)),
                _ => Err(format!(
                    "Invalid binary operation: {:?} {} {:?}",
                    lhs, operator, rhs
                )),
            },
            Operator::Modulo => match (lhs, rhs) {
                (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l % r)),
                _ => Err(format!(
                    "Invalid binary operation: {:?} {} {:?}",
                    lhs, operator, rhs
                )),
            },
            Operator::Equal => match (lhs, rhs) {
                (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l == r)),
                (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l == r)),
                (Value::String(l), Value::String(r)) => Ok(Value::Boolean(l == r)),
                _ => Err(format!(
                    "Invalid binary operation: {:?} {} {:?}",
                    lhs, operator, rhs
                )),
            },
            Operator::NotEqual => match (lhs, rhs) {
                (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l != r)),
                (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l != r)),
                (Value::String(l), Value::String(r)) => Ok(Value::Boolean(l != r)),
                _ => Err(format!(
                    "Invalid binary operation: {:?} {} {:?}",
                    lhs, operator, rhs
                )),
            },
            Operator::Greater => match (lhs, rhs) {
                (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l > r)),
                (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l > r)),
                _ => Err(format!(
                    "Invalid binary operation: {:?} {} {:?}",
                    lhs, operator, rhs
                )),
            },
            Operator::GreaterEqual => match (lhs, rhs) {
                (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l >= r)),
                (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l >= r)),
                _ => Err(format!(
                    "Invalid binary operation: {:?} {} {:?}",
                    lhs, operator, rhs
                )),
            },
            Operator::Less => match (lhs, rhs) {
                (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l < r)),
                (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l < r)),
                _ => Err(format!(
                    "Invalid binary operation: {:?} {} {:?}",
                    lhs, operator, rhs
                )),
            },
            Operator::LessEqual => match (lhs, rhs) {
                (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l <= r)),
                (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l <= r)),
                _ => Err(format!(
                    "Invalid binary operation: {:?} {} {:?}",
                    lhs, operator, rhs
                )),
            },
            Operator::LogicalAnd => match (lhs, rhs) {
                (Value::Boolean(l), Value::Boolean(r)) => Ok(Value::Boolean(*l && *r)),
                _ => Err(format!(
                    "Invalid binary operation: {:?} {} {:?}",
                    lhs, operator, rhs
                )),
            },
            Operator::LogicalOr => match (lhs, rhs) {
                (Value::Boolean(l), Value::Boolean(r)) => Ok(Value::Boolean(*l || *r)),
                _ => Err(format!(
                    "Invalid binary operation: {:?} {} {:?}",
                    lhs, operator, rhs
                )),
            },

            _ => Err(format!(
                "Unsupported binary operation: {:?} {} {:?}",
                lhs, operator, rhs
            )),
        }
    }

    fn eval_prefix_expression(
        &mut self,
        operator: &Operator,
        expression: &Expression,
    ) -> Result<ValueRef, String> {
        match operator {
            Operator::Negate => match expression {
                Expression::Literal {
                    value: Literal::Boolean(b),
                } => Ok(Value::Boolean(!b).into()),

                _ => Err(format!("Invalid prefix operation: !{:?}", expression)),
            },
            Operator::Minus => match expression {
                Expression::Literal { value } => match value {
                    Literal::Integer(i) => Ok(Value::Integer(-i).into()),
                    Literal::Float(f) => Ok(Value::Float(-f).into()),
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
    ) -> Result<ValueRef, String> {
        match operator {
            Operator::Increase => match expression {
                Expression::Variable { name } => {
                    let mut value = self.get_variable(name).unwrap();
                    let mut value = value.borrow_mut();
                    match value.deref_mut() {
                        Value::Integer(i) => {
                            *i += 1;
                            Ok(Value::Null.into())
                        }
                        _ => Err(format!("Invalidpostfix operation: {:?}++", expression)),
                    }
                }
                _ => Err(format!("Invalid postfix operation: {:?}++", expression)),
            },
            Operator::Decrease => match expression {
                Expression::Variable { name } => {
                    let mut value = self.get_variable(name).unwrap();
                    let mut value = value.borrow_mut();
                    match value.deref_mut() {
                        Value::Integer(i) => {
                            *i -= 1;
                            Ok(Value::Null.into())
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
    ) -> Result<ValueRef, String> {
        let value = self.eval_expression(left)?;
        let idx = self.eval_expression(index)?;

        let array = value.borrow();
        let idx = idx.borrow();

        match (array.deref(), idx.deref()) {
            (Value::Array(array), Value::Integer(i)) => {
                if *i < 0 || *i >= array.len() as i64 {
                    return Err(format!("Index out of bounds: {:?}[{:?}]", left, index));
                }
                Ok(array[*i as usize].clone())
            }
            _ => Err(format!("Invalid index operation: {:?}[{:?}]", left, index)),
        }
    }

    fn eval_call_expression(
        &mut self,
        expression: &Expression,
        args: &[Expression],
    ) -> Result<ValueRef, String> {
        match expression {
            Expression::Variable { name } => {
                match self.get_variable(name) {
                    Some(valule) => {
                        let value = valule.borrow();
                        match value.deref() {
                            Value::UserFunction(func) => self.eval_call_function(func, args),
                            _ => Err(format!(
                                "Invalid call operation: {:?}({:?})",
                                expression, args
                            )),
                        }
                    }
                    None => {
                        // when not variable, it should be a function
                        self.eval_call_function(name, args)
                    }
                    _ => Err(format!(
                        "Invalid call operation: {:?}({:?})",
                        expression, args
                    )),
                }
            }
            _ => Err(format!(
                "Invalid call operation: {:?}({:?})",
                expression, args
            )),
        }
    }

    fn eval_call_function(&mut self, name: &str, args: &[Expression]) -> Result<ValueRef, String> {
        let func = self
            .functions
            .get(name)
            .cloned()
            .ok_or(format!("Function not found: {}", name))?;

        self.stack.push(StackFrame::new());

        if args.len() != func.parameters.len() {
            return Err(format!(
                "Invalid function call: {:?}({:?}) , args.len() != parameters.len()",
                name, args
            ));
        }

        for (i, arg) in args.iter().enumerate() {
            let arg = self.eval_expression(arg)?;
            self.insert_variable(&func.parameters[i], arg);
        }

        if let Some(ControlFlow::Return(value)) = self.eval_block_statement(&func.body)? {
            self.stack.pop();
            return Ok(value);
        }

        self.stack.pop();

        Ok(Value::Null.into())
    }

    fn set_variable(&mut self, name: &str, value: ValueRef) {
        for frame in self.stack.iter_mut().rev() {
            if frame.locals.contains_key(name) {
                frame.locals.insert(name.to_string(), value);
                return;
            }
        }
    }

    fn insert_variable(&mut self, name: &str, value: impl Into<ValueRef>) {
        self.stack
            .last_mut()
            .unwrap()
            .locals
            .insert(name.to_string(), value.into());
    }

    fn get_variable(&self, name: &str) -> Option<ValueRef> {
        for frame in self.stack.iter().rev() {
            if let Some(value) = frame.locals.get(name) {
                return Some(value.clone());
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
        assert_eq!(result, 42);
    }

    #[test]
    fn test_eval_boolean_expression() {
        let mut evaluator = Evaluator::new();
        let expr = Expression::Literal {
            value: Literal::Boolean(true),
        };
        let result = evaluator.eval_expression(&expr).unwrap();
        assert_eq!(result, true);
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
        assert_eq!(result, 5);
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
        assert_eq!(result, -5);
    }

    #[test]
    fn test_eval_expression() {
        let expr = "1 + 2 * 3 - 4 / 2 + 5";

        let result = eval_expr(expr).unwrap();

        assert_eq!(result, 10);
    }

    #[test]
    fn test_eval_function() {
        let script = "
            fn sum(a, b) {
                return a + b;
            }
            return sum(1, 2);
        ";

        let result = eval(script).unwrap();

        assert_eq!(result, 3);
    }

    #[test]
    fn test_eval_fib_function() {
        let script = "
            fn fib(n) {
                if (n <= 0) {
                    return 0;
                }
                if (n <= 2) {
                    return 1;
                }
                return fib(n - 1) + fib(n - 2);
            }

            let f = fib;

            return f(10);
        ";

        let result = eval(script).unwrap();

        assert_eq!(result, 55);
    }

    #[test]
    fn test_eval_for_loop() {
        let script = "
            let sum = 0;
            for (let i = 0; i < 10; i++) {
                sum += i;
            }
            return sum;
        ";

        let result = eval(script).unwrap();

        assert_eq!(result, 45);
    }

    #[test]
    fn test_eval() {
        let inputs = vec![
            ("return 1 + 2 * 3 - 4 / 2 + 5;", 10),
            ("let x = 1; return x;", 1),
            ("let x = 1; let y = 2; return x + y;", 3),
            (
                "let sum = 0; for (let i = 0; i < 10; i++) { sum += i; } return sum;",
                45,
            ),
            (
                "let sum = 0; for (let i = 0; i < 10; i++) { if (i % 2 == 1) { sum += i; } } return sum;",
                25,
            ),
            (
                "let sum = 0; for (let i = 0; i < 10; i++) { if (i % 2 == 1) { sum += i; } if (i == 5) { break; } } return sum;",
                9,
            ),
            (
                "let sum = 0; for (let i = 0; i < 10; i++) { if (i % 2 == 0) { continue; } sum += i; } return sum;",
                25,
            ),
        ];

        for (script, ret) in inputs {
            let result = eval(script).unwrap();

            assert_eq!(result, ret);
        }
    }
}
