use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use crate::{
    ast::*,
    parser::Parser,
    value::{Callable, NativeFunction, Value},
    Object, ValueRef,
};

/// Eval program.
///
/// # Example
///
/// ```
/// # use liexpr::{eval, Value};
/// assert_eq!(eval("let x = 5; return x + 1;"), Ok(6.into()));
/// ```
pub fn eval(script: &str) -> Result<ValueRef, String> {
    Evaluator::eval_script(script, &mut Context::default())
}

/// Eval expression.
///
/// # Example
///
/// ```
/// # use liexpr::{eval_expr, Value};
/// assert_eq!(eval_expr("1 + 2 * 3 - 4"), Ok(3.into()));
/// ```
pub fn eval_expr(expr: &str) -> Result<ValueRef, String> {
    Evaluator::eval_expression(expr, &mut Context::default())
}

#[derive(Debug)]
pub struct Environment {
    variables: HashMap<String, ValueRef>,
}

impl Environment {
    fn new() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }

    pub fn define(&mut self, name: impl ToString, value: impl Into<ValueRef>) {
        self.variables.insert(name.to_string(), value.into());
    }

    pub fn define_function<Args: 'static>(
        &mut self,
        name: impl ToString,
        callable: impl Callable<Args>,
    ) {
        self.define(
            name.to_string(),
            Value::new_object(NativeFunction::new(
                name,
                Box::new(callable.into_function()),
            )),
        );
    }

    pub fn take(&mut self, name: impl AsRef<str>) -> Option<Value> {
        self.variables.remove(name.as_ref()).map(|v| v.take())
    }
}

impl Default for Environment {
    fn default() -> Self {
        Self::new()
    }
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

#[derive(Debug)]
pub struct Context {
    stack: Vec<StackFrame>,
    environment: Environment,
    functions: HashMap<String, Rc<Function>>,
}

impl Context {
    pub fn new(environment: Environment) -> Self {
        Self {
            stack: vec![StackFrame::new()],
            environment,
            functions: HashMap::new(),
        }
    }

    fn enter_scope(&mut self) {
        self.stack.push(StackFrame::new());
    }

    fn level_scope(&mut self) {
        self.stack.pop();
    }

    fn insert_variable(&mut self, name: &str, value: impl Into<ValueRef>) {
        self.stack
            .last_mut()
            .unwrap()
            .locals
            .insert(name.to_string(), value.into());
    }

    fn get_variable(&self, name: &str) -> Option<ValueRef> {
        if let Some(value) = self.environment.variables.get(name) {
            return Some(value.clone());
        }

        for frame in self.stack.iter().rev() {
            if let Some(value) = frame.locals.get(name) {
                return Some(value.clone());
            }
        }

        self.functions
            .get(name)
            .map(|_| Value::UserFunction(name.to_string()).into())
    }

    fn set_variable(&mut self, name: &str, value: ValueRef) -> Result<(), String> {
        for frame in self.stack.iter_mut().rev() {
            if let Some(old) = frame.locals.get_mut(name) {
                *old = value.clone();
                return Ok(());
            }
        }

        Err(format!("variable `{:?}` not found", name))
    }

    fn get_function(&self, name: &str) -> Option<Rc<Function>> {
        self.functions.get(name).cloned()
    }

    fn set_function(&mut self, name: &str, function: Rc<Function>) {
        self.functions.insert(name.to_string(), function);
    }

    pub fn into_environment(self) -> Environment {
        self.environment
    }
}

impl Default for Context {
    fn default() -> Context {
        Context::new(Environment::new())
    }
}

pub enum ControlFlow {
    Return(ValueRef),
    Break,
    Continue,
}

fn eval_binop(
    ctx: &mut Context,
    operator: &Operator,
    lhs: &Value,
    rhs: &Value,
) -> Result<Value, String> {
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

pub fn eval_expression(ctx: &mut Context, expr: &Expression) -> Result<ValueRef, String> {
    match expr {
        Expression::Literal { value } => match value {
            Literal::Null => Ok(Value::Null.into()),
            Literal::Boolean(b) => Ok(Value::Boolean(*b).into()),
            Literal::Integer(i) => Ok(Value::Integer(*i).into()),
            Literal::Float(f) => Ok(Value::Float(*f).into()),
            Literal::String(s) => Ok(Value::String(s.clone()).into()),
        },
        Expression::Variable { name } => {
            if let Some(value) = ctx.get_variable(name) {
                Ok(value.clone())
            } else {
                Err(format!("Variable not found: {}", name))
            }
        }
        Expression::Array { elements } => eval_array_expression(ctx, elements),
        Expression::Grouping { expression } => eval_expression(ctx, expression),
        Expression::Binary {
            left,
            operator,
            right,
        } => eval_binary_expression(ctx, left, operator, right),
        Expression::Prefix {
            operator,
            expression,
        } => eval_prefix_expression(ctx, operator, expression),
        Expression::Postfix {
            operator,
            expression,
        } => eval_postfix_expression(ctx, operator, expression),
        Expression::Index { callee, index } => eval_index_expression(ctx, callee, index),
        Expression::Call { callee, arguments } => eval_call_expression(ctx, callee, arguments),
    }
}
fn eval_binary_expression(
    ctx: &mut Context,
    left: &Expression,
    operator: &Operator,
    right: &Expression,
) -> Result<ValueRef, String> {
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
            let rhs = eval_expression(ctx, right)?;
            let lhs = eval_expression(ctx, left)?;
            let lhs = lhs.borrow();
            let rhs = rhs.borrow();
            eval_binop(ctx, operator, lhs.deref(), rhs.deref()).map(Into::into)
        }
        Operator::Assign => match left {
            Expression::Variable { name } => {
                let rhs = eval_expression(ctx, right)?;
                ctx.set_variable(name, rhs)?;
                Ok(Value::Null.into())
            }
            Expression::Binary {
                left: object,
                operator,
                right: prop,
            } if operator == &Operator::Access => match prop.deref() {
                Expression::Variable { name } => {
                    let mut lhs = eval_expression(ctx, object)?;
                    let mut lhs = lhs.borrow_mut();
                    let lhs = lhs.deref_mut();
                    let value = eval_expression(ctx, right)?;
                    match lhs {
                        Value::Object(obj) => {
                            obj.property_set(name, value)?;
                            Ok(Value::Null.into())
                        }
                        _ => Err(format!("Invalid assignment: {:?} = {:?}", left, right)),
                    }
                }
                _ => Err(format!("Invalid assignment: {:?} = {:?}", left, right)),
            },
            _ => Err(format!("Invalid assignment: {:?} = {:?}", left, right)),
        },

        Operator::PlusAssign => {
            let mut lhs = eval_expression(ctx, left)?;
            let rhs = eval_expression(ctx, right)?;

            let mut lhs = lhs.borrow_mut();
            let lhs = lhs.deref_mut();

            let ret = eval_binop(ctx, &Operator::Plus, lhs, &rhs.borrow())?;

            *lhs = ret;

            Ok(Value::Null.into())
        }

        Operator::MinusAssign => {
            let mut lhs = eval_expression(ctx, left)?;
            let rhs = eval_expression(ctx, right)?;

            let mut lhs = lhs.borrow_mut();
            let lhs = lhs.deref_mut();

            let ret = eval_binop(ctx, &Operator::Minus, lhs, &rhs.borrow())?;

            *lhs = ret;

            Ok(Value::Null.into())
        }
        Operator::MultiplyAssign => {
            let mut lhs = eval_expression(ctx, left)?;
            let rhs = eval_expression(ctx, right)?;

            let mut lhs = lhs.borrow_mut();
            let lhs = lhs.deref_mut();

            let ret = eval_binop(ctx, &Operator::Multiply, lhs, &rhs.borrow())?;

            *lhs = ret;

            Ok(Value::Null.into())
        }
        Operator::DivideAssign => {
            let mut lhs = eval_expression(ctx, left)?;
            let rhs = eval_expression(ctx, right)?;

            let mut lhs = lhs.borrow_mut();
            let lhs = lhs.deref_mut();

            let ret = eval_binop(ctx, &Operator::Divide, lhs, &rhs.borrow())?;

            *lhs = ret;

            Ok(Value::Null.into())
        }
        Operator::ModuloAssign => {
            let mut lhs = eval_expression(ctx, left)?;
            let rhs = eval_expression(ctx, right)?;

            let mut lhs = lhs.borrow_mut();
            let lhs = lhs.deref_mut();

            let ret = eval_binop(ctx, &Operator::Modulo, lhs, &rhs.borrow())?;

            *lhs = ret;

            Ok(Value::Null.into())
        }
        Operator::Access => {
            let lhs = eval_expression(ctx, left)?;
            match right {
                Expression::Variable { name } => {
                    let lhs = lhs.borrow();
                    let lhs = lhs.deref();
                    match lhs {
                        Value::Object(obj) => {
                            let value = obj.property_get(name)?;
                            Ok(value)
                        }
                        _ => Err(format!("Invalid access: {:?}.{:?}", left, right)),
                    }
                }
                _ => Err(format!("Invalid access: {:?}.{:?}", left, right)),
            }
        }
        _ => Err(format!(
            "Unsupported binary operation: {:?} {} {:?}",
            left, operator, right
        )),
    }
}

fn eval_prefix_expression(
    ctx: &mut Context,
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
    ctx: &mut Context,
    operator: &Operator,
    expression: &Expression,
) -> Result<ValueRef, String> {
    match operator {
        Operator::Increase => {
            let mut val = eval_expression(ctx, expression)?;
            let mut val = val.borrow_mut();
            match val.deref_mut() {
                Value::Integer(i) => {
                    *i += 1;
                    Ok(Value::Null.into())
                }
                _ => Err(format!("Invalid postfix operation: {:?}++", expression)),
            }
        }
        Operator::Decrease => {
            let mut val = eval_expression(ctx, expression)?;
            let mut val = val.borrow_mut();
            match val.deref_mut() {
                Value::Integer(i) => {
                    *i -= 1;
                    Ok(Value::Null.into())
                }
                _ => Err(format!("Invalid postfix operation: {:?}--", expression)),
            }
        }
        _ => Err(format!(
            "Unsupported postfix operation: {} {:?}",
            operator, expression
        )),
    }
}

fn eval_boolean_expression(ctx: &mut Context, expr: &Expression) -> Result<bool, String> {
    let value = eval_expression(ctx, expr)?;
    let value = value.borrow();
    match value.deref() {
        Value::Boolean(b) => Ok(*b),
        v => Err(format!("Expected boolean value, but found {v:?}")),
    }
}

fn eval_array_expression(ctx: &mut Context, elements: &[Expression]) -> Result<ValueRef, String> {
    let mut values = vec![];
    for element in elements {
        values.push(eval_expression(ctx, element)?);
    }
    Ok(Value::Array(values).into())
}

fn eval_index_expression(
    ctx: &mut Context,
    left: &Expression,
    index: &Expression,
) -> Result<ValueRef, String> {
    let value = eval_expression(ctx, left)?;
    let idx = eval_expression(ctx, index)?;

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
    ctx: &mut Context,
    expression: &Expression,
    args: &[Expression],
) -> Result<ValueRef, String> {
    match expression {
        Expression::Variable { name } => {
            match ctx.get_variable(name) {
                Some(valule) => {
                    let value = valule.borrow();
                    match value.deref() {
                        Value::UserFunction(func) => eval_call_function(ctx, func, args),
                        Value::Object(obj) => eval_object_call(ctx, obj.as_ref(), args),

                        _ => Err(format!(
                            "Invalid call operation: {:?}({:?})",
                            expression, args
                        )),
                    }
                }
                None => {
                    // when not variable, it should be a function
                    eval_call_function(ctx, name, args)
                }
            }
        }
        Expression::Binary {
            left,
            operator,
            right,
        } if operator == &Operator::Access => {
            let mut left = eval_expression(ctx, left)?;
            let mut lhs = left.borrow_mut();
            match &**right {
                Expression::Variable { name } => match lhs.deref_mut() {
                    Value::Boolean(value) => eval_object_method_call(ctx, value, name, args),
                    Value::Integer(value) => eval_object_method_call(ctx, value, name, args),
                    Value::Float(value) => eval_object_method_call(ctx, value, name, args),
                    Value::String(value) => eval_object_method_call(ctx, value, name, args),
                    Value::Object(obj) => eval_object_method_call(ctx, obj.as_mut(), name, args),
                    _ => Err(format!(
                        "Invalid call operation: {:?}({:?})",
                        expression, args
                    )),
                },
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

fn eval_call_function(
    ctx: &mut Context,
    name: &str,
    args: &[Expression],
) -> Result<ValueRef, String> {
    let func = ctx
        .get_function(name)
        .ok_or(format!("Function not found: {}", name))?;

    ctx.enter_scope();

    if args.len() != func.parameters.len() {
        return Err(format!(
            "Invalid function call: {:?}({:?}) , args.len() != parameters.len()",
            name, args
        ));
    }

    for (i, arg) in args.iter().enumerate() {
        let arg = eval_expression(ctx, arg)?;
        ctx.insert_variable(&func.parameters[i], arg);
    }

    if let Some(ControlFlow::Return(value)) = eval_block_statement(ctx, &func.body)? {
        ctx.level_scope();
        return Ok(value);
    }

    ctx.level_scope();

    Ok(Value::Null.into())
}

fn eval_object_call(
    ctx: &mut Context,
    object: &dyn Object,
    args: &[Expression],
) -> Result<ValueRef, String> {
    let args: Result<Vec<ValueRef>, String> =
        args.iter().map(|arg| eval_expression(ctx, arg)).collect();
    Object::call(object, &args?).map(|v| v.unwrap_or(Value::new_null().into()))
}

fn eval_object_method_call(
    ctx: &mut Context,
    object: &mut dyn Object,
    method: &str,
    args: &[Expression],
) -> Result<ValueRef, String> {
    let args: Result<Vec<ValueRef>, String> =
        args.iter().map(|arg| eval_expression(ctx, arg)).collect();

    Object::method_call(object, method, &args?).map(|v| v.unwrap_or(Value::new_null().into()))
}

fn eval_statement(ctx: &mut Context, statement: &Statement) -> Result<Option<ControlFlow>, String> {
    // println!("-> stmt: {statement:?}");
    match statement {
        Statement::Empty => {}
        Statement::Return { value } => {
            let ret = match value {
                Some(expr) => eval_expression(ctx, expr)?,
                None => Value::Null.into(),
            };
            return Ok(Some(ControlFlow::Return(ret)));
        }
        Statement::Let { name, value } => match value {
            Some(value) => {
                let value = eval_expression(ctx, value)?;
                ctx.insert_variable(name, value);
            }
            None => {
                let value = Value::Null;
                ctx.insert_variable(name, value);
            }
        },
        Statement::Expression { expression } => {
            eval_expression(ctx, expression)?;
        }
        Statement::Block(block) => {
            eval_block_statement(ctx, block)?;
        }
        Statement::For {
            initializer,
            condition,
            increment,
            body,
        } => {
            ctx.enter_scope();
            eval_statement(ctx, initializer)?;

            loop {
                if let Some(condition) = condition {
                    // when condition is false, finish loop
                    if !eval_boolean_expression(ctx, condition)? {
                        break;
                    }
                }

                if let Some(ctrl) = eval_block_statement(ctx, body)? {
                    match ctrl {
                        ControlFlow::Continue => {}
                        ControlFlow::Break => break,
                        ControlFlow::Return(_) => {
                            ctx.level_scope();
                            return Ok(Some(ctrl));
                        }
                    }
                }

                if let Some(increment) = increment {
                    eval_expression(ctx, increment)?;
                }
            }

            ctx.level_scope();
        }
        Statement::If {
            condition,
            then_branch,
            else_branch,
        } => match eval_boolean_expression(ctx, condition)? {
            true => {
                if let Some(ctrl) = eval_block_statement(ctx, then_branch)? {
                    return Ok(Some(ctrl));
                }
            }
            false => {
                if let Some(else_branch) = else_branch {
                    if let Some(ctrl) = eval_statement(ctx, else_branch)? {
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
    ctx: &mut Context,
    block: &BlockStatement,
) -> Result<Option<ControlFlow>, String> {
    ctx.enter_scope();
    for statement in &block.0 {
        if let Some(ctrl) = eval_statement(ctx, statement)? {
            ctx.level_scope();
            return Ok(Some(ctrl));
        }
    }

    ctx.level_scope();

    Ok(None)
}

pub fn eval_program(ctx: &mut Context, program: &Program) -> Result<ValueRef, String> {
    ctx.enter_scope();

    for (name, function) in &program.functions {
        ctx.set_function(name, function.clone());
    }

    for statement in &program.statements {
        if let Some(ctrl) = eval_statement(ctx, statement)? {
            match ctrl {
                ControlFlow::Return(value) => {
                    ctx.level_scope();
                    return Ok(value);
                }
                ControlFlow::Break => {
                    ctx.level_scope();
                    return Ok(Value::Null.into());
                }
                _ => {}
            }
        }
    }

    ctx.level_scope();

    Ok(Value::Null.into())
}

#[derive(Debug)]
struct Evaluator {}

impl Evaluator {
    pub fn eval_script(script: &str, ctx: &mut Context) -> Result<ValueRef, String> {
        let program = Parser::parse_program(script)?;
        eval_program(ctx, &program)
    }

    pub fn eval_expression(script: &str, ctx: &mut Context) -> Result<ValueRef, String> {
        let expr = Parser::parse_expression(script)?;
        eval_expression(ctx, &expr)
    }
}

#[cfg(test)]
mod tests {

    use std::sync::LazyLock;

    use crate::value::{MetaObject, MetaTable};

    use super::*;

    #[test]
    fn test_eval_integer_expression() {
        let mut ctx = Context::default();
        let expr = Expression::Literal {
            value: Literal::Integer(42),
        };
        let result = eval_expression(&mut ctx, &expr).unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    fn test_eval_boolean_expression() {
        let mut ctx = Context::default();
        let expr = Expression::Literal {
            value: Literal::Boolean(true),
        };
        let result = eval_expression(&mut ctx, &expr).unwrap();
        assert_eq!(result, true);
    }

    #[test]
    fn test_eval_binary_expression() {
        let mut ctx = Context::default();
        let expr = Expression::Binary {
            left: Box::new(Expression::Literal {
                value: Literal::Integer(2),
            }),
            operator: Operator::Plus,
            right: Box::new(Expression::Literal {
                value: Literal::Integer(3),
            }),
        };
        let result = eval_expression(&mut ctx, &expr).unwrap();
        assert_eq!(result, 5);
    }

    #[test]
    fn test_eval_prefix_expression() {
        let mut ctx = Context::default();
        let expr = Expression::Prefix {
            operator: Operator::Minus,
            expression: Box::new(Expression::Literal {
                value: Literal::Integer(5),
            }),
        };
        let result = eval_expression(&mut ctx, &expr).unwrap();
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
    fn test_eval_native_function() {
        fn fib(n: i64) -> i64 {
            if n <= 0 {
                return 0;
            }
            if n <= 2 {
                return 1;
            }
            fib(n - 1) + fib(n - 2)
        }

        let mut env = Environment::new();

        env.define_function("fib", fib);

        let result = Evaluator::eval_expression("fib(10)", &mut Context::new(env));

        assert_eq!(result, Ok(55.into()));
    }

    #[test]
    fn test_eval_string() {
        let script = r#"
            s.push("hello");
            s.push(" world");
            return s;
        "#;

        let s = ValueRef::new(Value::new_string(""));

        let mut env = Environment::new();

        env.define("s", s);

        let result = Evaluator::eval_script(script, &mut Context::new(env)).unwrap();

        assert_eq!(result, "hello world".to_string())
    }

    #[test]
    fn test_eval_object() {
        #[derive(Debug)]
        struct Request {
            headers: HashMap<String, Vec<String>>,
            body: String,
        }

        impl Request {
            fn new() -> Self {
                Request {
                    headers: HashMap::new(),
                    body: "".to_string(),
                }
            }
        }

        impl MetaObject for Request {
            fn meta_table() -> &'static MetaTable<Self> {
                static META: LazyLock<MetaTable<Request>> = LazyLock::new(|| {
                    MetaTable::build()
                        .with_method(
                            "set_header",
                            |this: &mut Request, name: String, value: String| {
                                this.headers.insert(name, vec![value]);
                            },
                        )
                        .with_method(
                            "add_header",
                            |this: &mut Request, name: String, value: String| {
                                this.headers.entry(name).or_default().push(value);
                            },
                        )
                        .with_property(
                            "body",
                            |this: &Request| this.body.clone(),
                            |this: &mut Request, body: String| {
                                this.body = body;
                            },
                        )
                        .fininal()
                });

                &META
            }
        }

        let script = r#"
            request.set_header("Content-Type", "application/json");
            request.add_header("foo", "bar");
            request.add_header("foo", "barbar");
            request.body = "ok";

            return request.body;
        "#;

        let req = ValueRef::with_object(Request::new());

        let mut env = Environment::new();

        env.define("request", req);

        let mut ctx = Context::new(env);

        let result = Evaluator::eval_script(script, &mut ctx).unwrap();

        assert_eq!(result, "ok");

        let req = ctx.into_environment().take("request").unwrap();
        let req = req.into_object::<Request>().unwrap();

        assert_eq!(req.body, "ok");
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
            (r#"return "hello".len();"#, 5),
        ];

        for (script, ret) in inputs {
            let result = eval(script).unwrap();

            assert_eq!(result, ret);
        }
    }
}
