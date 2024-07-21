use std::{
    cell::{Ref, RefCell, RefMut},
    ops::Deref,
    rc::Rc,
};

#[derive(Debug)]
pub enum Value {
    Null,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Array(Vec<ValueRef>),
    UserFunction(String),
    Object(Box<dyn Object>),
}

#[derive(Debug, Clone)]
pub struct ValueRef(Rc<RefCell<Value>>);

impl ValueRef {
    pub fn new(value: Value) -> Self {
        ValueRef(Rc::new(RefCell::new(value)))
    }
    pub fn borrow(&self) -> Ref<Value> {
        self.0.deref().borrow()
    }
    pub fn borrow_mut(&mut self) -> RefMut<Value> {
        self.0.deref().borrow_mut()
    }
}

impl From<Value> for ValueRef {
    fn from(value: Value) -> Self {
        ValueRef(Rc::new(RefCell::new(value)))
    }
}

impl PartialEq<bool> for ValueRef {
    fn eq(&self, other: &bool) -> bool {
        match self.borrow().deref() {
            Value::Boolean(v) => v == other,
            _ => false,
        }
    }
}

impl PartialEq<i64> for ValueRef {
    fn eq(&self, other: &i64) -> bool {
        match self.borrow().deref() {
            Value::Integer(v) => v == other,
            _ => false,
        }
    }
}

impl PartialEq<f64> for ValueRef {
    fn eq(&self, other: &f64) -> bool {
        match self.borrow().deref() {
            Value::Float(v) => v == other,
            _ => false,
        }
    }
}

impl PartialEq<String> for ValueRef {
    fn eq(&self, other: &String) -> bool {
        match self.borrow().deref() {
            Value::String(v) => v == other,
            _ => false,
        }
    }
}

impl PartialEq for ValueRef {
    fn eq(&self, other: &ValueRef) -> bool {
        match (self.borrow().deref(), other.borrow().deref()) {
            (Value::Null, Value::Null) => true,
            (Value::Boolean(l), Value::Boolean(r)) => l == r,
            (Value::Integer(l), Value::Integer(r)) => l == r,
            (Value::Float(l), Value::Float(r)) => l == r,
            (Value::String(l), Value::String(r)) => l == r,
            (Value::Array(l), Value::Array(r)) => {
                for (l, r) in l.iter().zip(r.iter()) {
                    if l != r {
                        return false;
                    }
                }
                true
            }
            _ => false,
        }
    }
}

impl PartialEq<Result<ValueRef, String>> for ValueRef {
    fn eq(&self, other: &Result<ValueRef, String>) -> bool {
        match other {
            Ok(v) => self == other,
            Err(_) => false,
        }
    }
}

pub trait Object: std::any::Any + std::fmt::Debug {
    fn property_get(&self, key: &str) -> Result<Option<ValueRef>, String> {
        Err(format!(
            "Unsupport property_get: {}.{}",
            std::any::type_name::<Self>(),
            key
        ))
    }
    fn property_set(&mut self, key: &str, value: ValueRef) -> Result<(), String> {
        Err(format!(
            "Unsupport property_set: {}.{}",
            std::any::type_name::<Self>(),
            key
        ))
    }
    fn call(&mut self, args: Vec<ValueRef>) -> Result<Option<ValueRef>, String> {
        Err(format!("Unsupport call: {}", std::any::type_name::<Self>()))
    }

    fn method_call(&mut self, name: &str, args: Vec<ValueRef>) -> Result<Option<ValueRef>, String> {
        Err(format!(
            "Unsupport method_call: {}.{}",
            std::any::type_name::<Self>(),
            name
        ))
    }
}
