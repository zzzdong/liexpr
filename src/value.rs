use std::{
    any::type_name,
    cell::{Ref, RefCell, RefMut},
    collections::HashMap,
    fmt,
    marker::PhantomData,
    ops::Deref,
    rc::Rc,
    sync::OnceLock,
};

#[derive(Debug, Default)]
pub enum Value {
    #[default]
    Null,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Array(Vec<ValueRef>),
    UserFunction(String),
    Object(Box<dyn Object>),
}

impl Value {
    pub fn new_null() -> Self {
        Value::Null
    }

    pub fn new_boolean(value: impl Into<bool>) -> Self {
        Value::Boolean(value.into())
    }

    pub fn new_integer(value: impl Into<i64>) -> Self {
        Value::Integer(value.into())
    }

    pub fn new_float(value: impl Into<f64>) -> Self {
        Value::Float(value.into())
    }

    pub fn new_string(value: impl ToString) -> Self {
        Value::String(value.to_string())
    }

    pub fn new_object(object: impl Object) -> Self {
        Value::Object(Box::new(object))
    }

    pub fn as_object(self) -> Option<Box<dyn Object>> {
        match self {
            Value::Object(object) => Some(object),
            _ => None,
        }
    }
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Value::new_boolean(value)
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Value::new_integer(value)
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Value::new_float(value)
    }
}

impl From<String> for Value {
    fn from(value: String) -> Self {
        Value::new_string(value)
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Null, Value::Null) => true,
            (Value::Boolean(l), Value::Boolean(r)) => l == r,
            (Value::Integer(l), Value::Integer(r)) => l == r,
            (Value::Float(l), Value::Float(r)) => l == r,
            (Value::String(l), Value::String(r)) => l == r,
            (Value::Array(l), Value::Array(r)) => {
                if l.len() != r.len() {
                    return false;
                }
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

#[derive(Debug, Clone)]
pub struct ValueRef(Rc<RefCell<Value>>);

impl ValueRef {
    pub fn new(value: Value) -> Self {
        ValueRef(Rc::new(RefCell::new(value)))
    }

    pub fn with_object(object: Box<dyn Object>) -> Self {
        ValueRef(Rc::new(RefCell::new(Value::Object(object))))
    }

    pub fn borrow(&self) -> Ref<'_, Value> {
        self.0.deref().borrow()
    }

    pub fn borrow_mut(&mut self) -> RefMut<'_, Value> {
        self.0.deref().borrow_mut()
    }

    pub fn try_downcast_ref<T: 'static>(&self) -> Result<Ref<T>, String> {
        let value = self.0.deref().borrow();
        Ref::filter_map(value, |value| match value {
            Value::Object(object) => (object as &dyn std::any::Any).downcast_ref::<T>(),
            _ => None,
        })
        .map_err(|_err| {
            format!(
                "downcart_ref::<{}> failed for {:?}",
                std::any::type_name::<T>(),
                self
            )
        })
    }

    pub fn try_downcast_mut<T: 'static>(&self) -> Result<RefMut<T>, String> {
        let value = self.0.deref().borrow_mut();
        RefMut::filter_map(value, |value| match value {
            Value::Object(object) => (object as &mut dyn std::any::Any).downcast_mut::<T>(),
            _ => None,
        })
        .map_err(|_err| {
            format!(
                "downcart_ref::<{}> failed for {:?}",
                std::any::type_name::<T>(),
                self
            )
        })
    }

    pub fn as_null(&self) -> Result<(), String> {
        match self.borrow().deref() {
            Value::Null => Ok(()),
            _ => Err(format!("{:?} is not a null", self)),
        }
    }

    pub fn as_boolean(&self) -> Result<bool, String> {
        match self.borrow().deref() {
            Value::Boolean(b) => Ok(*b),
            _ => Err(format!("{:?} is not a boolean", self)),
        }
    }

    pub fn as_integer(&self) -> Result<i64, String> {
        match self.borrow().deref() {
            Value::Integer(i) => Ok(*i),
            _ => Err(format!("{:?} is not a integer", self)),
        }
    }

    pub fn as_float(&self) -> Result<f64, String> {
        match self.borrow().deref() {
            Value::Float(f) => Ok(*f),
            _ => Err(format!("{:?} is not a float", self)),
        }
    }

    pub fn as_string(&self) -> Result<String, String> {
        match self.borrow().deref() {
            Value::String(s) => Ok(s.clone()),
            _ => Err(format!("{:?} is not a string", self)),
        }
    }

    pub fn take(self) -> Value {
        self.0.deref().take()
    }
}

impl<T> From<T> for ValueRef
where
    T: Into<Value>,
{
    fn from(value: T) -> Self {
        ValueRef::new(value.into())
    }
}

impl<T> PartialEq<T> for ValueRef
where
    Value: PartialEq<T>,
{
    fn eq(&self, other: &T) -> bool {
        self.borrow().deref() == other
    }
}

impl PartialEq for ValueRef {
    fn eq(&self, other: &ValueRef) -> bool {
        self.borrow().deref() == other.borrow().deref()
    }
}

pub trait Object: std::any::Any + std::fmt::Debug {
    fn property_get(&self, key: &str) -> Result<ValueRef, String> {
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
    fn call(&self, args: &[ValueRef]) -> Result<Option<ValueRef>, String> {
        Err(format!("Unsupport call: {}", std::any::type_name::<Self>()))
    }

    fn method_call(&mut self, name: &str, args: &[ValueRef]) -> Result<Option<ValueRef>, String> {
        Err(format!(
            "Unsupport method_call: {}.{}",
            std::any::type_name::<Self>(),
            name
        ))
    }
}

// #[derive(Debug, Clone, Copy)]
// pub struct Null;

// impl Object for Null {}

// impl Object for bool {}

// impl Object for i64 {}

// impl Object for f64 {}

// pub struct MetaTable {
//     name: String,
//     properties: BTreeMap<String, ValueRef>,
//     methods: BTreeMap<String, ValueRef>,
// }

// trait MetaObject {
//     fn meta_table(&self) -> &MetaTable;
// }

// impl<T: std::any::Any + std::fmt::Debug + MetaObject> Object for T {
//     fn property_get(&self, key: &str) -> Result<Option<ValueRef>, String> {
//         let meta_table = self.meta_table();
//         for (name, value) in meta_table.properties.iter() {
//             if name == key {
//                 return Ok(Some(value.clone()));
//             }
//         }
//         Err(format!(
//             "Unsupport property_get: {}.{}",
//             std::any::type_name::<Self>(),
//             key
//         ))
//     }

//     fn method_call(&mut self, name: &str, args: Vec<ValueRef>) -> Result<Option<ValueRef>, String> {
//         let meta_table = self.meta_table();
//         for (method_name, method) in meta_table.methods.iter() {
//             if method_name == name {
//                 return method.borrow_mut().call(args);
//             }
//         }
//         Err(format!(
//             "Unsupport method_call: {}.{}",
//             std::any::type_name::<Self>(),
//             name
//         ))
//     }
// }

pub struct NativeFunction {
    pub name: String,
    pub func: Box<dyn Function + Send + Sync>,
}

impl NativeFunction {
    pub fn new(name: impl ToString, func: Box<dyn Function + Send + Sync>) -> Self {
        Self {
            name: name.to_string(),
            func,
        }
    }
}

impl fmt::Debug for NativeFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<NativeFunction`{}`>", self.name)
    }
}

impl Object for NativeFunction {
    fn call(&self, args: &[ValueRef]) -> Result<Option<ValueRef>, String> {
        (self.func).call(args)
    }
}

/// Function trait for external functions.
pub trait Function: Send + 'static {
    fn call(&self, args: &[ValueRef]) -> Result<Option<ValueRef>, String>;
}

pub struct IntoFunction<F, Args> {
    func: F,
    _marker: PhantomData<fn(Args) -> ()>,
}

impl<F, Args> Function for IntoFunction<F, Args>
where
    F: Callable<Args> + Clone,
    Args: 'static,
{
    fn call(&self, args: &[ValueRef]) -> Result<Option<ValueRef>, String> {
        self.func.call(args)
    }
}

pub trait IntoRet {
    fn into_ret(self) -> Result<Option<ValueRef>, String>;
}

impl<T> IntoRet for T
where
    T: Into<ValueRef>,
{
    fn into_ret(self) -> Result<Option<ValueRef>, String> {
        Ok(Some(self.into()))
    }
}

impl<T> IntoRet for Result<T, String>
where
    T: Into<ValueRef>,
{
    fn into_ret(self) -> Result<Option<ValueRef>, String> {
        self.map(|v| Some(v.into()))
    }
}

impl<T> IntoRet for Result<Option<T>, String>
where
    T: Into<ValueRef>,
{
    fn into_ret(self) -> Result<Option<ValueRef>, String> {
        self.map(|v| v.map(Into::into))
    }
}

pub trait FromValue: Sized {
    fn from_value(value: &ValueRef) -> Result<Self, String>;
}

impl FromValue for ValueRef {
    fn from_value(value: &ValueRef) -> Result<ValueRef, String> {
        Ok(value.clone())
    }
}

impl FromValue for bool {
    fn from_value(value: &ValueRef) -> Result<bool, String> {
        match value.borrow().deref() {
            Value::Boolean(b) => Ok(*b),
            _ => Err(format!("Can not convert {:?} to bool", value)),
        }
    }
}

impl FromValue for i64 {
    fn from_value(value: &ValueRef) -> Result<i64, String> {
        match value.borrow().deref() {
            Value::Integer(i) => Ok(*i),
            _ => Err(format!("Can not convert {:?} to i64", value)),
        }
    }
}

impl FromValue for f64 {
    fn from_value(value: &ValueRef) -> Result<f64, String> {
        match value.borrow().deref() {
            Value::Float(f) => Ok(*f),
            _ => Err(format!("Can not convert {:?} to f64", value)),
        }
    }
}

impl FromValue for String {
    fn from_value(value: &ValueRef) -> Result<String, String> {
        match value.borrow().deref() {
            Value::String(s) => Ok(s.clone()),
            _ => Err(format!("Can not convert {:?} to String", value)),
        }
    }
}

pub trait Callable<Args>: Clone + Send + Sync + Sized + 'static {
    fn call(&self, args: &[ValueRef]) -> Result<Option<ValueRef>, String>;

    fn into_function(self) -> IntoFunction<Self, Args> {
        IntoFunction {
            func: self,
            _marker: PhantomData,
        }
    }
}

impl<F, Ret> Callable<&[ValueRef]> for F
where
    F: Fn(&[ValueRef]) -> Ret + Clone + Send + Sync + 'static,
    Ret: IntoRet,
{
    fn call(&self, args: &[ValueRef]) -> Result<Option<ValueRef>, String> {
        (self)(args).into_ret()
    }
}

impl<F, Ret> Callable<()> for F
where
    F: Fn() -> Ret + Clone + Send + Sync + 'static,
    Ret: IntoRet,
{
    fn call(&self, args: &[ValueRef]) -> Result<Option<ValueRef>, String> {
        self().into_ret()
    }
}

macro_rules! impl_callable {
    ($($idx: expr => $arg: ident),+) => {
        #[allow(non_snake_case)]
        impl<F, Ret, $($arg,)*> Callable<($($arg,)*)> for F
        where
            F: Fn($($arg,)*) -> Ret + Clone + Send + Sync + 'static,
            Ret: IntoRet,
            $( $arg: FromValue + 'static, )*
        {
            fn call(&self, args: &[ValueRef]) -> Result<Option<ValueRef>, String> {
                $(
                    let $arg = <$arg>::from_value(args.get($idx).ok_or(format!("Invalid argument {}", $idx))?)?;
                )*
                (self)($($arg,)*).into_ret()
            }
        }
    }
}

impl_callable!(0=>T0);
impl_callable!(0=>T0, 1=>T1);
impl_callable!(0=>T0, 1=>T1, 2=>T2);
impl_callable!(0=>T0, 1=>T1, 2=>T2, 3=>T3);
impl_callable!(0=>T0, 1=>T1, 2=>T2, 3=>T3, 4=>T4);
impl_callable!(0=>T0, 1=>T1, 2=>T2, 3=>T3, 4=>T4, 5=>T5);
impl_callable!(0=>T0, 1=>T1, 2=>T2, 3=>T3, 4=>T4, 5=>T5, 6=>T6);
impl_callable!(0=>T0, 1=>T1, 2=>T2, 3=>T3, 4=>T4, 5=>T5, 6=>T6, 7=>T7);
impl_callable!(0=>T0, 1=>T1, 2=>T2, 3=>T3, 4=>T4, 5=>T5, 6=>T6, 7=>T7, 8=>T8);
impl_callable!(0=>T0, 1=>T1, 2=>T2, 3=>T3, 4=>T4, 5=>T5, 6=>T6, 7=>T7, 8=>T8, 9=>T9);
impl_callable!(0=>T0, 1=>T1, 2=>T2, 3=>T3, 4=>T4, 5=>T5, 6=>T6, 7=>T7, 8=>T8, 9=>T9, 10=>T10);
impl_callable!(0=>T0, 1=>T1, 2=>T2, 3=>T3, 4=>T4, 5=>T5, 6=>T6, 7=>T7, 8=>T8, 9=>T9, 10=>T10, 11=>T11);
impl_callable!(0=>T0, 1=>T1, 2=>T2, 3=>T3, 4=>T4, 5=>T5, 6=>T6, 7=>T7, 8=>T8, 9=>T9, 10=>T10, 11=>T11, 12=>T12);
impl_callable!(0=>T0, 1=>T1, 2=>T2, 3=>T3, 4=>T4, 5=>T5, 6=>T6, 7=>T7, 8=>T8, 9=>T9, 10=>T10, 11=>T11, 12=>T12, 13=>T13);
impl_callable!(0=>T0, 1=>T1, 2=>T2, 3=>T3, 4=>T4, 5=>T5, 6=>T6, 7=>T7, 8=>T8, 9=>T9, 10=>T10, 11=>T11, 12=>T12, 13=>T13, 14=>T14);
impl_callable!(0=>T0, 1=>T1, 2=>T2, 3=>T3, 4=>T4, 5=>T5, 6=>T6, 7=>T7, 8=>T8, 9=>T9, 10=>T10, 11=>T11, 12=>T12, 13=>T13, 14=>T14, 15=>T15);

/* use this when [feature(macro_metavar_expr)](https://github.com/rust-lang/rust/pull/122808) is available
macro_rules! impl_callable_tuple {
    ($($arg: ident),+) => {
        #[allow(non_snake_case)]
        impl<F, Ret, $($arg,)*> Callable<($($arg,)*)> for F
        where
            F: Fn($($arg,)*) -> Ret + Clone + Send + 'static,
            Ret: IntoRet,
            $( $arg: FromValue, )*
        {
            fn call(&self, args: &[ValueRef]) -> Result<Option<Value>, String> {
                $(
                    let $arg = <$arg>::from_value(&args[${index()}])?;
                )*
                (self)($($arg,)*).into_ret()
            }
        }
    }
}
impl_callable_tuple!(T0);
impl_callable_tuple!(T0, T1);
impl_callable_tuple!(T0, T1, T2);
impl_callable_tuple!(T0, T1, T2, T3);
impl_callable_tuple!(T0, T1, T2, T3, T4);
impl_callable_tuple!(T0, T1, T2, T3, T4, T5);
impl_callable_tuple!(T0, T1, T2, T3, T4, T5, T6);
impl_callable_tuple!(T0, T1, T2, T3, T4, T5, T6, T7);
impl_callable_tuple!(T0, T1, T2, T3, T4, T5, T6, T7, T8);
impl_callable_tuple!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9);
impl_callable_tuple!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10);
impl_callable_tuple!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11);
impl_callable_tuple!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12);
impl_callable_tuple!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13);
impl_callable_tuple!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14);
impl_callable_tuple!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15);
impl_callable_tuple!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16);
 */

// trait Property {
//     fn name(&self) -> String;
//     fn getter(&self) -> Result<ValueRef, String>;
//     fn setter(&self, value: ValueRef) -> Result<(), String>;
// }

pub trait MetaObject
where
    Self: Sized,
{
    fn meta_table() -> &'static MetaTable<Self>;
}

impl<T> Object for T
where
    T: MetaObject + std::fmt::Debug + 'static,
{
    fn method_call(&mut self, name: &str, args: &[ValueRef]) -> Result<Option<ValueRef>, String> {
        if let Some(method) = Self::meta_table().methods.get(name) {
            return (method.func)(self, args);
        }

        Err(format!("Method {}.{} not found", type_name::<T>(), name))
    }

    fn property_get(&self, key: &str) -> Result<ValueRef, String> {
        if let Some(property) = Self::meta_table().properties.get(key) {
            if let Some(getter) = &property.getter {
                return (getter)(self);
            }
        }

        Err(format!(
            "Property {}.{} is not settable",
            type_name::<T>(),
            key
        ))
    }

    fn property_set(&mut self, key: &str, value: ValueRef) -> Result<(), String> {
        if let Some(property) = Self::meta_table().properties.get(key) {
            if let Some(setter) = &property.setter {
                return (setter)(self, value);
            }
        }

        Err(format!(
            "Property {}.{} is not gettable",
            type_name::<T>(),
            key
        ))
    }
}

struct Property<T> {
    name: String,
    getter: Option<Box<dyn Fn(&T) -> Result<ValueRef, String> + Send + Sync>>,
    setter: Option<Box<dyn Fn(&mut T, ValueRef) -> Result<(), String> + Send + Sync>>,
}

struct MethodFunction<T> {
    name: String,
    func: Box<dyn Fn(&mut T, &[ValueRef]) -> Result<Option<ValueRef>, String> + Send + Sync>,
}

pub struct MetaTable<T> {
    properties: HashMap<String, Property<T>>,
    methods: HashMap<String, MethodFunction<T>>,
}

impl<T> Default for MetaTable<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> MetaTable<T> {
    pub fn new() -> Self {
        MetaTable {
            properties: HashMap::new(),
            methods: HashMap::new(),
        }
    }

    pub fn build() -> MetaTableBuilder<T> {
        MetaTableBuilder::new()
    }
}

pub struct MetaTableBuilder<T> {
    inner: MetaTable<T>,
}

impl<T> Default for MetaTableBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> MetaTableBuilder<T> {
    pub fn new() -> Self {
        MetaTableBuilder {
            inner: MetaTable {
                properties: HashMap::new(),
                methods: HashMap::new(),
            },
        }
    }

    pub fn with_property(
        mut self,
        name: impl ToString,
        getter: Option<Box<dyn Fn(&T) -> Result<ValueRef, String> + Send + Sync>>,
        setter: Option<Box<dyn Fn(&mut T, ValueRef) -> Result<(), String> + Send + Sync>>,
    ) -> Self {
        self.inner.properties.insert(
            name.to_string(),
            Property {
                name: name.to_string(),
                getter,
                setter,
            },
        );
        self
    }

    pub fn with_method(
        mut self,
        name: impl ToString,
        func: impl Fn(&mut T, &[ValueRef]) -> Result<Option<ValueRef>, String> + Send + Sync + 'static,
    ) -> Self {
        self.inner.methods.insert(
            name.to_string(),
            MethodFunction {
                name: name.to_string(),
                func: Box::new(func),
            },
        );
        self
    }

    pub fn fininal(self) -> MetaTable<T> {
        self.inner
    }
}

impl MetaObject for String {
    fn meta_table() -> &'static MetaTable<Self> {
        static STRING_META: OnceLock<MetaTable<String>> = OnceLock::new();
        STRING_META.get_or_init(|| {
            MetaTable::build()
                .with_method(
                    "len",
                    Box::new(|this: &mut String, args: &[ValueRef]| {
                        if !args.is_empty() {
                            return Err(format!(
                                "Invalid argument for {}.{}",
                                type_name::<Self>(),
                                "len"
                            ));
                        }
                        Ok(Some(Value::Integer(this.len() as i64).into()))
                    }),
                )
                .with_method(
                    "starts_with",
                    Box::new(|this: &mut String, args: &[ValueRef]| {
                        let arg = args.first().ok_or(format!(
                            "Invalid argument for {}.{}",
                            type_name::<Self>(),
                            "starts_with"
                        ))?;
                        let arg = arg.borrow();
                        match arg.deref() {
                            Value::String(s) => {
                                Ok(Some(Value::Boolean(this.starts_with(s)).into()))
                            }
                            _ => Err(format!(
                                "Invalid argument for {}.{}",
                                type_name::<Self>(),
                                "starts_with"
                            )),
                        }
                    }),
                )
                .with_method(
                    "ends_with",
                    Box::new(|this: &mut String, args: &[ValueRef]| {
                        let arg = args.first().ok_or(format!(
                            "Invalid argument for {}.{}",
                            type_name::<Self>(),
                            "ends_with"
                        ))?;
                        let arg = arg.borrow();
                        match arg.deref() {
                            Value::String(s) => Ok(Some(Value::Boolean(this.ends_with(s)).into())),
                            _ => Err(format!(
                                "Invalid argument for {}.{}",
                                type_name::<Self>(),
                                "ends_with"
                            )),
                        }
                    }),
                )
                .with_method(
                    "push",
                    Box::new(|this: &mut String, args: &[ValueRef]| {
                        let arg = args.first().ok_or(format!(
                            "Invalid argument for {}.{}",
                            type_name::<Self>(),
                            "push"
                        ))?;
                        let arg = arg.borrow();
                        match arg.deref() {
                            Value::String(s) => {
                                this.push_str(s);
                                Ok(None)
                            }
                            _ => Err(format!(
                                "Invalid argument for {}.{}",
                                type_name::<Self>(),
                                "push"
                            )),
                        }
                    }),
                )
                .fininal()
        })
    }
}
