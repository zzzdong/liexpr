use std::{
    any::{type_name, TypeId},
    cell::{Ref, RefCell, RefMut},
    collections::HashMap,
    fmt,
    marker::PhantomData,
    ops::Deref,
    rc::Rc,
    sync::LazyLock,
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

    pub fn into_object<T>(self) -> Option<Box<T>>
    where
        T: Object,
    {
        match self {
            Value::Object(object) if TypeId::of::<T>() == (object).type_id() => unsafe {
                let ptr = Box::into_raw(object);

                Some(Box::from_raw(ptr as *mut T))
            },
            _ => None,
        }
    }
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Value::new_boolean(value)
    }
}

macro_rules! impl_from_integer_for_value {
    ($t: ty) => {
        impl From<$t> for Value {
            fn from(value: $t) -> Self {
                Value::new_integer(value as i64)
            }
        }
    };
}

impl_from_integer_for_value!(i8);
impl_from_integer_for_value!(u8);
impl_from_integer_for_value!(i16);
impl_from_integer_for_value!(u16);
impl_from_integer_for_value!(i32);
impl_from_integer_for_value!(u32);
impl_from_integer_for_value!(i64);
impl_from_integer_for_value!(u64);
impl_from_integer_for_value!(isize);
impl_from_integer_for_value!(usize);

impl From<f32> for Value {
    fn from(value: f32) -> Self {
        Value::new_float(value as f64)
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Value::new_float(value)
    }
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Value::new_string(value)
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

impl PartialEq<&str> for ValueRef {
    fn eq(&self, other: &&str) -> bool {
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

    pub fn with_object(object: impl Object) -> Self {
        ValueRef::new(Value::new_object(object))
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
                "downcast_mut::<{}> failed for {:?}",
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
    fn property_set(&mut self, key: &str, _value: ValueRef) -> Result<(), String> {
        Err(format!(
            "Unsupport property_set: {}.{}",
            std::any::type_name::<Self>(),
            key
        ))
    }
    fn call(&self, _args: &[ValueRef]) -> Result<Option<ValueRef>, String> {
        Err(format!("Unsupport call: {}", std::any::type_name::<Self>()))
    }

    fn method_call(&mut self, name: &str, _args: &[ValueRef]) -> Result<Option<ValueRef>, String> {
        Err(format!(
            "Unsupport method_call: {}.{}",
            std::any::type_name::<Self>(),
            name
        ))
    }
}

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

impl IntoRet for () {
    fn into_ret(self) -> Result<Option<ValueRef>, String> {
        Ok(None)
    }
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

macro_rules! impl_from_value_for_integer {
    ($t: ty) => {
        impl FromValue for $t {
            fn from_value(value: &ValueRef) -> Result<$t, String> {
                match value.borrow().deref() {
                    Value::Integer(i) => Ok(*i as $t),
                    _ => Err(format!("Can not convert {:?} to i64", value)),
                }
            }
        }
    };
}

impl_from_value_for_integer!(i8);
impl_from_value_for_integer!(i16);
impl_from_value_for_integer!(i32);
impl_from_value_for_integer!(i64);
impl_from_value_for_integer!(isize);
impl_from_value_for_integer!(u8);
impl_from_value_for_integer!(u16);
impl_from_value_for_integer!(u32);
impl_from_value_for_integer!(u64);
impl_from_value_for_integer!(usize);

impl FromValue for f32 {
    fn from_value(value: &ValueRef) -> Result<f32, String> {
        match value.borrow().deref() {
            Value::Float(f) => Ok(*f as f32),
            _ => Err(format!("Can not convert {:?} to f64", value)),
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
    fn call(&self, _args: &[ValueRef]) -> Result<Option<ValueRef>, String> {
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
            return (method.func).call(self, args);
        }

        Err(format!("Method {}.{} not found", type_name::<T>(), name))
    }

    fn property_get(&self, key: &str) -> Result<ValueRef, String> {
        if let Some(property) = Self::meta_table().properties.get(key) {
            if let Some(getter) = &property.getter {
                return (getter).call(self);
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
                return (setter).call(self, value);
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
    getter: Option<Box<dyn Getter<T> + Send + Sync>>,
    setter: Option<Box<dyn Setter<T> + Send + Sync>>,
}

impl<T: fmt::Debug> fmt::Debug for Property<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Property")
            .field("name", &self.name)
            .field("getter", &self.getter.is_some())
            .field("setter", &self.setter.is_some())
            .finish()
    }
}

struct Method<T> {
    name: String,
    func: Box<dyn MethodFunction<T> + Send + Sync>,
}

impl<T: fmt::Debug> fmt::Debug for Method<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Method").field("name", &self.name).finish()
    }
}

pub struct MetaTable<T> {
    properties: HashMap<String, Property<T>>,
    methods: HashMap<String, Method<T>>,
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

    pub fn with_property<Arg: 'static>(
        mut self,
        name: impl ToString,
        getter: impl GetterCallable<T> + 'static,
        setter: impl SetterCallable<T, Arg> + 'static,
    ) -> Self {
        self.inner.properties.insert(
            name.to_string(),
            Property {
                name: name.to_string(),
                getter: Some(Box::new(getter.into_function()) as Box<dyn Getter<T> + Send + Sync>),
                setter: Some(Box::new(setter.into_function()) as Box<dyn Setter<T> + Send + Sync>),
            },
        );
        self
    }

    pub fn with_getter_only(
        mut self,
        name: impl ToString,
        getter: impl GetterCallable<T> + 'static,
    ) -> Self {
        self.inner.properties.insert(
            name.to_string(),
            Property {
                name: name.to_string(),
                getter: Some(Box::new(getter.into_function()) as Box<dyn Getter<T> + Send + Sync>),
                setter: None,
            },
        );
        self
    }

    pub fn with_setter_only<Arg: 'static>(
        mut self,
        name: impl ToString,
        setter: impl SetterCallable<T, Arg> + 'static,
    ) -> Self {
        self.inner.properties.insert(
            name.to_string(),
            Property {
                name: name.to_string(),
                getter: None,
                setter: Some(Box::new(setter.into_function()) as Box<dyn Setter<T> + Send + Sync>),
            },
        );

        self
    }

    pub fn with_method<Args: 'static>(
        mut self,
        name: impl ToString,
        func: impl MethodCallable<T, Args>,
    ) -> Self {
        self.inner.methods.insert(
            name.to_string(),
            Method {
                name: name.to_string(),
                func: Box::new(func.into_function()),
            },
        );
        self
    }

    pub fn fininal(self) -> MetaTable<T> {
        self.inner
    }
}

static BOOLEAN_METATABLE: LazyLock<MetaTable<bool>> = LazyLock::new(|| {
    MetaTable::build()
        .with_method("to_string", |this: &mut bool| this.to_string())
        .fininal()
});

static INTEGER_METATABLE: LazyLock<MetaTable<i64>> = LazyLock::new(|| {
    MetaTable::build()
        .with_method("to_string", |this: &mut i64| Ok(this.to_string()))
        .with_method("abs", |this: &mut i64| Ok(this.abs()))
        .fininal()
});

static FLOAT_METATABLE: LazyLock<MetaTable<f64>> = LazyLock::new(|| {
    MetaTable::build()
        .with_method("to_string", |this: &mut f64| Ok(this.to_string()))
        .with_method("abs", |this: &mut f64| Ok(this.abs()))
        .fininal()
});

static STRING_METATABLE: LazyLock<MetaTable<String>> = LazyLock::new(|| {
    MetaTable::build()
        .with_method(
            "len",
            Box::new(|this: &mut String| Ok(this.chars().count() as i64)),
        )
        .with_method("starts_with", |this: &mut String, pat: String| {
            this.starts_with(&pat)
        })
        .with_method("ends_with", |this: &mut String, pat: String| {
            this.ends_with(&pat)
        })
        .with_method("push", |this: &mut String, other: String| {
            this.push_str(&other);
        })
        .fininal()
});

impl MetaObject for bool {
    fn meta_table() -> &'static MetaTable<Self> {
        &BOOLEAN_METATABLE
    }
}

impl MetaObject for i64 {
    fn meta_table() -> &'static MetaTable<Self> {
        &INTEGER_METATABLE
    }
}

impl MetaObject for f64 {
    fn meta_table() -> &'static MetaTable<Self> {
        &FLOAT_METATABLE
    }
}

impl MetaObject for String {
    fn meta_table() -> &'static MetaTable<Self> {
        &STRING_METATABLE
    }
}

/// A method function that can be called on a value.
pub trait MethodFunction<T>: Send + 'static {
    fn call(&self, this: &mut T, args: &[ValueRef]) -> Result<Option<ValueRef>, String>;
}

impl<T, F, Args> MethodFunction<T> for IntoFunction<F, Args>
where
    F: MethodCallable<T, Args> + Clone,
    Args: 'static,
{
    fn call(&self, this: &mut T, args: &[ValueRef]) -> Result<Option<ValueRef>, String> {
        self.func.call(this, args)
    }
}

pub trait MethodCallable<T, Args>: Clone + Send + Sync + Sized + 'static {
    fn call(&self, this: &mut T, args: &[ValueRef]) -> Result<Option<ValueRef>, String>;

    fn into_function(self) -> IntoFunction<Self, Args> {
        IntoFunction {
            func: self,
            _marker: PhantomData,
        }
    }
}

impl<T, F, Ret> MethodCallable<T, &[ValueRef]> for F
where
    F: Fn(&mut T, &[ValueRef]) -> Ret + Clone + Send + Sync + 'static,
    Ret: IntoRet,
{
    fn call(&self, this: &mut T, args: &[ValueRef]) -> Result<Option<ValueRef>, String> {
        (self)(this, args).into_ret()
    }
}

impl<T, F, Ret> MethodCallable<T, ()> for F
where
    F: Fn(&mut T) -> Ret + Clone + Send + Sync + 'static,
    Ret: IntoRet,
{
    fn call(&self, this: &mut T, _args: &[ValueRef]) -> Result<Option<ValueRef>, String> {
        self(this).into_ret()
    }
}

macro_rules! impl_method_callable {
    ($($idx: expr => $arg: ident),+) => {
        #[allow(non_snake_case)]
        impl<T, F, Ret, $($arg,)*> MethodCallable<T, ($($arg,)*)> for F
        where
            F: Fn(&mut T, $($arg,)*) -> Ret + Clone + Send + Sync + 'static,
            Ret: IntoRet,
            $( $arg: FromValue + 'static, )*
        {
            fn call(&self,this: &mut T, args: &[ValueRef]) -> Result<Option<ValueRef>, String> {
                $(
                    let $arg = <$arg>::from_value(args.get($idx).ok_or(format!("Invalid argument {}", $idx))?)?;
                )*
                (self)(this, $($arg,)*).into_ret()
            }
        }
    }
}

impl_method_callable!(0=>Arg0);
impl_method_callable!(0=>Arg0, 1=>Arg1);
impl_method_callable!(0=>Arg0, 1=>Arg1, 2=>Arg2);
impl_method_callable!(0=>Arg0, 1=>Arg1, 2=>Arg2, 3=>Arg3);
impl_method_callable!(0=>Arg0, 1=>Arg1, 2=>Arg2, 3=>Arg3, 4=>Arg4);
impl_method_callable!(0=>Arg0, 1=>Arg1, 2=>Arg2, 3=>Arg3, 4=>Arg4, 5=>Arg5);
impl_method_callable!(0=>Arg0, 1=>Arg1, 2=>Arg2, 3=>Arg3, 4=>Arg4, 5=>Arg5, 6=>Arg6);
impl_method_callable!(0=>Arg0, 1=>Arg1, 2=>Arg2, 3=>Arg3, 4=>Arg4, 5=>Arg5, 6=>Arg6, 7=>Arg7);
impl_method_callable!(0=>Arg0, 1=>Arg1, 2=>Arg2, 3=>Arg3, 4=>Arg4, 5=>Arg5, 6=>Arg6, 7=>Arg7, 8=>Arg8);
impl_method_callable!(0=>Arg0, 1=>Arg1, 2=>Arg2, 3=>Arg3, 4=>Arg4, 5=>Arg5, 6=>Arg6, 7=>Arg7, 8=>Arg8, 9=>Arg9);
impl_method_callable!(0=>Arg0, 1=>Arg1, 2=>Arg2, 3=>Arg3, 4=>Arg4, 5=>Arg5, 6=>Arg6, 7=>Arg7, 8=>Arg8, 9=>Arg9, 10=>Arg10);
impl_method_callable!(0=>Arg0, 1=>Arg1, 2=>Arg2, 3=>Arg3, 4=>Arg4, 5=>Arg5, 6=>Arg6, 7=>Arg7, 8=>Arg8, 9=>Arg9, 10=>Arg10, 11=>Arg11);
impl_method_callable!(0=>Arg0, 1=>Arg1, 2=>Arg2, 3=>Arg3, 4=>Arg4, 5=>Arg5, 6=>Arg6, 7=>Arg7, 8=>Arg8, 9=>Arg9, 10=>Arg10, 11=>Arg11, 12=>Arg12);
impl_method_callable!(0=>Arg0, 1=>Arg1, 2=>Arg2, 3=>Arg3, 4=>Arg4, 5=>Arg5, 6=>Arg6, 7=>Arg7, 8=>Arg8, 9=>Arg9, 10=>Arg10, 11=>Arg11, 12=>Arg12, 13=>Arg13);
impl_method_callable!(0=>Arg0, 1=>Arg1, 2=>Arg2, 3=>Arg3, 4=>Arg4, 5=>Arg5, 6=>Arg6, 7=>Arg7, 8=>Arg8, 9=>Arg9, 10=>Arg10, 11=>Arg11, 12=>Arg12, 13=>Arg13, 14=>Arg14);
impl_method_callable!(0=>Arg0, 1=>Arg1, 2=>Arg2, 3=>Arg3, 4=>Arg4, 5=>Arg5, 6=>Arg6, 7=>Arg7, 8=>Arg8, 9=>Arg9, 10=>Arg10, 11=>Arg11, 12=>Arg12, 13=>Arg13, 14=>Arg14, 15=>Arg15);

pub trait IntoValueRet {
    fn into_ret(self) -> Result<ValueRef, String>;
}

impl<T: Into<ValueRef>> IntoValueRet for T {
    fn into_ret(self) -> Result<ValueRef, String> {
        Ok(self.into())
    }
}

pub trait Getter<T>: Send + 'static {
    fn call(&self, this: &T) -> Result<ValueRef, String>;
}

impl<T, F, Args> Getter<T> for IntoFunction<F, Args>
where
    F: GetterCallable<T> + Clone,
    Args: 'static,
{
    fn call(&self, this: &T) -> Result<ValueRef, String> {
        self.func.call(this)
    }
}

pub trait GetterCallable<T>: Clone + Send + Sync + Sized + 'static {
    fn call(&self, this: &T) -> Result<ValueRef, String>;

    fn into_function(self) -> IntoFunction<Self, ()> {
        IntoFunction {
            func: self,
            _marker: PhantomData,
        }
    }
}

impl<T, F, Ret> GetterCallable<T> for F
where
    F: Fn(&T) -> Ret + Clone + Send + Sync + 'static,
    Ret: IntoValueRet,
{
    fn call(&self, this: &T) -> Result<ValueRef, String> {
        self(this).into_ret()
    }
}

pub trait Setter<T>: Send + 'static {
    fn call(&self, this: &mut T, value: ValueRef) -> Result<(), String>;
}

pub trait IntoVoidRet {
    fn into_ret(self) -> Result<(), String>;
}

impl IntoVoidRet for () {
    fn into_ret(self) -> Result<(), String> {
        Ok(())
    }
}

impl<T, F, Args> Setter<T> for IntoFunction<F, Args>
where
    F: SetterCallable<T, Args> + Clone,
    Args: 'static,
{
    fn call(&self, this: &mut T, value: ValueRef) -> Result<(), String> {
        self.func.call(this, value)
    }
}

pub trait SetterCallable<T, Args>: Clone + Send + Sync + Sized + 'static {
    fn call(&self, this: &mut T, value: ValueRef) -> Result<(), String>;

    fn into_function(self) -> IntoFunction<Self, Args> {
        IntoFunction {
            func: self,
            _marker: PhantomData,
        }
    }
}

impl<T, F, V, Ret> SetterCallable<T, V> for F
where
    F: Fn(&mut T, V) -> Ret + Clone + Send + Sync + 'static,
    V: FromValue,
    Ret: IntoVoidRet,
{
    fn call(&self, this: &mut T, value: ValueRef) -> Result<(), String> {
        let v = V::from_value(&value)?;
        self(this, v).into_ret()
    }
}
