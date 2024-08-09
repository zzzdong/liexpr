use std::str::Chars;

use crate::ast::Literal;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// EOF
    Eof,
    /// Literal
    Literal(Literal),
    /// Identifier
    Identifier(String),
    /// fn
    Fn,
    /// return
    Return,
    /// let
    Let,
    /// if
    If,
    /// else
    Else,
    /// for
    For,
    /// continue
    Continue,
    /// break
    Break,
    /// (
    LeftParen,
    /// )
    RightParen,
    /// [
    LeftBracket,
    /// ]
    RightBracket,
    /// {
    LeftBrace,
    /// }
    RightBrace,
    /// ,
    Comma,
    /// .
    Dot,
    /// -
    Minus,
    /// +
    Plus,
    /// ;
    Semicolon,
    /// /
    Slash,
    /// *
    Star,
    /// %
    Percent,
    /// ++
    PlusPlus,
    /// --
    MinusMinus,
    /// +=
    PlusEqual,
    /// -=
    MinusEqual,
    /// *=
    StarEqual,
    /// /=
    SlashEqual,
    /// %=
    PercentEqual,
    /// !
    Bang,
    /// !=
    BangEqual,
    /// =
    Equal,
    /// ==
    EqualEqual,
    /// >
    Greater,
    /// >=
    GreaterEqual,
    /// <
    Less,
    /// <=
    LessEqual,
    /// &&
    AndAnd,
    /// ||
    OrOr,
    /// unknown
    Unknown(char),
}

impl Token {
    pub fn single_char(c: char) -> Self {
        match c {
            '(' => Token::LeftParen,
            ')' => Token::RightParen,
            '{' => Token::LeftBrace,
            '}' => Token::RightBrace,
            '[' => Token::LeftBracket,
            ']' => Token::RightBracket,
            ',' => Token::Comma,
            ';' => Token::Semicolon,
            '.' => Token::Dot,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tokenizer<'i> {
    chars: Chars<'i>,
}

impl<'i> Tokenizer<'i> {
    pub fn new(input: &'i str) -> Tokenizer<'i> {
        let chars = input.chars();

        Tokenizer { chars }
    }

    fn next_char(&mut self) -> Option<char> {
        self.chars.next()
    }

    fn peek_char(&self) -> Option<char> {
        self.chars.clone().peekable().peek().copied()
    }

    fn starts_with(&self, pat: &str) -> bool {
        self.chars.as_str().starts_with(pat)
    }

    fn match_next(&self, expect: char) -> bool {
        if let Some(next) = self.peek_char() {
            if next == expect {
                return true;
            }
        }
        false
    }

    fn next_token(&mut self) -> Result<Token, String> {
        while let Some(c) = self.peek_char() {
            // println!("-> self:{self:?}, c: {c}");
            match c {
                ' ' | '\r' | '\t' | '\n' => {
                    // eat whitespace
                    self.next_char();
                }
                '"' => {
                    self.next_char();
                    return self
                        .eat_string()
                        .map(|s| Token::Literal(Literal::String(s)));
                }
                ch if ch.is_ascii_digit() => return self.eat_number().map(Token::Literal),
                ch if ch.is_ascii_alphabetic() || ch == '_' => {
                    let s = self.eat_identifier()?;
                    return match s.as_str() {
                        "fn" => Ok(Token::Fn),
                        "return" => Ok(Token::Return),
                        "let" => Ok(Token::Let),
                        "if" => Ok(Token::If),
                        "for" => Ok(Token::For),
                        "else" => Ok(Token::Else),
                        "break" => Ok(Token::Break),
                        "continue" => Ok(Token::Continue),
                        "null" => Ok(Token::Literal(Literal::Null)),
                        "true" => Ok(Token::Literal(Literal::Boolean(true))),
                        "false" => Ok(Token::Literal(Literal::Boolean(false))),
                        _ => Ok(Token::Identifier(s)),
                    };
                }
                ch => {
                    if self.starts_with("//") {
                        self.eat_comment();
                        continue;
                    }

                    if let Some(symbol) = self.eat_symbol(ch) {
                        return Ok(symbol);
                    }
                    return Err(format!("Unexpected character: {}", ch));
                }
            }
        }

        Ok(Token::Eof)
    }

    fn eat_symbol(&mut self, ch: char) -> Option<Token> {
        match ch {
            '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';' | '.' => {
                self.next_char();
                Some(Token::single_char(ch))
            }
            '!' => {
                self.next_char();
                if self.match_next('=') {
                    self.next_char();
                    Some(Token::BangEqual)
                } else {
                    Some(Token::Bang)
                }
            }
            '=' => {
                self.next_char();
                if self.match_next('=') {
                    self.next_char();
                    Some(Token::EqualEqual)
                } else {
                    Some(Token::Equal)
                }
            }
            '>' => {
                self.next_char();
                if self.match_next('=') {
                    self.next_char();
                    Some(Token::GreaterEqual)
                } else {
                    Some(Token::Greater)
                }
            }
            '<' => {
                self.next_char();
                if self.match_next('=') {
                    self.next_char();
                    Some(Token::LessEqual)
                } else {
                    Some(Token::Less)
                }
            }
            '+' => {
                self.next_char();
                if self.match_next('+') {
                    self.next_char();
                    Some(Token::PlusPlus)
                } else if self.match_next('=') {
                    self.next_char();
                    return Some(Token::PlusEqual);
                } else {
                    return Some(Token::Plus);
                }
            }
            '-' => {
                self.next_char();
                if self.match_next('-') {
                    self.next_char();
                    Some(Token::MinusMinus)
                } else if self.match_next('=') {
                    self.next_char();
                    return Some(Token::MinusEqual);
                } else {
                    return Some(Token::Minus);
                }
            }
            '*' => {
                self.next_char();
                if self.match_next('=') {
                    self.next_char();
                    Some(Token::StarEqual)
                } else {
                    Some(Token::Star)
                }
            }
            '/' => {
                self.next_char();
                if self.match_next('=') {
                    self.next_char();
                    Some(Token::SlashEqual)
                } else {
                    Some(Token::Slash)
                }
            }
            '%' => {
                self.next_char();
                if self.match_next('=') {
                    self.next_char();
                    Some(Token::PercentEqual)
                } else {
                    Some(Token::Percent)
                }
            }
            '&' => {
                self.next_char();
                if self.match_next('&') {
                    self.next_char();
                    Some(Token::AndAnd)
                } else {
                    Some(Token::Unknown(ch))
                }
            }
            '|' => {
                self.next_char();
                if self.match_next('|') {
                    self.next_char();
                    Some(Token::OrOr)
                } else {
                    Some(Token::Unknown(ch))
                }
            }
            _ => None,
        }
    }

    fn eat_string(&mut self) -> Result<String, String> {
        let mut s = String::new();

        while let Some(ch) = self.next_char() {
            match ch {
                '"' => return Ok(s),
                '\\' => {
                    const ESCAPED: [char; 5] = ['r', 'n', 't', '\\', '"'];
                    match self.peek_char() {
                        Some(peek) => {
                            if ESCAPED.contains(&peek) {
                                match peek {
                                    'r' => s.push('\r'),
                                    'n' => s.push('\n'),
                                    't' => s.push('\t'),
                                    '\\' => s.push('\\'),
                                    '"' => s.push('"'),
                                    _ => return Err("Invalid escape sequence".to_string()),
                                }
                                self.next_char();
                            } else {
                                s.push(ch);
                            }
                        }
                        None => return Err("Unterminated string".to_string()),
                    }
                }
                _ => s.push(ch),
            }
        }

        Err("Unterminated string".to_string())
    }

    fn eat_number(&mut self) -> Result<Literal, String> {
        let mut s = String::new();

        let mut has_dot = false;

        while let Some(ch) = self.peek_char() {
            match ch {
                '0'..='9' => {
                    self.next_char();
                    s.push(ch);
                }
                '.' => {
                    if has_dot {
                        return Err("Invalid number".to_string());
                    }
                    has_dot = true;
                    self.next_char();
                    s.push(ch);
                }
                _ => break,
            }
        }

        if has_dot {
            s.parse::<f64>()
                .map(Literal::Float)
                .map_err(|err| format!("Invalid float number: {}", err))
        } else {
            s.parse::<i64>()
                .map(Literal::Integer)
                .map_err(|err| format!("Invalid integer number: {}", err))
        }
    }

    fn eat_identifier(&mut self) -> Result<String, String> {
        let mut s = String::new();
        while let Some(ch) = self.peek_char() {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                self.next_char();
                s.push(ch);
            } else {
                break;
            }
        }
        Ok(s)
    }

    fn eat_comment(&mut self) {
        while let Some(ch) = self.next_char() {
            if ch == '\n' {
                return;
            }
        }
    }
}

impl<'i> Iterator for Tokenizer<'i> {
    type Item = Result<Token, String>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_token() {
            Ok(Token::Eof) => None,
            Ok(token) => Some(Ok(token)),
            Err(err) => Some(Err(err)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tokens(std::vec::IntoIter<Token>);

impl Tokens {
    pub fn new(input: &str) -> Result<Tokens, String> {
        let tokens: Result<Vec<Token>, String> = Tokenizer::new(input).collect();

        Ok(Tokens(tokens?.into_iter()))
    }
}

impl Iterator for Tokens {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer() {
        let source = "let x = 1.0; let arr = [1, 2, 3];";
        let tokenizer = Tokenizer::new(source);

        let tokens: Result<Vec<Token>, String> = tokenizer.into_iter().collect();

        assert_eq!(
            tokens,
            Ok(vec![
                Token::Let,
                Token::Identifier("x".to_string()),
                Token::Equal,
                Token::Literal(Literal::Float(1.0)),
                Token::Semicolon,
                Token::Let,
                Token::Identifier("arr".to_string()),
                Token::Equal,
                Token::LeftBracket,
                Token::Literal(Literal::Integer(1)),
                Token::Comma,
                Token::Literal(Literal::Integer(2)),
                Token::Comma,
                Token::Literal(Literal::Integer(3)),
                Token::RightBracket,
                Token::Semicolon,
            ])
        );
    }

    #[test]
    fn test_tokenizer_string() {
        let source = r#"let x = "hello\tworld"; "#;
        let tokenizer = Tokenizer::new(source);

        let tokens: Result<Vec<Token>, String> = tokenizer.into_iter().collect();

        assert_eq!(
            tokens,
            Ok(vec![
                Token::Let,
                Token::Identifier("x".to_string()),
                Token::Equal,
                Token::Literal(Literal::String("hello\tworld".to_string())),
                Token::Semicolon,
            ])
        );
    }
}
