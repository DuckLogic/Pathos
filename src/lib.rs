pub mod lexer;
mod ast;


#[cfg(feature="num-bigint")]
/// An arbitrary precision integer
pub type BigInt = num_bigint::BigInt;
#[cfg(fefature="rug")]
/// A rug BigInt
pub type BigInt = rug::Integer;
/// Fallback arbitrary precision integers,
/// when all dependencies are disabled
///
/// Stored as plain text
#[cfg(not(any(feature="num-bigint", feature="rug")))]
pub struct BigInt {
    pub text: String
}
