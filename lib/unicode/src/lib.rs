use thiserror::Error;

#[cfg(feature = "name-table")]
mod unicode_names;

pub const UNICODE_NAME_TABLE: Option<&[(&str, char)]> = {
    #[cfg(feature = "name-table")] {
        Some(self::unicode_names::NAMES)
    }
    #[cfg(not(feature = "name-table"))] {
        None
    }
};

#[derive(Debug, Error)]
pub enum NameResolveError {
    #[error("Unicode character names are currently unsupported")]
    Unsupported,
    #[error("Unknown unicode character name")]
    UnknownName,
}

pub const SUPPORTS_CHARACTER_NAMES: bool = UNICODE_NAME_TABLE.is_some();
pub fn resolve_name(name: &str) -> Result<char, NameResolveError> {
    match UNICODE_NAME_TABLE {
        Some(table) => {
            match table.binary_search_by_key(&name, |&(other_name, _)| other_name) {
                Ok(index) => Ok(table[index].1),
                Err(_) => Err(NameResolveError::UnknownName)
            }
        },
        None => Err(NameResolveError::Unsupported)
    }
}