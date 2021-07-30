Pathos: A Python Parser
========================
A straightforward and fast python parser, written in Rust.

## Features
- Recursive decent
   - Predictive parsing (where possible)
- Typed AST and visitors
   - AST is constructed through an extensible 'visitor' trait
   - If you want, you can av    oid constructing an AST and visit the source code directly
- Uses the [Logos](https://github.com/maciejhirsz/logos) lexer generator.
   - Unfortunately this means that all the text must be held in memory right now
      - This is a [limitation of Logos](https://github.com/maciejhirsz/logos/issues/159)

## TODO
NOTE: The CPython parser doesn't have any of these features, but I'd still like to have them ;)

- Safety against stack overflows
- Buffering input too big to fit in memory
- Automatic fuzzing w/ American Fuzzy Lop