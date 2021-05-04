Pathos
========
A simple recursive decent parser generator.


The complement to the [Logos](https://github.com/maciejhirsz/logos) lexer generator.

Designed for fast and straightforward parsing.


## Features
- Recursive decent
   - Predictive parsing (where possible)
   - Automatically generates code for all `LL(k)` rule
   - TODO: [Pratt parsing](http://craftinginterpreters.com/compiling-expressions.html#a-pratt-parser)
- Readable generated code, designed for editing by the user
- Auto-generates typed AST
