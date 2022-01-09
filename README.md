Pathos: A Python Parser [DEAD]
========================
This was my attempt at writing a recursive-decent parser by hand, for the python source code.

It worked fairly well (for a subset of code) but it was large and clunky.

I have since decided to try and avoid hand-written parsers ;)

### Parser generators
If you're into parser generators, take a look at:
1. [parol](https://lib.rs/crates/parol) - LL(k)
2. [pest](https://pest.rs/) - PEG
3. [tree-sitter](https://tree-sitter.github.io/tree-sitter/) - LR(1)
4. [lalrpop](https://lib.rs/crates/lalrpop) LR(1) or LALR(1)
5. [Antlr](https://www.antlr.org/) - Amazing magic but Rust bindings are experimental :(


If you're into parser combinators, take a look at:
1. [combine](https://lib.rs/crates/combine) - Looks neat
2. [chumsky](https://github.com/zesterer/chumsky/) - Excellent error recovery and messagses
2. [nom](https://lib.rs/crates/nom) - Stable


### ASTs
I did some work on [ASDLR](https://www.oilshell.org/blog/2016/12/11.html) ASTs here. That has since moved [to a seperate project](https://github.com/DuckLogic/rust-asdlr).
