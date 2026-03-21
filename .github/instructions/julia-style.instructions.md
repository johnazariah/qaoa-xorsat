---
applyTo: "**/*.jl"
---

# Julia Coding Standards — QAOA-XORSAT

## Style

- **Small composable functions** — each does one thing, named for what it returns or does
- **Multiple dispatch** — use types for dispatch, not `if typeof(x) == ...`
- **`|>` pipelines** — prefer `x |> f |> g` over `g(f(x))` when it reads better
- **Broadcasting** — prefer `f.(xs)` over `map(f, xs)` for simple element-wise ops
- **Descriptive names** — `contract_branch`, `build_factor_tree`, not `cb`, `bft`
- **No type piracy** — don't add methods to types you don't own unless it's idiomatic

## Types

- Types are for **dispatch and clarity**, not Java-style inheritance hierarchies
- Use `struct` (immutable) by default; `mutable struct` only when mutation is essential
- Parametric types for generality: `struct FactorTree{K,D}` where K and D are type parameters
- Use `abstract type` to define dispatch hierarchies, not data hierarchies

## Functions

- Pure functions preferred — minimise side effects
- Functions that modify arguments: name with `!` suffix (`contract_inplace!`)
- Short functions (~1-5 lines) can be one-liners: `tree_depth(t::FactorTree) = t.depth`
- Use keyword arguments for options, positional for core data

## Numeric Code

- Use `Float64` by default; `ComplexF64` for quantum state amplitudes
- Use `LinearAlgebra` stdlib for matrix/vector operations
- Avoid allocations in hot loops — preallocate and use in-place operations
- Use `@views` for array slices to avoid copies
- Consider `StaticArrays.jl` for small fixed-size tensors (4^p entries at moderate p)

## Error Handling

- Use exceptions at system boundaries (file I/O, user input, invalid parameters)
- Use return values and dispatch for expected control flow
- Validate `(k, D, p)` parameters early — assert `k ≥ 2`, `D ≥ 2`, `p ≥ 1`

## Module Organisation

```julia
module QaoaXorsat

# One include per concern
include("types.jl")        # Core types: FactorTree, TensorNetwork, etc.
include("tree.jl")         # Tree construction
include("tensors.jl")      # Tensor building and contraction
include("qaoa.jl")         # QAOA evaluation (combines tree + tensors)
include("optimization.jl") # Angle optimization

# Export public API only
export build_factor_tree, tree_size, tree_depth
export contract_tree, qaoa_expectation
export optimize_angles

end
```

## Documentation

- Docstrings on all exported functions (Julia `\"\"\"...\"\"\"`  style)
- Internal functions: self-explanatory names suffice, add a one-line comment only if non-obvious
- Mathematical notation in docstrings: use LaTeX where helpful
