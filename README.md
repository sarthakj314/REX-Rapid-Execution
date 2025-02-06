# Backprop Engine

A minimal single-header backpropagation library with operator overloading, color-coded graph visualization, and a global memory management scheme (all dynamic nodes freed at once).

## Features

- **Operator overloading** for `+ - * / pow`
- **Unary ops**: `sin`, `cos`, `tan`, `exp`, `log`
- **Color-coded** `draw_graph()` for easy visualization
- **No partial frees**: All intermediate nodes stored in a global container and freed via `free_all_vars()`

## How to Use

1. Include `backprop_engine.h` in exactly one `.cpp` file **with**:
   ```cpp
   #define BACKPROP_ENGINE_IMPLEMENTATION
   #include "backprop_engine.h"