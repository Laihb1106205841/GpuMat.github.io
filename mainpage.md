# How to use GpuMat

Welcome to the documentation for GpuMat. This document provides an overview of the project's features and usage instructions.

## Features

- 2D array
- Matrix on GPU
- Genetics

## Installation

To download GpuMat, follow these steps:

1. Clone the repository: https://github.com/Laihb1106205841/GpuMat.github.io.git
2. add #include "h.cu"
3. Read document on this site

## Example

Here's a simple example of how to use My Project:

```cpp

#include "matr.h"

int main() {
    const size_t rows = 8000;
    const int cols = 8000;

    Matrix<float> A(rows, cols);
    Matrix<float> B(rows, cols);

    for (float i = 0; i < rows; ++i) {
        for (float j = 0; j < cols; ++j) {
            A(i, j) = i+j;
            B(i, j) = j-j;
        }
    }

    Matrix<float> D = A * B;   // 重写*

}
```
