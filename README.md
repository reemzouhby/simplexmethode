# 📈 Simplex Method — Linear Programming Solver (Python)

This project implements the **Simplex algorithm**, a popular method in operations research for solving **Linear Programming (LP) problems** involving a linear objective function and a set of linear constraints.

It was developed as part of my Study at Sem 5 to understand optimization techniques and Python-based algorithm development.

---

## 📌 Problem Overview

The goal of the Simplex method is to:
- **Maximize** or **minimize** a linear objective function
- Subject to a set of linear constraints (inequalities or equalities)
- With non-negative variables

**Example problem**:
\[
\text{Maximize } Z = 3x_1 + 2x_2 \\
\text{Subject to:} \\
x_1 + x_2 \leq 4 \\
2x_1 + x_2 \leq 5 \\
x_1, x_2 \geq 0
\]

---

## 🔧 Features

- Solve standard maximization problems using the Simplex tableau
- Handles slack variables
- Step-by-step iteration display
- Outputs final optimal solution

---

## 🛠 Technologies Used

- **Python**
- `numpy` – For matrix operations
- (Optional: `pandas` for cleaner output)


