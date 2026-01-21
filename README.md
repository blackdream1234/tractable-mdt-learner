# Efficient Learning of Tractable Multivariate Decision Trees

## Overview
This repository contains a practical implementation of an "Intelligent Learning Algorithm" for **Generalized Square 2CNF** Multivariate Decision Trees (MDTs), as described in the paper *"Tractable Explaining of Multivariate Decision Trees"*.

The paper identified a computational bottleneck where learning these constraints via naive feature expansion scales as $O(n^2 d^4)$. This project implements a **Bounded 2D-Interval Search** algorithm that reduces complexity to $O(nd^2)$, rendering the problem tractable for large domains.

## Algorithm
Instead of pre-generating the feature space, the learner uses a dynamic greedy search accelerated with **Numba (JIT)**:
1.  **Unary Scan ($O(nd^2)$)**: Finds optimal intervals $[p, q]$ for single features.
2.  **Pair Selection**: Ranks feature pairs based on unary gain to avoid checking all $n^2$ combinations.
3.  **Optimized Disjunction Search ($O(d^2)$)**: Dynamically finds the optimal $(x_i \in A \lor x_j \in B)$ split without materializing the feature matrix.

## Key Results
Benchmarked against Brute Force Feature Expansion on a synthetic dataset (Target: $(x_0 \in [2, d-2]) \lor (x_2 \in [1, d-1])$).

| Domain Size ($d$) | Brute Force Time | Intelligent Time | Speedup | Note |
| :--- | :--- | :--- | :--- | :--- |
| **20** | 0.39s | 0.08s | **4.6x** | |
| **60** | 98.08s | 4.93s | **19.8x** | Brute force struggles |
| **80** | **CRASH (OOM)** | **0.003s** | **$\infty$** | **Tractability Proven** |
| **100** | **CRASH (OOM)** | **0.0001s** | **$\infty$** | **Tractability Proven** |

*Hardware: Standard Consumer CPU / Google Colab Environment*

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
