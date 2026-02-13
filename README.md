# Data Analysis and Unsupervised Learning in Julia

This repository contains a collection of Julia implementations for fundamental data science tasks, including synthetic data generation, linear/polynomial regression, and K-means clustering. The project explores the relationship between model complexity, data volume, and algorithmic optimization.

---

## Core Components

### 1. Regression and Curve Fitting
This section explores recovering ground truth parameters from noisy datasets using the Normal Equation and regularization techniques.

* **Linear Regression:** Implementations for Bivariate (1D) and Multivariate (2D+) linear models.
    * **The Normal Equation:** $$B = (X^T X)^{-1} X^T Y$$
* **Polynomial Curve Fitting:** A study of the bias-variance tradeoff. We demonstrate how low-degree polynomials underfit the target, while high-degree polynomials (e.g., $M=9$) overfit small datasets.

* **Regularization (Ridge Regression):** To mitigate overfitting, L2 regularization is implemented by adding a penalty term $\lambda$ to the diagonal of the design matrix:
    $$w^* = (\lambda I + \Phi^T \Phi)^{-1} \Phi^T Y$$

### 2. K-Means Clustering
The project implements two primary algorithmic approaches to the K-means clustering problem, minimizing the squared Euclidean distance between points and centroids.

* **Batch K-means (Lloyd's Algorithm):** * **E-Step:** Assigns all data points to the nearest centroid.
    * **M-Step:** Relocates centroids to the arithmetic mean of all assigned points.
    * Includes a stability fix to reinitialize empty clusters to a random data point.

* **Online K-means (Stochastic Update):**
    * Updates centroids incrementally after each individual data point.
    * Utilizes a learning rate $\eta$ for real-time updates: $$\mu_{k} \leftarrow \mu_{k} + \eta(x_n - \mu_{k})$$
    * Suitable for large-scale datasets where memory constraints prevent batch processing.

### 3. Data Simulation and Utilities
Tools designed to generate controlled synthetic data for benchmarking:
* **noisyLinearMultiDim:** Multi-dimensional linear data generation with Gaussian noise.
* **noisySin:** Sinusoidal data generation for non-linear regression testing.
* **circGauss:** Generates $N$ samples from a Circular Gaussian distribution $\mathcal{N}(\mu, \sigma^2 I)$.
* **histo:** A custom histogram implementation using nearest-neighbor binning logic.

---

## Analysis and Results

### Bias-Variance Tradeoff
Through iterative testing, we visualize how increasing the polynomial degree $M$ reduces training error but causes the Test Root Mean Square Error ($E_{RMS}$) to increase exponentially once the model begins to model noise.


**Key findings include:**
1. **Data Volume:** Increasing the number of observations ($N$) allows high-complexity models to generalize without needing regularization.
2. **Regularization:** In scenarios with sparse data, the $\lambda$ parameter effectively "shrinks" coefficients to prevent erratic model behavior.

### Algorithmic Comparison: Batch vs. Online
* **Batch K-means** is deterministic and typically converges in fewer iterations but requires the entire dataset to be loaded into memory.
* **Online K-means** offers higher flexibility for streaming data and can be more computationally efficient for massive datasets, though it requires careful tuning of the learning rate.

---

## Technical Setup

### Prerequisites
* Julia (Recommended v1.6 or higher)
* Required Packages: `Plots.jl`, `LinearAlgebra`, `Statistics`, `Random`

### Execution
To run a specific module, include the source file within the Julia REPL:
```julia
include("src/Problem5.jl")
# Example function call
w = polyfitLS(X_data, Y_data, 3)
