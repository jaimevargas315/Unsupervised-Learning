# Unsupervised Learning and Data Discretization in Julia

This repository contains implementations of fundamental unsupervised learning algorithms and data simulation utilities, specifically focusing on **K-means Clustering** variants and distance-based data processing.

---

## Core Implementations

### 1. K-Means Clustering Algorithms
The project implements two distinct approaches to partitioning $N$ observations into $K$ clusters by minimizing the squared Euclidean distance.

* **Batch K-means (Lloyd's Algorithm):** * **E-Step:** Assigns data points to the nearest centroid using the `compDistInd` utility.
    * **M-Step:** Updates centroids to the mean of all assigned points.
    * **Stability:** Features an automated reinitialization for empty clusters to prevent algorithmic failure.
* **Online K-means (Stochastic Update):**
    * Performs incremental updates for each data point: $\mu_{k} \leftarrow \mu_{k} + \eta(x_n - \mu_{k})$.
    * Includes dataset order randomization (shuffling) before each epoch to improve convergence.
    * Utilizes a learning rate $\eta$ and movement-based convergence tolerance.

### 2. Data Simulation and Processing
Tools designed to generate synthetic datasets and discretize continuous values:

* **circGauss:** Generates $N$ samples from a Circular Gaussian distribution $\mathcal{N}(\mu, \sigma^2 I)$ in multi-dimensional space.
* **histo:** A custom discretization tool that assigns data points to bin centers based on a nearest-neighbor (minimum absolute distance) logic.
* **compDistInd:** A distance-calculation utility that returns both a distance matrix and a boolean indicator matrix ($r_{nk}$) for point-to-cluster assignments.

---

## Usage

### Clustering Setup
Initialize your data as a `Vector{Vector{Float64}}` and provide an initial set of centroids.

```julia
# Batch K-means
final_centers, assignments, iterations = myKmeansBatch(data, initial_centers)

# Online K-means (Learning rate 0.01)
final_centers, epochs = myKmeansOnline(data, initial_centers, 0.01)
