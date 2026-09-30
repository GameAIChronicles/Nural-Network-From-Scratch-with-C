# Neural-Network-From-Scratch-with-C

A dependency-free Feedforward Neural Network built from scratch in pure C. Implements forward/backpropagation, gradient descent, and dynamic layer architecture using only standard libraries. Showcases underlying deep learning math and custom memory management without external frameworks. Perfect for studying low-level machine learning.

## What I Have Built So Far

* **Autograd Engine:** A custom `Value` struct system that automatically tracks operations (`+`, `-`, `*`, `/`, `exp`, `log`, `tanh`) to trace complex mathematical execution graphs.
* **Topological Sorting:** A graph evaluation algorithm that organizes mathematical nodes linearly so gradients accumulate perfectly backward during `.backward()` execution.
* **Hierarchical MLP Layout:** A fully flexible Multi-Layer Perceptron architecture that manages layer structs down to individual heap-allocated Neuron weights and biases, complete with customizable activation switches per layer.
* **Custom Softmax & NLL Loss:** Designed a fully linked multi-class `Softmax` allocation layer that avoids broken computational graphs, paired with a `Negative Log-Likelihood` function to train classifications directly.
* **Dynamic Graph Memory Management:** Written custom, fine-grained lifecycle cleanup loops inside the execution phase to prune operator nodes and isolate unlinked Softmax nodes, keeping runtime RAM footprints perfectly flat.

## What I Will Build Next

Now that the core calculation graph, network layers, multi-class math, and autograd backpropagation loops are completely stable, I will focus on these production-level milestones:

### 1. Smartwatch Data Pipeline (`unclean_smartwatch_health_data.csv`)
* **CSV Parser:** Write a dependency-free CSV reader in C using standard file I/O (`fopen`, `fgets`, `strtok`) to load health data into memory.
* **Data Cleaning & Imputation:** Handle missing values, filter out structural corruptions/outliers, and strip headers directly in C.
* **Feature Scaling:** Implement a normalization module (like Min-Max Scaling or Z-score Standardisation) to prevent raw smartwatch health metrics from breaking exponential functions.

### 2. Model Persistence Layer (Save & Load)
* **Serialization (`Save`):** Create a weight exporter that iterates through all trained parameters (`weights` and `biases`) and flushes them safely into a checkpoint file.
* **Deserialization (`Load`):** Build an initializer that reconstructs the network configuration from a saved checkpoint file, parsing structural matrices directly back into live `Value` structs to allow instant model inference without retraining.
