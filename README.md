# Neural-Network-From-Scratch-with-C

A dependency-free feedforward neural network built from scratch in pure C. It implements forward/backpropagation, gradient descent, and a dynamic layer architecture using only the standard library. It shows the math behind deep learning and hand-written memory management, without any external framework. Good for studying low-level machine learning.

## Results

Trained on the classic Iris dataset (150 rows, 4 features, 3 classes) with an 80/20 train/test split:

| Item | Value |
|---|---|
| Architecture | `4 -> 16 -> 16 -> 3` (tanh hidden layers, linear output + softmax) |
| Parameters | 403 |
| Optimizer | Stochastic gradient descent, 1 sample per step |
| Learning rate / epochs | 0.005 / 100 |
| Training loss | 1.198 (epoch 1) -> 0.081 (epoch 100) |
| Test loss | 0.057 |
| **Test accuracy** | **96.7% (29 of 30 unseen rows)** |

Notes on reading this result:
* The test set is only 30 rows, so each row is worth about 3.3%. Treat the number as roughly 93-100%.
* The random seed is based on the current time, so every run starts from different weights and results vary a little.

## What I Have Built

* **Autograd Engine:** A custom `Value` struct system that tracks operations (`+`, `-`, `*`, `/`, `exp`, `log`, `tanh`) and builds the computation graph automatically.
* **Topological Sorting:** Orders the graph nodes so gradients flow backward correctly during the backward pass.
* **Hierarchical MLP:** A flexible multi-layer perceptron made of layer structs, neuron structs, and heap-allocated weights and biases. The network shape is set by one array (for example `{4, 16, 16, 3}`), and the output layer skips the activation.
* **Softmax and NLL Loss:** A multi-class `Softmax` layer built from graph nodes (so gradients reach the network), paired with a negative log-likelihood loss.
* **Weight Initialization:** Weights are drawn uniformly from `[-1/sqrt(fan_in), +1/sqrt(fan_in)]`.
* **Dataset Loader:** A dependency-free CSV reader (`fopen`, `fgets`, `strtok`) that grows its arrays dynamically and applies min-max scaling to every feature.
* **Train/Test Split and Evaluation:** An 80/20 split, a prediction step (`find_max_index`), and reporting of test loss and accuracy on rows the network never trained on.
* **Dynamic Graph Memory Management:** Cleanup loops during training free the temporary operator nodes after every sample, including the isolated softmax nodes, so memory use stays flat across epochs.

## Dataset Format

The loader expects a plain text file with:
* no header row,
* comma-separated values,
* the feature columns first (`COLS`, default 4), then the class label as the last column,
* labels as whole numbers starting at 0 (`0`, `1`, `2`),
* **rows shuffled**. The network trains one row at a time, so a file sorted by class would make it forget earlier classes.

Example row: `5.9,3.2,4.8,1.8,1`

The Iris data used here comes from scikit-learn's `load_iris()`, with the species converted to 0/1/2 and the rows shuffled.

## Build and Run

1. Set the dataset path in `main()` (the `fopen(...)` call) to your own file.
2. Build with CMake (CLion) or directly with a compiler:

```bash
gcc main.c -o nn -lm
./nn
```

(`-lm` links the math library on Linux and macOS. It is not needed on Windows.)

## Lessons Learned

* **A flat loss is not always a bug.** My first dataset (a cleaned smartwatch file) kept the loss stuck at about 1.10, which equals ln(3), the loss of guessing 1/3 for every class. Before touching the code, I checked the data: per-class feature means, spreads and correlations were nearly identical, and random forest and gradient boosting models scored about 33%, the same as guessing. The labels had no learnable link to the features, so no code change could have fixed it.
* **A sanity test separates "bug" from "bad data".** I replaced the labels with a simple rule computed from one feature. The loss dropped quickly, which proved the autograd, softmax, backpropagation and update loop were correct.
* **Real data confirmed it.** On Iris, the same code reached 96.7% test accuracy.
* **Row order matters for one-sample-at-a-time training,** so data must be shuffled.

## Known Limitations

* Min-max scaling currently uses the minimum and maximum of **all** rows, including test rows. Cleaner practice is to compute them from the training rows only and apply those same numbers to the test rows.
* Training uses a batch size of 1, and the row order is not reshuffled between epochs.
* Softmax does not subtract the maximum logit, so very large logits could overflow `exp`.
* Graph nodes created in the test loop are not freed (not a problem for a short run).
* The dataset path is hardcoded in `main()`.

## What I Will Build Next

### 1. Better Evaluation
* Compute normalization statistics from the training rows only.
* Add a confusion matrix to see which classes get mixed up.
* Free the memory used during the test loop.
* Reshuffle the training rows every epoch.

### 2. Model Persistence Layer (Save & Load)
* **Serialization (`Save`):** Write all trained weights and biases to a checkpoint file.
* **Deserialization (`Load`):** Rebuild the network from the checkpoint into live `Value` structs, so the model can make predictions without retraining.

### 3. A Harder Dataset and Real-World Data Cleaning
* Try a dataset with more features, such as Wine (13 features), which needs only a change to `COLS` and the first value of the structure array.
