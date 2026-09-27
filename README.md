# Neural-Network-From-Scratch-with-C

A dependency-free Feedforward Neural Network built from scratch in pure C. Implements forward/backpropagation, gradient descent, and dynamic layer architecture using only standard libraries. Showcases underlying deep learning math and custom memory management without external frameworks. Perfect for studying low-level machine learning.

## What I Have Built So Far

* **Autograd Engine:** A custom `Value` struct system that automatically tracks operations like addition, multiplication, and `tanh` activations to manage the mathematical graph.
* **Topological Sorting:** A graph evaluation algorithm that organizes mathematical nodes linearly so gradients accumulate perfectly without over-writing or firing too early.
* **Hierarchical MLP Layout:** A structural architecture where an overall Multi-Layer Perceptron manages an array of flat Layer structures, which safely track individual heap-allocated Neuron weights and biases.
* **C Memory Safety:** Conquered low-level pointers by moving the execution graph to the permanent Heap (`malloc` / `realloc`) and managing array decay shapes manually.

## What I Will Build Next

Now that the core calculation graph and network layers are structurally stable and computing accurate gradients, I plan to focus on these milestones:

* **Gradient Descent Optimizer:** Write a weight-updating function that steps through all parameters to modify their internal values using the computed gradients (`data -= learning_rate * grad`).
* **Loss Function Tracking:** Add a Mean Squared Error (MSE) metric to calculate how far off the predictions are from the true training targets.
* **Training Loops:** Create an execution routine to stream datasets through the network over multiple epochs to actively minimize loss.
* **Automated Graph Clearing:** Write a graph traversal cleanup function to safely free up heap allocations at the end of each training step so the network does not leak system RAM.
