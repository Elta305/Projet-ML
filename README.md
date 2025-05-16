# Neural Network Implementation from First Principles

A modular neural network framework built from scratch using NumPy, progressing from linear regression to advanced architectures including autoencoders. Developed as part of the Machine Learning course at Sorbonne University, Master AI2D M1, 2024-2025.

## About the Project

This project implements a neural network framework without relying on deep learning libraries, focusing on fundamental principles and modular design. The implementation follows a design pattern where each component (layers, activations, optimizers, loss functions) is represented as a module with standardized interfaces.

Key features include:

- **Core Module Architecture**: Base abstract classes and implementations for network components
- **Optimization Algorithms**: Multiple gradient-based optimizers (SGD, SGD with Momentum, Adam)
- **Activation Functions**: Various non-linearities (ReLU, Sigmoid, TanH, Softmax)
- **Advanced Architectures**: Support for autoencoders and latent space analysis
- **Systematic Experimentation**: Comprehensive hyperparameter studies and model comparisons

## Usage

Run the main benchmark suite:

```sh
just benchmark
```

Generate visualizations:

```sh
just visualize
```

Run specific experiments:

```sh
# Linear models
just benchmark-linear-regression
just benchmark-linear-classification

# Non-linear models
just benchmark-non-linear-classification
just benchmark-non-linear-classification-adam

# MNIST experiments
just benchmark-mnist
just benchmark-optimizer-comparison
just benchmark-activation-functions
just benchmark-batch-size
just benchmark-hidden-size

# Autoencoder experiments
just benchmark-autoencoder-matrix
just benchmark-autoencoder-denoiser

# Visualization
just visualize-latent-space
just visualize-latent-centroids
just visualize-autoencoder-denoiser
```

Generate the paper:

```sh
just paper
```

## Project Structure

```
mlp/                # Core neural network implementation
├── __init__.py     # Package initialization
├── activation.py   # Activation functions (ReLU, Sigmoid, TanH, Softmax)
├── linear.py       # Linear layer implementation
├── loss.py         # Loss functions (MSE, CrossEntropy)
├── module.py       # Base module abstract class
├── optim.py        # Optimization algorithms
├── sequential.py   # Container for chaining modules
└── utils.py        # Utility functions

scripts/            # Benchmarking and visualization scripts
├── benchmark_*.py  # Performance evaluation scripts
└── visualize_*.py  # Result visualization scripts

paper/              # Report and figures
├── paper.typ      # Main report document
├── template.typ   # Report template
└── figures/       # Generated visualization figures

results/           # Experimental results
datasets/          # Training datasets (MNIST)
```

## Key Findings

Our experiments across multiple neural network architectures demonstrate:

1. **Activation Functions**: ReLU and TanH significantly outperform Sigmoid on MNIST classification
2. **Batch Size**: Medium batch sizes (64-128) provide optimal balance between performance and stability
3. **Optimizers**: Adam consistently outperforms SGD variants
4. **Autoencoder Analysis**: Latent space effectively captures digit-specific features with 96% classification accuracy
5. **Denoising Capabilities**: Models demonstrate remarkable generalization to noise levels beyond training conditions

## Authors

- [Paul Chambaz](https://www.linkedin.com/in/paul-chambaz-17235a158/)
- [Frédéric Li Combeau](https://www.linkedin.com/in/frederic-li-combeau/)

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.
