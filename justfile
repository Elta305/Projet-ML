run:
  @just benchmark-autoencoder-matrix

benchmark:
  just benchmark-linear-regression
  just benchmark-linear-classification
  just benchmark-non-linear-classification
  just benchmark-non-linear-classification-adam
  just benchmark-mnist
  just benchmark-optimizer-comparison
  just benchmark-activation-functions
  just benchmark-batch-size
  just benchmark-hidden-size
  just benchmark-autoencoder-matrix
  just benchmark-autoencoder-denoiser
  just benchmark-initialization
  just benchmark-hidden-layers_relu

visualize:
  just visualize-linear-regression
  just visualize-linear-classification
  just visualize-non-linear-classification
  just visualize-non-linear-classification-adam
  just visualize-optimizer-comparison
  just visualize-activation-functions
  just visualize-batch-size
  just visualize-hidden-size
  just visualize-autoencoder-matrix
  just visualize-autoencoder-examples
  just visualize-latent-space
  just visualize-latent-centroids
  just visualize-autoencoder-denoiser
  just visualize-initialization
  just visualize-hidden-layers_tanh

benchmark-linear-regression:
  @uv run scripts/benchmark_linear_regression.py

visualize-linear-regression:
  @uv run scripts/visualize_linear_regression.py

benchmark-linear-classification:
  @uv run scripts/benchmark_linear_classification.py

visualize-linear-classification:
  @uv run scripts/visualize_linear_classification.py

benchmark-non-linear-classification:
  @uv run scripts/benchmark_non_linear_classification.py

visualize-non-linear-classification:
  @uv run scripts/visualize_non_linear_classification.py

benchmark-non-linear-classification-adam:
  @uv run scripts/benchmark_non_linear_classification_adam.py

visualize-non-linear-classification-adam:
  @uv run scripts/visualize_non_linear_classification_adam.py

benchmark-mnist:
  @uv run scripts/benchmark_mnist.py

visualize-mnist:
  @uv run scripts/visualize_mnist.py

benchmark-optimizer-comparison:
  @uv run scripts/benchmark_optimizer_comparison.py

visualize-optimizer-comparison:
  @uv run scripts/visualize_optimizer_comparison.py

benchmark-activation-functions:
  @uv run scripts/benchmark_activation_functions.py

visualize-activation-functions:
  @uv run scripts/visualize_activation_functions.py

benchmark-batch-size:
  @uv run scripts/benchmark_batch_size.py

visualize-batch-size:
  @uv run scripts/visualize_batch_size.py

benchmark-hidden-size:
  @uv run scripts/benchmark_hidden_size.py

visualize-hidden-size:
  @uv run scripts/visualize_hidden_size.py

benchmark-autoencoder-matrix:
  @uv run scripts/benchmark_autoencoder_matrix.py

visualize-autoencoder-matrix:
  @uv run scripts/visualize_autoencoder_matrix.py

visualize-autoencoder-examples:
  @uv run scripts/visualize_autoencoder_examples.py

visualize-latent-space:
  @uv run scripts/visualize_latent_space.py

visualize-latent-centroids:
  @uv run scripts/visualize_latent_centroids.py

benchmark-autoencoder-denoiser:
  @uv run scripts/benchmark_autoencoder_denoiser.py

visualize-autoencoder-denoiser:
  @uv run scripts/visualize_autoencoder_denoiser.py

benchmark-initialization:
  @uv run scripts/benchmark_initialization.py

visualize-initialization:
  @uv run scripts/visualize_initialization.py

benchmark-hidden-layers_relu:
  @uv run scripts/benchmark_hidden_layers_relu.py

visualize-hidden-layers_tanh:
  @uv run scripts/visualize_hidden_layers_tanh.py

paper:
  @typst watch paper/paper.typ

publish:
  rm -fr dist
  rm -fr ML-paulchambaz-fredericlicombeau
  mkdir -p ML-paulchambaz-fredericlicombeau
  mkdir -p ML-paulchambaz-fredericlicombeau/mlp
  cp -r mlp/*.py ML-paulchambaz-fredericlicombeau/mlp
  mkdir -p ML-paulchambaz-fredericlicombeau/scripts
  cp -r scripts/*.py ML-paulchambaz-fredericlicombeau/scripts
  cp paper/paper.pdf ML-paulchambaz-fredericlicombeau
  cp README.md ML-paulchambaz-fredericlicombeau
  cp pyproject.toml ML-paulchambaz-fredericlicombeau
  cp uv.lock ML-paulchambaz-fredericlicombeau
  zip -r ML-paulchambaz-fredericlicombeau.zip ML-paulchambaz-fredericlicombeau
  rm -fr ML-paulchambaz-fredericlicombeau
  mkdir -p dist
  mv ML-paulchambaz-fredericlicombeau.zip dist/
