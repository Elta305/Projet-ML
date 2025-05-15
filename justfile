run:
  @just linear-regression

benchmark:
  just benchmark-linear-regression
  just benchmark-linear-classification
  just benchmark-non-linear-classification
  just benchmark-mnist

visualize:
  just visualize-linear-regression
  just visualize-linear-classification
  just visualize-non-linear-classification
  just visualize-mnist

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

test:
  @uv run pytest

paper:
  @typst watch paper/paper.typ
