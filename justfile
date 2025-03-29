run:
  @just linear-regression

linear-regression:
  @uv run scripts/train_linear_regression.py

linear-classification:
  @uv run scripts/train_linear_classification.py

non-linear-classification:
  @uv run scripts/train_non_linear_classification.py

non-linear-mlp-classification:
  @uv run scripts/train_non_linear_mlp_classification.py

mnist:
  @uv run scripts/train_mnist.py

test:
  @uv run pytest

paper:
  @typst watch paper/paper.typ
