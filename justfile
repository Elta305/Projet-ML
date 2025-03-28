run:
  uv run scripts/train_mnist.py

test:
  uv run pytest

paper:
  typst watch paper/paper.typ
