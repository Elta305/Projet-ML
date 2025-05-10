#import "template.typ": *
#import "@preview/cetz:0.3.1": canvas, draw
#import "@preview/cetz-plot:0.1.0": plot

#show: report.with(
  title: [ Neural Networks ],
  course: [ Machine Learning ],
  authors: ("Paul Chambaz", "Frédéric Li Combeau"),
  university: [ Sorbonne University ],
  reference: [ Master 1 -- #smallcaps[Ai2d] -- 2025 ],
  nb-columns: 2,
  abstract: [
    This project implements a neural network framework from first principles, beginning with linear regression and extending to more complex architectures.
  ]
)

== Linear Regression

Neural networks begin with linear regression. Let $x in RR^d$ be our input vector with dimension $d$, $W in RR^(p times d)$ a weight matrix mapping to $p$ outputs, and $b in RR^p$ a bias vector. The linear transformation computes:

$
hat(y) = bold(W) bold(x) + b
$

// TODO: this part is badly explained - we should just compute directly both sides (both fractions of partials in the gradients) to show where they come from since it would be cleaner
This computation forms the core of our linear layer. Learning occurs through gradient descent, which requires computing how changes in weights affect the loss. Let $cal(L)$ be our loss function, mean square error. The weight gradients are:

$
gradient_bold(W) cal(L)
= (partial cal(L))/(partial hat(y)) (partial hat(y))/(partial bold(W))
$

For a linear layer, $(partial hat(y))/(partial bold(W)_(i j)) = x_j$ for output $hat(y)_i$, which in matrix form is $(partial hat(y))/(partial bold(W)) = x^T$.

For MSE, defined as:

$
cal(L) (y, hat(y)) = 1/n sum_(i=0)^n (y_i - hat(y)_i)^2
$

This gradient is:

$
(partial cal(L))/(partial hat(y)) = -2/n (y - hat(y))
$

Our implementation seperates these concerns into distinct modules:
- A `Linear` module handling forward computation and gradient calculations ;
- A `MSELoss` module computing the loss and its gradient ;
- The base `Module` class defining interfaces for all layers, given by the handout.

The training process iteratively applies gradient descent:

#algorithm(
  title: [Gradient descent for linear regression],
  input: [$bold(x)$ examples, $y$ labels, $eta$, $cal(E)$],
  output: [$bold(W), b$],
  steps: (
    ([$bold(W) random RR^(p times d), b random RR^p$]),
    ([*For* $e in {1, ..., cal(E)}$]),
    (depth: 1, line: [$hat(y) <- bold(W) bold(x) + b$]),
    (depth: 1, line: [$cal(L) <- 1/n sum (y_i - hat(y)_i)^2$]),
    (depth: 1, line: [$gradient_bold(W) cal(L) <- -2/n (y-hat(y)) bold(x)^T$]),
    (depth: 1, line: [$gradient_b cal(L) <- sum -2/n (y-hat(y))$]),
    (depth: 1, line: [$bold(W) <- bold(W) - eta gradient_bold(W) cal(L)$]),
    (depth: 1, line: [$b <- b - eta gradient_b cal(L)$]),
  )
)

=== Results analysis
@fig-1 shows our linear regression results. The loss curve displays the expected pattern: rapid initial decrease followed by gradual convergence. Within 200 epochs, most improvement occurs, with minimal gains thereafter. The bottom plots contrast the best and worst models. Both capture linear relationships but with different slopes. The data points cluster tightly around both lines, showing both models learned useful patterns performance differences. This variability stems from random initialization and dataset generation.

These results indicate our implementation correctly applies the gradient descent algorithm. The smooth convergence indicates proper gradient computation and parameter updates.

== Linear and non-linear classification

#lorem(50)

#figure(caption: [
  This figure displays training results from $100$ linear regression trials
  ($n=100$) with $200$ samples per run ($sigma=200$), learning rate $0.01$
  ($eta=0.01$), and $1000$ epochs ($cal(E)=1000$). The top plot shows the
  interquartile mean loss and Q1-Q3 range during training, while the bottom
  plots contrast the best and worst performing models from the ensemble.
], image("./figures/linear_regression.svg")) <fig-1>

#figure(caption: [
], image("./figures/linear_classification.svg")) <fig-2>


#figure(caption: [
], image("./figures/non_linear_classification.svg")) <fig-3>
