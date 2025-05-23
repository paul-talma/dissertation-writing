# Summary

"When studying a given cognitive phenomenon we face the daunting task of narrowing down some vast set $\Pi$ of potential programs. The point of rational analysis is to focus our attention on those programs that solve the underlying problem well, and the point of boundedly rational analysis is to incorporate computational cost as a central component of what it would take to solve a problem well."

# Rational Analysis

- Addresses the identifiability problem: which models of the mind to even test?
- Premised on two assumptions:

  1. The mind performs probabilistic inference (in many of its functions).
  2. The standard of rationality for probabilistic inference is given by Bayesian methods.

# Challenges to Bayesian rational analysis

1. Bayesian inference intractable

   - Neither ontogeny nor phylogeny could equip us with the right algorithms

2. People often violate Bayesian norms

   - E.g. posterior matching: people's responses follow the posterior distribution, instead of maximizing expected utility.

In both cases, why should rational analysis do anything but lead us astray? People couldn't be Bayesians (because of intractability), and don't seem to approximate Bayesian rationality either.

# Approximation?

Why not say that that the mind just performs _approximate_ Bayesian inference? There are algorithms for approximate inference that are tractable, and that enjoy strong convergence results (e.g. monte carlo methods).

- Convergence results say nothing about the short run, sparse sample regime. In those conditions, other algorithms may fare far better—by Bayesian lights—than approximate Bayesian inference. What then is distinctly rational about these approximation algorithms?

# Bounds

Bayesian approximations may be _boundedly rational_, in that they maximize cost-adjusted utility.

Key components of this approach: - Need to specify a reasonable cost function. -
