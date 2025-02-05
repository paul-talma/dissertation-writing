There are several ways of construing the error of a linear value function.

We restrict our attention to linear value functions on large but finite state spaces.
In general, we have fewer parameters ($n$) than there are states ($k$) (that being the point of function approximation).
If we represent value functions as vectors of values, with an entry for each state/parameter, then the value functions live in a $k$-dimensional space, while the linear approximations thereof live in a $n < k$-dimensional subspace.

Let $v_\pi$ denote the true value function for policy $\pi$.
Then it is reasonable to model the distance between an approximation and the real thing by the length of the vector $v_\pi - v_\vec w$.
We don't want to define this length as euclidean distance, since this gives equal weight to each state.
Rather, we wish to weigh each state based on how much we care about getting its value right.
One reasonable way to do this is to weigh each state by its probability under the on-policy distribution.

- Side note: the idea that we don't need too precise an estimate of very low value states is right—though it's rather that we don't need too precise an estimate of states whose value is much lower than that of accessible states. In other words, we just need to know they're bad enough to ignore, we don't need to know precisely how bad.
- Note that the same applies to great states: we don't need to know exactly how good they are, if they are so good that they should always be chosen over their peers (i.e. if the imprecision we allow never lets another state be preferred to this one).

Thus, we define the length of a vector by the norm $\| v \|^2_\mu = \sum_{s \in \mathcal S} \mu(s) v(s)^2$.

Now, each policy has a projection onto the subspace of approximable policies.
The projection of the true value function is the approximable function closest to it.
But TD methods find another solution: a fixed point of the **Projected Bellman operator**.
The Bellman operator $B$ takes a value function $v$ to the function $Bv$ that assigns to $s$ its expected bootstrap value (i.e. its expected next reward + the estimated value of the next state).

The true value function uniquely satisfies the Bellman equation.
For other functiond, there is a gap between the value it assigns to a state and that state's bootstrap value—this is precisely the temporal difference that is used to correct the current value.
(The length of) that gap is the **Bellman error**.

Now, the subspace of approximable functions is not closed under the Bellman operator.
So we cannot in general move in the direction of $Bv$.
But consider the projection of $Bv$ onto the approximable subspace.
That we can move towards, and that is precisely what semi-gradient TD methods do.
Thus, we can define a Projected Bellman Operator (which is the composition of the projection and Bellman operators) and a corresponding Projected Bellman error, which measures the distance between an approximable function and its image under the Projected Bellman operator.
Semi-gradient TD methods converge to the fixed point of the PBO.

Note that this need not minimize the projected Bellman error.
Note also that this need not minimize error (distance from true function).

Thus, we have three notions of error, all potentially distinct:

- Mean squared error, distance to the true value function
- bellman error, distance to one's bootstrap
- projected bellman error, distance to the projection of one's bootstrap

Clearly, the mean squared error is the most fundamental, since it characterizes distance to the true value function.
But without knowing the true value function, it is impossible to compute.
By contrast, one's bootstrap is computable (certainly if one has a model, what if one doesn't?), but not in general representable.

- This means that, at least with a model, it is possible to compute, given $v_\pi$, the value of $Bv_\pi$ on any input.
  (well this is all computable so what does this even mean??)
- But despite being computable, there is no setting of the weights that will make _our_ value function be $Bv_\pi$.

Finally, the bootstrap projection is both computable and representable.
