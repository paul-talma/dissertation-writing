# Introduction

To learn from experience is to generalize.
Such, at any rate, is the assumption of many researchers in the machine learning community.
For example, [CITE: LFD] note that the difference between a machine learning model that has merely memorized its training data and one that has learned from it is that the latter, but not the former, can be expected to generalize.
[CITE: CIML], in a popular textbook on machine learning, goes so far as to say that "generalization is perhaps the most central concept in machine learning."
[CITE: AIMA] put the point succinctly: "in machine learning the goal is to generalize to new data that has not been seen previously, as measured by performance on a test set."
But although generalization has been extensively studied in the context of supervised learning, its proper interpretation in the context of reinforcement learning remains unclear.
This is situation is theoretically unsatisfying: if generalization is as central to learning many machine learning researchers claim, the "learning" in reinforcement learning will remain puzzling unless we can identify the dimensions of generalization appropriate to a reinforcement learner.

In this paper, we remedy this situation by laying out three concepts of generalization appropriate to reinforcement learning.
We explain how they relate to each other as well as to generalization in supervised learning, and draw some epistemological lessons.

# Generalization in supervised learning

We begin by situating generalization in the context of supervised learning, where it has received most attention.

Roughly speaking, a machine learning system is said to generalize, or generalize well, to the extent that its training performance matches its "real world" performance.
We will illustrate this rough characterization using a basic supervised learning setup.

Suppose that we are interested in classifying irises on the basis of a handful of features.
The first step in building a machine learning model is to define the task to be solved and collect relevant data.
In our case, the task is to predict which species a given iris belongs to on the basis of the length and width of its sepal and petals.
We thus collect data on the length and width of the sepal and petals of various irises, noting which of three species each specimen belongs to.
Our data then consists of a set $\mathcal D = \{ (\vec x_1, y_1), \dots, (\vec x_N, y_N) \}$ of $N$ datapoints.
Each datapoint $(\vec x_i, y_i)$ consists of a _feature vector_ $\vec x_i = (x_\text{sepal-length}, x_\text{sepal-width}, x_\text{petal-length}, x_\text{petal-width})$ recording the dimensions of the flower, together with an _index_ $y_i \in \{\text{setosa}, \text{virginica}, \text{versicolor}\}$ recording that iris' species.
The latter is often called the _target_ of classification: it represents how the model should classify that particular exemplar.
Since the aim is to classify irises on the basis of their features, we will call the feature vector the _input_ to the classification problem.

Once the task has been defined and the data collected, we select a machine learning model to train on the data.
The choice of model is as much art as science, and is guided by our experience using various models, our "feel" for the data, domain-specific knowledge about the task at hand, and a variety of practical considerations.
In our case, we might choose a decision tree as our model.
A decision tree classifies irises on the basis of a series of yes-no questions concerning the features of its input.
For example, given an input $\vec x$, the model might start by asking whether the sepal is wider than $w$.
If the answer is "yes," the model might then ask whether the petals are longer than $\ell$, whereas if the answer is "no," it might ask a different question.
After a few such questions, the model makes its decision: an iris with features $\vec x$ belongs to species $\hat y$.
More generally, we can think of our model as implementing a function $g$ from features $\vec x$ to classifications $\hat y$.
Which function our model implements depends on the specific questions the model asks and on the order in which they are asked, as represented by the model's tree structure.
For example, our model may ignore sepal length completely in making its decision, or it may ask about sepal length only if the petal width is less than a certain threshold.
These factors—the features taken into consideration, the thresholds, and the structure of the model—are collectively denoted by a vector of _trainable parameters_ $\vec \theta$.
The parameter vector determines the function instantiated by the model, and different values of $\vec \theta$ will in general result in different classification decision.
To make explicit the fact that the model's predictions depend not only on its input but also on its parameters, we display the parameters in our notation for the function instantiated by the model: $g(\vec x; \vec \theta) = \hat y$.

Having chosen our model, we must now train it on the data.
The process of training the model consists in selecting appropriate values for the components of $\theta$ on the basis of the training data.
Values of $\vec \theta$ are "appropriate" to the extent that a model with these parameters performs well.
And a model performs well to the extent that it succeeds on its task—in our case, to the extent that it correctly classifies irises.
Here, we come to a crucial distinction.
We do not need our model to categorize the irises in the training data—after all, we already know which species these belong to.
Instead, we care about the model's performance ourside of our iris sample—perhaps our model will help botanists categorize large quantities of as-yet-unseen irises in a high-stakes situation (we may suppose that only the petals of the _versicolor_ species are fit for human consumption, and that a certain restaurant wishes to garnish its signature dish with edible iris petals).
More precisely, we want the probability that our model will correctly classify an unseen iris sample correctly to be close to $1$.

<!--TODO: check how this relates to false positive and false negative rates, and their relative importance. -->

Crucially, this probability depends both on the model itself and on the true distribution of irises conditional on their features, a distribution which is _ex hypothesi_ unknown to us.
What we can know is whether our model classifies its training examples correctly.

[transition needed, this was patched up]

Since we do not generally know how a model will perform in the real world, we estimate its real world performance by evaluating it on a held-out _testing set_ of data.
Correspondingly, we estimate the degree of generalization of a model by computing the difference between its performance (or error) on the training and testing set.
Let us define the model's error on the training set, also known as the _in-sample error_, as

$$
E_\text{train} := \frac{1}{|\testdata|} \sum_{(\vec x, y) \in \testdata} \mathbb I(h(\vec x) \neq y)
$$

The function $\mathbb I (p)$ is called the _indicator function_ and returns $1$ if its argument is a true proposition and $0$ otherwise.
Thus, the training error measures the fraction of training instances that are misclassified by the model.
(Note that for simplicity we have used a very simple measure of error: the number of misclassified instances.
More sophisticated ways of measuring error may be appropriate in different contexts.)
Similarly, we define the model's error on the true distribution, also known as the _out-of-sample error_, as

$$
E_\text{out} := P_{\vec x \in \mathcal X} [h(\vec x) \neq f(\vec x)]
$$

This denotes the probability that the model $h$'s prediction on an input
$\vec x$ differs from the true classification $f(\vec x)$.

As discussed above, since the function $f$ is not known to us, we cannot compute the out-of-sample error.
We instead estimate it using the model's error on the held-out testing set:

$$
E_\text{test} := \frac{1}{|\testdata|} \sum_{(\vec x, y) \in \testdata} \mathbb I(h(\vec x) \neq y)
$$

As mentioned above, under certain assumptions (the most important of which being that the sample $\sample$ is drawn from the true distribution), the test error is an unbiased estimator of the out-of-sample error.

We thus arrive at a model's _generalization error_:

$$
E_\text{gen} = |E_{\text{train}} - E_{\text{out}}|
$$

which is operationalized as the _empirical generalization error_:

$$
E_\text{emp-gen} = |E_{\text{train}} - E_{\text{test}}|.
$$

In what follows, we shall talk of the generalization error indiscriminately, letting context disambiguate which of the two kinds we mean.
The important points to keep in mind are that (i) what we really care about is the out-of-sample generalization error, (ii) we cannot directly compute this error, (iii) the empirical generalization error, which we can compute, is an unbiased estimate of the out-of-sample generalization error.

Note that the generalization error is comparative: it quantifies how much worse (or better) the model does out of sample than in sample, not how well the model does out of sample.
In common usage, a model is said to generalize (or generalize well) if it has both a low training error and a low generalization error.

[Need a couple paragraphs to wrap up this section: step back from the formal details, summarize in an intuitive way, connect to learning.]

# Reinforcement learning: a primer

[FILL IN]

# Generalization in RL

With the basics of reinforcement learning on the table, we are ready to pose our central question: what is generalization in the context of reinforcement learning?
We will argue that there are three distinct concepts of generalization applicable to reinforcement learning.

Before exploring these notions of generalization, it is worth saying a word about why our question is not answered by our earlier discussion of generalization.
In other words, why think that the notion of generalization relevant to reinforcement learning is any different from that relevant to supervised learning?
Generalization in supervised machine learning is naturally framed, as we did above, in terms of the difference in performance on training and testing data.
But reinforcement learning is characteristically an _online_ learning paradigm: in many applications, there is no sharp distinction between training and testing.
Instead, the reinforcement learner is "dropped" in an environment, and it must simultaneously act and learn to act, bearing the consequences of its learning trials in real time.
Of course, it is possible to enforce a distinction between training and testing in reinforcement learning algorithms.
For example, [CITE: Mnih et al.] trained a reinforcement learning agent to play Atari video games at a high level.
The agent underwent a training phase, during which it developed a policy for each game.
After learning good policies, the agent underwent a testing phase, during which it followed the policies it had learned, undiluted by exploratory decisions.
Under such a regime, there is a clear distinction between training and testing.
But many applications of reinforcement learning do not support a distinction between training and testing.
This is particularly clear in the case of animal learning: in many tasks, the animal does not enjoy the luxury of a training period before being "deployed" in an environment.
But the same point also holds in many artificial cases, where we might want the agent to adapt to a changing environment in real-time.
In such cases, the definition of generalization introduced above finds no application.

In what follows, we will develop concepts of generalization that apply to both online and offline reinforcement learning algorithms.

## Within-environment generalization

## Cross-state generalization

In simple environments, the agent's policy may associate an action to each state, and in the process of learning such a policy, the agent may maintain a value function that assigns to each state or state-action pair a value.
This strategy will not work if the environment is too complex along either of two dimensions: if the state space is too large, or if the action space is too large.
For example, consider the classic control problem of balancing a pole upright by moving the cart to which the pole is attached (see figure X).
A natural way to think of a state in this environment is as a tuple $(\theta, x, v)$ recording the angle $\theta$ between the pole and the vertical and the position $x$ and velocity $v$ of the cart.
However, each of these variables is continuous: they can take an uncountably infinite number of values.
We cannot expect a finite mind or computer to separately encode its preferred response to each state.
Moreover, the algorithms discussed in the previous section for learning value functions require the agent to update its estimates each time a state is entered, which presumes a discrete state space. [Need to clarify here—the point is not quite about continuity, since if the state values were rationals, the same problem would arise. Where exactly do the relevant algorithms break down in the move to (i) a large state space, (ii) a countably infinite, discrete state space, (iii) a countably infinite, non-discrete state space (i.e. Q), and (iv) an uncountable, continuous state space (R)]
[Note well: the point can't be about the fact that we can't be sensitive to a continuous quantity: it's perfectly possible for our sensory apparatus to respond continuously to certain continuous inputs. Rather, the point is that the RL algorithms depend on updating your estimate of the value of a state each time you visit it, which would require an infinite amount of computation every second.]
Moreover, in environments with large state spaces, most states will only be encountered once, or not at all, in the lifetime of an organism.
If that lifetime is to last long, the organism must respond sensibly to states it has not yet seen.
And as [CITE: Sutton and Barto] put it: "To make sensible decisions in such states it is necessary to generalize from previous encounters with different states that are in some sense similar to the current one. In other words, the key issue is that of _generalization_."

Formally, this problem is approached through _parametrization_: there are too many states to store the value of each in memory, so we instead store a vector of parameters $\theta$ and compute a parametrized function $f(x; \theta)$ as the value of $x$.
For this to make sense, the number of parameters must be less than the number of states (else instead of storing the parameters, we could just store the state values).
Now, generalization comes into the picture in two ways here.
First, when the organism encounters a state, what it learns will transfer to other states.
Since it can only update its estimate of the value of the state by updating the parameter vector $\theta$, and since there are more states than parameters, updating the parameters will invariably change the value assigned to other states.
Thus, updates exhibit cross-state generalization.
The key question concerning this generalization is whether it works: whether states whose value estimates evolve together have similar values.
More precisely, when the agent increases its value estimate for state $s$, it will thereby increase its value estimate for some other states and decrease its estimate for others.
For learning to be effective, these other estimates must be updated in the appropriate direction, i.e. they must move closer to their true estimates.
Now, by the nature of function approximation, we cannot always make every estimate move in the right direction.
If some estimates become more accurate, others will have to be less accurate.
But it would seem reasonable to concentrate our attention on estimates of states that we care about.
How much we care about a state is given by a distribution of _interest_ across states—often this is just taken to be the on-policy distribution.
Note that how much we care about a state may change as a result of our new estimate of its value (once we update the policy with respect to the new estimated value function).

In any case, ideally, we would like for updates to the parameter vector to generalize well, in the sense that when we update the parameters after visiting state $s$, the estimated values of states that we are likely to encounter moves in the right direction.
We quantify this desideratum by defining the weighted squared error of an estimated value function, which is the sum of squared distances between estimated and true state values, weighted by their importance.

## Cross-environment generalization

# Relation to overfitting

# Relation to knowledge transfer

# Objections and replies

A skeptic might argue that generalization plays no fundamental role in reinforcement learning: either it reduces to other, already-understood concepts, or it is not theoretically or practically significant in reinforcement learning algorithms.

Two sources of skepticism:

- Lack of distinction between training and testing
- Lack of need for the notion in unsupervised learning

# Quotes

- "Generalization is perhaps the most central concept in machine learning." Daumé, CIML: 9
- Couldn't find a specific quote in _Learning From Data_ but the idea that learning is generalization is all over the book.
- "If our model performs well on predicting the labels of the examples from the test set, we say that our model _generalizes well_ or, simply, that it's good." Burkov 2019, The 100 Page Machine Learning Book: §5.6
- "The true measure of a hypothesis is not how it does on the training set, but rather how well it handles inputs it has not yet seen. ... We say that $h$ generalizes well if it accurately predicts the outputs of the test set." Russell and Norvig, §19.2
- "in machine learning the goal is to generalize to new data that has not been seen previously, as measured by performance on a test set." Russell and Norvig, §21.3
  - "a great deal of effort in deep learning research has gone into finding network architectures that generalize well"
- "Learning is the process of converting experience into expertise or knowledge" Understanding Machine Learning: 19
- "While the preceding “learning by memorization” approach is sometimes useful, it lacks an important aspect of learning systems–the ability to label unseen e-mail messages. A successful learner should be able to progress from individual examples to broader _generalization_." UML
- "To make sensible decisions in such states it is necessary to generalize from previous encounters with different states that are in some sense similar to the current one. In other words, the key issue is that of _generalization_." Sutton and Barto: 195
  $$
