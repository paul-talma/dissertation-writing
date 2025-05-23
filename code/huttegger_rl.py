import random


class Environment:
    def __init__(
        self,
        states: list[int],
        transition_probabilities: list[list[list[float]]],
        rewards: list[list[list[tuple[float, float]]]],
        initial_state=0,
    ) -> None:
        self.states = states
        self.transition_probabilities = transition_probabilities
        self.rewards = rewards
        self.current_state = initial_state

    def step(self, act: int):
        transition_probabilities = self.transition_probabilities[self.current_state][
            act
        ]
        next_state = random.choices(
            population=self.states, weights=transition_probabilities, k=1
        )[0]
        mu, sigma = self.rewards[self.current_state][act][next_state]
        reward = random.gauss(mu, sigma)
        self.current_state = next_state
        return next_state, reward


class BasicRLAgent:
    def __init__(self, actions: list[int]) -> None:
        self.actions = actions
        self.choice_propensities = [1 for _ in self.actions]

    def choose_act(self):
        choice_probabilities = self._normalize()
        a = random.choices(population=self.actions, weights=choice_probabilities)[0]
        return a

    def update(self, act, reward):
        self.choice_propensities[act] += reward

    def _normalize(self):
        total = sum(self.choice_propensities)
        return [self.choice_propensities[a] / total for a in self.actions]


class SophisticatedRLAgent:
    def __init__(self, actions: list[int]) -> None:
        self.actions = actions
        self.last_act = None
        self.consitional_choice_propensities = {
            a: {b: 1 for b in self.actions} for a in self.actions
        }

    def choose_act(self):
        choice_probabilities = self._normalize()
        a = random.choices(population=self.actions, weights=choice_probabilities)
        self.last_act = a
        return a

    def _normalize(self):
        pass


def loop(agent: BasicRLAgent, environment: Environment, T: int):
    rewards = []
    for iter in range(T):
        a = agent.choose_act()
        _, reward = environment.step(a)
        agent.update(act=a, reward=reward)
        rewards.append(reward)
