import gymnasium as gym
import numpy as np
from typing import Union, Optional


class NavigationfSSPUDE(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    def __init__(self, N_x: int = 3, N_y: int = 3, penalty=-100):
        self.N_x = N_x
        self.N_y = N_y
        self.n_dead_ends = 1
        self.n_states = N_x * N_y + self.n_dead_ends
        self.n_actions = 4
        self.observation_space = gym.spaces.Discrete(self.n_states)
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.penalty = penalty
        self.s0 = N_x * N_y - 1  # initial state (bottom-right corner)
        self.g = N_x - 1  # goal state (top-right corner)
        self.dead_end_states = [self.n_states - 1]  # last state is dead-end
        self.actions_dict = {0: "UP", 1: "DOWN", 2: "RIGHT", 3: "LEFT"}
        self.states = np.arange(self.n_states)
        self.actions = np.arange(self.n_actions)
        self.goal_reward = 0
        self.min_dead_end_prob = 0.1
        self.max_dead_end_prob = 0.9

        self.P = self._create_transition_matrix()
        self.assert_transition_probabilities()

    def _create_transition_matrix(self):
        P = np.zeros((self.n_states, self.n_actions, self.n_states))

        # mid_rows = [self.N_y // 2] if self.N_y % 2 == 1 else [self.N_y // 2 - 1, self.N_y // 2]
        mid_rows = list(range(self.N_y))[1:-1]

        states = list((range(self.n_states - 1)))
        states.remove(self.g)

        dead_end_probs = np.linspace(
            self.min_dead_end_prob, self.max_dead_end_prob, num=self.N_x
        ).round(2)

        for s in states:  # Exclude dead-end state
            x, y = s % self.N_x, s // self.N_x

            # UP
            P[s, 0, s if y == 0 else s - self.N_x] = 1

            # DOWN
            P[s, 1, s if y == self.N_y - 1 else s + self.N_x] = 1

            # RIGHT
            P[s, 2, s if x == self.N_x - 1 else s + 1] = 1

            # LEFT
            P[s, 3, s if x == 0 else s - 1] = 1

            # Add probability to transition to dead-end state only for intermediate row(s)

            if y in mid_rows:
                dead_end_prob = dead_end_probs[x]  # Increases with distance from left
                for a in range(self.n_actions):
                    P[s, a, self.dead_end_states[0]] = dead_end_prob
                    P[s, a, P[s, a].nonzero()[0][0]] *= 1 - dead_end_prob

        # Goal state
        P[self.g, :, self.g] = 1

        # Dead-end state
        P[self.dead_end_states[0], :, self.dead_end_states[0]] = 1

        return P

    def assert_transition_probabilities(self):
        for s in range(self.n_states):
            for a in range(self.n_actions):
                total_prob = np.sum(self.P[s, a, :])
                if ~np.isclose(total_prob, 1.0, atol=1e-6):
                    print(
                        f"Transition probabilities for state {s}, action {a} sum to {total_prob}, not 1.0"
                    )
        print("All transition probabilities sum to 1.0 as expected.")

    # Add this line at the end of the __init__ method:

    def step(self, action):
        next_state = self.move(action)
        reward = self.reward_fun(self.s, action)
        done = False
        if self.s == self.g:
            done = True
        elif self.s in self.dead_end_states:
            done = True
        self.s = next_state
        return self.s, reward, done, {}

    def move(self, action):
        return np.random.choice(self.n_states, p=self.P[self.s, action])

    def reward_fun(self, state, action, s_next: int = None):
        if state == self.g:
            return self.goal_reward
        elif state in self.dead_end_states:
            return self.penalty
        else:
            return -1

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.s = self.s0
        return self.s, {}


class NavigationfSSPUDEtoSSP(NavigationfSSPUDE):
    def __init__(self, penalty=-100):
        super().__init__()
        self.action_space = gym.spaces.Discrete(
            self.action_space.n + 1
        )  # adding a_stop
        self.n_actions = self.n_actions + 1
        self.actions_dict = {0: "UP", 1: "DOWN", 2: "RIGHT", 3: "LEFT", 4: "A_STOP"}
        self.P = self.calculate_new_probs()
        self.penalty = penalty

    def calculate_new_probs(self):
        new_probs = []
        for state in range(self.n_states):
            state_probs = np.vstack(
                (self.P[state, :, :], np.zeros((1, self.P.shape[2])))
            )
            state_probs[-1, self.g] = 1.0
            new_probs.append(state_probs)

        return np.array(new_probs)

    def reward_fun(self, s: int, a: int, s_next: int = None) -> float:
        if a == (self.action_space.n - 1):  # a_stop action gets penalty
            reward = self.penalty
        elif s == self.g:
            reward = 0
        else:
            reward = -1

        return reward


class NavigationfSSPUDEtoMAXPROB(NavigationfSSPUDE):
    def __init__(self, N_x: int = 3, N_y: int = 3):
        super().__init__(N_x=N_x, N_y=N_y)

    def step(self, action):
        next_state = self.move(action)
        reward = self.reward_fun(self.s, action, next_state)

        done = False
        if self.s == self.g:
            done = True

        self.s = next_state
        return self.s, reward, done, {}

    def reward_fun(self, s: int, a: int, s_next: int = None) -> float:
        reward = 0
        if s != self.g and s_next == self.g:
            reward = 1

        return reward
