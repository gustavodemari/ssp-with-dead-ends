import numpy as np
from tqdm import tqdm


class QLearning:
    """
    Q-Learning algorithm implementation.

    Attributes
    ----------
    gamma : float
        Discount factor for future rewards.
    learning_rate : float
        Learning rate for the Q-learning algorithm.
    epsilon : float
        Exploration rate for the epsilon-greedy policy.
    n_max_timesteps : int
        Maximum number of timesteps for training.
    n_max_timesteps_per_episode : int
        Maximum number of timesteps per episode.
    verbose : bool
        If True, enables verbose output.
    writer : object, optional
        Optional writer object for logging.
    init_method : str
        Method to initialize the Q-table, default is "zeros".
    """

    def __init__(
        self,
        gamma=0.9,
        learning_rate=0.1,
        epsilon=0.1,
        n_max_timesteps=10000,
        n_max_timesteps_per_episode=100,
        verbose=False,
        writer=None,
        init_method="zeros",
    ):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.n_max_timesteps = n_max_timesteps
        self.n_max_timesteps_per_episode = n_max_timesteps_per_episode
        self.n_episodes = 0
        self.verbose = verbose
        self.writer = writer
        self.init_method = init_method

        self.Q_s_a = None
        self.V_s = None
        self.policy = None
        self.N_s_a = None

    def fit(self, env):
        if self.init_method == "ones":
            self.Q_s_a = np.ones((env.n_states, env.n_actions))  # optimistic init
            self.Q_s_a[env.g, :] = 0.0
        elif self.init_method == "zeros":
            self.Q_s_a = np.zeros((env.n_states, env.n_actions))

        self.N_s_a = np.zeros((env.n_states, env.n_actions))
        cumulative_goals_achieved = 0
        total_timesteps = 0

        self.Q_s_a[env.g, :] = 0.0

        while total_timesteps < self.n_max_timesteps:
            state, _ = env.reset()
            sum_rewards_per_episode = 0
            sum_discounted_reward_per_episode = 0

            for t in range(self.n_max_timesteps_per_episode):
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(env.n_actions)
                else:
                    action = self.Q_s_a[state].argmax()

                next_state, reward, done, info = env.step(action)

                self.N_s_a[state, action] += 1
                if state == env.g:
                    cumulative_goals_achieved += 1

                V_state = self.Q_s_a[state].max()
                V_next_state = self.Q_s_a[next_state].max()

                if state != env.g:
                    self.Q_s_a[state, action] = (1 - self.learning_rate) * self.Q_s_a[
                        state, action
                    ] + self.learning_rate * (reward + self.gamma * V_next_state)

                sum_rewards_per_episode += reward
                sum_discounted_reward_per_episode += self.gamma * reward

                V_init = self.Q_s_a[env.s0].max()
                V_dead_end = self.Q_s_a[env.dead_end_states[0]].max()
                self.V_s = self.Q_s_a.max(axis=1)
                self.policy = self.Q_s_a.argmax(axis=1)

                state = next_state
                total_timesteps += 1

                if done:
                    break

            self.n_episodes += 1


class QLearningIntrinsicMotivation:
    """
    Q-Learning Intrinnsic Motivation algorithm implementation.

    Attributes
    ----------
    gamma : float
        Discount factor for future rewards.
    learning_rate : float
        Learning rate for the Q-learning algorithm.
    epsilon : float
        Exploration rate for the epsilon-greedy policy.
    n_max_timesteps : int
        Maximum number of timesteps for training.
    n_max_timesteps_per_episode : int
        Maximum number of timesteps per episode.
    verbose : bool
        If True, enables verbose output.
    writer : object, optional
        Optional writer object for logging.
    beta : float
        Intrinsic motivation reward factor.
    init_method : str
        Method to initialize the Q-table, default is "zeros".
    """

    def __init__(
        self,
        gamma=0.9,
        learning_rate=0.1,
        epsilon=0.1,
        n_max_timesteps=10000,
        n_max_timesteps_per_episode=100,
        verbose=False,
        writer=None,
        beta=0.1,
        init_method="zeros",
    ):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.n_max_timesteps = n_max_timesteps
        self.n_max_timesteps_per_episode = n_max_timesteps_per_episode
        self.n_episodes = 0
        self.verbose = verbose
        self.writer = writer
        self.beta = beta
        self.init_method = init_method

        self.Q_s_a = None
        self.V_s = None
        self.policy = None
        self.N_s_a = None

    def fit(self, env):
        if self.init_method == "ones":
            self.Q_s_a = np.ones((env.n_states, env.n_actions))  # optimistic init
            self.Q_s_a[env.g, :] = 0.0
        elif self.init_method == "zeros":
            self.Q_s_a = np.zeros((env.n_states, env.n_actions))

        self.N_s_a = np.zeros((env.n_states, env.n_actions))
        cumulative_goals_achieved = 0
        total_timesteps = 0

        while total_timesteps < self.n_max_timesteps:
            state, _ = env.reset()
            sum_rewards_per_episode = 0
            sum_discounted_reward_per_episode = 0
            # env.render()
            for t in range(self.n_max_timesteps_per_episode):
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(env.n_actions)
                else:
                    action = self.Q_s_a[state].argmax()

                next_state, reward, done, _ = env.step(action)
                self.N_s_a[state, action] += 1

                reward_intrinsic = self.beta * (
                    self.N_s_a[state].sum() / (total_timesteps + 1)
                )

                reward = reward + reward_intrinsic

                if state == env.g:
                    cumulative_goals_achieved += 1

                V_state = self.Q_s_a[state].max()
                V_next_state = self.Q_s_a[next_state].max()

                if state != env.g:
                    self.Q_s_a[state, action] = (1 - self.learning_rate) * self.Q_s_a[
                        state, action
                    ] + self.learning_rate * (reward + self.gamma * V_next_state)

                self.Q_s_a[state, action] = max(0, self.Q_s_a[state, action])

                sum_rewards_per_episode += reward
                sum_discounted_reward_per_episode += self.gamma * reward

                V_init = self.Q_s_a[env.s0].max()
                V_dead_end = self.Q_s_a[env.dead_end_states[0]].max()
                self.V_s = self.Q_s_a.max(axis=1)
                self.policy = self.Q_s_a.argmax(axis=1)

                state = next_state
                total_timesteps += 1

                if done:
                    break

            self.n_episodes += 1


class ValueIterationFinitePenalty:
    """
    Value Iteration Finite Penalty algorithm implementation.

    Attributes
    ----------
    gamma : float
        Discount factor for future rewards.
    theta : float
        Threshold for convergence.
    n_max_iter : int
        Maximum number iterations
    penalty: float
        Penalty value for the algorithm.
    verbose : bool
        If True, enables verbose output.
    writer : object, optional
        Optional writer object for logging.
    """

    def __init__(
        self,
        gamma=0.9,
        theta=0.01,
        n_max_iter=1000,
        penalty=-100,
        verbose=False,
        writer=None,
    ):
        self.gamma = gamma
        self.Q_s_a = None
        self.V_s = None
        self.policy = None
        self.theta = theta
        self.n_max_iter = n_max_iter
        self.penalty = penalty
        self.verbose = verbose
        self.writer = writer

        if self.verbose:
            print(self.__dict__)

    def fit(self, env):
        states = list(range(env.n_states))
        actions = list(range(env.n_actions))
        goal_state = env.g
        self.Q_s_a = np.zeros((env.n_states, env.n_actions))
        self.Q_s_a[goal_state, :] = 0.0  # goal state value is 0

        for i in tqdm(range(self.n_max_iter)):
            delta = 0.0

            for state in states:
                if state == goal_state:
                    self.Q_s_a[goal_state, :] = 0.0
                    continue

                V_state_before_update = self.Q_s_a[state].max()

                for action in actions:
                    Q_s_a_sum = 0.0

                    for next_state in states:
                        reward = env.reward_fun(state, action, next_state)
                        V_next_state = self.Q_s_a[next_state].max()
                        probability = env.P[state, action, next_state]

                        Q_s_a_sum += probability * (
                            reward + (self.gamma * V_next_state)
                        )

                    self.Q_s_a[state, action] = Q_s_a_sum
                    self.Q_s_a = np.maximum(self.Q_s_a, self.penalty)

                V_state_after_update = self.Q_s_a[state].max()

                V_diff = np.abs(V_state_after_update - V_state_before_update)
                delta = max(delta, V_diff)
                V_init = self.Q_s_a[env.s0].max()
                self.V_s = self.Q_s_a.max(axis=1)
                self.policy = self.Q_s_a.argmax(axis=1)

            if delta < self.theta:
                break

        self.V_s = self.Q_s_a.max(axis=1)
        self.policy = self.Q_s_a.argmax(axis=1)


class QLearningMinCMaxP:
    """
    Q-Learning MinC MaxP algorithm implementation.

    Attributes
    ----------
    gamma : float
        Discount factor for future rewards.
    learning_rate : float
        Learning rate for the Q-learning algorithm.
    epsilon : float
        Exploration rate for the epsilon-greedy policy.
    n_max_timesteps : int
        Maximum number of timesteps for training.
    n_max_timesteps_per_episode : int
        Maximum number of timesteps per episode.
    verbose : bool
        If True, enables verbose output.
    writer : object, optional
        Optional writer object for logging.
    threshold : float
        Threshold for selecting the MAXPROB actions.
    beta : float
        Intrinsic motivation reward factor.
    """

    def __init__(
        self,
        gamma=0.9,
        learning_rate=0.1,
        epsilon=0.1,
        n_max_timesteps_per_episode=100,
        n_max_timesteps=10000,
        verbose=False,
        writer=None,
        threshold=0.01,
        beta=-0.1,
    ):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.n_max_timesteps = n_max_timesteps
        self.n_max_timesteps_per_episode = n_max_timesteps_per_episode
        self.n_episodes = 0
        self.verbose = verbose
        self.writer = writer
        self.beta = beta

        self.Q_s_a = None
        self.V_s = None
        self.policy = None
        self.N_s_a = None
        self.threshold = threshold

    def fit_maxprob(self, env):
        params = {
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "epsilon": self.epsilon,
            "n_max_timesteps": self.n_max_timesteps,
            "n_max_timesteps_per_episode": self.n_max_timesteps_per_episode,
            "verbose": self.verbose,
            "beta": self.beta,
            "init_method": "ones",
        }

        algo_maxprob = QLearningIntrinsicMotivation(**params)
        algo_maxprob.fit(env)
        return algo_maxprob

    def fit(self, env_maxprob, env):
        self.Q_s_a = np.zeros((env.n_states, env.n_actions))
        self.N_s_a = np.zeros((env.n_states, env.n_actions))
        cumulative_goals_achieved = 0
        total_timesteps = 0

        # run maxprob
        self.algo_maxprob = self.fit_maxprob(env_maxprob)
        self.P_g = self.algo_maxprob.Q_s_a.max(axis=-1)
        zero_states = np.argwhere(self.P_g == 0)
        self.Q_s_a[zero_states, :] = 0

        selected_actions = self.algo_maxprob.Q_s_a > (
            self.algo_maxprob.Q_s_a.max(axis=-1) - self.threshold
        ).reshape(-1, 1)
        avaliable_actions = [
            np.flatnonzero(action_each_state) for action_each_state in selected_actions
        ]

        while total_timesteps < self.n_max_timesteps:
            state, _ = env.reset()
            sum_rewards_per_episode = 0
            sum_discounted_reward_per_episode = 0

            # env.render()
            for t in range(self.n_max_timesteps):
                A_s = avaliable_actions[state]

                if np.random.rand() < self.epsilon:
                    action = np.random.choice(A_s)
                else:
                    action_index = self.Q_s_a[state, A_s].argmax()
                    action = A_s[action_index]

                next_state, reward, done, _ = env.step(action)
                self.N_s_a[state, action] += 1
                if state == env.g:
                    cumulative_goals_achieved += 1

                A_next_state = avaliable_actions[next_state]
                V_next_state = self.Q_s_a[next_state, A_next_state].max()

                if state not in zero_states:
                    self.Q_s_a[state, action] = self.Q_s_a[
                        state, action
                    ] + self.learning_rate * (
                        reward + self.gamma * V_next_state - self.Q_s_a[state, action]
                    )

                sum_rewards_per_episode += reward
                sum_discounted_reward_per_episode += self.gamma * reward

                self.V_s = np.array(
                    [
                        self.Q_s_a[state, avaliable_action].max()
                        for state, avaliable_action in enumerate(avaliable_actions)
                    ]
                )
                self.policy = np.array(
                    [
                        avaliable_action[self.Q_s_a[state, avaliable_action].argmax()]
                        for state, avaliable_action in enumerate(avaliable_actions)
                    ]
                )
                V_init = self.V_s[env.s0]
                V_dead_end = self.Q_s_a[env.dead_end_states[0]].max()

                state = next_state
                total_timesteps += 1

                if done:
                    break

            self.n_episodes += 1


class QLearningFinitePenalty:
    """
    Q-Learning Finite Penalty algorithm implementation.

    Attributes
    ----------
    gamma : float
        Discount factor for future rewards.
    learning_rate : float
        Learning rate for the Q-learning algorithm.
    epsilon : float
        Exploration rate for the epsilon-greedy policy.
    n_max_timesteps : int
        Maximum number of timesteps for training.
    n_max_timesteps_per_episode : int
        Maximum number of timesteps per episode.
    verbose : bool
        If True, enables verbose output.
    writer : object, optional
        Optional writer object for logging.
    penalty: float
        Penalty value for the algorithm.
    """

    def __init__(
        self,
        gamma=0.9,
        learning_rate=0.1,
        epsilon=0.1,
        n_max_timesteps=10000,
        n_max_timesteps_per_episode=100,
        n_episodes=1000,
        penalty=-100,
        verbose=False,
        writer=None,
    ):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.n_max_timesteps = n_max_timesteps
        self.n_max_timesteps_per_episode = n_max_timesteps_per_episode
        self.n_episodes = n_episodes
        self.verbose = verbose
        self.writer = writer
        self.penalty = penalty

        self.Q_s_a = None
        self.V_s = None
        self.policy = None
        self.N_s_a = None

    def fit(self, env):
        self.Q_s_a = np.zeros((env.n_states, env.n_actions))
        self.N_s_a = np.zeros((env.n_states, env.n_actions))
        cumulative_goals_achieved = 0
        total_timesteps = 0
        # self.Q_s_a[env.g, :] = 1.0

        while total_timesteps < self.n_max_timesteps:
            state, _ = env.reset()
            sum_rewards_per_episode = 0
            sum_discounted_reward_per_episode = 0
            # env.render()
            for t in range(self.n_max_timesteps):
                # action = np.random.choice(env.n_actions)
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(env.n_actions)
                else:
                    action = self.Q_s_a[state].argmax()

                next_state, reward, done, _ = env.step(action)
                self.N_s_a[state, action] += 1
                if state == env.g:
                    cumulative_goals_achieved += 1

                V_state = max(self.penalty, self.Q_s_a[state].max())
                V_next_state = max(self.penalty, self.Q_s_a[next_state].max())
                self.Q_s_a[state, action] = max(
                    self.penalty,
                    self.Q_s_a[state, action]
                    + self.learning_rate
                    * (reward + self.gamma * V_next_state - self.Q_s_a[state, action]),
                )

                sum_rewards_per_episode += reward
                sum_discounted_reward_per_episode += self.gamma * reward

                V_init = max(self.penalty, self.Q_s_a[env.s0].max())
                V_dead_end = self.Q_s_a[env.dead_end_states[0]].max()
                self.V_s = np.clip(self.Q_s_a, self.penalty, None).max(axis=1)
                self.policy = np.clip(self.Q_s_a, self.penalty, None).argmax(axis=1)

                state = next_state
                total_timesteps += 1

                if done:
                    break

            self.n_episodes += 1


class GoalProbabilityCostIteration:
    """
    Goal Probability Cost Iteration algorithm implementation.

    Attributes
    ----------
    gamma : float
        Discount factor for future rewards.
    theta : float
        Threshold for convergence.
    n_max_iter : int
        Maximum number iterations
    penalty: float
        Penalty value for the algorithm.
    verbose : bool
        If True, enables verbose output.
    writer : object, optional
        Optional writer object for logging.
    """

    def __init__(
        self, gamma=0.9, theta=0.01, n_max_iter=1000, verbose=False, writer=None
    ):
        self.gamma = gamma
        self.Q_s_a = None
        self.V_s = None
        self.policy = None
        self.theta = theta
        self.n_max_iter = n_max_iter
        self.verbose = verbose
        self.writer = writer

        if self.verbose:
            print(self.__dict__)

    def fit_goal_prob(self, env):
        states = list(range(env.n_states))
        actions = list(range(env.n_actions))
        goal_state = env.g
        P_s_a = np.zeros((env.n_states, env.n_actions))
        P_s_a[goal_state, :] = 1.0  # goal state value is 0

        for i in tqdm(range(self.n_max_iter)):
            delta = 0.0

            for state in states:
                if state == goal_state:
                    continue

                V_state_before_update = P_s_a[state].max()

                for action in actions:
                    P_s_a_sum = 0.0

                    for next_state in states:
                        V_next_state = P_s_a[next_state].max()
                        probability = env.P[state, action, next_state]

                        P_s_a_sum += probability * V_next_state

                    P_s_a[state, action] = P_s_a_sum

                V_state_after_update = P_s_a[state].max()

                V_diff = np.abs(V_state_after_update - V_state_before_update)
                delta = max(delta, V_diff)
                V_init = P_s_a[env.s0].max()
                P_s = P_s_a.max(axis=1)
                policy = P_s_a.argmax(axis=1)

            if delta < self.theta:
                break

        return P_s, P_s_a, policy

    def fit(self, env_maxprob, env):
        P_s, P_s_a, policy_mp = self.fit_goal_prob(env_maxprob)
        self.P_s = P_s
        self.P_s_a = P_s_a
        self.algo_maxprob = type("test", (object,), {})()
        self.algo_maxprob.Q_s_a = P_s_a
        self.algo_maxprob.V_s = P_s
        self.algo_maxprob.policy = policy_mp

        states = list(range(env.n_states))
        actions = list(range(env.n_actions))
        goal_state = env.g
        self.Q_s_a = np.zeros((env.n_states, env.n_actions))

        for i in range(self.n_max_iter):
            delta = 0.0
            Q_s_a_before_update = self.Q_s_a.copy()

            for s in states:
                V_state_before_update = self.Q_s_a[s].max()

                best_actions = np.flatnonzero(self.P_s_a[s] == self.P_s[s])
                if self.P_s[s] == 0:
                    self.Q_s_a[s, :] = 0
                else:
                    for a in best_actions:
                        denom = P_s[s]
                        num = 0
                        next_states = np.flatnonzero(env.P[s, a])

                        for s_next in next_states:
                            best_actions_next = np.flatnonzero(
                                P_s_a[s_next] == P_s_a[s_next].max()
                            )
                            num += (
                                env.P[s, a, s_next]
                                * P_s[s_next]
                                * (
                                    env.reward_fun(s, a, s_next)
                                    + self.Q_s_a[s_next, best_actions_next].max()
                                )
                            )

                        self.Q_s_a[s, a] = num / denom

                V_state_after_update = self.Q_s_a[s].max()

                V_diff = np.abs(V_state_after_update - V_state_before_update)
                delta = max(delta, V_diff)
                V_init = self.Q_s_a[env.s0].max()
                maxprob_mask = self.P_s_a == self.P_s.reshape(-1, 1)
                Q_s_a_masked = np.where(maxprob_mask, self.Q_s_a, -np.inf)
                self.V_s = Q_s_a_masked.max(axis=1)
                self.policy = Q_s_a_masked.argmax(axis=1)

            converged = np.allclose(self.Q_s_a, Q_s_a_before_update)

            if converged:
                break

            maxprob_mask = self.P_s_a == self.P_s.reshape(-1, 1)
            Q_s_a_masked = np.where(maxprob_mask, self.Q_s_a, -np.inf)
            self.V_s = Q_s_a_masked.max(axis=1)
            self.policy = Q_s_a_masked.argmax(axis=1)
