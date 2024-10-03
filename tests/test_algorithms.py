from pytest import approx

from sspde.algorithms import ValueIterationFinitePenalty, QLearningFinitePenalty
from sspde.envs import NavigationfSSPUDE

import numpy as np
SEED = 0
np.random.seed(SEED)

class TestAlgorithms:
    env = NavigationfSSPUDE(N_x=4, N_y=3)
    def test_vi_fp(self):
        vi_fp = ValueIterationFinitePenalty(gamma=1.0, n_max_iter=1000, penalty=-100)
        vi_fp.fit(self.env)
        assert vi_fp.V_s is not None
        assert vi_fp.V_s.shape == (self.env.n_states,)
        assert vi_fp.policy is not None
        assert vi_fp.Q_s_a[-2, 3] == approx(-17.7)

    def test_ql_fp(self):
        ql_fp = QLearningFinitePenalty(gamma=1.0, penalty=-100, n_max_timesteps=100000)
        ql_fp.fit(self.env)
        assert ql_fp.Q_s_a is not None
        assert ql_fp.Q_s_a.shape == (self.env.n_states, self.env.n_actions)


