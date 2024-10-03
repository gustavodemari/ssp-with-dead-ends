import streamlit as st
import numpy as np

from sspde.envs import NavigationfSSPUDE, NavigationfSSPUDEtoMAXPROB
from sspde.utils import plot_navigation_env, plot_q_values
from sspde.algorithms import (
    ValueIterationFinitePenalty,
    GoalProbabilityCostIteration,
    QLearningFinitePenalty,
    QLearningMinCMaxP,
)


# Main Streamlit app
def main():
    st.title("SSP with dead-ends")

    # Display the environment
    st.subheader("Environment")
    st.sidebar.subheader("Environment parameters")

    N_x = st.sidebar.slider("X coord size", 3, 20)
    N_y = st.sidebar.slider("Y coord size", 3, 9)

    if st.sidebar.button(f"Create Env with size {N_x}x{N_y}", type="primary"):
        st.session_state.env_mp = NavigationfSSPUDEtoMAXPROB(N_x=N_x, N_y=N_y)
        st.session_state.env = NavigationfSSPUDE(N_x=N_x, N_y=N_y)
        st.session_state.fig = plot_navigation_env(st.session_state.env)

        st.pyplot(st.session_state.fig)

    st.sidebar.subheader("Algorithm parameters")

    algorithm = st.sidebar.selectbox(
        "Choose algorithm", ["vi-fp", "gpci", "q-learning-fp", "q-learning-minc-maxp"]
    )

    match algorithm:
        case "vi-fp":
            penalty = st.sidebar.slider("penalty", 1, 200, value=100)
            gamma = st.sidebar.slider("gamma", 0.1, 1.0, value=1.0)
            n_max_iter = st.sidebar.slider("n_max_iter", 100, 1000, value=1000)
            algo = ValueIterationFinitePenalty(
                gamma=gamma, n_max_iter=n_max_iter, penalty=-penalty
            )
        case "gpci":
            gamma = st.sidebar.slider("gamma", 0.1, 1.0, value=1.0)
            n_max_iter = st.sidebar.slider("n_max_iter", 100, 1000, value=1000)
            algo = GoalProbabilityCostIteration(gamma=gamma, n_max_iter=n_max_iter)
        case "q-learning-fp":
            penalty = st.sidebar.slider("penalty", 1, 200, value=100)
            epsilon = st.sidebar.slider("epsilon", 0.1, 1.0, value=0.3)
            gamma = st.sidebar.slider("gamma", 0.1, 1.0, value=1.0)
            learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.9, value=0.1)
            n_max_timesteps = st.sidebar.slider(
                "n_max_timesteps", 100000, 1000000, value=300000
            )
            algo = QLearningFinitePenalty(
                gamma=gamma,
                learning_rate=learning_rate,
                n_max_timesteps=n_max_timesteps,
                penalty=-penalty,
                epsilon=epsilon,
            )
        case "q-learning-minc-maxp":
            gamma = st.sidebar.slider("gamma", 0.1, 1.0, value=1.0)
            learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.9, value=0.1)
            n_max_timesteps = st.sidebar.slider(
                "n_max_timesteps", 100000, 1000000, value=300000
            )
            algo = QLearningMinCMaxP(
                learning_rate=learning_rate,
                n_max_timesteps=n_max_timesteps,
            )

    is_fp_algorithm = "fp" in algorithm

    st.subheader("Solution")

    if st.sidebar.button(
        f"Solve the grid {N_x}x{N_y} using {algorithm}", type="primary"
    ):
        with st.spinner("Wait for it..."):
            if is_fp_algorithm:
                algo.fit(st.session_state.env)
            else:
                algo.fit(st.session_state.env_mp, st.session_state.env)

        q_values_plot = plot_q_values(st.session_state.env, algo.Q_s_a * -1)

        st.pyplot(st.session_state.fig)
        print(algo.Q_s_a)

        if not is_fp_algorithm:
            q_values_mp_plot = plot_q_values(
                st.session_state.env, algo.algo_maxprob.Q_s_a
            )
            st.subheader("Solution - MAXPROB")
            st.pyplot(q_values_mp_plot)

        st.subheader("Solution - MinCost")
        st.pyplot(q_values_plot)


if __name__ == "__main__":
    np.random.seed(0)
    main()
