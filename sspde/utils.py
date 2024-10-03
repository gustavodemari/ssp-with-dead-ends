import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.transforms as transforms
import numpy as np


def draw_heatmap(data):
    fig = plt.figure(figsize=(6, 6))
    ax = sns.heatmap(
        data,
        fmt=".2f",
        cmap="coolwarm_r",
        cbar=False,
        square=True,
        linewidths=0.5,
        linecolor="black",
        annot=True,
    )
    ax.set_title("Value Function")
    fig.add_subplot(ax)
    return fig


def draw_heatmap_with_de(data, dead_end_data):
    fig, axs = plt.subplots(1, 2)
    sns.heatmap(
        data,
        fmt=".2f",
        cmap="coolwarm_r",
        cbar=False,
        square=True,
        linewidths=0.5,
        linecolor="black",
        annot=True,
        ax=axs[0],
    )

    sns.heatmap(
        dead_end_data,
        fmt=".2f",
        cmap="coolwarm_r",
        cbar=False,
        square=True,
        linewidths=0.5,
        linecolor="black",
        annot=True,
        ax=axs[1],
    )

    return fig


def draw_state_counts(data):
    fig = plt.figure(figsize=(4 * data.shape[0], 2 * data.shape[1]))
    ax = sns.heatmap(
        data,
        fmt="d",
        cmap="coolwarm_r",
        cbar=False,
        square=True,
        linewidths=0.5,
        linecolor="black",
        annot=True,
        vmin=0,
    )

    ax.set_title("State counts")
    fig.add_subplot(ax)
    return fig


def draw_state_counts_with_de(data, dead_end_data):
    N_x = data.shape[0]
    N_y = data.shape[1]
    fig, axs = plt.subplots(1, 2, figsize=(2 * N_y, 2 * N_x))
    sns.heatmap(
        data,
        fmt="d",
        cmap="coolwarm_r",
        cbar=False,
        square=True,
        linewidths=0.5,
        linecolor="black",
        annot=True,
        ax=axs[0],
        vmin=0,
    )

    sns.heatmap(
        dead_end_data,
        fmt="d",
        cmap="coolwarm_r",
        cbar=False,
        square=True,
        linewidths=0.5,
        linecolor="black",
        annot=True,
        ax=axs[1],
        vmin=0,
    )
    fig.tight_layout()

    return fig


def draw_policy_value_plot(policy, value_function):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Policy and Value Function")

    # Get the shape of the policy and value function
    policy_shape = policy.shape
    value_shape = value_function.shape

    # Create the gridworld plot
    if policy_shape == value_shape:
        grid_shape = policy_shape
    else:
        raise ValueError("Policy and value function shapes do not match.")

    action_dict = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0)}
    action_labels = {0: "↑", 1: "↓", 2: "→", 3: "←"}

    akws = {"ha": "left", "va": "top"}

    sns.heatmap(
        value_function,
        cmap="coolwarm_r",
        cbar=True,
        linewidths=0.5,
        linecolor="black",
        ax=ax,
        square=True,
        annot=True,
        fmt=".2f",
        # vmin=-np.max(np.abs(value_function)),
        # vmax=np.max(np.abs(value_function)),
        annot_kws=akws,
    )

    for t in ax.texts:
        trans = t.get_transform()
        offs = transforms.ScaledTranslation(
            -0.48, -0.48, transforms.IdentityTransform()
        )
        t.set_transform(offs + trans)

    # Add arrows for the policy and display the value function
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            action = policy[i, j]
            dx, dy = action_dict[action]
            ax.arrow(
                j + 0.5,
                i + 0.5,
                dx * 0.3,
                dy * 0.3,
                head_width=0.1,
                head_length=0.1,
                fc="black",
                ec="black",
            )
            # ax.text(
            #     j + 0.3 + dx * 0.0,
            #     i + 0.3 + dy * 0.0,
            #     f"{value_function[i, j]:.2f}",
            #     ha="center",
            #     va="center",
            #     color="k",
            # )

    # plt.show()
    return fig


def draw_policy_value_plot_with_de(policy, value_function, dead_end_value_function):
    # Get the shape of the policy and value function
    policy_shape = policy.shape
    value_shape = value_function.shape

    # fig, ax = plt.subplots(figsize=(6, 6))
    N_x = value_function.shape[0]
    N_y = value_function.shape[1]
    fig, axs = plt.subplots(1, 2, figsize=(2 * N_y, 2 * N_x))
    axs[0].set_title("Policy and Value Function")
    axs[1].set_title("Dead End Value Function")

    # Create the gridworld plot
    if policy_shape == value_shape:
        grid_shape = policy_shape
    else:
        raise ValueError("Policy and value function shapes do not match.")

    action_dict = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0)}
    action_labels = {0: "↑", 1: "↓", 2: "→", 3: "←"}

    akws = {"ha": "left", "va": "top"}

    sns.heatmap(
        value_function,
        cmap="coolwarm_r",
        cbar=False,
        linewidths=0.5,
        linecolor="black",
        ax=axs[0],
        square=True,
        annot=True,
        fmt=".2f",
        # vmin=-np.max(np.abs(value_function)),
        # vmax=np.max(np.abs(value_function)),
        annot_kws=akws,
    )

    for t in axs[0].texts:
        trans = t.get_transform()
        offs = transforms.ScaledTranslation(
            -0.48, -0.48, transforms.IdentityTransform()
        )
        t.set_transform(offs + trans)

    sns.heatmap(
        dead_end_value_function,
        cmap="coolwarm_r",
        cbar=False,
        linewidths=0.5,
        linecolor="black",
        ax=axs[1],
        square=True,
        annot=True,
        fmt=".2f",
        # vmin=-np.max(np.abs(value_function)),
        # vmax=np.max(np.abs(value_function)),
        annot_kws=akws,
    )

    # plt.colorbar()

    # Add arrows for the policy and display the value function
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            action = policy[i, j]
            if action < 4:
                dx, dy = action_dict[action]

                axs[0].arrow(
                    j + 0.5,
                    i + 0.5,
                    dx * 0.3,
                    dy * 0.3,
                    head_width=0.1,
                    head_length=0.1,
                    fc="black",
                    ec="black",
                )
            # ax.text(
            #     j + 0.3 + dx * 0.0,
            #     i + 0.3 + dy * 0.0,
            #     f"{value_function[i, j]:.2f}",
            #     ha="center",
            #     va="center",
            #     color="k",
            # )

    # plt.show()
    plt.tight_layout()
    return fig


def draw_transition_matrix(env):
    states = [f"S{i}" for i in range(env.n_states)]
    actions = [f"A{j}" for j in range(env.n_actions)]

    # Plot heatmaps for each action
    fig, axes = plt.subplots(1, len(actions), figsize=(20, 5))

    for i, action in enumerate(actions):
        ax = axes[i]
        sns.heatmap(
            env.P[:, i, :],
            annot=True,
            cmap="YlGnBu",
            cbar=True,
            xticklabels=states,
            yticklabels=states,
            ax=ax,
        )
        ax.set_title(f"Transition Matrix for Action {action}")
        ax.set_xlabel("Next State")
        ax.set_ylabel("Current State")

    plt.tight_layout()
    plt.show()


def draw_navigation(env):
    fig = plt.figure(figsize=(2 * env.N_x, 2 * env.N_y))
    dead_ends_mat = np.zeros((env.N_x, env.N_y))

    ax = sns.heatmap(
        dead_ends_mat,
        fmt=".2f",
        cmap="coolwarm",
        cbar=False,
        square=True,
        linewidths=0.5,
        linecolor="black",
        annot=True,
        vmin=0,
        vmax=1,
    )
    # ax.set_title("Dead ends probability map")
    fig.add_subplot(ax)
    return fig


def draw_navigation_with_ade(env):
    fig = plt.figure(figsize=(2 * env.N_x, 2 * env.N_y))
    dead_ends_mat = np.zeros((env.N_x, env.N_y))
    dead_ends_mat[1, [1, 2]] = 0.5
    ax = sns.heatmap(
        dead_ends_mat,
        fmt=".2f",
        cmap="coolwarm",
        cbar=False,
        square=True,
        linewidths=0.5,
        linecolor="black",
        annot=True,
        vmin=0,
        vmax=1,
    )
    # ax.set_title("Dead ends probability map")
    fig.add_subplot(ax)
    return fig


def draw_navigation_with_ude(env):
    fig = plt.figure(figsize=(2 * env.N_x, 2 * env.N_y))
    dead_end_state = env.dead_end_states[0]

    action_choice = 0  # any action goes to dead end, so this could be any action
    dead_ends_mat = env.P[:, action_choice, dead_end_state][
        :dead_end_state
    ]  # all states minus the dead-end state
    # dead_ends_mat = dead_ends_mat.reshape((env.N_x, env.N_y))
    dead_ends_mat = dead_ends_mat.reshape((env.N_y, env.N_x))
    # dead_ends_mat[1, [0, 1,2]] = 0.5
    ax = sns.heatmap(
        dead_ends_mat,
        fmt=".2f",
        cmap="coolwarm",
        cbar=False,
        square=True,
        linewidths=0.5,
        linecolor="black",
        annot=True,
        vmin=0,
        vmax=1,
        annot_kws={"size": 20},
    )
    # ax.set_title("Dead ends probability map")
    fig.add_subplot(ax)
    return fig


def plot_navigation_env(env):
    env_name = env.__class__.__name__.lower()

    match env_name:
        case env_name if "ade" in env_name:
            fig = draw_navigation_with_ade(env)
        case env_name if "ude" in env_name:
            fig = draw_navigation_with_ude(env)
        case _:
            fig = draw_navigation(env)

    return fig


def plot_policy_value_plot(env, policy, value_function):
    gridsize = env.N_x * env.N_y
    if gridsize == env.n_states:
        policy = policy.reshape(env.N_y, env.N_x)
        value_function = value_function.reshape(env.N_y, env.N_x)
        fig = draw_policy_value_plot(policy, value_function)
    elif gridsize == env.n_states - 1:
        policy = policy[:-1].reshape(env.N_y, env.N_x)
        dead_end_value_function = value_function[-1].reshape(-1, 1)
        value_function = value_function[:-1].reshape(env.N_y, env.N_x)
        fig = draw_policy_value_plot_with_de(
            policy, value_function, dead_end_value_function
        )

    return fig


def plot_state_counts(env, state_counts):
    gridsize = env.N_x * env.N_y
    if gridsize == env.n_states:
        state_counts = state_counts.reshape(env.N_y, env.N_x)
        fig = draw_state_counts(state_counts)
    elif gridsize == env.n_states - 1:
        state_counts_de = state_counts[-1].reshape(-1, 1)
        state_counts = state_counts[:-1].reshape(env.N_y, env.N_x)
        fig = draw_state_counts_with_de(state_counts, state_counts_de)

    return fig


def plot_q_values(env, Q_s_a):
    gridsize = env.N_x * env.N_y
    if gridsize == env.n_states:
        Q_s_a = Q_s_a.reshape(env.N_y, env.N_x, -1)
    elif gridsize == env.n_states - 1:
        Q_s_a = Q_s_a[:-1].reshape(env.N_y, env.N_x, -1)

    fig, ax = plt.subplots(figsize=(2 * env.N_x, 2 * env.N_y))
    ax.set_xlim(-0.5, env.N_x - 0.5)
    ax.set_ylim(-0.5, env.N_y - 0.5)
    ax.set_xticks(np.arange(-0.5, env.N_x, 1))
    ax.set_yticks(np.arange(-0.5, env.N_y, 1))
    ax.grid(True)

    action_directions = {
        0: (0, -0.1),  # UP
        1: (0, 0.1),  # DOWN
        2: (0.1, 0),  # RIGHT
        3: (-0.1, 0),  # LEFT
    }

    # Plot arrows and annotate with Q-values
    for i in range(Q_s_a.shape[0]):
        for j in range(Q_s_a.shape[1]):
            values = Q_s_a[i, j]
            for action, q_value in enumerate(values):
                if action < 4:
                    direction = action_directions[action]
                    ax.arrow(
                        j + direction[0] * 0.5,
                        i + direction[1] * 0.5,
                        dx=direction[0] * 0.5,
                        dy=direction[1] * 0.5,
                        head_width=0.05,
                        head_length=0.05,
                        fc="black",
                        ec="black",
                    )
                    ax.text(
                        j + direction[0] * 3.0,
                        i + direction[1] * 3.0,
                        f"{q_value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=12,
                    )

    # Set labels and title
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.gca().invert_yaxis()

    return fig
