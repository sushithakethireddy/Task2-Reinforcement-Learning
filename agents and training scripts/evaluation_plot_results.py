"""
Evaluation and Visualisation for Chef's Hat Partial Observability Experiments.

Student ID: 16387858
Variant: Partial Observability (ID mod 7 = 4)

"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def smooth(data, window=20):
    if len(data) < window:
        return np.array(data)
    kernel = np.ones(window) / float(window)
    return np.convolve(data, kernel, mode="valid")


def load_metrics(exp_name):
    """
    Load training stats from results/<exp_name>/<exp_name>_stats.json
    This matches where training_train.py actually saves them.
    """
    path = os.path.join(RESULTS_DIR, exp_name, f"{exp_name}_stats.json")
    if not os.path.exists(path):
        print(f"WARNING: Not found: {path}")
        return None
    with open(path, "r") as f:
        data = json.load(f)

    # Normalise keys: training saves 'episode_scores', plots expect 'episode_rewards'
    scores = data.get("episode_scores", data.get("episode_rewards", []))
    # Derive win_rates: a score > 0 counts as a win (1.0), else 0
    win_rates = [1.0 if s > 0 else 0.0 for s in scores]

    return {
        "episode_rewards": scores,
        "win_rates": win_rates,
        "actor_losses": data.get("actor_losses", []),
        "critic_losses": data.get("critic_losses", []),
        "entropies": data.get("entropies", []),
        "episode_lengths": data.get("episode_lengths", [])
    }


def set_plot_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "lines.linewidth": 1.8
    })


# FIGURE 1 – Learning curves by observability level
def plot_exp1_learning_curves():
    exp_names = [
        "exp1_ppo_lstm_level1",
        "exp1_ppo_lstm_level2",
        "exp1_ppo_lstm_level3",
        "exp1_ppo_lstm_level4"
    ]
    labels = [
        "Level 1 - Full Observation",
        "Level 2 - Opponent Hands Hidden",
        "Level 3 - Hands and History Hidden",
        "Level 4 - Minimal Observation"
    ]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    fig, ax = plt.subplots(figsize=(10, 5))
    found_any = False

    for exp_name, label, color in zip(exp_names, labels, colors):
        metrics = load_metrics(exp_name)
        if metrics is None or len(metrics["episode_rewards"]) == 0:
            continue
        rewards = smooth(metrics["episode_rewards"], window=30)
        ax.plot(rewards, label=label, color=color)
        found_any = True

    if not found_any:
        ax.text(0.5, 0.5, "No data found. Run training first.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="gray")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Smoothed Episode Reward")
    ax.set_title("Learning Curves by Observability Level\n"
                 "(PPO-LSTM Agent, Smoothing Window=30)")
    ax.legend(loc="lower right")
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "fig1_exp1_learning_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# FIGURE 2 – Final win rate bar chart

def plot_exp1_win_rate_bar():
    exp_names = [
        "exp1_ppo_lstm_level1",
        "exp1_ppo_lstm_level2",
        "exp1_ppo_lstm_level3",
        "exp1_ppo_lstm_level4"
    ]
    labels = ["Full Obs\n(L1)", "Opp Hands\nHidden (L2)",
              "Hands+Hist\nHidden (L3)", "Minimal\nObs (L4)"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    win_rates = []
    for exp_name in exp_names:
        metrics = load_metrics(exp_name)
        if metrics is not None and len(metrics["win_rates"]) > 0:
            win_rates.append(float(np.mean(metrics["win_rates"][-50:])))
        else:
            win_rates.append(0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, win_rates, color=colors, edgecolor="black",
                  linewidth=0.7, alpha=0.85)

    for bar, val in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, max(win_rates) * 1.4 + 0.05 if max(win_rates) > 0 else 1.0)
    ax.set_ylabel("Final Win Rate (Last 50 Episodes Avg)")
    ax.set_title("Final Win Rate by Observability Level\n(PPO-LSTM Agent)")
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "fig2_exp1_win_rate_bar.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

# FIGURE 3 – PPO vs PPO-LSTM memory comparison
def plot_exp2_memory_comparison():
    exp_configs = [
        ("exp2_PPO_level2",      "PPO - No Memory",        "#F44336", "--"),
        ("exp2_PPO_LSTM_level2", "PPO-LSTM - With Memory", "#2196F3", "-")
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for exp_name, label, color, ls in exp_configs:
        metrics = load_metrics(exp_name)
        if metrics is None or len(metrics["episode_rewards"]) == 0:
            continue
        rewards = smooth(metrics["episode_rewards"], window=30)
        axes[0].plot(rewards, label=label, color=color, linestyle=ls)

        win_rates = smooth(metrics["win_rates"], window=20)
        axes[1].plot(win_rates, label=label, color=color, linestyle=ls)

    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Smoothed Episode Reward")
    axes[0].set_title("Episode Reward: PPO vs PPO-LSTM\n(Level 2 Partial Obs)")
    axes[0].legend()

    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Rolling Win Rate (smoothed)")
    axes[1].set_title("Win Rate: PPO vs PPO-LSTM\n(Level 2 Partial Obs)")
    axes[1].legend()

    plt.suptitle("Effect of LSTM Memory Under Partial Observability",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "fig3_exp2_memory_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# FIGURE 4 – Seed robustness
def plot_exp3_seed_robustness():
    seed_names  = ["exp3_ppo_lstm_seed42", "exp3_ppo_lstm_seed123", "exp3_ppo_lstm_seed456"]
    seed_labels = ["Seed 42", "Seed 123", "Seed 456"]
    seed_colors = ["#9C27B0", "#673AB7", "#3F51B5"]

    fig, ax = plt.subplots(figsize=(10, 5))
    all_rewards = []
    min_len = None

    for exp_name, label, color in zip(seed_names, seed_labels, seed_colors):
        metrics = load_metrics(exp_name)
        if metrics is None or len(metrics["episode_rewards"]) == 0:
            continue
        rewards = smooth(metrics["episode_rewards"], window=30)
        ax.plot(rewards, label=label, color=color, alpha=0.6, linewidth=1.2)
        all_rewards.append(rewards)
        if min_len is None or len(rewards) < min_len:
            min_len = len(rewards)

    if len(all_rewards) > 1 and min_len:
        truncated = np.array([r[:min_len] for r in all_rewards])
        mean_r = np.mean(truncated, axis=0)
        std_r  = np.std(truncated, axis=0)
        x = np.arange(min_len)
        ax.plot(x, mean_r, color="black", linewidth=2.5, label="Mean (3 seeds)")
        ax.fill_between(x, mean_r - std_r, mean_r + std_r,
                        alpha=0.2, color="black", label="Mean ± 1 Std")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Smoothed Episode Reward")
    ax.set_title("PPO-LSTM Robustness Across 3 Random Seeds\n"
                 "(Level 2 Partial Observability)")
    ax.legend()
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "fig4_exp3_seed_robustness.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# FIGURE 5 – Hyperparameter sensitivity
def plot_exp4_hyperparameter_sensitivity():
    configs = [
        ("exp4_ppo_lstm_lr1e3_h128", "LR=1e-3, H=128"),
        ("exp4_ppo_lstm_lr3e4_h128", "LR=3e-4, H=128"),
        ("exp4_ppo_lstm_lr1e4_h128", "LR=1e-4, H=128"),
        ("exp4_ppo_lstm_lr3e4_h64",  "LR=3e-4, H=64"),
        ("exp4_ppo_lstm_lr3e4_h256", "LR=3e-4, H=256"),
    ]
    labels    = [c[1] for c in configs]
    win_rates = []

    for exp_name, _ in configs:
        metrics = load_metrics(exp_name)
        if metrics is not None and len(metrics["win_rates"]) > 0:
            win_rates.append(float(np.mean(metrics["win_rates"][-50:])))
        else:
            win_rates.append(0.0)

    colors = ["#FF5722", "#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, win_rates, color=colors, edgecolor="black",
                  linewidth=0.7, alpha=0.85)

    for bar, val in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(0, max(win_rates) * 1.4 + 0.05 if max(win_rates) > 0 else 1.0)
    ax.set_ylabel("Final Win Rate (Last 50 Episodes Avg)")
    ax.set_title("Hyperparameter Sensitivity Analysis\n"
                 "(PPO-LSTM, Level 2 Partial Observability)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "fig5_exp4_hyperparams.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

# FIGURE 6 – Policy entropy
def plot_entropy_analysis():
    exp_configs = [
        ("exp2_PPO_level2",      "PPO",      "#F44336"),
        ("exp2_PPO_LSTM_level2", "PPO-LSTM", "#2196F3")
    ]

    fig, ax = plt.subplots(figsize=(9, 4))
    found = False

    for exp_name, label, color in exp_configs:
        metrics = load_metrics(exp_name)
        if metrics is None or len(metrics.get("entropies", [])) == 0:
            continue
        entropies = smooth(metrics["entropies"], window=5)
        ax.plot(entropies, label=label, color=color)
        found = True

    if not found:
        ax.text(0.5, 0.5, "No entropy data found.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="gray")

    ax.set_xlabel("Update Step")
    ax.set_ylabel("Policy Entropy")
    ax.set_title("Policy Entropy During Training\n"
                 "(Higher = More Exploration, Level 2 Partial Obs)")
    ax.legend()
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "fig6_entropy_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

# FIGURE 7 – Episode lengths

def plot_episode_lengths():
    """
    Plot Actor and Critic losses over training updates for PPO and PPO-LSTM.
    Lower and stable loss = better convergence.
    """
    exp_configs = [
        ("exp2_PPO_level2",      "PPO",      "#F44336", "--"),
        ("exp2_PPO_LSTM_level2", "PPO-LSTM", "#2196F3", "-")
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    found = False

    for exp_name, label, color, ls in exp_configs:
        metrics = load_metrics(exp_name)
        if metrics is None:
            continue

        actor_losses  = metrics.get("actor_losses", [])
        critic_losses = metrics.get("critic_losses", [])

        if len(actor_losses) > 1:
            smoothed_actor  = smooth(actor_losses,  window=min(5, len(actor_losses)))
            axes[0].plot(smoothed_actor,  label=label, color=color, linestyle=ls)
            found = True

        if len(critic_losses) > 1:
            smoothed_critic = smooth(critic_losses, window=min(5, len(critic_losses)))
            axes[1].plot(smoothed_critic, label=label, color=color, linestyle=ls)

    if not found:
        for ax in axes:
            ax.text(0.5, 0.5, "No loss data found.",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12, color="gray")

    axes[0].set_xlabel("Update Step")
    axes[0].set_ylabel("Actor Loss")
    axes[0].set_title("Actor Loss: PPO vs PPO-LSTM\n(Level 2 Partial Obs)")
    axes[0].legend()

    axes[1].set_xlabel("Update Step")
    axes[1].set_ylabel("Critic Loss")
    axes[1].set_title("Critic Loss: PPO vs PPO-LSTM\n(Level 2 Partial Obs)")
    axes[1].legend()

    plt.suptitle("Actor and Critic Loss During Training\n"
                 "(Lower and Stable = Better Convergence)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "fig7_episode_lengths.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# FIGURE 8 – Summary 2x2
def plot_summary_figure():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Chef's Hat RL - Partial Observability Variant Summary\n"
        "Student ID: 16387858 | Variant: ID mod 7 = 4",
        fontsize=13, fontweight="bold"
    )

    # Top-left: Observability level learning curves
    ax = axes[0][0]
    colors_4 = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    labels_4 = ["L1-Full", "L2-Opp Hidden", "L3-Hist Hidden", "L4-Minimal"]
    for level, color, label in zip([1, 2, 3, 4], colors_4, labels_4):
        metrics = load_metrics(f"exp1_ppo_lstm_level{level}")
        if metrics and len(metrics["episode_rewards"]) > 0:
            r = smooth(metrics["episode_rewards"], window=30)
            ax.plot(r, color=color, label=label)
    ax.set_title("Exp 1: Observability Level Effect")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Smoothed Reward")
    ax.legend(fontsize=8)

    # Top-right: PPO vs LSTM
    ax = axes[0][1]
    for exp_name, label, color, ls in [
        ("exp2_PPO_level2",      "PPO",      "#F44336", "--"),
        ("exp2_PPO_LSTM_level2", "PPO-LSTM", "#2196F3", "-")
    ]:
        metrics = load_metrics(exp_name)
        if metrics and len(metrics["episode_rewards"]) > 0:
            r = smooth(metrics["episode_rewards"], window=30)
            ax.plot(r, label=label, color=color, linestyle=ls)
    ax.set_title("Exp 2: Memory vs No Memory")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Smoothed Reward")
    ax.legend()

    # Bottom-left: Seed robustness
    ax = axes[1][0]
    seed_rewards = []
    seed_min_len = None
    for seed, color in zip([42, 123, 456], ["#9C27B0", "#673AB7", "#3F51B5"]):
        metrics = load_metrics(f"exp3_ppo_lstm_seed{seed}")
        if metrics and len(metrics["episode_rewards"]) > 0:
            r = smooth(metrics["episode_rewards"], window=30)
            ax.plot(r, color=color, alpha=0.6, label=f"Seed {seed}")
            seed_rewards.append(r)
            if seed_min_len is None or len(r) < seed_min_len:
                seed_min_len = len(r)
    if len(seed_rewards) > 1 and seed_min_len:
        trunc  = np.array([r[:seed_min_len] for r in seed_rewards])
        mean_r = np.mean(trunc, axis=0)
        ax.plot(np.arange(seed_min_len), mean_r, color="black",
                linewidth=2, label="Mean")
    ax.set_title("Exp 3: Seed Robustness")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Smoothed Reward")
    ax.legend(fontsize=8)

    # Bottom-right: Win rate bar
    ax = axes[1][1]
    wr_vals   = []
    wr_labels = []
    wr_colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    for level in [1, 2, 3, 4]:
        metrics = load_metrics(f"exp1_ppo_lstm_level{level}")
        if metrics and len(metrics["win_rates"]) > 0:
            wr_vals.append(float(np.mean(metrics["win_rates"][-50:])))
        else:
            wr_vals.append(0.0)
        wr_labels.append(f"L{level}")
    bars = ax.bar(wr_labels, wr_vals, color=wr_colors, edgecolor="black",
                  linewidth=0.7, alpha=0.85)
    for bar, val in zip(bars, wr_vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.005,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("Final Win Rate by Obs Level")
    ax.set_ylabel("Win Rate")
    ax.set_xlabel("Observability Level")

    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "fig8_summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# Summary table
def print_metrics_table():
    print("\nRESULTS SUMMARY TABLE")
    
    print(f"{'Experiment':<40} {'Final Reward':>12} {'Win Rate':>12}")
 

    all_exps = [
        ("exp1_ppo_lstm_level1",  "Exp1 Level1 Full Obs"),
        ("exp1_ppo_lstm_level2",  "Exp1 Level2 Opp Hidden"),
        ("exp1_ppo_lstm_level3",  "Exp1 Level3 Hist Hidden"),
        ("exp1_ppo_lstm_level4",  "Exp1 Level4 Minimal"),
        ("exp2_PPO_level2",       "Exp2 PPO (no memory)"),
        ("exp2_PPO_LSTM_level2",  "Exp2 PPO-LSTM (memory)"),
        ("exp3_ppo_lstm_seed42",  "Exp3 Seed 42"),
        ("exp3_ppo_lstm_seed123", "Exp3 Seed 123"),
        ("exp3_ppo_lstm_seed456", "Exp3 Seed 456"),
    ]

    for exp_name, display_name in all_exps:
        metrics = load_metrics(exp_name)
        if metrics is None:
            print(f"{'  ' + display_name:<40} {'N/A':>12} {'N/A':>12}")
            continue
        rewards = metrics["episode_rewards"]
        win_rates = metrics["win_rates"]
        avg_reward = float(np.mean(rewards[-100:])) if len(rewards) >= 100 else float(np.mean(rewards)) if rewards else 0.0
        avg_wr     = float(np.mean(win_rates[-50:])) if len(win_rates) >= 50 else float(np.mean(win_rates)) if win_rates else 0.0
        print(f"{'  ' + display_name:<40} {avg_reward:>12.4f} {avg_wr:>12.3f}")

    print("=" * 70)


if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)
    set_plot_style()

    print("Generating all figures")
    plot_exp1_learning_curves()
    plot_exp1_win_rate_bar()
    plot_exp2_memory_comparison()
    plot_exp3_seed_robustness()
    plot_exp4_hyperparameter_sensitivity()
    plot_entropy_analysis()
    plot_episode_lengths()
    plot_summary_figure()
    print_metrics_table()

    print(f"\nAll figures saved to: {PLOTS_DIR}")