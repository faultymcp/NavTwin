"""
Visualization Module — Generates publication-quality evaluation charts
for the NavTwin paper.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List
import os


def set_style():
    """Set publication-ready matplotlib style."""
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


COLORS = {
    'sarah': '#E74C3C',
    'alex': '#3498DB',
    'maya': '#2ECC71',
    'james': '#9B59B6',
    'baseline': '#95A5A6',
    'adaptive': '#E67E22',
}


def moving_average(data, window=15):
    """Compute moving average with given window."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_rl_convergence(results: Dict, output_dir: str):
    """
    Figure: RL weight adaptation convergence across scenarios.
    Shows how the RL agent learns to select appropriate weight
    configurations for different user types.
    """
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('RL Agent Weight Adaptation Convergence', fontsize=14, fontweight='bold')

    for idx, (key, result) in enumerate(results.items()):
        ax = axes[idx // 2][idx % 2]
        ts = result["time_series"]
        weights = np.array(ts["weights"])

        if len(weights) == 0:
            continue

        n = len(weights)
        x = np.arange(n)

        labels = ['$w_{PPS}$', '$w_{ECS}$', '$w_{SPS}$', '$w_{ES}$']
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6']

        for i in range(4):
            smoothed = moving_average(weights[:, i], window=10)
            ax.plot(np.arange(len(smoothed)), smoothed, color=colors[i],
                   label=labels[i], linewidth=1.5, alpha=0.8)

        ax.set_xlabel('Journey Number')
        ax.set_ylabel('Weight Value')
        ax.set_title(key.capitalize(), fontweight='bold')
        ax.set_ylim(-0.05, 0.65)
        ax.legend(loc='upper right', ncol=2, framealpha=0.8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'rl_convergence.png')
    plt.savefig(path)
    plt.close()
    return path


def plot_reward_and_stress(results: Dict, output_dir: str):
    """
    Figure: Reward signal and stress levels over time for all users.
    Shows the RL agent's learning progress and stress reduction.
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Rewards
    for key, result in results.items():
        ts = result["time_series"]
        rewards = moving_average(ts["rewards"], window=15)
        ax1.plot(np.arange(len(rewards)), rewards,
                label=key.capitalize(), color=COLORS[key], linewidth=1.5)

    ax1.set_xlabel('Journey Number')
    ax1.set_ylabel('Reward (moving avg)')
    ax1.set_title('Reward Signal Convergence', fontweight='bold')
    ax1.legend(framealpha=0.8)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Stress
    for key, result in results.items():
        ts = result["time_series"]
        stress = moving_average(ts["stress"], window=15)
        ax2.plot(np.arange(len(stress)), stress,
                label=key.capitalize(), color=COLORS[key], linewidth=1.5)

    ax2.set_xlabel('Journey Number')
    ax2.set_ylabel('Stress Level (moving avg)')
    ax2.set_title('User Stress Over Time', fontweight='bold')
    ax2.legend(framealpha=0.8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'reward_stress.png')
    plt.savefig(path)
    plt.close()
    return path


def plot_acceptance_completion(results: Dict, output_dir: str):
    """
    Figure: Acceptance and completion rates — first half vs second half.
    Demonstrates adaptation effectiveness.
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    users = list(results.keys())
    x = np.arange(len(users))
    width = 0.35

    # Acceptance rates
    first_half = [results[u]["metrics"]["first_half"]["acceptance_rate"] for u in users]
    second_half = [results[u]["metrics"]["second_half"]["acceptance_rate"] for u in users]

    bars1 = ax1.bar(x - width/2, first_half, width, label='First 50%',
                    color='#BDC3C7', edgecolor='white')
    bars2 = ax1.bar(x + width/2, second_half, width, label='Second 50%',
                    color='#3498DB', edgecolor='white')

    ax1.set_xlabel('User Scenario')
    ax1.set_ylabel('Acceptance Rate')
    ax1.set_title('Route Acceptance: Before vs After Adaptation', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([u.capitalize() for u in users])
    ax1.legend()
    ax1.set_ylim(0, 1.1)

    # Add value labels
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

    # Completion rates
    first_half_c = [results[u]["metrics"]["first_half"]["completion_rate"] for u in users]
    second_half_c = [results[u]["metrics"]["second_half"]["completion_rate"] for u in users]

    bars3 = ax2.bar(x - width/2, first_half_c, width, label='First 50%',
                    color='#BDC3C7', edgecolor='white')
    bars4 = ax2.bar(x + width/2, second_half_c, width, label='Second 50%',
                    color='#2ECC71', edgecolor='white')

    ax2.set_xlabel('User Scenario')
    ax2.set_ylabel('Completion Rate')
    ax2.set_title('Journey Completion: Before vs After Adaptation', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([u.capitalize() for u in users])
    ax2.legend()
    ax2.set_ylim(0, 1.1)

    for bar in bars3:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars4:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'acceptance_completion.png')
    plt.savefig(path)
    plt.close()
    return path


def plot_baseline_comparison(results: Dict, baseline_results: Dict, output_dir: str):
    """
    Figure: Adaptive RL vs Static Weights comparison.
    Key figure for demonstrating the value of the RL approach.
    """
    set_style()
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    metrics = ['avg_score', 'acceptance_rate', 'completion_rate', 'avg_stress']
    titles = ['Avg Route Score', 'Acceptance Rate', 'Completion Rate', 'Avg Stress']
    colors_pair = ['#3498DB', '#95A5A6']

    users = list(results.keys())

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        x = np.arange(len(users))
        width = 0.35

        adaptive_vals = []
        static_vals = []
        for u in users:
            if metric == 'avg_score':
                adaptive_vals.append(results[u]["metrics"]["overall"]["avg_score"])
            elif metric == 'avg_stress':
                adaptive_vals.append(results[u]["metrics"]["overall"]["avg_stress"])
            else:
                adaptive_vals.append(results[u]["metrics"]["overall"][metric])
            static_vals.append(baseline_results[u][metric])

        ax.bar(x - width/2, adaptive_vals, width, label='Adaptive (RL)',
               color=colors_pair[0], edgecolor='white')
        ax.bar(x + width/2, static_vals, width, label='Static',
               color=colors_pair[1], edgecolor='white')

        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([u[0].upper() for u in users])  # First letter
        ax.legend(fontsize=8)

        if metric == 'avg_stress':
            ax.set_ylim(0, 0.8)
        else:
            ax.set_ylim(0, 1.1)

    fig.suptitle('Adaptive RL Weights vs Static Equal Weights', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'baseline_comparison.png')
    plt.savefig(path)
    plt.close()
    return path


def plot_score_dimensions(results: Dict, output_dir: str):
    """
    Figure: Radar/bar chart of the four scoring dimensions per user.
    Shows how different users achieve different score profiles.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    users = list(results.keys())
    dimensions = ['PPS', 'ECS', 'SPS', 'ES']
    x = np.arange(len(dimensions))
    width = 0.18

    for i, user in enumerate(users):
        log = results[user]["journey_log"]
        if not log:
            continue
        # Average scores across last 50 journeys
        recent = log[-50:]
        vals = [np.mean([j[d.lower()] for j in recent]) for d in dimensions]
        ax.bar(x + i * width, vals, width, label=user.capitalize(),
               color=COLORS[user], edgecolor='white')

    ax.set_xlabel('Score Dimension')
    ax.set_ylabel('Average Score (last 50 journeys)')
    ax.set_title('Score Profile by User Scenario', fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(['Personal\nPreference', 'Environmental\nComfort',
                        'Success\nProbability', 'Efficiency'])
    ax.legend()
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    path = os.path.join(output_dir, 'score_dimensions.png')
    plt.savefig(path)
    plt.close()
    return path


def plot_epsilon_decay(results: Dict, output_dir: str):
    """Figure: Epsilon decay showing exploration → exploitation transition."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    # All users have same epsilon schedule, just show one
    first_key = list(results.keys())[0]
    epsilon = results[first_key]["time_series"]["epsilon"]
    ax.plot(epsilon, color='#E74C3C', linewidth=2)
    ax.set_xlabel('Journey Number')
    ax.set_ylabel('ε (Exploration Rate)')
    ax.set_title('ε-Greedy Exploration Decay', fontweight='bold')
    ax.fill_between(range(len(epsilon)), epsilon, alpha=0.1, color='#E74C3C')

    plt.tight_layout()
    path = os.path.join(output_dir, 'epsilon_decay.png')
    plt.savefig(path)
    plt.close()
    return path


def plot_action_distribution(results: Dict, output_dir: str):
    """Figure: Which weight configurations each user type converges to."""
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Weight Configuration Selection Distribution', fontsize=14, fontweight='bold')

    for idx, (key, result) in enumerate(results.items()):
        ax = axes[idx // 2][idx % 2]
        stats = result["rl_stats"]
        actions = stats["action_distribution"]
        configs = list(actions.keys())
        counts = list(actions.values())

        # Color by type: efficiency (red) -> balanced (yellow) -> comfort (blue)
        cmap = plt.cm.RdYlBu
        colors = [cmap(i / 9) for i in range(10)]

        ax.bar(range(len(configs)), counts, color=colors, edgecolor='white')
        ax.set_xlabel('Weight Configuration (0=Efficiency → 9=Comfort)')
        ax.set_ylabel('Times Selected')
        ax.set_title(f'{key.capitalize()}', fontweight='bold')
        ax.set_xticks(range(10))
        ax.set_xticklabels([str(i) for i in range(10)])

    plt.tight_layout()
    path = os.path.join(output_dir, 'action_distribution.png')
    plt.savefig(path)
    plt.close()
    return path


def generate_summary_table(results: Dict, baseline_results: Dict) -> str:
    """Generate a markdown summary table of all metrics."""
    lines = []
    lines.append("## Evaluation Summary")
    lines.append("")
    lines.append("### Table: Overall Performance Metrics (Adaptive RL vs Static Baseline)")
    lines.append("")
    lines.append("| Metric | " + " | ".join(k.capitalize() for k in results.keys()) + " |")
    lines.append("|" + "---|" * (len(results) + 1))

    for metric_name, getter in [
        ("Acceptance Rate (Adaptive)", lambda r: f"{r['metrics']['overall']['acceptance_rate']:.3f}"),
        ("Acceptance Rate (Static)", lambda r: None),
        ("Completion Rate (Adaptive)", lambda r: f"{r['metrics']['overall']['completion_rate']:.3f}"),
        ("Completion Rate (Static)", lambda r: None),
        ("Avg Stress (Adaptive)", lambda r: f"{r['metrics']['overall']['avg_stress']:.3f}"),
        ("Avg Stress (Static)", lambda r: None),
        ("Avg Reward", lambda r: f"{r['metrics']['overall']['avg_reward']:.3f}"),
        ("Improvement: Acceptance", lambda r: f"{r['metrics']['improvement']['acceptance_rate']:+.3f}"),
        ("Improvement: Completion", lambda r: f"{r['metrics']['improvement']['completion_rate']:+.3f}"),
        ("Improvement: Stress", lambda r: f"{r['metrics']['improvement']['stress_reduction']:+.3f}"),
    ]:
        vals = []
        for key in results.keys():
            if "Static" in metric_name:
                metric_key = metric_name.split("(")[0].strip().lower().replace(" ", "_")
                if "acceptance" in metric_key:
                    vals.append(f"{baseline_results[key]['acceptance_rate']:.3f}")
                elif "completion" in metric_key:
                    vals.append(f"{baseline_results[key]['completion_rate']:.3f}")
                elif "stress" in metric_key:
                    vals.append(f"{baseline_results[key]['avg_stress']:.3f}")
                else:
                    vals.append("—")
            else:
                vals.append(getter(results[key]))
        lines.append(f"| {metric_name} | " + " | ".join(vals) + " |")

    return "\n".join(lines)


def generate_all_plots(results: Dict, baseline_results: Dict,
                       output_dir: str) -> List[str]:
    """Generate all evaluation figures."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    print("\nGenerating visualizations...")
    paths.append(plot_rl_convergence(results, output_dir))
    print("  ✓ RL convergence plot")
    paths.append(plot_reward_and_stress(results, output_dir))
    print("  ✓ Reward & stress plot")
    paths.append(plot_acceptance_completion(results, output_dir))
    print("  ✓ Acceptance/completion plot")
    paths.append(plot_baseline_comparison(results, baseline_results, output_dir))
    print("  ✓ Baseline comparison plot")
    paths.append(plot_score_dimensions(results, output_dir))
    print("  ✓ Score dimensions plot")
    paths.append(plot_epsilon_decay(results, output_dir))
    print("  ✓ Epsilon decay plot")
    paths.append(plot_action_distribution(results, output_dir))
    print("  ✓ Action distribution plot")

    return paths
