import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# eda plot -set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})

# 1 real computed data from previous results
metrics = {
    'BERTScore (Fluency)': 86.04,
    'SBERT (Meaning)': 62.00,
    'SPECS (Spatial)': 54.57,
    'CLIPScore (Visual)': 31.49,
    'POPE F1 (Existence)': 25.05
}

safety_errors = {
    'General Hallucination\n(CHAIR-i 0.5)': 44.93,
    'Navigation Hallucination\n(Critical Hazards)': 57.10
}

title_suffix = "\nLLaVA v1.5 Supermarket Results Evaluation"

# plot 1: safety gap (fluency of model vs reality)
def plot_safety_gap():
    plt.figure(figsize=(10, 6))
    # compare linguistic capabilities vs grounding capabilities
    labels = ['BERTScore\n(Fluency)', 'SBERT\n(Meaning)', 'SPECS\n(Spatial)', 'POPE F1\n(Object Reality)']
    values = [metrics['BERTScore (Fluency)'], metrics['SBERT (Meaning)'],
              metrics['SPECS (Spatial)'], metrics['POPE F1 (Existence)']]

    colors = ['#2ecc71', '#3498db', '#f1c40f', '#e74c3c']  # green, blue, yellow, red
    bars = plt.bar(labels, values, color=colors, edgecolor='black', alpha=0.8)
    plt.ylabel('Score (0-100%)')
    plt.title(f'The "Fluency-Grounding Gap" {title_suffix}', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    # add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 2,
                 f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('llava_eval_1_safety_gap.png', dpi=300)
    print("Generated Plot 1: Safety Gap")


# plot 2: hallucination severity (General vs navigation)
def plot_hallucination_severity():
    plt.figure(figsize=(8, 6))

    labels = list(safety_errors.keys())
    values = list(safety_errors.values())

    # red for danger
    colors = ['#e67e22', '#c0392b']

    bars = plt.bar(labels, values, color=colors, edgecolor='black', width=0.5)

    plt.ylabel('Error Rate (%) - LOWER is Better')
    plt.title(f'Hallucination Severity Analysis {title_suffix}', fontsize=14, fontweight='bold')
    plt.ylim(0, 70)

    # add line for Acceptable Safety Limit
    plt.axhline(y=10, color='green', linestyle='--', linewidth=2, label='Target Safety Threshold (<10%)')
    plt.legend()

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 2,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold', color='darkred')

    plt.tight_layout()
    plt.savefig('llava_eval_2_hallucination.png', dpi=300)
    print("Generated Plot 2: Hallucination Severity")


# plot 3: the comprehensive radar chart
def plot_radar_profile():
    # Variables
    labels = ['Fluency\n(BERT)', 'Meaning\n(SBERT)', 'Spatial\n(SPECS)',
              'Safety\n(100 - NavError)', 'Accuracy\n(POPE)', 'Visual\n(CLIP)']

    # calculate Safety Score (100 - Error) to make "Bigger = Better" on radar
    nav_safety_score = 100 - safety_errors['Navigation Hallucination\n(Critical Hazards)']

    stats = [
        metrics['BERTScore (Fluency)'],
        metrics['SBERT (Meaning)'],
        metrics['SPECS (Spatial)'],
        nav_safety_score,
        metrics['POPE F1 (Existence)'],
        metrics['CLIPScore (Visual)']
    ]

    # close the loop for radar chart
    stats = np.concatenate((stats, [stats[0]]))
    labels_plot = np.concatenate((labels, [labels[0]]))

    # plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    ax.plot(np.linspace(0, 2 * np.pi, len(stats)), stats, 'o-', linewidth=2, color='#8e44ad')
    ax.fill(np.linspace(0, 2 * np.pi, len(stats)), stats, alpha=0.25, color='#8e44ad')

    ax.set_thetagrids(np.degrees(np.linspace(0, 2 * np.pi, len(labels))), labels)
    ax.set_title(f'Holistic Model Profile {title_suffix}', fontsize=15, fontweight='bold', y=1.08)

    # set range
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig('llava_eval_3_radar.png', dpi=300)
    print("Generated Plot 3: Radar Profile")

if __name__ == "__main__":
    plot_safety_gap()
    plot_hallucination_severity()
    plot_radar_profile()
