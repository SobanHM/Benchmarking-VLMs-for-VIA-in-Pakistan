import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. load (computed results using metric/safety_granular.py)
data = {
    'Category': [
        'Entrances_Exits',
        'Level_Changes',
        'Trip_Hazards',
        'Structural_Obstacles',
        'Dynamic_Obstacles'
    ],
    'False Alarm (Anxiety)': [48.6, 15.9, 66.7, 7.9, 4.2],
    'Missed Hazard (Danger)': [64.0, 54.9, 80.0, 56.8, 5.2]
}

df = pd.DataFrame(data)

# seaborn plotting
df_melt = df.melt(id_vars="Category", var_name="Error Type", value_name="Rate")


sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.2
})

fig, ax = plt.subplots(figsize=(12, 7.5))

# soft orange for anxiety (false alarms or hallucination)
# strong red for danger (missed hazards/low recall)
palette = {"False Alarm (Anxiety)": "#FFB347", "Missed Hazard (Danger)": "#FF6961"}

# eda plots
bar_plot = sns.barplot(
    x="Category", y="Rate", hue="Error Type",
    data=df_melt, palette=palette, edgecolor=".2", linewidth=1.5
)

# titles
plt.title("Safety Audit Results: LLaVA v1.5 (supermarket: Context-Aware NLI)", fontsize=18, fontweight='bold', pad=20, color='#2c3e50')
plt.ylabel("Error Rate (%) - Lower is Better", fontweight='bold')
plt.xlabel("")
plt.ylim(0, 105)

# add Value labels on Bars
for container in bar_plot.containers:
    bar_plot.bar_label(container, fmt='%.1f%%', padding=4, fontweight='bold', fontsize=10)

# sdd "Safe Threshold" line
plt.axhline(20, color='green', linestyle='--', linewidth=2, label="Safe Threshold (<20%)")

plt.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.95, shadow=True, borderpad=1)


interpretation_text = (
    "Interpretation (Brief):\n"
    "Dynamic Obstacles: SUCCESS (4% Error). The model reliably detects people.\n"
    "Entrances/Exits: UNRELIABLE. High omission (64%) poses navigation risks.\n"
    "Trip Hazards: BROKEN. High false alarms (66%) indicate 'Paranoid AI' behavior."
)

# place text box at the bottom
plt.figtext(0.4, 0.015, interpretation_text, wrap=True, horizontalalignment='left', fontsize=10, family='monospace',
            bbox=dict(facecolor='#f8f9fa', edgecolor='#7f8c8d', boxstyle='round,pad=1', alpha=1.0))

plt.tight_layout(rect=[0, 0.16, 1, 1])
plt.savefig('safety_granular_Dashboard.png', dpi=300)
print("Generated 'final_safety_dashboard.png'")
plt.show()
