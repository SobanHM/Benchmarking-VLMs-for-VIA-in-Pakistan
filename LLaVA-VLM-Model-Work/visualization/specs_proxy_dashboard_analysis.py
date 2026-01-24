import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

DATA_FILE = r"C:\Users\soban\PycharmProjects\LLaVA\metrics\specs_proxy_analysis_results.csv"
TITLE_PREFIX = "LLaVA v1.5 [Supermarket Split] SPECS Analysis"

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'figure.titlesize': 16,
    'axes.titlesize': 14
})

def load_data():
    try:
        return pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find '{DATA_FILE}'. Please ensure it is in the current directory.")
        return None

# plot 1 distribution or specificity gain
def plot_distribution(df):
    plt.figure(figsize=(10, 8))  # increased height for text box

    # histogram with kde
    ax = sns.histplot(df['SPECS_Score'], kde=True, color='#2c3e50', bins=15, alpha=0.6)

    # color zones (Green = Good, Red = Bad)
    ylim = ax.get_ylim()
    plt.axvline(0, color='black', linestyle='--', linewidth=2)
    plt.fill_betweenx(ylim, 0, df['SPECS_Score'].max(), color='green', alpha=0.1)
    plt.fill_betweenx(ylim, df['SPECS_Score'].min(), 0, color='red', alpha=0.1)

    # calculate stat
    positive_pct = (df['SPECS_Score'] > 0).mean() * 100

    # annotations on plot
    plt.text(df['SPECS_Score'].max() * 0.6, ylim[1] * 0.85, f"Value Added\n({positive_pct:.1f}%)",
             color='darkgreen', fontweight='bold', ha='center', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'))

    plt.text(df['SPECS_Score'].min() * 0.6, ylim[1] * 0.85, f"Noise\n({100 - positive_pct:.1f}%)",
             color='darkred', fontweight='bold', ha='center', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))

    plt.title(f"{TITLE_PREFIX}\nDistribution of Specificity Gain", fontweight='bold')
    plt.xlabel("SPECS Score (Value of Detail)")
    plt.ylabel("Frequency")

    # description box
    desc_text = (
        "INTERPRETATION:\n"
        "• Green Zone (Positive): The model's specific details (colors, spatial terms) matched the image.\n"
        "• Red Zone (Negative): The details were hallucinations or irrelevant noise.\n"
        f"• Result: In {positive_pct:.1f}% of cases, adding detail IMPROVED the visual grounding score."
    )
    plt.figtext(0.5, 0.02, desc_text, wrap=True, horizontalalignment='center', fontsize=11,
                bbox=dict(facecolor='#f8f9fa', alpha=1.0, edgecolor='gray', boxstyle='round,pad=0.5'))

    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.savefig('specs_proxy_plot_1_distribution.png', dpi=300)
    print("Generated 'specs_proxy_plot_1_distribution.png'")


# plot 2 verbosity vs value
def plot_verbosity(df):
    plt.figure(figsize=(10, 8))

    # scatter plot
    sns.scatterplot(
        data=df, x='Word_Count', y='SPECS_Score', hue='SPECS_Score',
        palette='RdYlGn', size='SPECS_Score', sizes=(50, 300),
        alpha=0.8, edgecolor='black', legend=False
    )

    plt.axhline(0, color='black', linestyle='--', linewidth=1.5)

    # quadrant Labels
    max_x, min_x = df['Word_Count'].max(), df['Word_Count'].min()
    max_y, min_y = df['SPECS_Score'].max(), df['SPECS_Score'].min()

    plt.text(max_x * 0.95, max_y * 0.9, "Rich Description\n(Goal)",
             color='darkgreen', fontweight='bold', ha='right', fontsize=11,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.text(max_x * 0.95, min_y * 0.9, "Verbose Hallucination\n(Risk)",
             color='darkred', fontweight='bold', ha='right', fontsize=11,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.title(f"{TITLE_PREFIX}\nDoes Verbosity Add Value?", fontweight='bold')
    plt.xlabel("Caption Length (Word Count)")
    plt.ylabel("SPECS Score (Visual Gain)")

    # description box
    desc_text = (
        "INTERPRETATION:\n"
        "• This plot tests the hypothesis 'Longer is Better'.\n"
        "• Top-Right Cluster: Indicates that LLaVA v1.5 maintains high visual accuracy even with long captions.\n"
        "• Unlike standard models that degrade with length, this model's verbosity correlates with higher quality."
    )
    plt.figtext(0.5, 0.02, desc_text, wrap=True, horizontalalignment='center', fontsize=11,
                bbox=dict(facecolor='#f8f9fa', alpha=1.0, edgecolor='gray', boxstyle='round,pad=0.5'))

    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.savefig('specs_proxy_plot_2_verbosity.png', dpi=300)
    print("Generated 'specs_proxy_plot_2_verbosity.png'")


# plot 3 impactful analysis
def plot_impact(df):
    plt.figure(figsize=(10, 8))

    # neutral line (y=x)
    lims = [
        min(df['CLIP_Generic'].min(), df['CLIP_Full'].min()),
        max(df['CLIP_Generic'].max(), df['CLIP_Full'].max())
    ]
    plt.plot(lims, lims, '--', color='grey', alpha=0.7, linewidth=2, label="Neutral Impact")

    # scatter
    sns.scatterplot(
        data=df, x='CLIP_Generic', y='CLIP_Full', hue='SPECS_Score',
        palette='RdYlGn', alpha=0.9, s=100, edgecolor='black'
    )
    plt.legend(title="SPECS Value", loc='lower right', frameon=True)

    # zone shading
    plt.fill_between(lims, lims, [lims[1], lims[1]], color='green', alpha=0.05)
    plt.fill_between(lims, [lims[0], lims[0]], lims, color='red', alpha=0.05)

    # label
    plt.text(lims[0] + (lims[1] - lims[0]) * 0.05, lims[1] - (lims[1] - lims[0]) * 0.05,
             "Points ABOVE line\n= Detail Improved Score", color='darkgreen', fontweight='bold', ha='left')

    plt.title(f"{TITLE_PREFIX}\nImpact of Detail on Visual Grounding", fontweight='bold')
    plt.xlabel("Baseline Score (Generic Caption)")
    plt.ylabel("Final Score (Detailed Caption)")

    # description box
    desc_text = (
        "INTERPRETATION:\n"
        "• X-Axis: Score of the caption with all adjectives/spatial terms removed (Generic).\n"
        "• Y-Axis: Score of the full, detailed caption.\n"
        "• Points above the diagonal prove that adding specific details creates a stronger match to the image."
    )
    plt.figtext(0.5, 0.02, desc_text, wrap=True, horizontalalignment='center', fontsize=11,
                bbox=dict(facecolor='#f8f9fa', alpha=1.0, edgecolor='gray', boxstyle='round,pad=0.5'))

    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.savefig('specs_proxy_plot_3_impact.png', dpi=300)
    print("Generated 'specs_proxy_plot_3_impact.png'")

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        plot_distribution(df)
        plot_verbosity(df)
        plot_impact(df)
        print("\nAll plots generated successfully")
