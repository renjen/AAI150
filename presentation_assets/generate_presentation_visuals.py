import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set font family and clean style
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

# -------------------------------------------------------------
# SLIDE 7: How We Built the Model (Random Forest 300 Trees Voting)
# -------------------------------------------------------------
def create_slide_07_visual():
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    # Title Banner / Subtitle
    ax.text(5, 5.1, "Ensemble Voting Architecture: 300 Decision Trees", 
            ha='center', va='center', fontsize=16, fontweight='bold', color='#1e293b')
    ax.text(5, 4.7, "Each tree inspects a random subset of lab tests and casts an independent vote", 
            ha='center', va='center', fontsize=11, color='#64748b')

    # Input Box: Wine Chemistry
    box_in = patches.FancyBboxPatch((0.4, 1.8), 2.2, 2.2, boxstyle="round,pad=0.1,rounding_size=0.15",
                                    ec="#3b82f6", fc="#eff6ff", lw=2)
    ax.add_patch(box_in)
    ax.text(1.5, 3.5, "Input Sample", ha='center', va='center', fontsize=12, fontweight='bold', color='#1e40af')
    ax.text(1.5, 2.7, "11 Lab Tests\n• Alcohol\n• Volatile Acidity\n• Chlorides\n• Density, etc.", 
            ha='center', va='center', fontsize=10, color='#334155', linespacing=1.4)

    # Arrows from Input to Trees
    ax.annotate('', xy=(3.3, 3.8), xytext=(2.7, 3.1), arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=2, shrinkA=5, shrinkB=5))
    ax.annotate('', xy=(3.3, 2.8), xytext=(2.7, 2.8), arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=2, shrinkA=5, shrinkB=5))
    ax.annotate('', xy=(3.3, 1.8), xytext=(2.7, 2.5), arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=2, shrinkA=5, shrinkB=5))

    # Tree 1 Box
    t1 = patches.FancyBboxPatch((3.4, 3.3), 2.3, 1.1, boxstyle="round,pad=0.1,rounding_size=0.1", ec="#10b981", fc="#ecfdf5", lw=1.5)
    ax.add_patch(t1)
    ax.text(4.55, 4.0, "Decision Tree #1", ha='center', va='center', fontsize=10, fontweight='bold', color='#065f46')
    ax.text(4.55, 3.55, "Alcohol > 10.5% & Low Salt?\nVote: GOOD (+1)", ha='center', va='center', fontsize=8.5, color='#047857')

    # Tree 2 Box
    t2 = patches.FancyBboxPatch((3.4, 2.1), 2.3, 1.1, boxstyle="round,pad=0.1,rounding_size=0.1", ec="#10b981", fc="#ecfdf5", lw=1.5)
    ax.add_patch(t2)
    ax.text(4.55, 2.8, "Decision Tree #2", ha='center', va='center', fontsize=10, fontweight='bold', color='#065f46')
    ax.text(4.55, 2.35, "High Sugar & Low Density?\nVote: GOOD (+1)", ha='center', va='center', fontsize=8.5, color='#047857')

    # Ellipsis for trees 3-299
    ax.text(4.55, 1.8, "•  •  •  (300 Trees Total)  •  •  •", ha='center', va='center', fontsize=10, fontweight='bold', color='#64748b')

    # Tree 300 Box
    t3 = patches.FancyBboxPatch((3.4, 0.5), 2.3, 1.1, boxstyle="round,pad=0.1,rounding_size=0.1", ec="#f59e0b", fc="#fffbeb", lw=1.5)
    ax.add_patch(t3)
    ax.text(4.55, 1.2, "Decision Tree #300", ha='center', va='center', fontsize=10, fontweight='bold', color='#92400e')
    ax.text(4.55, 0.75, "High Volatile Acidity?\nVote: ORDINARY (0)", ha='center', va='center', fontsize=8.5, color='#b45309')

    # Arrows from Trees to Voting Box
    ax.annotate('', xy=(6.6, 3.0), xytext=(5.8, 3.8), arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=2, shrinkA=5, shrinkB=5))
    ax.annotate('', xy=(6.6, 2.8), xytext=(5.8, 2.7), arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=2, shrinkA=5, shrinkB=5))
    ax.annotate('', xy=(6.6, 2.6), xytext=(5.8, 1.1), arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=2, shrinkA=5, shrinkB=5))

    # Voting / Tally Box
    vbox = patches.FancyBboxPatch((6.7, 1.7), 1.6, 2.2, boxstyle="round,pad=0.1,rounding_size=0.15", ec="#6366f1", fc="#eef2ff", lw=2)
    ax.add_patch(vbox)
    ax.text(7.5, 3.4, "Consensus Tally", ha='center', va='center', fontsize=11, fontweight='bold', color='#3730a3')
    ax.text(7.5, 2.7, "72% of Trees\nVote 'Good'\n\nScore: 0.72\n(Threshold = 0.50)", 
            ha='center', va='center', fontsize=9.5, color='#4338ca', linespacing=1.3)

    # Arrow to Output
    ax.annotate('', xy=(8.7, 2.8), xytext=(8.35, 2.8), arrowprops=dict(arrowstyle="->", color="#6366f1", lw=3, shrinkA=4, shrinkB=4))

    # Final Output Badge
    out = patches.FancyBboxPatch((8.8, 2.1), 1.0, 1.4, boxstyle="round,pad=0.1,rounding_size=0.15", ec="#16a34a", fc="#dcfce7", lw=2.5)
    ax.add_patch(out)
    ax.text(9.3, 3.0, "FINAL\nVERDICT", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#15803d')
    ax.text(9.3, 2.4, "FLAG FOR\nTASTING", ha='center', va='center', fontsize=9, fontweight='heavy', color='#166534')

    plt.tight_layout()
    plt.savefig("/Users/bloodsailadmiralamunra/Documents/GitHub/AAI150/presentation_assets/slide_07_model_trees.png")
    plt.savefig("/Users/bloodsailadmiralamunra/Documents/GitHub/MAAI-USD-ASC/Final/presentation_assets/slide_07_model_trees.png")
    plt.close()
    print("Slide 7 visual created.")

# -------------------------------------------------------------
# SLIDE 8: How Well It Works (Confusion Matrix as 4 Plain Boxes)
# -------------------------------------------------------------
def create_slide_08_visual():
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    # Title & Subtitle
    ax.text(5, 5.1, "Test Cellar Results: 1,064 Unseen Wines Tested", 
            ha='center', va='center', fontsize=16, fontweight='bold', color='#1e293b')
    ax.text(5, 4.7, "What happens when every wine batch is screened through the model", 
            ha='center', va='center', fontsize=11, color='#64748b')

    # Column / Row Headers
    ax.text(3.1, 4.25, "MODEL SAYS: 'FLAG AS GOOD'", ha='center', va='center', fontsize=11, fontweight='bold', color='#1d4ed8')
    ax.text(6.9, 4.25, "MODEL SAYS: 'PASS OVER'", ha='center', va='center', fontsize=11, fontweight='bold', color='#475569')

    ax.text(0.5, 3.1, "ACTUAL\nGOOD WINES\n(202 Total)", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#065f46')
    ax.text(0.5, 1.4, "ACTUAL\nORDINARY\n(862 Total)", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#475569')

    # Box 1: Caught Good Wines (TP)
    b1 = patches.FancyBboxPatch((1.4, 2.3), 3.4, 1.6, boxstyle="round,pad=0.1,rounding_size=0.15", ec="#16a34a", fc="#dcfce7", lw=2)
    ax.add_patch(b1)
    ax.text(3.1, 3.45, "CAUGHT GOOD WINES", ha='center', va='center', fontsize=12, fontweight='bold', color='#15803d')
    ax.text(3.1, 2.9, "145 Bottles", ha='center', va='center', fontsize=22, fontweight='black', color='#166534')
    ax.text(3.1, 2.45, "72% of all great wines successfully saved for reserve", ha='center', va='center', fontsize=8.5, color='#14532d')

    # Box 2: Missed Good Wines (FN)
    b2 = patches.FancyBboxPatch((5.2, 2.3), 3.4, 1.6, boxstyle="round,pad=0.1,rounding_size=0.15", ec="#ef4444", fc="#fee2e2", lw=2)
    ax.add_patch(b2)
    ax.text(6.9, 3.45, "MISSED GOOD WINES", ha='center', va='center', fontsize=12, fontweight='bold', color='#b91c1c')
    ax.text(6.9, 2.9, "57 Bottles", ha='center', va='center', fontsize=22, fontweight='black', color='#991b1b')
    ax.text(6.9, 2.45, "28% of good wines sent to table wine untasted", ha='center', va='center', fontsize=8.5, color='#7f1d1d')

    # Box 3: False Alarms (FP)
    b3 = patches.FancyBboxPatch((1.4, 0.5), 3.4, 1.6, boxstyle="round,pad=0.1,rounding_size=0.15", ec="#f59e0b", fc="#fef3c7", lw=2)
    ax.add_patch(b3)
    ax.text(3.1, 1.65, "FALSE ALARMS", ha='center', va='center', fontsize=12, fontweight='bold', color='#b45309')
    ax.text(3.1, 1.1, "123 Bottles", ha='center', va='center', fontsize=22, fontweight='black', color='#92400e')
    ax.text(3.1, 0.65, "Ordinary wines sent to tasting (judges filter these out)", ha='center', va='center', fontsize=8.5, color='#78350f')

    # Box 4: Screened Out (TN)
    b4 = patches.FancyBboxPatch((5.2, 0.5), 3.4, 1.6, boxstyle="round,pad=0.1,rounding_size=0.15", ec="#94a3b8", fc="#f1f5f9", lw=2)
    ax.add_patch(b4)
    ax.text(6.9, 1.65, "CORRECTLY SCREENED OUT", ha='center', va='center', fontsize=12, fontweight='bold', color='#475569')
    ax.text(6.9, 1.1, "739 Bottles", ha='center', va='center', fontsize=22, fontweight='black', color='#334155')
    ax.text(6.9, 0.65, "Ordinary wines bottled immediately (no tasting needed)", ha='center', va='center', fontsize=8.5, color='#1e293b')

    # Summary Badges on the right margin
    badge1 = patches.FancyBboxPatch((8.8, 2.4), 1.1, 1.4, boxstyle="round,pad=0.08,rounding_size=0.1", ec="#3b82f6", fc="#eff6ff", lw=1.5)
    ax.add_patch(badge1)
    ax.text(9.35, 3.4, "TASTING CUT", ha='center', va='center', fontsize=8, fontweight='bold', color='#1e40af')
    ax.text(9.35, 2.95, "75%", ha='center', va='center', fontsize=16, fontweight='black', color='#1d4ed8')
    ax.text(9.35, 2.6, "Less Tasting\nVolume", ha='center', va='center', fontsize=7, color='#1e3a8a')

    badge2 = patches.FancyBboxPatch((8.8, 0.6), 1.1, 1.4, boxstyle="round,pad=0.08,rounding_size=0.1", ec="#10b981", fc="#ecfdf5", lw=1.5)
    ax.add_patch(badge2)
    ax.text(9.35, 1.6, "FLAG PURITY", ha='center', va='center', fontsize=8, fontweight='bold', color='#065f46')
    ax.text(9.35, 1.15, "54%", ha='center', va='center', fontsize=16, fontweight='black', color='#047857')
    ax.text(9.35, 0.8, "vs. 19% Base\nRate (3x Boost)", ha='center', va='center', fontsize=7, color='#064e3b')

    plt.tight_layout()
    plt.savefig("/Users/bloodsailadmiralamunra/Documents/GitHub/AAI150/presentation_assets/slide_08_confusion_matrix_plain.png")
    plt.savefig("/Users/bloodsailadmiralamunra/Documents/GitHub/MAAI-USD-ASC/Final/presentation_assets/slide_08_confusion_matrix_plain.png")
    plt.close()
    print("Slide 8 visual created.")

# -------------------------------------------------------------
# SLIDE 9: What the Model Looks At (Top 5 Permutation Feature Importance)
# -------------------------------------------------------------
def create_slide_09_visual():
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    features = ['Alcohol\n(Ripeness & Maturity)', 
                'Chlorides\n(Saltiness & Astringency)', 
                'Density\n(Body & Sugar-Alcohol Balance)', 
                'Volatile Acidity\n(Vinegar Spoilage Marker)', 
                'Sulphates\n(Freshness & Oxidation Control)'][::-1]
    
    importance = [0.2461, 0.0382, 0.0304, 0.0241, 0.0163][::-1]
    colors = ['#cbd5e1', '#94a3b8', '#64748b', '#3b82f6', '#1d4ed8']

    bars = ax.barh(features, importance, color=colors, height=0.6, edgecolor='none')
    
    # Value annotations on bars
    for bar, val in zip(bars, importance):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f"-{val:.3f} F1 drop", 
                va='center', ha='left', fontsize=10, fontweight='bold', color='#1e293b')

    ax.set_xlim(0, 0.28)
    ax.set_xlabel("F1-Score Drop When Measurement is Scrambled on 1,064 Test Wines", fontsize=10, fontweight='bold', color='#475569', labelpad=10)
    ax.set_title("Top 5 Chemical Signals Driving Quality Predictions", fontsize=14, fontweight='bold', color='#1e293b', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.tick_params(axis='y', labelsize=9.5)
    ax.tick_params(axis='x', labelsize=9)
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("/Users/bloodsailadmiralamunra/Documents/GitHub/AAI150/presentation_assets/slide_09_feature_importance_top5.png")
    plt.savefig("/Users/bloodsailadmiralamunra/Documents/GitHub/MAAI-USD-ASC/Final/presentation_assets/slide_09_feature_importance_top5.png")
    plt.close()
    print("Slide 9 visual created.")

# -------------------------------------------------------------
# SLIDE 11: Limitations (3 Key Operational Reality Pillars)
# -------------------------------------------------------------
def create_slide_11_visual():
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis('off')

    # Title
    ax.text(5, 4.8, "Key Limitations & Operational Realities", 
            ha='center', va='center', fontsize=16, fontweight='bold', color='#1e293b')
    ax.text(5, 4.4, "What every winery manager must consider before deploying the model", 
            ha='center', va='center', fontsize=11, color='#64748b')

    # Card 1: One Region Only
    c1 = patches.FancyBboxPatch((0.5, 0.6), 2.7, 3.4, boxstyle="round,pad=0.1,rounding_size=0.15", ec="#3b82f6", fc="#f8fafc", lw=2)
    ax.add_patch(c1)
    ax.text(1.85, 3.5, "1. Geography", ha='center', va='center', fontsize=13, fontweight='bold', color='#1d4ed8')
    ax.text(1.85, 3.1, "Vinho Verde, Portugal", ha='center', va='center', fontsize=10, fontweight='semibold', color='#475569')
    
    text1 = ("• Cool, Atlantic maritime climate\n"
             "• High natural acid, low alcohol\n"
             "• Distinctive regional profile\n\n"
             "Warning: Do not apply to\n"
             "warm regions (Napa, Rioja)\n"
             "without local retraining.")
    ax.text(1.85, 1.8, text1, ha='center', va='center', fontsize=9, color='#334155', linespacing=1.4)

    # Card 2: Subjective Palates
    c2 = patches.FancyBboxPatch((3.65, 0.6), 2.7, 3.4, boxstyle="round,pad=0.1,rounding_size=0.15", ec="#f59e0b", fc="#f8fafc", lw=2)
    ax.add_patch(c2)
    ax.text(5.0, 3.5, "2. Human Palates", ha='center', va='center', fontsize=13, fontweight='bold', color='#d97706')
    ax.text(5.0, 3.1, "Expert Rating Variance", ha='center', va='center', fontsize=10, fontweight='semibold', color='#475569')
    
    text2 = ("• Median score of 3 judges\n"
             "• Palate fatigue and order bias\n"
             "• Subjective style preferences\n\n"
             "Result: Inherent label noise\n"
             "along the border between\n"
             "score 6 and score 7.")
    ax.text(5.0, 1.8, text2, ha='center', va='center', fontsize=9, color='#334155', linespacing=1.4)

    # Card 3: Operational Trade-off
    c3 = patches.FancyBboxPatch((6.8, 0.6), 2.7, 3.4, boxstyle="round,pad=0.1,rounding_size=0.15", ec="#ef4444", fc="#f8fafc", lw=2)
    ax.add_patch(c3)
    ax.text(8.15, 3.5, "3. Missed Bottles", ha='center', va='center', fontsize=13, fontweight='bold', color='#dc2626')
    ax.text(8.15, 3.1, "The 28% Miss Rate", ha='center', va='center', fontsize=10, fontweight='semibold', color='#475569')
    
    text3 = ("• Cuts tasting flights by ~75%\n"
             "• Misses 57 of 202 good wines\n"
             "• Missed wines become table wine\n\n"
             "Solution: Wineries can dial\n"
             "down cutoff threshold to\n"
             "catch more good bottles.")
    ax.text(8.15, 1.8, text3, ha='center', va='center', fontsize=9, color='#334155', linespacing=1.4)

    plt.tight_layout()
    plt.savefig("/Users/bloodsailadmiralamunra/Documents/GitHub/AAI150/presentation_assets/slide_11_limitations_cards.png")
    plt.savefig("/Users/bloodsailadmiralamunra/.gemini/antigravity/brain/f454c265-efe2-4773-bad4-b99fadf8bef6/slide_11_limitations_cards.png")
    plt.close()
    print("Slide 11 visual created.")

if __name__ == '__main__':
    create_slide_07_visual()
    create_slide_08_visual()
    create_slide_09_visual()
    create_slide_11_visual()
