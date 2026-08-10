import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. Create figure canvas
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
fig.patch.set_facecolor('#F2F7F4')  # Light sage green background
ax.set_facecolor('#F2F7F4')

# Hide axis borders & ticks
ax.axis('off')
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)

# Main Title
ax.text(6, 5.4, "Unsupervised Learning", fontsize=22, fontweight='bold', 
        ha='center', va='center', color='#000000', family='sans-serif')

# 2. Input Box (Top Left Green Header)
rect_input = patches.FancyBboxPatch((0.5, 4.0), 1.8, 1.0, boxstyle="round,pad=0.05,rounding_size=0.1",
                                    facecolor='#1E7E43', edgecolor='none')
ax.add_patch(rect_input)
ax.text(1.4, 4.5, "INPUT\nRAW DATA", fontsize=11, fontweight='bold', color='white', ha='center', va='center')

# Input Shapes (Replacing emojis with clear vector shapes)
# Group 1: Circles
ax.plot(1.1, 3.4, 'o', color='#1E7E43', markersize=14)
ax.plot(1.7, 3.4, 'o', color='#1E7E43', markersize=14)
# Group 2: Triangles
ax.plot(1.1, 2.7, '^', color='#2196F3', markersize=14)
ax.plot(1.7, 2.7, '^', color='#2196F3', markersize=14)
# Group 3: Squares
ax.plot(1.1, 2.0, 's', color='#FF9800', markersize=14)
ax.plot(1.7, 2.0, 's', color='#FF9800', markersize=14)

# Subtext Notes
ax.text(3.4, 1.2, "•  Unknown Output\n•  No Training Data Set", fontsize=11, color='#111111', ha='left', va='center')

# 3. Middle Process Nodes
middle_labels = ["Interpretation", "Algorithm", "Processing"]
x_positions = [3.8, 6.0, 8.2]

for x, label in zip(x_positions, middle_labels):
    # Light green tag above node
    rect_tag = patches.FancyBboxPatch((x-0.8, 3.8), 1.6, 0.4, boxstyle="square,pad=0",
                                      facecolor='#C8E6C9', edgecolor='none')
    ax.add_patch(rect_tag)
    ax.text(x, 4.0, label, fontsize=11, color='#111111', ha='center', va='center')

    # Connecting vertical line from tag down to circle
    ax.plot([x, x], [3.8, 2.8], color='#1E7E43', lw=2)

    # Green circle node
    circle = plt.Circle((x, 2.3), 0.45, color='#1E7E43')
    ax.add_patch(circle)

# Draw simple white inner details for process nodes
ax.plot(3.8, 2.3, 'o', color='white', markersize=8)  # Interpretation icon
ax.plot(6.0, 2.3, 's', color='white', markersize=8)  # Algorithm icon
ax.plot(8.2, 2.3, '*', color='white', markersize=10) # Processing icon

# Horizontal Process Arrows
ax.annotate("", xy=(3.2, 2.3), xytext=(2.2, 2.3), arrowprops=dict(arrowstyle="->", lw=2, color='#111111'))
ax.annotate("", xy=(5.4, 2.3), xytext=(4.4, 2.3), arrowprops=dict(arrowstyle="->", lw=2, color='#111111'))
ax.annotate("", xy=(7.6, 2.3), xytext=(6.6, 2.3), arrowprops=dict(arrowstyle="->", lw=2, color='#111111'))

# 4. Output Section
ax.annotate("", xy=(9.2, 2.3), xytext=(8.8, 2.3), arrowprops=dict(arrowstyle="-", lw=2, color='#111111'))

# Split lines to grouped outputs
ax.plot([9.2, 9.2], [1.0, 3.6], color='#111111', lw=2)
ax.annotate("", xy=(9.7, 3.6), xytext=(9.2, 3.6), arrowprops=dict(arrowstyle="->", lw=2, color='#111111'))
ax.annotate("", xy=(9.7, 2.3), xytext=(9.2, 2.3), arrowprops=dict(arrowstyle="->", lw=2, color='#111111'))
ax.annotate("", xy=(9.7, 1.0), xytext=(9.2, 1.0), arrowprops=dict(arrowstyle="->", lw=2, color='#111111'))

# Output Box (Top Right Green Header)
rect_output = patches.FancyBboxPatch((9.5, 4.0), 1.8, 1.0, boxstyle="round,pad=0.05,rounding_size=0.1",
                                     facecolor='#1E7E43', edgecolor='none')
ax.add_patch(rect_output)
ax.text(10.4, 4.5, "Output", fontsize=12, fontweight='bold', color='white', ha='center', va='center')

# Grouped Output Shapes (Circles, Triangles, Squares)
ax.plot(10.2, 3.6, 'o', color='#1E7E43', markersize=16)
ax.plot(10.2, 2.3, '^', color='#2196F3', markersize=16)
ax.plot(10.2, 1.0, 's', color='#FF9800', markersize=16)

# Bottom Footer Tag
rect_footer = patches.FancyBboxPatch((4.9, 0.3), 2.2, 0.45, boxstyle="square,pad=0",
                                     facecolor='#C8E6C9', edgecolor='none')
ax.add_patch(rect_footer)
ax.text(6.0, 0.52, "Model training", fontsize=12, color='#111111', ha='center', va='center')

# Save PNG image with crisp formatting
plt.tight_layout()
plt.savefig('unsupervised_learning_fixed.png', dpi=300, bbox_inches='tight')
plt.show()