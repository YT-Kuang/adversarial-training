import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ['FGSM', 'BIM', 'PGD']
baseline_accuracy = [0.93, 0.93, 0.93]
adversarial_accuracy = [0.41, 0.19, 0.19]
robust_accuracy = [0.83, 0.66, 0.66]

# Bar width and positions
bar_width = 0.25
x = np.arange(len(categories))

# Define custom pastel colors
custom_colors = ['#ff69b4', '#ffd700', '#add8e6']

# Plotting with new colors
plt.figure(figsize=(10, 6))
plt.bar(x - bar_width, baseline_accuracy, width=bar_width, label='Baseline CNN Accuracy', color=custom_colors[0])
plt.bar(x, adversarial_accuracy, width=bar_width, label='Adversarial Sample Accuracy', color=custom_colors[1])
plt.bar(x + bar_width, robust_accuracy, width=bar_width, label='Robust CNN Accuracy', color=custom_colors[2])

# Labels and title
plt.xlabel('Attack Methods', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('CNN Accuracy under Different Attack Scenarios', fontsize=14)
plt.xticks(x, categories)
plt.ylim(0, 1)
plt.legend()

# Display plot
plt.tight_layout()
plt.show()