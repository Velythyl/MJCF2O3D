import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.switch_backend("Tkagg")

# Define the RGBA colors
data = {
    "right_hip_y": (1.0, 0.0, 0.0, 1.0),
    "left_hip_y": (1.0, 0.75, 0.0, 1.0),
    "right_shoulder1": (0.5, 1.0, 0.0, 1.0),
    "left_shoulder1": (0.0, 1.0, 0.25, 1.0),
    "right_knee": (0.0, 1.0, 1.0, 1.0),
    "left_knee": (0.0, 0.25, 1.0, 1.0),
    "right_elbow": (0.5, 0.0, 1.0, 1.0),
    "left_elbow": (1.0, 0.0, 0.75, 1.0),
}

# Extract labels and colors
labels = list(data.keys())
colors = list(data.values())
colors = list(map(np.array, colors))
colors = list(map(lambda x: (x.astype(int)*255).astype(float)/255, colors))

# Create the bar plot
plt.figure(figsize=(8, 4))
sns.barplot(x=np.arange(len(labels)), y=[1]*len(labels), palette=colors)
plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45, ha='right')
plt.yticks([])  # Hide y-axis
plt.title("Seaborn Plot of RGBA Colors")
plt.show()
