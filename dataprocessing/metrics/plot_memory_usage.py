import numpy as np
import matplotlib.pyplot as plt

# Load the data from the log file
data = np.genfromtxt('metrics/memory_usage.log', delimiter=',', names=True, dtype=None, encoding=None)

# Extract columns
pages = data['page']
ram_used_gb = data['ram_used_gb']
ram_percent = data['ram_percent']
gpu_used_mb = data['gpu_used_mb']
gpu_total_mb = data['gpu_total_mb']

# Create subplots for multiple graphs
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Plot RAM Used (GB)
axs[0].plot(pages, ram_used_gb, label='RAM Used (GB)', color='blue')
axs[0].set_title('RAM Usage Over Pages')
axs[0].set_xlabel('Page')
axs[0].set_ylabel('RAM Used (GB)')
axs[0].grid(True)

# Plot RAM Percent
axs[1].plot(pages, ram_percent, label='RAM Percent', color='green')
axs[1].set_title('RAM Percentage Over Pages')
axs[1].set_xlabel('Page')
axs[1].set_ylabel('RAM Percent (%)')
axs[1].grid(True)

# Plot GPU Used (MB)
axs[2].plot(pages, gpu_used_mb, label='GPU Used (MB)', color='red')
axs[2].set_title('GPU Usage Over Pages')
axs[2].set_xlabel('Page')
axs[2].set_ylabel('GPU Used (MB)')
axs[2].grid(True)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('memory_usage_graphs.png')

# Show the plot (if running in an environment that supports it)
plt.show()