"""Plot power and temperature from data/local_gpu_trace.csv on twin axes."""

import matplotlib.pyplot as plt
import pandas as pd

from configs import DATA_DIR

TRACE_PATH = DATA_DIR / "local_gpu_trace.csv"

df = pd.read_csv(TRACE_PATH)

# GPUs report power in jagged micro-spikes; a rolling average smooths it
df['Power_Smooth'] = df['Power (W)'].rolling(window=5, min_periods=1).mean()

fig, ax1 = plt.subplots(figsize=(12, 6))

color1 = '#d62728'
ax1.set_xlabel('Time (Seconds)', fontweight='bold')
ax1.set_ylabel('Power Draw (Watts)', color=color1, fontweight='bold')
ax1.plot(df['Time (s)'], df['Power_Smooth'], color=color1, linewidth=2, label='Power (W)')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle='--', alpha=0.5)

ax2 = ax1.twinx()
color2 = '#1f77b4'
ax2.set_ylabel('Temperature (°C)', color=color2, fontweight='bold')
ax2.plot(df['Time (s)'], df['Temperature (C)'], color=color2, linewidth=3, label='Temp (°C)')
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Multi-Job LLM Workload: Power vs. Thermal Stacking', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.show()
