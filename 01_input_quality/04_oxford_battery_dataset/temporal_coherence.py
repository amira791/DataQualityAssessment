# %% Simple Oxford Battery Degradation Analysis in Python
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

# %% Load the data
print("Loading data...")

# Load the .mat file
data = loadmat('C:\\Users\\admin\\Desktop\\DR2\\11 All Datasets\\05 Oxford Battery Degradation Dataset\\Oxford_Battery_Degradation_Dataset_1.mat')

# The data structure is a bit different in Python
# Let's explore what's in the file
print("Variables in file:", data.keys())

# The actual battery data is usually under 'data' or similar
# Let's find the right key
battery_data = None
for key in data.keys():
    if key not in ['__header__', '__version__', '__globals__']:
        battery_data = data[key]
        print(f"Using data from: {key}")
        break

# %% Extract data for all cells
print("\nExtracting capacity data...")

# Create figure for subplots
fig, axes = plt.subplots(2, 4, figsize=(16, 10))
axes = axes.flatten()

# Store summary data
summary = []

# Loop through all 8 cells
for cell in range(8):  # Cells are 0-7 in Python indexing
    cell_idx = cell + 1  # Cell number for display
    cell_data = battery_data[0, cell]['cell' + str(cell_idx)]
    
    # Get all cycle names
    cycle_names = []
    for name in cell_data.dtype.names:
        if name.startswith('cyc'):
            cycle_names.append(name)
    
    # Sort cycles by number
    cycle_numbers = []
    for name in cycle_names:
        num = int(name.replace('cyc', ''))
        cycle_numbers.append(num)
    
    # Sort both lists together
    sorted_pairs = sorted(zip(cycle_numbers, cycle_names))
    cycle_numbers = [pair[0] for pair in sorted_pairs]
    cycle_names = [pair[1] for pair in sorted_pairs]
    
    # Extract capacities
    capacities = []
    valid_cycles = []
    
    for i, (cycle_num, cycle_name) in enumerate(zip(cycle_numbers, cycle_names)):
        try:
            # Get 1C discharge data
            c1dc_data = cell_data[cycle_name][0, 0]['C1dc'][0, 0]
            
            # Capacity is the last value in q array
            q_data = c1dc_data['q'].flatten()
            if len(q_data) > 0:
                cap = q_data[-1]
                capacities.append(cap)
                valid_cycles.append(cycle_num)
        except:
            # Skip if data not available
            continue
    
    # Convert to numpy arrays
    valid_cycles = np.array(valid_cycles)
    capacities = np.array(capacities)
    
    # Plot in subplot
    ax = axes[cell]
    ax.plot(valid_cycles, capacities, 'b-', linewidth=2)
    ax.plot(valid_cycles, capacities, 'bo', markersize=4)
    
    # Add reference lines
    ax.axhline(y=740, color='k', linestyle='--', linewidth=1, label='Nominal 740mAh')
    ax.axhline(y=592, color='r', linestyle='--', linewidth=1, label='80% (EOL)')
    
    # Check for non-monotonic behavior (capacity increases)
    for j in range(1, len(capacities)):
        if capacities[j] > capacities[j-1] + 0.1:  # Increased by more than 0.1 mAh
            ax.plot(valid_cycles[j], capacities[j], 'ro', 
                   markersize=10, markerfacecolor='r')
    
    # Labels and formatting
    ax.set_xlabel('Cycle Number')
    ax.set_ylabel('Capacity (mAh)')
    ax.set_title(f'Cell {cell_idx}')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2100)
    ax.set_ylim(500, 800)
    
    # Add text with initial and final capacity
    if len(capacities) > 0:
        ax.text(100, 550, f'Start: {capacities[0]:.0f} mAh\nEnd: {capacities[-1]:.0f} mAh', 
                fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
    
    # Store for summary
    increases = sum(1 for j in range(1, len(capacities)) if capacities[j] > capacities[j-1] + 0.1)
    summary.append({
        'cell': cell_idx,
        'initial': capacities[0] if len(capacities) > 0 else 0,
        'final': capacities[-1] if len(capacities) > 0 else 0,
        'loss': capacities[0] - capacities[-1] if len(capacities) > 0 else 0,
        'increases': increases,
        'cycles': len(valid_cycles)
    })

# Add legend to first subplot
axes[0].legend(loc='lower left', fontsize=8)

# Main title
plt.suptitle('Oxford Battery Dataset - Capacity Degradation Over Time', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% Print summary table
print("\n" + "="*50)
print("DEGRADATION SUMMARY")
print("="*50)
print(f"{'Cell':<6} {'Initial':<10} {'Final':<10} {'Loss':<10} {'Cycles':<8} {'Status'}")
print(f"{'':<6} {'(mAh)':<10} {'(mAh)':<10} {'(mAh)':<10} {'':<8} {' '}")
print("-"*60)

for s in summary:
    if s['increases'] > 0:
        status = f"⚠️ {s['increases']} inc"
    else:
        status = "✓ Good"
    
    print(f"Cell {s['cell']:<3} {s['initial']:<10.0f} {s['final']:<10.0f} "
          f"{s['loss']:<10.0f} {s['cycles']:<8} {status}")

print("="*50)

# %% Additional plot: All cells together for comparison
plt.figure(figsize=(12, 6))
colors = plt.cm.tab10(np.linspace(0, 1, 8))

for cell in range(8):
    cell_idx = cell + 1
    cell_data = battery_data[0, cell]['cell' + str(cell_idx)]
    
    # Get cycle names and numbers
    cycle_names = [name for name in cell_data.dtype.names if name.startswith('cyc')]
    cycle_numbers = [int(name.replace('cyc', '')) for name in cycle_names]
    
    # Sort
    sorted_pairs = sorted(zip(cycle_numbers, cycle_names))
    cycle_numbers = [pair[0] for pair in sorted_pairs]
    cycle_names = [pair[1] for pair in sorted_pairs]
    
    # Get capacities
    capacities = []
    valid_cycles = []
    
    for cycle_num, cycle_name in zip(cycle_numbers, cycle_names):
        try:
            c1dc_data = cell_data[cycle_name][0, 0]['C1dc'][0, 0]
            q_data = c1dc_data['q'].flatten()
            if len(q_data) > 0:
                capacities.append(q_data[-1])
                valid_cycles.append(cycle_num)
        except:
            continue
    
    if len(capacities) > 0:
        plt.plot(valid_cycles, capacities, 'o-', color=colors[cell], 
                linewidth=1.5, markersize=4, label=f'Cell {cell_idx}')

plt.axhline(y=740, color='k', linestyle='--', linewidth=1, label='Nominal')
plt.axhline(y=592, color='r', linestyle='--', linewidth=1, label='80% EOL')
plt.xlabel('Cycle Number')
plt.ylabel('Capacity (mAh)')
plt.title('All Cells - Capacity Degradation Comparison')
plt.legend(loc='best', ncol=2)
plt.grid(True, alpha=0.3)
plt.xlim(0, 2100)
plt.ylim(500, 800)
plt.tight_layout()
plt.show()

print("\nAnalysis complete!")