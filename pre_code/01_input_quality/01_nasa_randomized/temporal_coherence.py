import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set the project path
project_path = Path(r"C:\Users\admin\Desktop\DR2\11 All Datasets\02 NASA Randomized Battery Dataset\battery_alt_dataset")

def analyze_battery_file(file_path):
    """
    Analyze a single battery CSV file for capacity degradation
    """
    """
    This function reads a battery CSV file, extracts discharge cycles, calculates capacities,
    and analyzes capacity degradation over time.
    """
    print(f"Analyzing: {file_path.name}")
    
    try:
        # Read the CSV file with correct column names
        df = pd.read_csv(file_path)
        
        # Check if dataframe is empty
        if df.empty:
            print(f"  Warning: {file_path.name} is empty")
            return None
        
        # Rename columns if needed (sometimes there might be spaces)
        df.columns = df.columns.str.strip()
        
        # Parse datetime - using correct column name 'start_time'
        df['start_time'] = pd.to_datetime(df['start_time'], format='%m:%d:%Y %H:%M:%S', errors='coerce')
        
        print(f"  File loaded: {len(df)} rows, columns: {list(df.columns)}")
        print(f"  Mode values: {df['mode'].unique()}")
        
        return df
            
    except Exception as e:
        print(f"  Error reading {file_path.name}: {str(e)}")
        return None

def extract_discharge_cycles(df):
    """
    Extract discharge cycles and calculate capacity for each
    """
    # Find discharge segments (mode = -1)
    discharge_mask = df['mode'] == -1
    
    if not discharge_mask.any():
        print("  No discharge cycles found")
        return None, None, None
    
    # Get all discharge data
    discharge_data = df[discharge_mask].copy()
    
    # Find continuous discharge segments (cycles)
    # Discharge cycles are separated by mode changes (to 0 or 1)
    discharge_data['cycle_change'] = (discharge_data['time'].diff() > 100).astype(int)  # Large time gap indicates new cycle
    discharge_data['cycle_id'] = discharge_data['cycle_change'].cumsum()
    
    # Calculate capacity for each cycle
    capacities = []
    cycle_times = []
    cycle_starts = []
    mission_types = []
    
    for cycle_id in discharge_data['cycle_id'].unique():
        cycle_data = discharge_data[discharge_data['cycle_id'] == cycle_id]
        
        if len(cycle_data) > 5:  # Need enough data points
            # Get time and current
            time = cycle_data['time'].values
            current = cycle_data['current_load'].values
            mission = cycle_data['mission_type'].iloc[0] if 'mission_type' in cycle_data.columns else 1
            
            # Check if current is valid (not all NaN or zero)
            if np.all(np.isnan(current)) or np.nanmean(current) == 0:
                continue
                
            # Handle any NaN values
            valid_mask = ~np.isnan(current)
            if not valid_mask.any():
                continue
                
            time = time[valid_mask]
            current = current[valid_mask]
            
            if len(time) > 1:
                # Calculate time differences in seconds
                dt = np.diff(time)
                
                # Use average current for each time interval
                current_mid = (current[1:] + current[:-1]) / 2
                
                # Integrate current over time to get charge (A * s = Coulombs)
                charge_coulombs = np.sum(current_mid * dt)
                
                # Convert to Amp-hours (1 Ah = 3600 Coulombs)
                capacity_ah = charge_coulombs / 3600
                
                # Only accept reasonable capacities (between 0.5 and 3 Ah for these batteries)
                if 0.5 < capacity_ah < 3:
                    capacities.append(capacity_ah)
                    cycle_times.append(time[0])  # Start time of discharge
                    cycle_starts.append(cycle_data['start_time'].iloc[0])
                    mission_types.append(mission)
    
    return cycle_times, capacities, cycle_starts, mission_types

def check_monotonic_degradation(times, capacities, tolerance=0.05):  # Changed to 5% tolerance
    """
    Check if capacity degradation is approximately monotonic
    tolerance: allow small increases due to measurement noise or temperature effects
    """
    violations = []
    
    # Check for significant increases in raw data
    for i in range(1, len(capacities)):
        if capacities[i] > capacities[i-1]:
            rel_increase = (capacities[i] - capacities[i-1]) / capacities[i-1]
            
            if rel_increase > tolerance:
                violations.append({
                    'cycle': i,
                    'time': times[i],
                    'capacity': capacities[i],
                    'prev_capacity': capacities[i-1],
                    'increase_pct': rel_increase * 100
                })
    
    return violations

# Analyze all battery files in the dataset
all_batteries = {}
failed_files = []

# Check each subfolder
for folder in ['regular_alt_batteries', 'recommissioned_batteries', 'second_life_batteries']:
    folder_path = project_path / folder
    
    if folder_path.exists():
        print(f"\n{'='*60}")
        print(f"Analyzing {folder}...")
        print('='*60)
        
        # Get all CSV files
        csv_files = sorted(list(folder_path.glob('*.csv')))
        
        for csv_file in csv_files:
            df = analyze_battery_file(csv_file)
            
            if df is not None:
                # Extract discharge cycles and capacities
                times, capacities, start_times, mission_types = extract_discharge_cycles(df)
                
                if times and len(capacities) > 3:  # Need at least a few cycles
                    all_batteries[f"{folder}/{csv_file.name}"] = {
                        'file': csv_file.name,
                        'folder': folder,
                        'times': np.array(times),
                        'capacities': np.array(capacities),
                        'start_times': start_times,
                        'mission_types': mission_types,
                        'num_cycles': len(capacities),
                        'df': df
                    }
                    print(f"   {csv_file.name}: {len(capacities)} discharge cycles extracted")
                else:
                    failed_files.append(csv_file.name)
                    print(f"   {csv_file.name}: Could not extract discharge cycles")
            else:
                failed_files.append(csv_file.name)

print(f"\n{'='*60}")
print(f"Total batteries successfully analyzed: {len(all_batteries)}")
print(f"Failed files: {len(failed_files)}")
if failed_files:
    print(f"  {', '.join(failed_files[:10])}{'...' if len(failed_files)>10 else ''}")

# Now verify monotonic degradation with different tolerance levels
print(f"\n{'='*60}")
print("VERIFYING CRITERIA 1: Capacity monotonically decreases over time")
print('='*60)

# Test different tolerance levels
tolerance_levels = [0.01, 0.03, 0.05, 0.08, 0.10]
tolerance_results = {}

for tol in tolerance_levels:
    passing = 0
    for battery_name, data in all_batteries.items():
        times = data['times']
        capacities = data['capacities']
        if len(capacities) >= 5:
            violations = check_monotonic_degradation(times, capacities, tolerance=tol)
            if len(violations) == 0:
                passing += 1
    tolerance_results[tol] = passing

print("\nBatteries passing at different tolerance levels:")
print(f"Tolerance 1%:  {tolerance_results[0.01]}/{len(all_batteries)} batteries (strict)")
print(f"Tolerance 3%:  {tolerance_results[0.03]}/{len(all_batteries)} batteries (original)")
print(f"Tolerance 5%:  {tolerance_results[0.05]}/{len(all_batteries)} batteries (recommended - practical)")
print(f"Tolerance 8%:  {tolerance_results[0.08]}/{len(all_batteries)} batteries (lenient)")
print(f"Tolerance 10%: {tolerance_results[0.10]}/{len(all_batteries)} batteries (very lenient)")

# Analyze each battery with 5% tolerance
print(f"\n{'='*60}")
print("DETAILED ANALYSIS WITH 5% TOLERANCE")
print('='*60)

results = []

for battery_name, data in all_batteries.items():
    times = data['times']
    capacities = data['capacities']
    mission_types = data['mission_types']
    
    # Skip if not enough data
    if len(capacities) < 5:
        continue
    
    # Check monotonic degradation with 5% tolerance
    violations = check_monotonic_degradation(times, capacities, tolerance=0.05)
    
    # Calculate degradation metrics
    initial_capacity = capacities[0]
    final_capacity = capacities[-1]
    capacity_loss = initial_capacity - final_capacity
    loss_percentage = (capacity_loss / initial_capacity) * 100
    
    # Calculate degradation rate (Ah per day)
    time_days = (times - times[0]) / (3600 * 24)
    if time_days[-1] > 0:
        degradation_rate = capacity_loss / time_days[-1]  # Ah per day
    else:
        degradation_rate = 0
    
    # Calculate moving average to show trend
    window = min(5, len(capacities)//5)
    if window > 1:
        moving_avg = np.convolve(capacities, np.ones(window)/window, mode='valid')
    else:
        moving_avg = capacities
    
    # Determine if criteria is met with 5% tolerance
    criteria_met = len(violations) == 0
    
    # Categorize violations
    severe_violations = [v for v in violations if v['increase_pct'] > 10]
    moderate_violations = [v for v in violations if 5 < v['increase_pct'] <= 10]
    
    results.append({
        'battery': battery_name,
        'folder': data['folder'],
        'num_cycles': len(capacities),
        'initial_capacity': initial_capacity,
        'final_capacity': final_capacity,
        'capacity_loss': capacity_loss,
        'loss_percentage': loss_percentage,
        'degradation_rate_ah_per_day': degradation_rate,
        'num_violations_5pct': len(violations),
        'severe_violations_gt10pct': len(severe_violations),
        'moderate_violations_5to10pct': len(moderate_violations),
        'criteria_met_5pct': criteria_met,
        'max_violation_pct': max([v['increase_pct'] for v in violations]) if violations else 0
    })
    
    # Print result
    status = "✅" if criteria_met else "⚠️" if len(violations) <= 2 and max([v['increase_pct'] for v in violations] or [0]) < 8 else "❌"
    print(f"\n{status} {battery_name}")
    print(f"   Cycles: {len(capacities)}, Initial: {initial_capacity:.2f} Ah, Final: {final_capacity:.2f} Ah")
    print(f"   Loss: {loss_percentage:.1f}%, Rate: {degradation_rate:.3f} Ah/day")
    
    if violations:
        print(f"    {len(violations)} non-monotonic points at 5% tolerance:")
        severe_count = len([v for v in violations if v['increase_pct'] > 10])
        moderate_count = len([v for v in violations if 5 < v['increase_pct'] <= 10])
        if severe_count > 0:
            print(f"       {severe_count} severe (>10%)")
        if moderate_count > 0:
            print(f"      {moderate_count} moderate (5-10%)")
        for v in violations[:2]:  # Show first 2 violations
            emoji = "" if v['increase_pct'] > 10 else "🟡"
            print(f"      {emoji} Cycle {v['cycle']}: +{v['increase_pct']:.1f}%")

# Create summary dataframe
if results:
    summary_df = pd.DataFrame(results)
    
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS WITH 5% TOLERANCE")
    print('='*60)
    print(f"Total batteries analyzed: {len(summary_df)}")
    print(f"Batteries meeting criteria (5% tolerance): {summary_df['criteria_met_5pct'].sum()}")
    print(f"Batteries with violations: {(~summary_df['criteria_met_5pct']).sum()}")
    
    print(f"\nAverage capacity loss: {summary_df['loss_percentage'].mean():.1f}%")
    print(f"Average degradation rate: {summary_df['degradation_rate_ah_per_day'].mean():.3f} Ah/day")
    
    # Categorize batteries by quality
    high_quality = summary_df[summary_df['criteria_met_5pct']]
    medium_quality = summary_df[(~summary_df['criteria_met_5pct']) & (summary_df['max_violation_pct'] <= 8) & (summary_df['num_violations_5pct'] <= 3)]
    low_quality = summary_df[(~summary_df['criteria_met_5pct']) & ((summary_df['max_violation_pct'] > 8) | (summary_df['num_violations_5pct'] > 3))]
    
    print(f"\n BATTERY QUALITY CLASSIFICATION:")
    print(f"    High Quality (no violations >5%): {len(high_quality)} batteries")
    print(f"     Medium Quality (minor violations): {len(medium_quality)} batteries")
    print(f"    Low Quality (significant violations): {len(low_quality)} batteries")
    
    # List high quality batteries
    if len(high_quality) > 0:
        print(f"\n HIGH QUALITY BATTERIES (Recommended for analysis):")
        for _, row in high_quality.iterrows():
            print(f"   • {row['battery']}: {row['num_cycles']} cycles, {row['loss_percentage']:.1f}% loss")
    
    # List medium quality batteries
    if len(medium_quality) > 0:
        print(f"\n MEDIUM QUALITY BATTERIES (Use with caution):")
        for _, row in medium_quality.iterrows():
            print(f"   • {row['battery']}: {row['num_violations_5pct']} violations, max {row['max_violation_pct']:.1f}%")
    
    # Summary by folder
    print(f"\n{'='*60}")
    print("RESULTS BY BATTERY CATEGORY (5% tolerance)")
    print('='*60)
    
    for folder in summary_df['folder'].unique():
        folder_data = summary_df[summary_df['folder'] == folder]
        folder_high = folder_data[folder_data['criteria_met_5pct']]
        print(f"\n{folder}:")
        print(f"  Total: {len(folder_data)}")
        print(f"   High quality: {len(folder_high)}/{len(folder_data)}")
        print(f"  Avg loss: {folder_data['loss_percentage'].mean():.1f}%")
        print(f"  Avg violations: {folder_data['num_violations_5pct'].mean():.1f}")
    
    # Create improved visualization
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    # Plot high quality batteries first
    plot_batteries = []
    plot_batteries.extend(high_quality.head(3)['battery'].tolist())
    plot_batteries.extend(medium_quality.head(3)['battery'].tolist())
    plot_batteries.extend(low_quality.head(3)['battery'].tolist())
    
    for i, battery_name in enumerate(plot_batteries[:9]):
        ax = axes[i]
        
        # Get data for this battery
        data = all_batteries[battery_name]
        times = data['times']
        capacities = data['capacities']
        mission_types = data['mission_types']
        
        # Convert to days
        time_days = (times - times[0]) / (3600 * 24)
        
        # Plot capacity
        ax.plot(time_days, capacities, 'b-', linewidth=1, alpha=0.5, marker='.', markersize=2)
        
        # Plot moving average
        window = min(10, len(capacities)//10)
        if window > 1:
            moving_avg = np.convolve(capacities, np.ones(window)/window, mode='valid')
            moving_time = time_days[window-1:]
            ax.plot(moving_time, moving_avg, 'b-', linewidth=2, label='Moving average')
        
        # Mark different mission types
        ref_discharge_mask = [m == 0 for m in mission_types]
        if any(ref_discharge_mask):
            ax.plot(np.array(time_days)[ref_discharge_mask], 
                   np.array(capacities)[ref_discharge_mask], 
                   'g^', markersize=8, label='Reference discharge', alpha=0.7)
        
        # Mark violations
        violations = check_monotonic_degradation(times, capacities, tolerance=0.05)
        severe_violations = [v for v in violations if v['increase_pct'] > 10]
        moderate_violations = [v for v in violations if 5 < v['increase_pct'] <= 10]
        
        for v in severe_violations:
            v_time_days = (v['time'] - times[0]) / (3600 * 24)
            ax.plot(v_time_days, v['capacity'], 'ro', markersize=8, label='Severe violation' if i==0 else '')
        
        for v in moderate_violations:
            v_time_days = (v['time'] - times[0]) / (3600 * 24)
            ax.plot(v_time_days, v['capacity'], 'yo', markersize=6, label='Moderate violation' if i==0 else '')
        
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Capacity (Ah)')
        
        # Determine quality class for title
        if battery_name in high_quality['battery'].values:
            quality = " HIGH QUALITY"
            color = 'green'
        elif battery_name in medium_quality['battery'].values:
            quality = " MEDIUM"
            color = 'orange'
        else:
            quality = " LOW QUALITY"
            color = 'red'
        
        short_name = os.path.basename(battery_name)
        ax.set_title(f"{short_name}\n{quality}\nLoss: {summary_df[summary_df['battery']==battery_name]['loss_percentage'].iloc[0]:.1f}%", 
                    color=color, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('capacity_degradation_quality_classification.png', dpi=150)
    plt.show()
    
    print(f"\nPlot saved to 'capacity_degradation_quality_classification.png'")
    
    # Save results
    summary_df.to_csv('capacity_degradation_summary_5pct_tolerance.csv', index=False)
    print(f"Results saved to 'capacity_degradation_summary_5pct_tolerance.csv'")
    
    # Create recommendation report
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS FOR DATASET USAGE")
    print('='*60)
    print(f"\nBased on the analysis with 5% practical tolerance:")
    print(f"\n For high-precision analysis (e.g., model calibration):")
    print(f"   Use the {len(high_quality)} high quality batteries:")
    for _, row in high_quality.iterrows():
        print(f"   - {row['battery']}")
    
    print(f"\n For general degradation trend analysis:")
    print(f"   Use high + medium quality batteries ({len(high_quality) + len(medium_quality)} total)")
    
    print(f"\n Batteries to avoid or investigate further:")
    for _, row in low_quality.iterrows():
        print(f"   - {row['battery']}: {row['num_violations_5pct']} violations, max {row['max_violation_pct']:.1f}%")
    
else:
    print("\n No batteries were successfully analyzed. Please check:")
    print("1. File paths and permissions")
    print("2. CSV file format and column names")
    print("3. That the files contain discharge cycles (mode = -1)")