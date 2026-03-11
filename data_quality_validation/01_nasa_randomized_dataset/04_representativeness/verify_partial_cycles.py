"""
verify_partial_cycles.py

This script detects and analyzes partial charge-discharge cycles in the NASA Randomized Battery Dataset.
Partial cycles are common in real-world applications where batteries are rarely fully depleted or fully charged.

WHAT WE ANALYZE:
1. Cycle completeness (full vs partial cycles)
2. Depth of Discharge (DoD) distribution
3. State of Charge (SoC) ranges covered
4. Pattern of partial cycling (if any)

CYCLE DEFINITIONS:
- Full cycle: Complete charge-discharge sequence with SoC from near 0% to near 100%
- Partial cycle: Incomplete charge or discharge (doesn't reach full range)
- Micro-cycle: Very small charge-discharge swings
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration
VOLTAGE_MIN_GUESS = 6.0  # Approximate minimum voltage (near 0% SoC)
VOLTAGE_MAX_GUESS = 8.4  # Approximate maximum voltage (near 100% SoC)
SOC_THRESHOLD_FULL = 0.85  # 85% of voltage range considered "full" cycle

# Mode values
MODE_DISCHARGE = -1
MODE_REST = 0
MODE_CHARGE = 1

def estimate_soc_from_voltage(voltage, vmin=VOLTAGE_MIN_GUESS, vmax=VOLTAGE_MAX_GUESS):
    """
    Rough estimate of State of Charge from voltage.
    This is approximate - actual SoC depends on chemistry, temperature, etc.
    """
    if pd.isna(voltage):
        return None
    # Simple linear interpolation (not accurate for all chemistries, but good enough for pattern detection)
    soc = (voltage - vmin) / (vmax - vmin) * 100
    return np.clip(soc, 0, 100)

def detect_cycles(df):
    """
    Detect charge-discharge cycles from mode and voltage data.
    Returns list of cycles with their characteristics.
    """
    cycles = []
    
    if 'mode' not in df.columns or 'voltage_charger' not in df.columns:
        return cycles
    
    # Find transitions between modes
    mode_changes = df['mode'].diff().fillna(0) != 0
    change_indices = df.index[mode_changes].tolist()
    
    # Add start and end indices
    if len(df) > 0:
        change_indices = [0] + change_indices + [len(df)-1]
        change_indices = sorted(set(change_indices))
    
    # Identify cycle segments
    current_cycle = []
    cycle_start_idx = 0
    
    for i in range(1, len(change_indices)):
        start_idx = change_indices[i-1]
        end_idx = change_indices[i]
        
        segment_modes = df.loc[start_idx:end_idx, 'mode'].value_counts()
        
        # If we have both charge and discharge in this segment, it's a cycle
        if (MODE_CHARGE in segment_modes.index and MODE_DISCHARGE in segment_modes.index):
            # Extract cycle data
            cycle_data = df.loc[start_idx:end_idx].copy()
            
            # Get voltage range during this cycle
            voltages = cycle_data['voltage_charger'].dropna()
            if len(voltages) > 0:
                v_min = voltages.min()
                v_max = voltages.max()
                
                # Estimate SoC range
                soc_min = estimate_soc_from_voltage(v_min)
                soc_max = estimate_soc_from_voltage(v_max)
                
                # Calculate depth of discharge (DoD)
                dod = soc_max - soc_min if soc_max > soc_min else soc_min - soc_max
                
                # Determine if cycle is full or partial
                is_full = dod >= (SOC_THRESHOLD_FULL * 100)
                
                # Get duration
                if 'time' in cycle_data.columns:
                    time_col = 'time'
                elif 'relative_time' in cycle_data.columns:
                    time_col = 'relative_time'
                else:
                    time_col = None
                
                duration = None
                if time_col and len(cycle_data) > 1:
                    time_vals = cycle_data[time_col].dropna()
                    if len(time_vals) > 1:
                        duration = time_vals.max() - time_vals.min()
                
                cycles.append({
                    'start_idx': int(start_idx),
                    'end_idx': int(end_idx),
                    'start_time': float(df.iloc[start_idx][time_col]) if time_col and time_col in df.columns else start_idx,
                    'end_time': float(df.iloc[end_idx][time_col]) if time_col and time_col in df.columns else end_idx,
                    'duration_seconds': float(duration) if duration else None,
                    'voltage_min': float(v_min),
                    'voltage_max': float(v_max),
                    'soc_min': float(soc_min),
                    'soc_max': float(soc_max),
                    'depth_of_discharge': float(dod),
                    'is_full_cycle': bool(is_full),
                    'cycle_type': 'full' if is_full else 'partial'
                })
    
    return cycles

def analyze_partial_cycles(file_path):
    """
    Analyze partial cycles in a single CSV file.
    """
    results = {
        'file': os.path.basename(file_path),
        'file_size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2),
        'total_rows': 0,
        'cycles_detected': 0,
        'full_cycles': 0,
        'partial_cycles': 0,
        'cycle_details': [],
        'dod_distribution': {},
        'soc_coverage': {},
        'error': None
    }
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, low_memory=False)
        results['total_rows'] = len(df)
        
        if len(df) == 0:
            results['error'] = "File is empty"
            return results
        
        # Sort by time if available
        time_col = None
        if 'time' in df.columns:
            time_col = 'time'
        elif 'relative_time' in df.columns:
            time_col = 'relative_time'
        
        if time_col and df[time_col].dtype == 'object':
            df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
            df = df.sort_values(by=time_col).reset_index(drop=True)
        
        # Detect cycles
        cycles = detect_cycles(df)
        results['cycles_detected'] = len(cycles)
        
        if cycles:
            # Count cycle types
            full_cycles = [c for c in cycles if c['is_full_cycle']]
            partial_cycles = [c for c in cycles if not c['is_full_cycle']]
            
            results['full_cycles'] = len(full_cycles)
            results['partial_cycles'] = len(partial_cycles)
            
            # Calculate DoD distribution
            dods = [c['depth_of_discharge'] for c in cycles]
            results['dod_distribution'] = {
                'min': float(np.min(dods)),
                'max': float(np.max(dods)),
                'mean': float(np.mean(dods)),
                'median': float(np.median(dods)),
                'std': float(np.std(dods)),
                'histogram': {
                    '0-10%': int(sum(1 for d in dods if d < 10)),
                    '10-20%': int(sum(1 for d in dods if 10 <= d < 20)),
                    '20-30%': int(sum(1 for d in dods if 20 <= d < 30)),
                    '30-40%': int(sum(1 for d in dods if 30 <= d < 40)),
                    '40-50%': int(sum(1 for d in dods if 40 <= d < 50)),
                    '50-60%': int(sum(1 for d in dods if 50 <= d < 60)),
                    '60-70%': int(sum(1 for d in dods if 60 <= d < 70)),
                    '70-80%': int(sum(1 for d in dods if 70 <= d < 80)),
                    '80-90%': int(sum(1 for d in dods if 80 <= d < 90)),
                    '90-100%': int(sum(1 for d in dods if d >= 90))
                }
            }
            
            # Calculate SoC coverage
            all_soc = []
            for cycle in cycles:
                all_soc.extend([cycle['soc_min'], cycle['soc_max']])
            
            if all_soc:
                results['soc_coverage'] = {
                    'min_soc': float(np.min(all_soc)),
                    'max_soc': float(np.max(all_soc)),
                    'range_covered': float(np.max(all_soc) - np.min(all_soc))
                }
            
            # Store cycle details (limited to first 20 for report)
            results['cycle_details'] = cycles[:20]
            
            # Calculate partial cycle ratio
            results['partial_cycle_ratio'] = len(partial_cycles) / len(cycles) if cycles else 0
            
    except Exception as e:
        results['error'] = str(e)
    
    return results

def scan_folder(folder_path, folder_name):
    """
    Scan all CSV files in a folder for partial cycle analysis.
    """
    print(f"\n{'='*80}")
    print(f"SCANNING FOLDER: {folder_name}")
    print(f"{'='*80}")
    
    folder_results = {
        'folder': folder_name,
        'files_checked': 0,
        'files_with_errors': 0,
        'files_with_cycles': 0,
        'total_cycles': 0,
        'total_full_cycles': 0,
        'total_partial_cycles': 0,
        'file_details': [],
        'summary_stats': {
            'avg_partial_ratio': 0,
            'dod_histogram': {},
            'file_with_most_partial': None
        }
    }
    
    if not os.path.exists(folder_path):
        print(f"  FOLDER NOT FOUND: {folder_path}")
        return folder_results
    
    csv_files = list(Path(folder_path).glob("*.csv"))
    folder_results['files_checked'] = len(csv_files)
    
    if not csv_files:
        print(f"  No CSV files found")
        return folder_results
    
    print(f"  Found {len(csv_files)} CSV files")
    
    partial_ratios = []
    max_partial_ratio = 0
    max_partial_file = None
    
    # Global DoD histogram
    global_dod_hist = {}
    
    for csv_file in sorted(csv_files):
        print(f"  Analyzing: {csv_file.name}")
        results = analyze_partial_cycles(csv_file)
        folder_results['file_details'].append(results)
        
        if results['error']:
            folder_results['files_with_errors'] += 1
            print(f"    ERROR: {results['error']}")
        else:
            if results['cycles_detected'] > 0:
                folder_results['files_with_cycles'] += 1
                folder_results['total_cycles'] += results['cycles_detected']
                folder_results['total_full_cycles'] += results['full_cycles']
                folder_results['total_partial_cycles'] += results['partial_cycles']
                
                # Track partial ratio
                partial_ratio = results.get('partial_cycle_ratio', 0)
                partial_ratios.append(partial_ratio)
                
                if partial_ratio > max_partial_ratio:
                    max_partial_ratio = partial_ratio
                    max_partial_file = results['file']
                
                # Update global histogram
                for range_name, count in results['dod_distribution'].get('histogram', {}).items():
                    global_dod_hist[range_name] = global_dod_hist.get(range_name, 0) + count
                
                print(f"    Cycles: {results['cycles_detected']} total, "
                      f"{results['full_cycles']} full, {results['partial_cycles']} partial")
                print(f"    Partial cycle ratio: {results.get('partial_cycle_ratio', 0)*100:.1f}%")
            else:
                print(f"    No cycles detected")
    
    # Calculate summary statistics
    if partial_ratios:
        folder_results['summary_stats']['avg_partial_ratio'] = float(np.mean(partial_ratios))
        folder_results['summary_stats']['max_partial_ratio'] = float(max_partial_ratio)
        folder_results['summary_stats']['file_with_most_partial'] = max_partial_file
        folder_results['summary_stats']['dod_histogram'] = global_dod_hist
    
    return folder_results

def print_summary(all_results):
    """
    Print comprehensive summary of partial cycle analysis.
    """
    print("\n" + "="*100)
    print("PARTIAL CYCLE ANALYSIS - SUMMARY")
    print("="*100)
    print("\nCycle detection parameters:")
    print(f"  • Voltage range assumed: {VOLTAGE_MIN_GUESS}V - {VOLTAGE_MAX_GUESS}V")
    print(f"  • Full cycle threshold: {SOC_THRESHOLD_FULL*100}% of voltage range")
    print("="*100)
    
    total_files = 0
    files_with_errors = 0
    files_with_cycles = 0
    total_cycles = 0
    total_full = 0
    total_partial = 0
    all_partial_ratios = []
    
    # Aggregate histogram
    total_hist = {}
    
    for folder_result in all_results:
        print(f"\nFolder: {folder_result['folder']}")
        print("-" * 60)
        print(f"  Files checked: {folder_result['files_checked']}")
        print(f"  Files with errors: {folder_result['files_with_errors']}")
        print(f"  Files with cycles: {folder_result['files_with_cycles']}")
        print(f"  Total cycles: {folder_result['total_cycles']}")
        print(f"    Full cycles: {folder_result['total_full_cycles']}")
        print(f"    Partial cycles: {folder_result['total_partial_cycles']}")
        print(f"  Partial cycle ratio: {folder_result['summary_stats']['avg_partial_ratio']*100:.1f}% average")
        
        if folder_result['summary_stats']['file_with_most_partial']:
            print(f"  Most partial cycles: {folder_result['summary_stats']['file_with_most_partial']} "
                  f"({folder_result['summary_stats']['max_partial_ratio']*100:.1f}%)")
        
        total_files += folder_result['files_checked']
        files_with_errors += folder_result['files_with_errors']
        files_with_cycles += folder_result['files_with_cycles']
        total_cycles += folder_result['total_cycles']
        total_full += folder_result['total_full_cycles']
        total_partial += folder_result['total_partial_cycles']
        
        if folder_result['summary_stats']['avg_partial_ratio'] > 0:
            all_partial_ratios.append(folder_result['summary_stats']['avg_partial_ratio'])
        
        # Aggregate histogram
        for range_name, count in folder_result['summary_stats']['dod_histogram'].items():
            total_hist[range_name] = total_hist.get(range_name, 0) + count
    
    print("\n" + "="*100)
    print("DEPTH OF DISCHARGE (DoD) DISTRIBUTION")
    print("="*100)
    
    if total_hist:
        print("\nDoD Range    | Count | Percentage")
        print("-" * 40)
        for range_name in ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', 
                           '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']:
            count = total_hist.get(range_name, 0)
            percentage = (count / total_cycles * 100) if total_cycles > 0 else 0
            bar = '█' * int(percentage / 2)
            print(f"  {range_name:6s} | {count:5d} | {percentage:5.1f}% {bar}")
    
    print("\n" + "="*100)
    print("FINAL QUALITY ASSESSMENT - PARTIAL CYCLE PRESENCE")
    print("="*100)
    
    # Calculate overall partial cycle ratio
    overall_partial_ratio = total_partial / total_cycles if total_cycles > 0 else 0
    
    print(f"\nOverall statistics:")
    print(f"  • Total cycles across all files: {total_cycles}")
    print(f"  • Full cycles: {total_full} ({total_full/total_cycles*100:.1f}%)")
    print(f"  • Partial cycles: {total_partial} ({total_partial/total_cycles*100:.1f}%)")
    
    # Quality assessment based on partial cycle presence
    if overall_partial_ratio > 0.3:
        print("\nRESULT: EXCELLENT (++)")
        print("="*40)
        print("High presence of partial cycles (>30%):")
        print("  ✓ Excellent representation of real-world usage")
        print("  ✓ Batteries are not always fully cycled")
        print("  ✓ Dataset captures realistic operating patterns")
        print("\nThe dataset EXCELLENTLY satisfies the partial cycle criterion.")
        
    elif overall_partial_ratio > 0.1:
        print("\nRESULT: GOOD (+)")
        print("="*40)
        print("Moderate presence of partial cycles (10-30%):")
        print("  ✓ Some real-world variation present")
        print("  ⚠ Mostly full cycles but partials exist")
        print("\nThe dataset GOODly satisfies the partial cycle criterion.")
        
    elif overall_partial_ratio > 0.01:
        print("\nRESULT: ACCEPTABLE (o)")
        print("="*40)
        print(f"Few partial cycles ({overall_partial_ratio*100:.1f}%):")
        print("  • Dataset primarily uses full cycles")
        print("  • Some partial cycles present")
        print("\nThe dataset ACCEPTABLY satisfies the partial cycle criterion.")
        
    else:
        print("\nRESULT: POOR (-)")
        print("="*40)
        print("Very few or no partial cycles detected:")
        print("  ✗ Dataset uses almost exclusively full cycles")
        print("  ✗ Poor representation of real-world usage")
        print("\nThe dataset POORLY satisfies the partial cycle criterion.")
        print("Recommendation: Supplement with datasets containing partial cycles")

def save_report(all_results, output_file="partial_cycles_report.txt"):
    """
    Save detailed report to file.
    """
    with open(output_file, 'w', encoding='ascii', errors='replace') as f:
        f.write("="*100 + "\n")
        f.write("NASA BATTERY DATASET - PARTIAL CYCLE ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*100 + "\n\n")
        f.write(f"Analysis parameters:\n")
        f.write(f"  Voltage range: {VOLTAGE_MIN_GUESS}V - {VOLTAGE_MAX_GUESS}V\n")
        f.write(f"  Full cycle threshold: {SOC_THRESHOLD_FULL*100}% of range\n\n")
        
        for folder_result in all_results:
            f.write(f"\nFOLDER: {folder_result['folder']}\n")
            f.write("-"*60 + "\n")
            
            for file_result in folder_result['file_details']:
                f.write(f"\nFile: {file_result['file']} ({file_result['file_size_mb']} MB)\n")
                f.write(f"  Rows: {file_result['total_rows']:,}\n")
                
                if file_result['error']:
                    f.write(f"  ERROR: {file_result['error']}\n")
                    continue
                
                if file_result['cycles_detected'] > 0:
                    f.write(f"  Cycles detected: {file_result['cycles_detected']}\n")
                    f.write(f"    Full cycles: {file_result['full_cycles']}\n")
                    f.write(f"    Partial cycles: {file_result['partial_cycles']}\n")
                    f.write(f"    Partial cycle ratio: {file_result.get('partial_cycle_ratio', 0)*100:.1f}%\n")
                    
                    if file_result.get('dod_distribution'):
                        f.write(f"  Depth of Discharge (DoD) distribution:\n")
                        for range_name, count in file_result['dod_distribution'].get('histogram', {}).items():
                            if count > 0:
                                f.write(f"    {range_name}: {count} cycles\n")
                    
                    # Write first few cycle details
                    if file_result['cycle_details']:
                        f.write(f"  Sample cycles (first 5):\n")
                        for i, cycle in enumerate(file_result['cycle_details'][:5]):
                            f.write(f"    Cycle {i+1}: {cycle['cycle_type']}, "
                                   f"DoD={cycle['depth_of_discharge']:.1f}%, "
                                   f"voltage [{cycle['voltage_min']:.2f}V - {cycle['voltage_max']:.2f}V]\n")
                else:
                    f.write(f"  No cycles detected\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*100 + "\n")
    
    print(f"\nDetailed report saved to: {output_file}")

def main():
    base_path = r"C:\Users\admin\Desktop\DR2\11 All Datasets\02 NASA Randomized Battery Dataset\battery_alt_dataset"
    
    folders = [
        ('regular_alt_batteries', os.path.join(base_path, 'regular_alt_batteries')),
        ('recommissioned_batteries', os.path.join(base_path, 'recommissioned_batteries')),
        ('second_life_batteries', os.path.join(base_path, 'second_life_batteries'))
    ]
    
    print("="*100)
    print("NASA RANDOMIZED BATTERY DATASET - PARTIAL CYCLE ANALYSIS")
    print("="*100)
    print("\nAnalyzing charge-discharge cycles for partial cycles:")
    print("  • Detecting full vs partial cycles")
    print("  • Calculating Depth of Discharge (DoD)")
    print("  • Measuring real-world usage representation")
    print(f"  • Full cycle threshold: {SOC_THRESHOLD_FULL*100}% of voltage range")
    
    all_results = []
    for folder_name, folder_path in folders:
        if os.path.exists(folder_path):
            results = scan_folder(folder_path, folder_name)
            all_results.append(results)
    
    print_summary(all_results)
    save_report(all_results)
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)

if __name__ == "__main__":
    main()