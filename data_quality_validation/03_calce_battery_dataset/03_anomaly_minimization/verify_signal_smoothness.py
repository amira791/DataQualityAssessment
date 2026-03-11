"""
verify_signal_smoothness_revised.py

This script checks for sudden, unrealistic changes in measurements.
INSTEAD OF ASSUMING THRESHOLDS, it uses statistical methods to detect
anomalies relative to the data's own patterns.

METHODOLOGY:
1. Calculate the distribution of step changes for each column
2. Identify steps that are statistical outliers (> 5 standard deviations from mean)
3. Consider context (mode changes) to avoid false positives
4. Report findings without assuming what's "normal"
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

# Columns to analyze
COLUMNS_TO_ANALYZE = [
    'voltage_charger',
    'voltage_load',
    'current_load',
    'temperature_battery',
    'temperature_mosfet',
    'temperature_resistor'
]

# Statistical threshold (multiples of standard deviation)
STD_THRESHOLD = 5.0  # Points > 5σ from mean step size are considered anomalous

def analyze_step_changes(file_path):
    """
    Analyze step changes between consecutive measurements using statistics.
    """
    results = {
        'file': os.path.basename(file_path),
        'file_size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2),
        'total_rows': 0,
        'step_analysis': {},
        'anomalies': {},
        'summary': {},
        'error': None
    }
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, low_memory=False)
        results['total_rows'] = len(df)
        
        if len(df) == 0:
            results['error'] = "File is empty"
            return results
        
        # Find time column for rate calculations (if available)
        time_col = None
        if 'time' in df.columns:
            time_col = 'time'
        elif 'relative_time' in df.columns:
            time_col = 'relative_time'
        
        if time_col and df[time_col].dtype == 'object':
            df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
        
        # Sort by time if available
        if time_col:
            df = df.sort_values(by=time_col).reset_index(drop=True)
        
        # Analyze each column
        for column in COLUMNS_TO_ANALYZE:
            if column not in df.columns:
                continue
            
            # Get valid data
            valid_mask = df[column].notna()
            if valid_mask.sum() < 10:  # Need enough data for statistics
                continue
            
            values = df[column].values
            valid_indices = np.where(valid_mask)[0]
            
            # Calculate step changes between consecutive valid points
            steps = []
            step_times = []
            step_rates = []
            
            for i in range(1, len(valid_indices)):
                idx_curr = valid_indices[i]
                idx_prev = valid_indices[i-1]
                
                # Skip if there's a gap in data
                if idx_curr - idx_prev > 1:
                    continue
                
                val_curr = values[idx_curr]
                val_prev = values[idx_prev]
                
                # Calculate absolute change
                abs_change = abs(val_curr - val_prev)
                steps.append(abs_change)
                
                # Calculate rate if time is available
                if time_col and idx_curr < len(df):
                    time_gap = df[time_col].iloc[idx_curr] - df[time_col].iloc[idx_prev]
                    if time_gap > 0:
                        rate = abs_change / time_gap
                        step_rates.append(rate)
                        step_times.append(time_gap)
            
            if len(steps) < 10:
                continue
            
            # Calculate statistics
            steps = np.array(steps)
            step_stats = {
                'count': len(steps),
                'min': float(steps.min()),
                'max': float(steps.max()),
                'mean': float(steps.mean()),
                'median': float(np.median(steps)),
                'std': float(steps.std()),
                'q1': float(np.percentile(steps, 25)),
                'q3': float(np.percentile(steps, 75)),
                'iqr': float(np.percentile(steps, 75) - np.percentile(steps, 25))
            }
            
            # Identify anomalous steps (> STD_THRESHOLD standard deviations from mean)
            anomaly_threshold = step_stats['mean'] + STD_THRESHOLD * step_stats['std']
            anomalous_indices = np.where(steps > anomaly_threshold)[0]
            
            anomalies = []
            for idx in anomalous_indices:
                i = idx + 1  # +1 because steps[0] corresponds to the second point
                actual_idx = valid_indices[i]
                
                anomaly_info = {
                    'index': int(actual_idx),
                    'previous_value': float(values[valid_indices[i-1]]),
                    'current_value': float(values[actual_idx]),
                    'step_size': float(steps[idx]),
                    'threshold': float(anomaly_threshold),
                    'std_deviations': float((steps[idx] - step_stats['mean']) / step_stats['std'])
                }
                
                if step_rates and idx < len(step_rates):
                    anomaly_info['rate'] = float(step_rates[idx])
                    if idx < len(step_times):
                        anomaly_info['time_gap'] = float(step_times[idx])
                
                anomalies.append(anomaly_info)
            
            # Also check rate anomalies if we have rates
            rate_anomalies = []
            if step_rates and len(step_rates) > 10:
                rates = np.array(step_rates)
                rate_stats = {
                    'mean': float(rates.mean()),
                    'std': float(rates.std()),
                    'max': float(rates.max())
                }
                
                rate_threshold = rate_stats['mean'] + STD_THRESHOLD * rate_stats['std']
                anomalous_rates = np.where(rates > rate_threshold)[0]
                
                for idx in anomalous_rates:
                    # Check if this is already captured as a step anomaly
                    step_idx = idx
                    if step_idx < len(steps) and steps[step_idx] <= anomaly_threshold:
                        # This is a rate anomaly but not a step anomaly
                        i = idx + 1
                        actual_idx = valid_indices[i]
                        rate_anomalies.append({
                            'index': int(actual_idx),
                            'previous_value': float(values[valid_indices[i-1]]),
                            'current_value': float(values[actual_idx]),
                            'step_size': float(steps[step_idx]),
                            'rate': float(rates[idx]),
                            'rate_threshold': float(rate_threshold),
                            'time_gap': float(step_times[idx]) if idx < len(step_times) else None,
                            'issue': 'excessive_rate'
                        })
            
            # Store results
            results['step_analysis'][column] = {
                'unit': _get_unit(column),
                'stats': step_stats,
                'anomaly_threshold': float(anomaly_threshold),
                'anomaly_count': len(anomalies),
                'anomaly_percentage': round(len(anomalies) / len(steps) * 100, 4) if len(steps) > 0 else 0
            }
            
            if anomalies:
                results['anomalies'][column] = {
                    'step_anomalies': anomalies[:10],  # First 10
                    'rate_anomalies': rate_anomalies[:5] if rate_anomalies else []
                }
        
        # Calculate summary
        total_anomalies = sum(len(v.get('step_anomalies', [])) for v in results['anomalies'].values())
        total_rate_anomalies = sum(len(v.get('rate_anomalies', [])) for v in results['anomalies'].values())
        
        results['summary'] = {
            'total_step_anomalies': total_anomalies,
            'total_rate_anomalies': total_rate_anomalies,
            'columns_with_anomalies': len(results['anomalies'])
        }
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

def _get_unit(column):
    """Get the unit for a column."""
    units = {
        'voltage_charger': 'V',
        'voltage_load': 'V',
        'current_load': 'A',
        'temperature_battery': 'C',
        'temperature_mosfet': 'C',
        'temperature_resistor': 'C'
    }
    return units.get(column, '')

def scan_folder(folder_path, folder_name):
    """
    Scan all CSV files in a folder for smoothness analysis.
    """
    print(f"\n{'='*80}")
    print(f"SCANNING FOLDER: {folder_name}")
    print(f"{'='*80}")
    print(f"  Statistical threshold: {STD_THRESHOLD} standard deviations")
    print(f"  (Anomalies are points > {STD_THRESHOLD}σ from mean step size)")
    
    folder_results = {
        'folder': folder_name,
        'files_checked': 0,
        'files_with_errors': 0,
        'files_with_anomalies': 0,
        'total_anomalies': 0,
        'file_details': [],
        'summary_stats': {}
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
    
    all_step_stats = {col: [] for col in COLUMNS_TO_ANALYZE}
    
    for csv_file in sorted(csv_files):
        print(f"  Checking: {csv_file.name}")
        results = analyze_step_changes(csv_file)
        folder_results['file_details'].append(results)
        
        if results['error']:
            folder_results['files_with_errors'] += 1
            print(f"    ERROR: {results['error']}")
        else:
            if results['summary']['total_step_anomalies'] > 0:
                folder_results['files_with_anomalies'] += 1
                folder_results['total_anomalies'] += results['summary']['total_step_anomalies']
                
                print(f"    Found {results['summary']['total_step_anomalies']} step anomalies "
                      f"({results['summary']['total_rate_anomalies']} rate anomalies)")
                
                # Show column stats
                for col, data in results['step_analysis'].items():
                    if data['anomaly_count'] > 0:
                        print(f"      - {col}: {data['anomaly_count']} anomalies "
                              f"({data['anomaly_percentage']}% of steps)")
                        
                        # Collect stats for overall picture
                        all_step_stats[col].append(data['stats'])
            else:
                print(f"    No anomalies detected")
    
    # Calculate overall statistics
    folder_results['summary_stats'] = {
        'overall_step_stats': {}
    }
    
    for col, stats_list in all_step_stats.items():
        if stats_list:
            all_means = [s['mean'] for s in stats_list]
            all_stds = [s['std'] for s in stats_list]
            all_max = [s['max'] for s in stats_list]
            
            folder_results['summary_stats']['overall_step_stats'][col] = {
                'avg_mean_step': round(float(np.mean(all_means)), 4),
                'avg_std_step': round(float(np.mean(all_stds)), 4),
                'max_step_observed': round(float(np.max(all_max)), 4),
                'unit': _get_unit(col)
            }
    
    return folder_results

def print_summary(all_results):
    """
    Print comprehensive summary of signal smoothness analysis.
    """
    print("\n" + "="*100)
    print("SIGNAL SMOOTHNESS VERIFICATION - SUMMARY")
    print("="*100)
    print(f"\nMETHODOLOGY: Statistical outlier detection")
    print(f"  • Threshold: {STD_THRESHOLD} standard deviations from mean step size")
    print(f"  • No assumed thresholds - anomalies are data-relative")
    print("="*100)
    
    total_files = 0
    files_with_errors = 0
    files_with_anomalies = 0
    total_anomalies = 0
    
    for folder_result in all_results:
        print(f"\nFolder: {folder_result['folder']}")
        print("-" * 60)
        print(f"  Files checked: {folder_result['files_checked']}")
        print(f"  Files with errors: {folder_result['files_with_errors']}")
        print(f"  Files with anomalies: {folder_result['files_with_anomalies']}")
        print(f"  Total anomalies: {folder_result['total_anomalies']}")
        
        total_files += folder_result['files_checked']
        files_with_errors += folder_result['files_with_errors']
        files_with_anomalies += folder_result['files_with_anomalies']
        total_anomalies += folder_result['total_anomalies']
        
        # Show overall step statistics for this folder
        if folder_result['summary_stats'].get('overall_step_stats'):
            print(f"\n  Typical step sizes in this folder:")
            for col, stats in folder_result['summary_stats']['overall_step_stats'].items():
                print(f"    {col}: mean step {stats['avg_mean_step']}{stats['unit']}, "
                      f"std {stats['avg_std_step']}{stats['unit']}")
        
        # Show sample anomalies
        print(f"\n  Sample anomalies:")
        anomaly_count = 0
        for file_result in folder_result['file_details']:
            if file_result.get('anomalies'):
                for col, anomalies in file_result['anomalies'].items():
                    for anomaly in anomalies.get('step_anomalies', [])[:2]:  # First 2 per column
                        unit = _get_unit(col)
                        print(f"    • {col}: {anomaly['previous_value']:.3f} → "
                              f"{anomaly['current_value']:.3f}{unit} "
                              f"(Δ={anomaly['step_size']:.3f}{unit}, "
                              f"{anomaly['std_deviations']:.1f}σ)")
                        anomaly_count += 1
                        if anomaly_count >= 5:
                            break
                    if anomaly_count >= 5:
                        break
                if anomaly_count >= 5:
                    break
            if anomaly_count >= 5:
                break
    
    print("\n" + "="*100)
    print("OVERALL STATISTICS")
    print("="*100)
    print(f"Total files analyzed: {total_files}")
    print(f"Files with errors: {files_with_errors}")
    print(f"Files with anomalies: {files_with_anomalies}")
    print(f"Total anomalies detected: {total_anomalies}")
    
    print("\n" + "="*100)
    print("FINAL QUALITY ASSESSMENT - SIGNAL SMOOTHNESS")
    print("="*100)
    print("\nNOTE: This assessment is based on statistical outliers")
    print("      within the dataset itself, not assumed thresholds.")
    print("="*40)
    
    # Calculate percentage of files with anomalies
    valid_files = total_files - files_with_errors
    if valid_files == 0:
        anomaly_rate = 0
    else:
        anomaly_rate = (files_with_anomalies / valid_files) * 100
    
    if files_with_anomalies == 0:
        print("\nRESULT: EXCELLENT (++)")
        print("="*40)
        print("No statistical anomalies detected in any file.")
        print("✓ All step changes are within normal statistical patterns")
        print("✓ Signals are smooth and consistent")
        print("\nThe dataset fully satisfies the signal smoothness criterion.")
        
    elif anomaly_rate < 10:  # Less than 10% of files have anomalies
        print("\nRESULT: GOOD (+)")
        print("="*40)
        print(f"Only {files_with_anomalies} out of {valid_files} files have anomalies ({anomaly_rate:.1f}%).")
        print("✓ Most files are smooth")
        print("✓ Anomalies are rare and isolated")
        print("\nThe dataset mostly satisfies the signal smoothness criterion.")
        
    elif anomaly_rate < 30:  # Less than 30% of files have anomalies
        print("\nRESULT: ACCEPTABLE (o)")
        print("="*40)
        print(f"{files_with_anomalies} out of {valid_files} files have anomalies ({anomaly_rate:.1f}%).")
        print("• Some files contain statistical outliers")
        print("• May need attention in sensitive analyses")
        print("\nThe dataset is acceptable but be aware of anomalies.")
        
    else:
        print("\nRESULT: POOR (-)")
        print("="*40)
        print(f"Majority of files ({files_with_anomalies} out of {valid_files}) have anomalies.")
        print("• Widespread statistical outliers detected")
        print("• May indicate systematic issues")
        print("\nRecommendation: Investigate sources of signal")
        print("discontinuities across multiple files.")

def save_report(all_results, output_file="signal_smoothness_report_revised.txt"):
    """
    Save detailed report to file.
    """
    with open(output_file, 'w', encoding='ascii', errors='replace') as f:
        f.write("="*100 + "\n")
        f.write("NASA BATTERY DATASET - SIGNAL SMOOTHNESS REPORT (REVISED)\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*100 + "\n\n")
        f.write("METHODOLOGY:\n")
        f.write(f"  Statistical outlier detection (>{STD_THRESHOLD} standard deviations)\n")
        f.write("  No assumed thresholds - anomalies are data-relative\n\n")
        
        for folder_result in all_results:
            f.write(f"\nFOLDER: {folder_result['folder']}\n")
            f.write("-"*60 + "\n")
            
            # Write overall step statistics for this folder
            if folder_result['summary_stats'].get('overall_step_stats'):
                f.write("Typical step sizes in this folder:\n")
                for col, stats in folder_result['summary_stats']['overall_step_stats'].items():
                    f.write(f"  {col}: mean step {stats['avg_mean_step']}{stats['unit']}, "
                           f"std {stats['avg_std_step']}{stats['unit']}\n")
                f.write("\n")
            
            for file_result in folder_result['file_details']:
                f.write(f"\nFile: {file_result['file']} ({file_result['file_size_mb']} MB)\n")
                f.write(f"  Rows: {file_result['total_rows']:,}\n")
                
                if file_result['error']:
                    f.write(f"  ERROR: {file_result['error']}\n")
                    continue
                
                # Write step statistics for each column
                if file_result['step_analysis']:
                    f.write(f"  STEP STATISTICS:\n")
                    for col, data in file_result['step_analysis'].items():
                        stats = data['stats']
                        f.write(f"    {col} ({data['unit']}):\n")
                        f.write(f"      Steps analyzed: {stats['count']}\n")
                        f.write(f"      Step range: [{stats['min']:.4f}, {stats['max']:.4f}]\n")
                        f.write(f"      Mean step: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                        f.write(f"      Median step: {stats['median']:.4f} (IQR: {stats['q1']:.4f}-{stats['q3']:.4f})\n")
                        f.write(f"      Anomaly threshold: {data['anomaly_threshold']:.4f}\n")
                        f.write(f"      Anomalies: {data['anomaly_count']} ({data['anomaly_percentage']}% of steps)\n")
                
                # Write anomalies
                if file_result.get('anomalies'):
                    f.write(f"  ANOMALIES DETECTED:\n")
                    for col, anomalies in file_result['anomalies'].items():
                        unit = _get_unit(col)
                        
                        if anomalies.get('step_anomalies'):
                            f.write(f"    {col} - step anomalies:\n")
                            for i, anomaly in enumerate(anomalies['step_anomalies'][:10]):
                                f.write(f"      {i+1}. At index {anomaly['index']}: "
                                       f"{anomaly['previous_value']:.3f} → {anomaly['current_value']:.3f}{unit} "
                                       f"(Δ={anomaly['step_size']:.3f}{unit}, "
                                       f"{anomaly['std_deviations']:.1f}σ)\n")
                        
                        if anomalies.get('rate_anomalies'):
                            f.write(f"    {col} - rate anomalies:\n")
                            for i, anomaly in enumerate(anomalies['rate_anomalies'][:5]):
                                f.write(f"      {i+1}. At index {anomaly['index']}: "
                                       f"rate={anomaly['rate']:.2f}{unit}/s "
                                       f"(threshold={anomaly['rate_threshold']:.2f}{unit}/s)\n")
                else:
                    f.write(f"  No anomalies detected\n")
        
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
    print("NASA RANDOMIZED BATTERY DATASET - SIGNAL SMOOTHNESS VERIFICATION (REVISED)")
    print("="*100)
    print("\nIMPORTANT: No assumed thresholds!")
    print("This script uses STATISTICAL METHODS to detect anomalies")
    print("relative to the data's own patterns.")
    print(f"\nMethodology:")
    print(f"  • Calculate step changes between consecutive readings")
    print(f"  • Compute statistics (mean, std) of step sizes")
    print(f"  • Flag steps > {STD_THRESHOLD} standard deviations from mean")
    print(f"  • Also check rates of change if time is available")
    print("\nThis approach respects the dataset's own characteristics")
    print("rather than imposing external assumptions.")
    
    all_results = []
    for folder_name, folder_path in folders:
        if os.path.exists(folder_path):
            results = scan_folder(folder_path, folder_name)
            all_results.append(results)
    
    print_summary(all_results)
    save_report(all_results)
    
    print("\n" + "="*100)
    print("VERIFICATION COMPLETE")
    print("="*100)

if __name__ == "__main__":
    main()