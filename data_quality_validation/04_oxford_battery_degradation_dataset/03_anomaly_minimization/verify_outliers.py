"""
verify_outliers.py

This script detects outliers in the NASA Randomized Battery Dataset by identifying
samples that deviate significantly from local signal patterns.

METHODOLOGY:
1. Use rolling window statistics to establish local patterns
2. Identify points that deviate by > N standard deviations from local mean
3. Consider context (mode changes) to avoid false positives
4. Separate analysis for different operational modes

OUTLIER DEFINITION:
A point is considered an outlier if:
- It deviates by > 5 standard deviations from the local rolling mean
- The deviation is not explained by a mode change
- The point is isolated (not part of a sustained change)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

# Configuration
ROLLING_WINDOW = 50  # Number of points for local statistics
STD_THRESHOLD = 5.0  # Number of standard deviations for outlier detection
MIN_SEGMENT_LENGTH = 10  # Minimum length to consider as a segment (not outlier)

# Columns to analyze for outliers
COLUMNS_TO_ANALYZE = [
    'voltage_charger',
    'voltage_load',
    'current_load',
    'temperature_battery',
    'temperature_mosfet',
    'temperature_resistor'
]

def detect_outliers_in_segment(data, column, mode_value=None):
    """
    Detect outliers in a continuous segment of data.
    """
    if len(data) < ROLLING_WINDOW:
        return []
    
    outliers = []
    values = data[column].values
    
    # Calculate rolling statistics
    rolling_mean = pd.Series(values).rolling(window=ROLLING_WINDOW, center=True).mean()
    rolling_std = pd.Series(values).rolling(window=ROLLING_WINDOW, center=True).std()
    
    for i in range(len(values)):
        if pd.isna(rolling_mean[i]) or pd.isna(rolling_std[i]) or rolling_std[i] == 0:
            continue
        
        # Calculate deviation
        dev = abs(values[i] - rolling_mean[i]) / rolling_std[i]
        
        if dev > STD_THRESHOLD:
            # Check if this is part of a sustained change
            # Look at surrounding points
            start_idx = max(0, i - MIN_SEGMENT_LENGTH)
            end_idx = min(len(values), i + MIN_SEGMENT_LENGTH + 1)
            segment = values[start_idx:end_idx]
            
            # If most points in the local segment are also deviating,
            # this might be a real change, not an outlier
            segment_mean = np.mean(segment)
            segment_std = np.std(segment)
            if segment_std > 0:
                segment_deviations = [abs(x - segment_mean) / segment_std > STD_THRESHOLD for x in segment]
                if np.mean(segment_deviations) > 0.3:  # 30% of segment also deviating
                    continue  # Skip - likely a real change
            
            outliers.append({
                'index': data.index[i],
                'position': i,
                'value': float(values[i]),
                'local_mean': float(rolling_mean[i]),
                'local_std': float(rolling_std[i]),
                'deviation_std': float(dev),
                'mode': mode_value
            })
    
    return outliers

def analyze_outliers(file_path):
    """
    Analyze outliers for a single CSV file.
    """
    results = {
        'file': os.path.basename(file_path),
        'file_size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2),
        'total_rows': 0,
        'outliers_by_column': {},
        'outlier_summary': {},
        'error': None
    }
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, low_memory=False)
        results['total_rows'] = len(df)
        
        if len(df) == 0:
            results['error'] = "File is empty"
            return results
        
        # Ensure time column exists and sort
        time_col = None
        if 'time' in df.columns:
            time_col = 'time'
        elif 'relative_time' in df.columns:
            time_col = 'relative_time'
        
        if time_col:
            if df[time_col].dtype == 'object':
                df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
            df = df.sort_values(by=time_col).reset_index(drop=True)
        
        # Analyze each column
        for column in COLUMNS_TO_ANALYZE:
            if column not in df.columns:
                continue
            
            column_outliers = []
            
            # If mode column exists, analyze by mode segment
            if 'mode' in df.columns:
                # Split by mode to avoid false positives at mode boundaries
                for mode_val in [-1, 0, 1]:
                    mode_data = df[df['mode'] == mode_val].copy()
                    if len(mode_data) > ROLLING_WINDOW:
                        outliers = detect_outliers_in_segment(mode_data, column, mode_val)
                        column_outliers.extend(outliers)
            else:
                # Analyze entire dataset as one segment
                outliers = detect_outliers_in_segment(df, column)
                column_outliers.extend(outliers)
            
            if column_outliers:
                # Calculate statistics
                outlier_values = [o['value'] for o in column_outliers]
                outlier_deviations = [o['deviation_std'] for o in column_outliers]
                
                results['outliers_by_column'][column] = {
                    'count': len(column_outliers),
                    'percentage': round(len(column_outliers) / len(df) * 100, 4),
                    'min_value': round(float(min(outlier_values)), 3),
                    'max_value': round(float(max(outlier_values)), 3),
                    'mean_value': round(float(np.mean(outlier_values)), 3),
                    'max_deviation': round(float(max(outlier_deviations)), 2),
                    'mean_deviation': round(float(np.mean(outlier_deviations)), 2),
                    'outliers': column_outliers[:10]  # First 10 outliers as samples
                }
                
                # Group by mode if available
                if 'mode' in df.columns:
                    mode_counts = {}
                    for o in column_outliers:
                        mode = o.get('mode')
                        mode_counts[mode] = mode_counts.get(mode, 0) + 1
                    results['outliers_by_column'][column]['by_mode'] = mode_counts
        
        # Calculate overall statistics
        total_outliers = sum(len(v['outliers']) for v in results['outliers_by_column'].values())
        results['outlier_summary'] = {
            'total_outliers': total_outliers,
            'columns_with_outliers': len(results['outliers_by_column']),
            'outlier_density': round(total_outliers / (len(df) * len(COLUMNS_TO_ANALYZE)) * 100, 4) if len(df) > 0 else 0
        }
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

def scan_folder(folder_path, folder_name):
    """
    Scan all CSV files in a folder for outlier detection.
    """
    print(f"\n{'='*80}")
    print(f"SCANNING FOLDER: {folder_name}")
    print(f"{'='*80}")
    print(f"  Rolling window: {ROLLING_WINDOW} points")
    print(f"  Threshold: {STD_THRESHOLD} standard deviations")
    
    folder_results = {
        'folder': folder_name,
        'files_checked': 0,
        'files_with_errors': 0,
        'files_with_outliers': 0,
        'total_outliers': 0,
        'file_details': [],
        'summary_stats': {
            'avg_outlier_percentage': 0,
            'worst_file': None,
            'worst_outlier_percentage': 0
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
    
    outlier_percentages = []
    worst_pct = 0
    worst_file = None
    
    for csv_file in sorted(csv_files):
        print(f"  Checking: {csv_file.name}")
        results = analyze_outliers(csv_file)
        folder_results['file_details'].append(results)
        
        if results['error']:
            folder_results['files_with_errors'] += 1
            print(f"    ERROR: {results['error']}")
        else:
            if results['outlier_summary']['total_outliers'] > 0:
                folder_results['files_with_outliers'] += 1
                folder_results['total_outliers'] += results['outlier_summary']['total_outliers']
                
                # Calculate outlier percentage for this file
                total_points = results['total_rows'] * len(COLUMNS_TO_ANALYZE)
                outlier_pct = (results['outlier_summary']['total_outliers'] / total_points) * 100
                outlier_percentages.append(outlier_pct)
                
                if outlier_pct > worst_pct:
                    worst_pct = outlier_pct
                    worst_file = results['file']
                
                print(f"    Found {results['outlier_summary']['total_outliers']} outliers "
                      f"({outlier_pct:.4f}% of measurements)")
                
                # Show top columns with outliers
                for col, data in results['outliers_by_column'].items():
                    if data['count'] > 0:
                        print(f"      - {col}: {data['count']} outliers "
                              f"(max dev {data['max_deviation']}σ)")
            else:
                print(f"    No outliers detected")
    
    # Calculate averages
    if outlier_percentages:
        folder_results['summary_stats']['avg_outlier_percentage'] = round(np.mean(outlier_percentages), 4)
        folder_results['summary_stats']['worst_file'] = worst_file
        folder_results['summary_stats']['worst_outlier_percentage'] = round(worst_pct, 4)
    
    return folder_results

def print_summary(all_results):
    """
    Print comprehensive summary of outlier analysis.
    """
    print("\n" + "="*100)
    print("OUTLIER DETECTION - SUMMARY")
    print("="*100)
    print(f"\nDetection parameters:")
    print(f"  • Rolling window: {ROLLING_WINDOW} points")
    print(f"  • Threshold: {STD_THRESHOLD} standard deviations")
    print(f"  • Minimum segment length: {MIN_SEGMENT_LENGTH} points")
    print("="*100)
    
    total_files = 0
    files_with_errors = 0
    files_with_outliers = 0
    total_outliers = 0
    all_outlier_pcts = []
    
    for folder_result in all_results:
        print(f"\nFolder: {folder_result['folder']}")
        print("-" * 60)
        print(f"  Files checked: {folder_result['files_checked']}")
        print(f"  Files with errors: {folder_result['files_with_errors']}")
        print(f"  Files with outliers: {folder_result['files_with_outliers']}")
        print(f"  Total outliers: {folder_result['total_outliers']:,}")
        print(f"  Avg outlier percentage: {folder_result['summary_stats']['avg_outlier_percentage']}%")
        
        if folder_result['summary_stats']['worst_file']:
            print(f"  Worst file: {folder_result['summary_stats']['worst_file']} "
                  f"({folder_result['summary_stats']['worst_outlier_percentage']}% outliers)")
        
        total_files += folder_result['files_checked']
        files_with_errors += folder_result['files_with_errors']
        files_with_outliers += folder_result['files_with_outliers']
        total_outliers += folder_result['total_outliers']
        
        if folder_result['summary_stats']['avg_outlier_percentage'] > 0:
            all_outlier_pcts.append(folder_result['summary_stats']['avg_outlier_percentage'])
        
        # Show detailed outlier info for first file
        print(f"\n  Sample outlier details (first file with outliers):")
        for file_result in folder_result['file_details']:
            if file_result.get('outliers_by_column'):
                print(f"    File: {file_result['file']}")
                for col, data in file_result['outliers_by_column'].items():
                    if data['count'] > 0:
                        print(f"      {col}: {data['count']} outliers ({data['percentage']}%)")
                        print(f"        Values: {data['min_value']} to {data['max_value']}")
                        print(f"        Max deviation: {data['max_deviation']}σ")
                        if 'by_mode' in data:
                            print(f"        By mode: {data['by_mode']}")
                break
    
    print("\n" + "="*100)
    print("OVERALL STATISTICS")
    print("="*100)
    print(f"Total files analyzed: {total_files}")
    print(f"Files with errors: {files_with_errors}")
    print(f"Files with outliers: {files_with_outliers}")
    print(f"Total outliers detected: {total_outliers:,}")
    
    if all_outlier_pcts:
        avg_pct = np.mean(all_outlier_pcts)
        print(f"Average outlier percentage: {avg_pct:.4f}%")
    
    print("\n" + "="*100)
    print("FINAL QUALITY ASSESSMENT - OUTLIER DETECTION")
    print("="*100)
    
    # Determine quality based on outlier prevalence
    if files_with_outliers == 0:
        print("\nRESULT: EXCELLENT (++)")
        print("="*40)
        print("No outliers detected in any file.")
        print("✓ All measurements follow local patterns")
        print("✓ No significant deviations found")
        print("\nThe dataset fully satisfies the outlier criterion.")
        
    elif all_outlier_pcts and np.mean(all_outlier_pcts) < 0.01:  # Less than 0.01% outliers
        print("\nRESULT: GOOD (+)")
        print("="*40)
        print(f"Very few outliers detected ({np.mean(all_outlier_pcts):.4f}% of measurements).")
        print("✓ Outliers are extremely rare")
        print("✓ Do not significantly impact data quality")
        print("\nThe dataset mostly satisfies the outlier criterion.")
        
    elif all_outlier_pcts and np.mean(all_outlier_pcts) < 0.1:  # Less than 0.1% outliers
        print("\nRESULT: ACCEPTABLE (o)")
        print("="*40)
        print(f"Some outliers detected ({np.mean(all_outlier_pcts):.4f}% of measurements).")
        print("• Outliers are present but rare")
        print("• May need to be handled in sensitive analyses")
        print("\nThe dataset is acceptable but be aware of outliers.")
        
    else:
        print("\nRESULT: POOR (-)")
        print("="*40)
        print(f"Significant number of outliers detected.")
        if all_outlier_pcts:
            print(f"  • Average outlier rate: {np.mean(all_outlier_pcts):.4f}%")
        print(f"  • {files_with_outliers} files contain outliers")
        print("\nRecommendation: Investigate outliers and consider")
        print("filtering or robust statistical methods for analysis.")

def save_report(all_results, output_file="outliers_report.txt"):
    """
    Save detailed report to file.
    """
    with open(output_file, 'w', encoding='ascii', errors='replace') as f:
        f.write("="*100 + "\n")
        f.write("NASA BATTERY DATASET - OUTLIER DETECTION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*100 + "\n\n")
        f.write(f"Detection parameters:\n")
        f.write(f"  Rolling window: {ROLLING_WINDOW} points\n")
        f.write(f"  Threshold: {STD_THRESHOLD} standard deviations\n")
        f.write(f"  Minimum segment length: {MIN_SEGMENT_LENGTH} points\n\n")
        
        for folder_result in all_results:
            f.write(f"\nFOLDER: {folder_result['folder']}\n")
            f.write("-"*60 + "\n")
            
            for file_result in folder_result['file_details']:
                f.write(f"\nFile: {file_result['file']} ({file_result['file_size_mb']} MB)\n")
                f.write(f"  Rows: {file_result['total_rows']:,}\n")
                
                if file_result['error']:
                    f.write(f"  ERROR: {file_result['error']}\n")
                    continue
                
                if file_result['outliers_by_column']:
                    f.write(f"  OUTLIERS DETECTED:\n")
                    for col, data in file_result['outliers_by_column'].items():
                        f.write(f"    {col}:\n")
                        f.write(f"      Count: {data['count']} ({data['percentage']}% of rows)\n")
                        f.write(f"      Value range: [{data['min_value']}, {data['max_value']}]\n")
                        f.write(f"      Max deviation: {data['max_deviation']}σ\n")
                        
                        if 'by_mode' in data:
                            f.write(f"      By mode: {data['by_mode']}\n")
                        
                        # Write sample outliers
                        if data['outliers']:
                            f.write(f"      Sample outliers (first 5):\n")
                            for i, outlier in enumerate(data['outliers'][:5]):
                                f.write(f"        {i+1}. value={outlier['value']:.3f}, "
                                       f"local_mean={outlier['local_mean']:.3f}, "
                                       f"dev={outlier['deviation_std']:.1f}σ")
                                if outlier.get('mode') is not None:
                                    f.write(f", mode={outlier['mode']}")
                                f.write("\n")
                else:
                    f.write(f"  No outliers detected\n")
        
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
    print("NASA RANDOMIZED BATTERY DATASET - OUTLIER DETECTION")
    print("="*100)
    print("\nDetecting samples that deviate significantly from local patterns:")
    print(f"  • Rolling window: {ROLLING_WINDOW} points")
    print(f"  • Threshold: {STD_THRESHOLD} standard deviations")
    print(f"  • Analyzing columns: {', '.join(COLUMNS_TO_ANALYZE)}")
    print("  • Context-aware: considers mode changes to avoid false positives")
    
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