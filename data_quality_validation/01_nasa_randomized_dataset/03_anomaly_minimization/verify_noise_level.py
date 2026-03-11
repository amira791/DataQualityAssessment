"""
verify_noise_level.py

This script quantifies the high-frequency noise level in measurements by analyzing
the signal's high-frequency components and calculating the noise floor.

WHAT WE MEASURE:
1. High-frequency noise amplitude (standard deviation of high-pass filtered signal)
2. Signal-to-Noise Ratio (SNR) estimation
3. Comparison of noise levels across different operating modes
4. Identification of excessively noisy channels

METHODOLOGY:
- Use differencing to isolate high-frequency components (removes slow trends)
- Calculate rolling statistics to separate signal from noise
- Compare noise levels during stable periods (rest mode) vs. dynamic periods
- Establish baseline noise floor for each sensor type
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import signal, stats

# Configuration
MIN_SEGMENT_LENGTH = 100  # Minimum points needed for noise analysis
REST_MODE_VALUE = 0  # Mode value for rest (stable period)

# Columns to analyze for noise
COLUMNS_TO_ANALYZE = [
    'voltage_charger',
    'voltage_load',
    'current_load',
    'temperature_battery',
    'temperature_mosfet',
    'temperature_resistor'
]

# Expected noise floors (based on typical sensor specifications)
# These are GUIDELINES, not strict thresholds
EXPECTED_NOISE_FLOORS = {
    'voltage_charger': {'typical': 0.005, 'unit': 'V', 'description': 'Charger voltage noise'},
    'voltage_load': {'typical': 0.005, 'unit': 'V', 'description': 'Load voltage noise'},
    'current_load': {'typical': 0.01, 'unit': 'A', 'description': 'Current noise'},
    'temperature_battery': {'typical': 0.05, 'unit': 'C', 'description': 'Battery temperature noise'},
    'temperature_mosfet': {'typical': 0.05, 'unit': 'C', 'description': 'MOSFET temperature noise'},
    'temperature_resistor': {'typical': 0.05, 'unit': 'C', 'description': 'Resistor temperature noise'}
}

def high_pass_filter(data, alpha=0.1):
    """
    Simple high-pass filter using exponential differencing.
    Removes low-frequency trends to isolate high-frequency noise.
    """
    if len(data) < 2:
        return np.array([])
    
    # Use differencing as a simple high-pass filter
    # This removes slow trends and leaves high-frequency components
    high_freq = np.diff(data)
    
    return high_freq

def analyze_noise_segment(segment_data, column_name, mode_name=""):
    """
    Analyze noise in a continuous segment of data.
    """
    if len(segment_data) < MIN_SEGMENT_LENGTH:
        return None
    
    # Remove NaN values
    clean_data = segment_data.dropna().values
    if len(clean_data) < MIN_SEGMENT_LENGTH:
        return None
    
    # Get high-frequency components (noise)
    high_freq = high_pass_filter(clean_data)
    
    if len(high_freq) < MIN_SEGMENT_LENGTH - 1:
        return None
    
    # Calculate noise statistics
    noise_stats = {
        'segment_length': len(clean_data),
        'noise_rms': float(np.sqrt(np.mean(high_freq**2))),
        'noise_std': float(np.std(high_freq)),
        'noise_mean': float(np.mean(high_freq)),
        'noise_max': float(np.max(np.abs(high_freq))),
        'noise_percentiles': {
            '1': float(np.percentile(np.abs(high_freq), 1)),
            '5': float(np.percentile(np.abs(high_freq), 5)),
            '50': float(np.percentile(np.abs(high_freq), 50)),
            '95': float(np.percentile(np.abs(high_freq), 95)),
            '99': float(np.percentile(np.abs(high_freq), 99))
        }
    }
    
    # Calculate Signal-to-Noise Ratio (SNR) estimation
    # SNR = 20 * log10(signal_rms / noise_rms)
    signal_rms = float(np.std(clean_data))
    if noise_stats['noise_rms'] > 0:
        noise_stats['snr_db'] = float(20 * np.log10(signal_rms / noise_stats['noise_rms']))
    else:
        noise_stats['snr_db'] = float('inf')
    
    # Check if noise is Gaussian (normal distribution)
    # Kurtosis near 3 indicates Gaussian noise
    from scipy import stats
    noise_stats['kurtosis'] = float(stats.kurtosis(high_freq))
    noise_stats['skewness'] = float(stats.skew(high_freq))
    
    # Shapiro-Wilk test for normality (if enough samples)
    if len(high_freq) < 5000:
        try:
            _, p_value = stats.shapiro(high_freq[:5000])  # Limit to 5000 samples
            noise_stats['normality_p_value'] = float(p_value)
        except:
            noise_stats['normality_p_value'] = None
    
    return noise_stats

def analyze_file_noise(file_path):
    """
    Analyze noise levels for a single CSV file.
    """
    results = {
        'file': os.path.basename(file_path),
        'file_size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2),
        'total_rows': 0,
        'noise_by_column': {},
        'rest_mode_noise': {},
        'dynamic_noise': {},
        'noise_summary': {},
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
        
        # Analyze each column
        for column in COLUMNS_TO_ANALYZE:
            if column not in df.columns:
                continue
            
            column_results = {
                'unit': EXPECTED_NOISE_FLOORS.get(column, {}).get('unit', ''),
                'overall_noise': None,
                'by_mode': {},
                'rest_noise': None,
                'dynamic_noise': None,
                'is_excessively_noisy': False
            }
            
            # Overall noise analysis (entire dataset)
            overall_stats = analyze_noise_segment(df[column], column)
            if overall_stats:
                column_results['overall_noise'] = overall_stats
            
            # If mode column exists, analyze by operational mode
            if 'mode' in df.columns:
                for mode_val in [-1, 0, 1]:
                    mode_data = df[df['mode'] == mode_val][column]
                    
                    if len(mode_data) >= MIN_SEGMENT_LENGTH:
                        mode_name = { -1: 'discharge', 0: 'rest', 1: 'charge' }.get(mode_val, f'mode_{mode_val}')
                        mode_stats = analyze_noise_segment(mode_data, column, mode_name)
                        
                        if mode_stats:
                            column_results['by_mode'][mode_name] = mode_stats
                            
                            # Store rest mode noise separately (usually lowest noise)
                            if mode_val == REST_MODE_VALUE:
                                column_results['rest_noise'] = mode_stats
                            
                            # Store dynamic mode noise (discharge/charge)
                            if mode_val in [-1, 1]:
                                if column_results['dynamic_noise'] is None:
                                    column_results['dynamic_noise'] = mode_stats
            
            # Compare noise levels with expected typical values
            expected = EXPECTED_NOISE_FLOORS.get(column, {}).get('typical', float('inf'))
            if column_results.get('rest_noise') and expected != float('inf'):
                noise_rms = column_results['rest_noise']['noise_rms']
                column_results['noise_vs_expected'] = noise_rms / expected if expected > 0 else float('inf')
                column_results['is_excessively_noisy'] = noise_rms > (expected * 5)  # 5x expected is excessive
            
            results['noise_by_column'][column] = column_results
        
        # Calculate summary statistics
        total_columns = len(results['noise_by_column'])
        noisy_columns = sum(1 for col in results['noise_by_column'].values() if col.get('is_excessively_noisy', False))
        
        results['noise_summary'] = {
            'columns_analyzed': total_columns,
            'excessively_noisy_columns': noisy_columns,
            'average_noise_floor': {}
        }
        
        # Calculate average noise floor across columns
        for column, col_results in results['noise_by_column'].items():
            if col_results.get('rest_noise'):
                results['noise_summary']['average_noise_floor'][column] = {
                    'rest_noise_rms': col_results['rest_noise']['noise_rms'],
                    'unit': col_results['unit']
                }
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

def scan_folder(folder_path, folder_name):
    """
    Scan all CSV files in a folder for noise analysis.
    """
    print(f"\n{'='*80}")
    print(f"SCANNING FOLDER: {folder_name}")
    print(f"{'='*80}")
    
    folder_results = {
        'folder': folder_name,
        'files_checked': 0,
        'files_with_errors': 0,
        'files_with_noise_data': 0,
        'file_details': [],
        'noise_summary': {
            'by_column': {},
            'overall_rating': None
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
    
    # Collect noise stats across files
    noise_stats_by_column = {col: [] for col in COLUMNS_TO_ANALYZE}
    
    for csv_file in sorted(csv_files):
        print(f"  Analyzing noise: {csv_file.name}")
        results = analyze_file_noise(csv_file)
        folder_results['file_details'].append(results)
        
        if results['error']:
            folder_results['files_with_errors'] += 1
            print(f"    ERROR: {results['error']}")
        else:
            folder_results['files_with_noise_data'] += 1
            
            # Collect noise stats
            for column, col_results in results['noise_by_column'].items():
                if col_results.get('rest_noise'):
                    noise_stats_by_column[column].append({
                        'file': results['file'],
                        'noise_rms': col_results['rest_noise']['noise_rms'],
                        'snr_db': col_results['rest_noise'].get('snr_db'),
                        'kurtosis': col_results['rest_noise'].get('kurtosis')
                    })
            
            # Print brief summary for this file
            noisy = [col for col, res in results['noise_by_column'].items() if res.get('is_excessively_noisy')]
            if noisy:
                print(f"    Noisy columns: {', '.join(noisy)}")
            else:
                print(f"    Noise levels normal")
    
    # Calculate summary statistics across files
    for column, stats_list in noise_stats_by_column.items():
        if stats_list:
            noise_values = [s['noise_rms'] for s in stats_list]
            snr_values = [s['snr_db'] for s in stats_list if s['snr_db'] is not None and s['snr_db'] != float('inf')]
            
            folder_results['noise_summary']['by_column'][column] = {
                'files_with_data': len(stats_list),
                'noise_rms_mean': float(np.mean(noise_values)),
                'noise_rms_median': float(np.median(noise_values)),
                'noise_rms_std': float(np.std(noise_values)),
                'noise_rms_min': float(np.min(noise_values)),
                'noise_rms_max': float(np.max(noise_values)),
                'snr_mean': float(np.mean(snr_values)) if snr_values else None,
                'unit': EXPECTED_NOISE_FLOORS.get(column, {}).get('unit', ''),
                'expected_typical': EXPECTED_NOISE_FLOORS.get(column, {}).get('typical', None)
            }
    
    return folder_results

def print_summary(all_results):
    """
    Print comprehensive summary of noise analysis.
    """
    print("\n" + "="*100)
    print("SIGNAL NOISE LEVEL ANALYSIS - SUMMARY")
    print("="*100)
    print("\nNoise is measured as RMS of high-frequency components (after trend removal)")
    print("Lower noise = cleaner signal | SNR > 30dB is excellent")
    print("="*100)
    
    total_files = 0
    files_with_errors = 0
    
    # Aggregate noise stats across folders
    all_noise_stats = {col: [] for col in COLUMNS_TO_ANALYZE}
    
    for folder_result in all_results:
        print(f"\nFolder: {folder_result['folder']}")
        print("-" * 60)
        print(f"  Files checked: {folder_result['files_checked']}")
        print(f"  Files with errors: {folder_result['files_with_errors']}")
        print(f"  Files with noise data: {folder_result['files_with_noise_data']}")
        
        total_files += folder_result['files_checked']
        files_with_errors += folder_result['files_with_errors']
        
        # Print noise summary for this folder
        if folder_result['noise_summary']['by_column']:
            print(f"\n  Noise levels by sensor (during rest mode):")
            for column, stats in folder_result['noise_summary']['by_column'].items():
                expected = stats.get('expected_typical')
                noise = stats['noise_rms_median']
                unit = stats['unit']
                
                # Compare to expected
                if expected:
                    ratio = noise / expected
                    if ratio < 2:
                        status = "✓ NORMAL"
                    elif ratio < 5:
                        status = "⚠ ELEVATED"
                    else:
                        status = "✗ EXCESSIVE"
                else:
                    status = ""
                
                print(f"    {column}: {noise:.4f}{unit} (median) {status}")
                if stats['snr_mean']:
                    print(f"      SNR: {stats['snr_mean']:.1f}dB")
        
        # Collect stats for overall
        for column, stats in folder_result['noise_summary']['by_column'].items():
            all_noise_stats[column].append(stats['noise_rms_median'])
    
    print("\n" + "="*100)
    print("OVERALL NOISE ASSESSMENT")
    print("="*100)
    
    # Determine overall noise quality
    total_columns_assessed = 0
    excessive_columns = 0
    elevated_columns = 0
    normal_columns = 0
    
    for column, noise_values in all_noise_stats.items():
        if noise_values:
            total_columns_assessed += 1
            median_noise = np.median(noise_values)
            expected = EXPECTED_NOISE_FLOORS.get(column, {}).get('typical')
            
            if expected:
                ratio = median_noise / expected
                if ratio < 2:
                    normal_columns += 1
                    status = "NORMAL"
                elif ratio < 5:
                    elevated_columns += 1
                    status = "ELEVATED"
                else:
                    excessive_columns += 1
                    status = "EXCESSIVE"
            else:
                status = "UNKNOWN"
            
            unit = EXPECTED_NOISE_FLOORS.get(column, {}).get('unit', '')
            print(f"\n{column}:")
            print(f"  Median noise floor: {median_noise:.4f}{unit}")
            print(f"  Status: {status}")
    
    print("\n" + "="*100)
    print("FINAL QUALITY ASSESSMENT - NOISE LEVEL")
    print("="*100)
    
    # Calculate overall score
    if excessive_columns == 0 and elevated_columns == 0:
        print("\nRESULT: EXCELLENT (++)")
        print("="*40)
        print("All sensors show excellent noise characteristics:")
        print("  ✓ Noise levels within expected range")
        print("  ✓ Clean signals with good SNR")
        print("  ✓ No excessive noise in any channel")
        
    elif excessive_columns == 0 and elevated_columns <= total_columns_assessed * 0.3:
        print("\nRESULT: GOOD (+)")
        print("="*40)
        print("Most sensors have acceptable noise levels:")
        print(f"  ✓ {normal_columns} sensors with NORMAL noise")
        print(f"  ⚠ {elevated_columns} sensors with ELEVATED noise")
        print("  ✗ No sensors with EXCESSIVE noise")
        print("\nElevated noise may be due to normal operation, not sensor issues.")
        
    elif excessive_columns <= total_columns_assessed * 0.2:
        print("\nRESULT: ACCEPTABLE (o)")
        print("="*40)
        print("Some sensors show elevated or excessive noise:")
        print(f"  ✓ {normal_columns} sensors with NORMAL noise")
        print(f"  ⚠ {elevated_columns} sensors with ELEVATED noise")
        print(f"  ✗ {excessive_columns} sensors with EXCESSIVE noise")
        print("\nData is still usable but noisy channels may need filtering.")
        
    else:
        print("\nRESULT: POOR (-)")
        print("="*40)
        print("Multiple sensors show excessive noise levels:")
        print(f"  ✓ {normal_columns} sensors with NORMAL noise")
        print(f"  ⚠ {elevated_columns} sensors with ELEVATED noise")
        print(f"  ✗ {excessive_columns} sensors with EXCESSIVE noise")
        print("\nNoise levels may significantly impact measurement quality.")
        print("Recommendation: Apply low-pass filtering before analysis.")

def save_report(all_results, output_file="noise_level_report.txt"):
    """
    Save detailed report to file.
    """
    with open(output_file, 'w', encoding='ascii', errors='replace') as f:
        f.write("="*100 + "\n")
        f.write("NASA BATTERY DATASET - NOISE LEVEL ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*100 + "\n\n")
        f.write("Noise is measured as RMS of high-frequency components (after trend removal)\n")
        f.write("Lower noise = cleaner signal | SNR > 30dB is excellent\n\n")
        
        for folder_result in all_results:
            f.write(f"\nFOLDER: {folder_result['folder']}\n")
            f.write("-"*60 + "\n")
            
            for file_result in folder_result['file_details']:
                f.write(f"\nFile: {file_result['file']} ({file_result['file_size_mb']} MB)\n")
                f.write(f"  Rows: {file_result['total_rows']:,}\n")
                
                if file_result['error']:
                    f.write(f"  ERROR: {file_result['error']}\n")
                    continue
                
                for column, col_results in file_result['noise_by_column'].items():
                    f.write(f"\n  {column}:\n")
                    
                    if col_results.get('rest_noise'):
                        rest = col_results['rest_noise']
                        f.write(f"    Rest mode noise:\n")
                        f.write(f"      RMS: {rest['noise_rms']:.6f}{col_results['unit']}\n")
                        f.write(f"      SNR: {rest.get('snr_db', 'N/A'):.1f}dB\n")
                        f.write(f"      Noise distribution: 95% < {rest['noise_percentiles']['95']:.6f}{col_results['unit']}\n")
                        
                        # Check if noise is Gaussian
                        if abs(rest.get('kurtosis', 3)) < 2:
                            f.write(f"      Noise appears Gaussian (kurtosis={rest['kurtosis']:.2f})\n")
                        else:
                            f.write(f"      Noise non-Gaussian (kurtosis={rest['kurtosis']:.2f})\n")
                    
                    if col_results.get('dynamic_noise'):
                        dyn = col_results['dynamic_noise']
                        f.write(f"    Dynamic mode noise:\n")
                        f.write(f"      RMS: {dyn['noise_rms']:.6f}{col_results['unit']}\n")
                    
                    if col_results.get('is_excessively_noisy'):
                        f.write(f"    ⚠ EXCESSIVE NOISE - {col_results['noise_vs_expected']:.1f}x expected\n")
        
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
    print("NASA RANDOMIZED BATTERY DATASET - NOISE LEVEL ANALYSIS")
    print("="*100)
    print("\nAnalyzing high-frequency noise in measurements:")
    print("  • Using high-pass filtering to isolate noise")
    print("  • Comparing rest mode (quiet) vs dynamic mode noise")
    print("  • Calculating SNR and noise statistics")
    print("  • Identifying excessively noisy channels")
    
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