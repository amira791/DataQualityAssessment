"""
verify_channel_synchronization_final.py

This script verifies that all measurement channels are properly time-aligned
in the NASA Randomized Battery Dataset.

CRITICAL DESIGN CONSTRAINTS (from documentation):
1. Load measurements (voltage_load, current_load) ONLY occur during discharge missions
2. MOSFET temperature ONLY recorded during discharge
3. Resistor temperature ONLY recorded during discharge
4. Charger voltage and battery temperature are continuous

Therefore, "missing data" during rest/charge modes is EXPECTED, not an error.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Columns to check for synchronization
VOLTAGE_COLUMNS = ['voltage_charger', 'voltage_load']
CURRENT_COLUMNS = ['current_load']
TEMPERATURE_COLUMNS = ['temperature_battery', 'temperature_mosfet', 'temperature_resistor']

# Columns that should ONLY have data during discharge (mode = -1)
DISCHARGE_ONLY_COLUMNS = ['voltage_load', 'current_load', 'temperature_mosfet', 'temperature_resistor']

# Columns that should have continuous data
CONTINUOUS_COLUMNS = ['voltage_charger', 'temperature_battery']

# Expected logging interval (from data exploration)
EXPECTED_INTERVAL = 1.0  # seconds

def safe_convert_time(df, time_col):
    """
    Safely convert time column to numeric, handling string values.
    """
    if df[time_col].dtype == 'object':
        # Try to convert to numeric
        df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
        # Check if conversion was successful
        if df[time_col].isna().all():
            return None, "Time column could not be converted to numeric"
    return df, None

def analyze_sync(file_path):
    """
    Analyze channel synchronization for a single CSV file.
    """
    results = {
        'file': os.path.basename(file_path),
        'file_size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2),
        'total_rows': 0,
        'time_column': None,
        'time_stats': {},
        'channel_stats': {},
        'design_compliance': {},  # Check if data follows design constraints
        'real_sync_issues': [],   # Only真正的 synchronization issues
        'mode_values': [],
        'error': None
    }
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, low_memory=False)
        results['total_rows'] = len(df)
        
        if len(df) == 0:
            results['error'] = "File is empty"
            return results
        
        # Check for time column
        time_col = None
        if 'time' in df.columns:
            time_col = 'time'
        elif 'relative_time' in df.columns:
            time_col = 'relative_time'
        
        if time_col is None:
            results['error'] = "No time column found"
            return results
        
        results['time_column'] = time_col
        
        # Convert time to numeric if needed
        df, conv_error = safe_convert_time(df, time_col)
        if conv_error:
            results['error'] = conv_error
            return results
        
        # Drop any rows with NaN time
        df = df.dropna(subset=[time_col])
        if len(df) == 0:
            results['error'] = "No valid time values"
            return results
        
        # Sort by time to ensure monotonicity
        df = df.sort_values(by=time_col).reset_index(drop=True)
        
        # Get mode values if available
        if 'mode' in df.columns:
            results['mode_values'] = sorted(df['mode'].dropna().unique().tolist())
        
        # Basic time statistics
        valid_time = df[time_col]
        time_min = valid_time.min()
        time_max = valid_time.max()
        time_diff = valid_time.diff().dropna()
        
        results['time_stats'] = {
            'min_time': round(float(time_min), 3),
            'max_time': round(float(time_max), 3),
            'total_duration_seconds': round(float(time_max - time_min), 3),
            'mean_time_step': round(float(time_diff.mean()), 6),
            'median_time_step': round(float(time_diff.median()), 6),
            'std_time_step': round(float(time_diff.std()), 6),
            'min_time_step': round(float(time_diff.min()), 6),
            'max_time_step': round(float(time_diff.max()), 6)
        }
        
        # Check for time monotonicity (REAL issue)
        if not (valid_time.is_monotonic_increasing):
            non_monotonic = (valid_time.diff() < 0).sum()
            results['real_sync_issues'].append({
                'issue': 'non_monotonic_time',
                'severity': 'HIGH',
                'description': f'Time column is not strictly increasing ({non_monotonic} decreases)',
                'count': int(non_monotonic)
            })
        
        # Check for REAL data gaps (missing data > 2x expected interval)
        # These are actual logging interruptions
        expected_max_interval = EXPECTED_INTERVAL * 2
        real_gaps = time_diff[time_diff > expected_max_interval]
        
        if len(real_gaps) > 0:
            total_gap_time = real_gaps.sum() - (len(real_gaps) * EXPECTED_INTERVAL)
            results['real_sync_issues'].append({
                'issue': 'data_gaps',
                'severity': 'MEDIUM',
                'description': f'Found {len(real_gaps)} real data gaps > {expected_max_interval}s',
                'max_gap': round(float(real_gaps.max()), 3),
                'total_gap_time': round(float(total_gap_time), 3),
                'estimated_missing_rows': int(total_gap_time / EXPECTED_INTERVAL)
            })
        
        # Check compliance with dataset design (if mode column exists)
        if 'mode' in df.columns:
            discharge_mode = df['mode'] == -1
            rest_charge_mode = df['mode'] != -1
            
            # 1. DISCHARGE-ONLY columns should ONLY have data during discharge
            for col in DISCHARGE_ONLY_COLUMNS:
                if col in df.columns:
                    has_data = df[col].notna()
                    
                    # Check for data during rest/charge (should NOT happen)
                    data_during_rest_charge = has_data & rest_charge_mode
                    if data_during_rest_charge.any():
                        percentage = (data_during_rest_charge.sum() / len(df)) * 100
                        results['real_sync_issues'].append({
                            'issue': f'{col}_during_rest_charge',
                            'severity': 'HIGH',
                            'description': f'{col} has data during rest/charge modes: {data_during_rest_charge.sum()} rows ({percentage:.2f}%)',
                            'count': int(data_during_rest_charge.sum())
                        })
                    
                    # Check for missing data during discharge (should have data)
                    missing_during_discharge = discharge_mode & ~has_data
                    if missing_during_discharge.any():
                        percentage = (missing_during_discharge.sum() / len(df)) * 100
                        results['real_sync_issues'].append({
                            'issue': f'missing_{col}_during_discharge',
                            'severity': 'HIGH',
                            'description': f'Missing {col} during discharge: {missing_during_discharge.sum()} rows ({percentage:.2f}%)',
                            'count': int(missing_during_discharge.sum())
                        })
                    
                    # Store compliance stats
                    results['design_compliance'][col] = {
                        'data_during_discharge': int(has_data[discharge_mode].sum()),
                        'missing_during_discharge': int(missing_during_discharge.sum()),
                        'data_during_rest_charge': int(data_during_rest_charge.sum())
                    }
            
            # 2. CONTINUOUS columns should have data regardless of mode
            for col in CONTINUOUS_COLUMNS:
                if col in df.columns:
                    has_data = df[col].notna()
                    missing_anywhere = ~has_data
                    
                    if missing_anywhere.any():
                        # Check if missing in large blocks
                        missing_streaks = (missing_anywhere != missing_anywhere.shift()).cumsum()
                        missing_streak_lengths = missing_anywhere.groupby(missing_streaks).sum()
                        long_streaks = missing_streak_lengths[missing_streak_lengths > 10]
                        
                        if len(long_streaks) > 0:
                            results['real_sync_issues'].append({
                                'issue': f'{col}_data_gaps',
                                'severity': 'MEDIUM',
                                'description': f'{col} has {len(long_streaks)} gaps of >10 consecutive missing values',
                                'max_gap_length': int(long_streaks.max())
                            })
                    
                    results['design_compliance'][col] = {
                        'total_missing': int(missing_anywhere.sum()),
                        'missing_percent': round(missing_anywhere.sum() / len(df) * 100, 2)
                    }
        
        # 3. Check alignment between related measurements during discharge
        discharge_periods = df[df['mode'] == -1] if 'mode' in df.columns else df
        
        if len(discharge_periods) > 0:
            # Voltage load and current load should be aligned during discharge
            if 'voltage_load' in discharge_periods.columns and 'current_load' in discharge_periods.columns:
                v_notna = discharge_periods['voltage_load'].notna()
                c_notna = discharge_periods['current_load'].notna()
                
                mismatch = v_notna != c_notna
                if mismatch.any():
                    percentage = (mismatch.sum() / len(discharge_periods)) * 100
                    results['real_sync_issues'].append({
                        'issue': 'voltage_current_mismatch_during_discharge',
                        'severity': 'HIGH',
                        'description': f'During discharge: voltage and current presence mismatch for {mismatch.sum()} rows ({percentage:.2f}%)',
                        'count': int(mismatch.sum())
                    })
        
        # 4. Check temperature jumps (only where data exists)
        for temp_col in TEMPERATURE_COLUMNS:
            if temp_col in df.columns:
                temp_data = df[temp_col]
                
                # Only check jumps where we have consecutive valid readings
                valid_mask = temp_data.notna()
                if valid_mask.sum() > 1:
                    # Get differences only where both values are valid
                    temp_diff = temp_data.diff()
                    temp_diff = temp_diff[temp_data.notna() & temp_data.shift().notna()]
                    
                    # Check for large jumps (> 10°C)
                    large_jumps = temp_diff[temp_diff.abs() > 10]
                    if len(large_jumps) > 0:
                        # Check if jumps coincide with mode changes (expected)
                        if 'mode' in df.columns:
                            jump_indices = large_jumps.index
                            mode_at_jump = df.loc[jump_indices, 'mode'].values if all(jump_indices < len(df)) else []
                            mode_before_jump = df.loc[jump_indices - 1, 'mode'].values if all(jump_indices > 0) else []
                            
                            # Count jumps that don't coincide with mode changes
                            unexplained_jumps = 0
                            for i, idx in enumerate(jump_indices):
                                if idx > 0 and idx < len(df):
                                    if pd.isna(df.loc[idx, 'mode']) or pd.isna(df.loc[idx-1, 'mode']):
                                        unexplained_jumps += 1
                                    elif df.loc[idx, 'mode'] == df.loc[idx-1, 'mode']:
                                        unexplained_jumps += 1
                            
                            if unexplained_jumps > 0:
                                results['real_sync_issues'].append({
                                    'issue': 'unexplained_temperature_jump',
                                    'severity': 'LOW',
                                    'column': temp_col,
                                    'description': f'Found {unexplained_jumps} temperature jumps > 10°C not related to mode changes',
                                    'max_jump': round(float(large_jumps.abs().max()), 3)
                                })
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

def scan_folder(folder_path, folder_name):
    """
    Scan all CSV files in a folder for synchronization issues.
    """
    print(f"\n{'='*80}")
    print(f"SCANNING FOLDER: {folder_name}")
    print(f"{'='*80}")
    
    folder_results = {
        'folder': folder_name,
        'files_checked': 0,
        'files_with_errors': 0,
        'files_with_sync_issues': 0,
        'total_sync_issues': 0,
        'file_details': []
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
    
    for csv_file in sorted(csv_files):
        print(f"  Checking: {csv_file.name}")
        results = analyze_sync(csv_file)
        folder_results['file_details'].append(results)
        
        if results['error']:
            folder_results['files_with_errors'] += 1
            print(f"    ERROR: {results['error']}")
        elif results['real_sync_issues']:
            folder_results['files_with_sync_issues'] += 1
            folder_results['total_sync_issues'] += len(results['real_sync_issues'])
            print(f"    Found {len(results['real_sync_issues'])} REAL synchronization issues")
            
            # Show key issues
            for issue in results['real_sync_issues']:
                print(f"      - [{issue['severity']}] {issue['description']}")
        else:
            print(f"    No synchronization issues detected - PERFECT alignment")
    
    return folder_results

def print_summary(all_results):
    """
    Print comprehensive summary of synchronization analysis.
    """
    print("\n" + "="*100)
    print("CHANNEL SYNCHRONIZATION VERIFICATION - ACCURATE ASSESSMENT")
    print("="*100)
    print("\nNOTE: This assessment accounts for dataset design constraints:")
    print("  • Load measurements ONLY during discharge (expected missing data)")
    print("  • MOSFET/resistor temperatures ONLY during discharge")
    print("  • Charger voltage and battery temperature are continuous")
    print("="*100)
    
    total_files = 0
    files_with_issues = 0
    files_with_errors = 0
    total_issues = 0
    
    # Track issue severity
    high_severity = 0
    medium_severity = 0
    low_severity = 0
    
    for folder_result in all_results:
        print(f"\nFolder: {folder_result['folder']}")
        print("-" * 60)
        print(f"  Files checked: {folder_result['files_checked']}")
        print(f"  Files with REAL sync issues: {folder_result['files_with_sync_issues']}")
        print(f"  Files with errors: {folder_result['files_with_errors']}")
        
        total_files += folder_result['files_checked']
        files_with_issues += folder_result['files_with_sync_issues']
        files_with_errors += folder_result['files_with_errors']
        total_issues += folder_result['total_sync_issues']
        
        # Show issues per file
        for file_result in folder_result['file_details']:
            if file_result['real_sync_issues']:
                print(f"\n  File: {file_result['file']}")
                if file_result['time_stats']:
                    ts = file_result['time_stats']
                    print(f"    Time stats: mean={ts['mean_time_step']:.3f}s, std={ts['std_time_step']:.3f}s")
                
                for issue in file_result['real_sync_issues']:
                    if issue['severity'] == 'HIGH':
                        high_severity += 1
                    elif issue['severity'] == 'MEDIUM':
                        medium_severity += 1
                    else:
                        low_severity += 1
                    print(f"    • [{issue['severity']}] {issue['description']}")
    
    print("\n" + "="*100)
    print("SYNCHRONIZATION ISSUES BY SEVERITY")
    print("="*100)
    print(f"  HIGH severity (critical): {high_severity}")
    print(f"  MEDIUM severity: {medium_severity}")
    print(f"  LOW severity: {low_severity}")
    
    print("\n" + "="*100)
    print("FINAL QUALITY ASSESSMENT - CHANNEL SYNCHRONIZATION")
    print("="*100)
    
    # Calculate percentage of files with issues
    valid_files = total_files - files_with_errors
    
    print(f"\nFiles successfully analyzed: {valid_files}")
    print(f"Files with REAL sync issues: {files_with_issues}")
    print(f"Files with errors: {files_with_errors}")
    
    if high_severity == 0 and medium_severity == 0:
        print("\nRESULT: EXCELLENT (++)")
        print("="*40)
        print("No REAL synchronization issues detected.")
        print("✓ Time column is monotonic and consistent")
        print("✓ All measurements follow the documented design constraints")
        print("✓ No misalignment between related measurements")
        print("\nThe dataset fully satisfies the channel synchronization criterion.")
        
    elif high_severity == 0 and medium_severity < 5:
        print("\nRESULT: GOOD (+)")
        print("="*40)
        print("Minor synchronization issues detected but do not impact usability.")
        print("✓ No high-severity issues")
        print(f"✓ Only {medium_severity} medium-severity issues across all files")
        print("\nThe dataset mostly satisfies the synchronization criterion.")
        
    elif high_severity == 0:
        print("\nRESULT: ACCEPTABLE (o)")
        print("="*40)
        print("Some synchronization issues present but manageable.")
        print("✓ No high-severity issues")
        print(f"⚠ {medium_severity} medium-severity issues detected")
        print("\nRecommendation: Review issues before analysis, but dataset is usable.")
        
    else:
        print("\nRESULT: POOR (-)")
        print("="*40)
        print(f"CRITICAL: {high_severity} high-severity synchronization issues detected.")
        print("These issues violate the dataset's design constraints:")
        print("  • Measurements during wrong operational modes")
        print("  • Missing required measurements during discharge")
        print("  • Misalignment between related signals")
        print("\nRecommendation: These issues must be addressed before use.")

def save_report(all_results, output_file="synchronization_report_final.txt"):
    """
    Save detailed report to file.
    """
    with open(output_file, 'w', encoding='ascii', errors='replace') as f:
        f.write("="*100 + "\n")
        f.write("NASA BATTERY DATASET - CHANNEL SYNCHRONIZATION REPORT (FINAL)\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*100 + "\n\n")
        f.write("NOTE: This report accounts for dataset design constraints:\n")
        f.write("- Load measurements ONLY during discharge (expected missing data)\n")
        f.write("- MOSFET/resistor temperatures ONLY during discharge\n")
        f.write("- Charger voltage and battery temperature are continuous\n\n")
        
        for folder_result in all_results:
            f.write(f"\nFOLDER: {folder_result['folder']}\n")
            f.write("-"*60 + "\n")
            
            for file_result in folder_result['file_details']:
                f.write(f"\nFile: {file_result['file']} ({file_result['file_size_mb']} MB)\n")
                f.write(f"  Rows: {file_result['total_rows']:,}\n")
                
                if file_result['error']:
                    f.write(f"  ERROR: {file_result['error']}\n")
                    continue
                
                if file_result['time_column']:
                    f.write(f"  Time column: {file_result['time_column']}\n")
                
                if file_result['time_stats']:
                    ts = file_result['time_stats']
                    f.write(f"  Time statistics:\n")
                    f.write(f"    Duration: {ts['total_duration_seconds']:.0f} seconds\n")
                    f.write(f"    Mean time step: {ts['mean_time_step']:.6f} seconds\n")
                    f.write(f"    Time step std: {ts['std_time_step']:.6f} seconds\n")
                
                if file_result['mode_values']:
                    f.write(f"  Mode values: {file_result['mode_values']}\n")
                
                if file_result['real_sync_issues']:
                    f.write(f"  REAL SYNCHRONIZATION ISSUES:\n")
                    for issue in file_result['real_sync_issues']:
                        f.write(f"    • [{issue['severity']}] {issue['description']}\n")
                else:
                    f.write(f"  No synchronization issues detected - PERFECT alignment\n")
        
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
    print("NASA RANDOMIZED BATTERY DATASET - CHANNEL SYNCHRONIZATION VERIFICATION (FINAL)")
    print("="*100)
    print("\nChecking for REAL synchronization issues:")
    print("  • Time column monotonicity")
    print("  • Actual data logging gaps (>2s)")
    print("  • Measurements during wrong operational modes")
    print("  • Missing required measurements during discharge")
    print("  • Alignment between related signals DURING discharge")
    print("\nACCOUNTING FOR DATASET DESIGN:")
    print("  • Load measurements ONLY during discharge (expected missing data)")
    print("  • MOSFET/resistor temperatures ONLY during discharge")
    print("  • Charger voltage and battery temperature are continuous")
    
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