"""
verify_calendar_aging.py

This script detects whether calendar aging data is present in the NASA Randomized Battery Dataset.
Calendar aging refers to battery degradation during storage/rest periods, not just during cycling.

WHAT WE LOOK FOR:
1. Long rest periods (days/weeks) between cycles
2. Capacity measurements after rest periods
3. Temperature-controlled storage conditions
4. Reference discharges to measure degradation over time

CALENDAR AGING INDICATORS:
- Extended periods with mode = 0 (rest) lasting hours/days
- Reference discharges (mission_type = 0) performed after rest periods
- Temperature data during rest periods
- Capacity degradation that correlates with rest time, not just cycle count
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration
MIN_REST_HOURS_FOR_CALENDAR = 12  # Minimum rest period to consider for calendar aging
MIN_CALENDAR_EVENTS = 3  # Minimum number of calendar aging events to confirm presence

# Mode values
MODE_DISCHARGE = -1
MODE_REST = 0
MODE_CHARGE = 1

def detect_calendar_aging(file_path):
    """
    Detect calendar aging indicators in a single CSV file.
    """
    results = {
        'file': os.path.basename(file_path),
        'file_size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2),
        'total_rows': 0,
        'has_calendar_aging': False,
        'calendar_aging_score': 0,  # 0-100, confidence score
        'calendar_aging_indicators': [],
        'rest_periods': [],
        'reference_discharges': [],
        'error': None
    }
    
    indicators = []
    score = 0
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, low_memory=False)
        results['total_rows'] = len(df)
        
        if len(df) == 0:
            results['error'] = "File is empty"
            return results
        
        # Check for required columns
        if 'mode' not in df.columns:
            results['error'] = "No mode column found"
            return results
        
        # Find time column
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
        
        # Analyze rest periods (potential calendar aging)
        rest_periods = []
        in_rest = False
        rest_start = None
        rest_start_idx = None
        
        for i, row in df.iterrows():
            mode = row['mode']
            
            if mode == MODE_REST and not in_rest:
                # Start of rest period
                in_rest = True
                rest_start = row[time_col] if time_col else i
                rest_start_idx = i
            elif mode != MODE_REST and in_rest:
                # End of rest period
                in_rest = False
                rest_end = row[time_col] if time_col else i
                
                if time_col:
                    rest_duration = rest_end - rest_start
                else:
                    rest_duration = i - rest_start_idx
                
                # Convert to hours if time column exists
                if time_col and rest_duration > 0:
                    rest_hours = rest_duration / 3600
                    
                    if rest_hours >= MIN_REST_HOURS_FOR_CALENDAR:
                        rest_periods.append({
                            'start_idx': int(rest_start_idx),
                            'end_idx': int(i),
                            'start_time': float(rest_start),
                            'end_time': float(rest_end),
                            'duration_seconds': float(rest_duration),
                            'duration_hours': float(rest_hours),
                            'duration_days': float(rest_hours / 24)
                        })
        
        results['rest_periods'] = rest_periods
        
        # Find reference discharges (mission_type = 0)
        if 'mission_type' in df.columns:
            reference_discharges = df[df['mission_type'] == 0].index.tolist()
            results['reference_discharges'] = {
                'count': len(reference_discharges),
                'indices': reference_discharges[:10]  # First 10
            }
            
            if len(reference_discharges) >= MIN_CALENDAR_EVENTS:
                indicators.append(f"Has {len(reference_discharges)} reference discharges")
                score += 30
        
        # Check for long rest periods
        if len(rest_periods) >= MIN_CALENDAR_EVENTS:
            indicators.append(f"Has {len(rest_periods)} long rest periods (> {MIN_REST_HOURS_FOR_CALENDAR} hours)")
            score += 30
            
            # Check if rest periods are distributed throughout the test
            if len(rest_periods) > 5:
                indicators.append("Multiple rest periods throughout test")
                score += 20
        
        # Check for temperature data during rest (indicates controlled environment)
        if 'temperature_battery' in df.columns:
            # Get temperature during rest periods
            rest_temps = []
            for period in rest_periods:
                period_data = df.iloc[period['start_idx']:period['end_idx'] + 1]
                temps = period_data['temperature_battery'].dropna()
                if len(temps) > 0:
                    rest_temps.extend(temps.values)
            
            if rest_temps:
                temp_std = np.std(rest_temps)
                if temp_std < 2:  # Stable temperature during rest (controlled environment)
                    indicators.append(f"Stable temperature during rest (std={temp_std:.2f}°C)")
                    score += 20
        
        # Check for capacity measurements at different times
        if 'voltage_charger' in df.columns and time_col:
            # Look for similar voltage patterns at different times
            # This is a weak indicator, but can suggest calendar aging checks
            unique_voltages = df['voltage_charger'].dropna().unique()
            if len(unique_voltages) > 100:  # Enough variation
                # Check if there are voltage measurements spread over time
                time_range = df[time_col].max() - df[time_col].min() if time_col else 0
                if time_range > 30 * 24 * 3600:  # More than 30 days
                    indicators.append(f"Test duration > 30 days ({time_range/86400:.1f} days)")
                    score += 10
        
        # Determine if calendar aging is present
        results['calendar_aging_indicators'] = indicators
        results['calendar_aging_score'] = min(score, 100)
        results['has_calendar_aging'] = score >= 50  # Threshold for confidence
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

def scan_folder(folder_path, folder_name):
    """
    Scan all CSV files in a folder for calendar aging indicators.
    """
    print(f"\n{'='*80}")
    print(f"SCANNING FOLDER: {folder_name}")
    print(f"{'='*80}")
    print(f"  Minimum rest period for calendar aging: {MIN_REST_HOURS_FOR_CALENDAR} hours")
    
    folder_results = {
        'folder': folder_name,
        'files_checked': 0,
        'files_with_errors': 0,
        'files_with_calendar_aging': 0,
        'total_calendar_score': 0,
        'file_details': [],
        'summary_stats': {
            'total_rest_periods': 0,
            'total_reference_discharges': 0,
            'max_test_duration_days': 0,
            'has_calendar_aging': False
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
    
    total_rest = 0
    total_ref = 0
    max_duration = 0
    
    for csv_file in sorted(csv_files):
        print(f"  Analyzing: {csv_file.name}")
        results = detect_calendar_aging(csv_file)
        folder_results['file_details'].append(results)
        
        if results['error']:
            folder_results['files_with_errors'] += 1
            print(f"    ERROR: {results['error']}")
        else:
            # Aggregate statistics
            folder_results['total_calendar_score'] += results['calendar_aging_score']
            
            if results['has_calendar_aging']:
                folder_results['files_with_calendar_aging'] += 1
                print(f"    ✓ Calendar aging detected (score: {results['calendar_aging_score']})")
                
                # Show indicators
                for indicator in results['calendar_aging_indicators']:
                    print(f"      - {indicator}")
            else:
                print(f"    ✗ No calendar aging detected (score: {results['calendar_aging_score']})")
            
            # Count rest periods and reference discharges
            total_rest += len(results['rest_periods'])
            total_ref += results.get('reference_discharges', {}).get('count', 0)
            
            # Check test duration
            if results['rest_periods']:
                max_period = max([p['duration_days'] for p in results['rest_periods']], default=0)
                max_duration = max(max_duration, max_period)
    
    folder_results['summary_stats']['total_rest_periods'] = total_rest
    folder_results['summary_stats']['total_reference_discharges'] = total_ref
    folder_results['summary_stats']['max_test_duration_days'] = max_duration
    folder_results['summary_stats']['has_calendar_aging'] = folder_results['files_with_calendar_aging'] > 0
    
    return folder_results

def print_summary(all_results):
    """
    Print comprehensive summary of calendar aging analysis.
    """
    print("\n" + "="*100)
    print("CALENDAR AGING ANALYSIS - SUMMARY")
    print("="*100)
    print("\nCalendar aging indicators:")
    print("  • Long rest periods (>12 hours) between cycles")
    print("  • Reference discharges to measure degradation")
    print("  • Stable temperature during rest (controlled environment)")
    print("  • Extended test duration (>30 days)")
    print("="*100)
    
    total_files = 0
    files_with_errors = 0
    files_with_calendar = 0
    total_rest = 0
    total_ref = 0
    max_duration = 0
    
    for folder_result in all_results:
        print(f"\nFolder: {folder_result['folder']}")
        print("-" * 60)
        print(f"  Files checked: {folder_result['files_checked']}")
        print(f"  Files with errors: {folder_result['files_with_errors']}")
        print(f"  Files with calendar aging: {folder_result['files_with_calendar_aging']}")
        print(f"  Total rest periods: {folder_result['summary_stats']['total_rest_periods']}")
        print(f"  Total reference discharges: {folder_result['summary_stats']['total_reference_discharges']}")
        print(f"  Max rest duration: {folder_result['summary_stats']['max_test_duration_days']:.1f} days")
        
        total_files += folder_result['files_checked']
        files_with_errors += folder_result['files_with_errors']
        files_with_calendar += folder_result['files_with_calendar_aging']
        total_rest += folder_result['summary_stats']['total_rest_periods']
        total_ref += folder_result['summary_stats']['total_reference_discharges']
        max_duration = max(max_duration, folder_result['summary_stats']['max_test_duration_days'])
    
    print("\n" + "="*100)
    print("OVERALL STATISTICS")
    print("="*100)
    print(f"Total files analyzed: {total_files}")
    print(f"Files with calendar aging indicators: {files_with_calendar}")
    print(f"Total long rest periods detected: {total_rest}")
    print(f"Total reference discharges: {total_ref}")
    print(f"Maximum rest duration: {max_duration:.1f} days")
    
    print("\n" + "="*100)
    print("FINAL QUALITY ASSESSMENT - CALENDAR AGING COVERAGE")
    print("="*100)
    
    # Calculate percentage of files with calendar aging
    valid_files = total_files - files_with_errors
    calendar_percentage = (files_with_calendar / valid_files * 100) if valid_files > 0 else 0
    
    # Determine if calendar aging is present
    has_calendar = files_with_calendar > 0 and total_rest >= MIN_CALENDAR_EVENTS
    
    if has_calendar and calendar_percentage > 50:
        print("\nRESULT: EXCELLENT (++)")
        print("="*40)
        print("Strong calendar aging presence detected:")
        print(f"  ✓ {files_with_calendar} files show calendar aging indicators")
        print(f"  ✓ {total_rest} long rest periods across the dataset")
        print(f"  ✓ {total_ref} reference discharges for degradation measurement")
        print("\nThe dataset EXCELLENTLY covers calendar aging.")
        print("This allows studying degradation during storage periods.")
        
    elif has_calendar and calendar_percentage > 20:
        print("\nRESULT: GOOD (+)")
        print("="*40)
        print("Calendar aging indicators present in some files:")
        print(f"  ✓ {files_with_calendar} files show calendar aging")
        print(f"  ✓ {total_rest} rest periods detected")
        print(f"  ⚠ Not all files include calendar aging")
        print("\nThe dataset GOODly covers calendar aging.")
        print("Some files can be used for calendar aging studies.")
        
    elif has_calendar:
        print("\nRESULT: ACCEPTABLE (o)")
        print("="*40)
        print("Calendar aging minimally present:")
        print(f"  • {files_with_calendar} files show some calendar aging")
        print(f"  • {total_rest} rest periods detected")
        print("\nThe dataset ACCEPTABLY covers calendar aging.")
        print("Limited data for calendar aging studies, but some exists.")
        
    else:
        print("\nRESULT: POOR (-)")
        print("="*40)
        print("No significant calendar aging indicators detected:")
        print("  ✗ Very few or no long rest periods")
        print("  ✗ Batteries continuously cycled")
        print("\nThe dataset POORLY covers calendar aging.")
        print("This dataset focuses on CYCLE AGING only.")
        print("\nBased on the documentation, this matches expectations:")
        print("  • 'accelerated life testing' - typically focuses on cycle aging")
        print("  • Continuous cycling until failure")
        print("  • No mention of storage periods in the README")

def save_report(all_results, output_file="calendar_aging_report.txt"):
    """
    Save detailed report to file.
    """
    with open(output_file, 'w', encoding='ascii', errors='replace') as f:
        f.write("="*100 + "\n")
        f.write("NASA BATTERY DATASET - CALENDAR AGING ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*100 + "\n\n")
        f.write(f"Analysis parameters:\n")
        f.write(f"  Minimum rest period: {MIN_REST_HOURS_FOR_CALENDAR} hours\n")
        f.write(f"  Minimum calendar events: {MIN_CALENDAR_EVENTS}\n\n")
        
        for folder_result in all_results:
            f.write(f"\nFOLDER: {folder_result['folder']}\n")
            f.write("-"*60 + "\n")
            
            for file_result in folder_result['file_details']:
                f.write(f"\nFile: {file_result['file']} ({file_result['file_size_mb']} MB)\n")
                f.write(f"  Rows: {file_result['total_rows']:,}\n")
                
                if file_result['error']:
                    f.write(f"  ERROR: {file_result['error']}\n")
                    continue
                
                f.write(f"  Calendar aging detected: {file_result['has_calendar_aging']}\n")
                f.write(f"  Confidence score: {file_result['calendar_aging_score']}/100\n")
                
                if file_result['calendar_aging_indicators']:
                    f.write(f"  Indicators:\n")
                    for indicator in file_result['calendar_aging_indicators']:
                        f.write(f"    • {indicator}\n")
                
                if file_result['rest_periods']:
                    f.write(f"  Long rest periods:\n")
                    for i, period in enumerate(file_result['rest_periods'][:5]):  # First 5
                        f.write(f"    {i+1}. {period['duration_hours']:.1f} hours ({period['duration_days']:.2f} days)\n")
                
                if file_result.get('reference_discharges', {}).get('count', 0) > 0:
                    ref = file_result['reference_discharges']
                    f.write(f"  Reference discharges: {ref['count']}\n")
        
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
    print("NASA RANDOMIZED BATTERY DATASET - CALENDAR AGING ANALYSIS")
    print("="*100)
    print("\nChecking for calendar aging indicators:")
    print("  • Long rest periods between cycles")
    print("  • Reference discharges for degradation measurement")
    print("  • Stable temperature during rest")
    print("  • Extended test duration")
    print(f"\nBased on documentation, this is likely a CYCLE AGING dataset")
    print(f"(continuous cycling until failure, minimal storage periods)")
    
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