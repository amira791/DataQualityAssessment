"""
verify_null_values.py

This script quantifies missing values for each critical measurement channel,
accounting for the dataset design where some columns are only recorded during
specific operational modes.

DESIGN CONSTRAINTS (from documentation):
- Load measurements (voltage_load, current_load) ONLY occur during discharge missions
- MOSFET temperature ONLY recorded during discharge
- Resistor temperature ONLY recorded during discharge
- Charger voltage and battery temperature are continuous (should have data always)
- Mission type only recorded during discharge

Therefore, "missing" data during rest/charge modes is EXPECTED for discharge-only columns.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Define column categories based on dataset design
CONTINUOUS_COLUMNS = [
    'voltage_charger',
    'temperature_battery',
    'time',
    'mode'
]

DISCHARGE_ONLY_COLUMNS = [
    'voltage_load',
    'current_load',
    'temperature_mosfet',
    'temperature_resistor',
    'mission_type'
]

ALL_COLUMNS = CONTINUOUS_COLUMNS + DISCHARGE_ONLY_COLUMNS

def analyze_null_values(file_path):
    """
    Analyze null values for a single CSV file, accounting for design constraints.
    """
    results = {
        'file': os.path.basename(file_path),
        'file_size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2),
        'total_rows': 0,
        'mode_distribution': {},
        'continuous_columns': {},
        'discharge_columns': {},
        'expected_nulls': {},  # Nulls that are expected by design
        'unexpected_nulls': {},  # Nulls that should have data
        'overall_stats': {},
        'error': None
    }
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, low_memory=False)
        results['total_rows'] = len(df)
        
        if len(df) == 0:
            results['error'] = "File is empty"
            return results
        
        # Check if mode column exists (needed for context)
        has_mode = 'mode' in df.columns
        if not has_mode:
            results['error'] = "No mode column found - cannot contextualize nulls"
            return results
        
        # Get mode distribution
        mode_counts = df['mode'].value_counts().sort_index()
        results['mode_distribution'] = {
            'discharge': int(mode_counts.get(-1, 0)),
            'rest': int(mode_counts.get(0, 0)),
            'charge': int(mode_counts.get(1, 0)),
            'other': int(mode_counts.get(pd.NA, 0))
        }
        
        # Calculate total discharge rows for context
        discharge_rows = results['mode_distribution']['discharge']
        total_rows = results['total_rows']
        
        # Analyze continuous columns (should have minimal nulls)
        for col in CONTINUOUS_COLUMNS:
            if col in df.columns:
                null_count = df[col].isna().sum()
                null_percent = (null_count / total_rows) * 100
                
                # Get additional stats
                if col != 'mode' and col != 'time':  # Skip non-numeric for stats
                    valid_data = df[col].dropna()
                    if len(valid_data) > 0:
                        stats = {
                            'min': round(float(valid_data.min()), 3),
                            'max': round(float(valid_data.max()), 3),
                            'mean': round(float(valid_data.mean()), 3)
                        }
                    else:
                        stats = {}
                else:
                    stats = {}
                
                results['continuous_columns'][col] = {
                    'total_rows': total_rows,
                    'null_count': int(null_count),
                    'null_percent': round(null_percent, 4),
                    'stats': stats
                }
                
                # For continuous columns, any null is unexpected
                if null_count > 0:
                    results['unexpected_nulls'][col] = {
                        'count': int(null_count),
                        'percent': round(null_percent, 4),
                        'expected': 0,
                        'context': 'Continuous column should have data for all rows'
                    }
            else:
                results['continuous_columns'][col] = {
                    'status': 'column_not_found',
                    'note': f'Column {col} not in file'
                }
        
        # Analyze discharge-only columns
        for col in DISCHARGE_ONLY_COLUMNS:
            if col in df.columns:
                # Total nulls in this column
                null_count = df[col].isna().sum()
                null_percent = (null_count / total_rows) * 100
                
                # During discharge, this column should have data
                discharge_data = df[df['mode'] == -1][col]
                discharge_nulls = discharge_data.isna().sum()
                discharge_rows_with_data = len(discharge_data) - discharge_nulls
                
                # During rest/charge, nulls are expected
                non_discharge_data = df[df['mode'] != -1][col]
                non_discharge_nulls = non_discharge_data.isna().sum()
                non_discharge_rows = len(non_discharge_data)
                
                # Calculate percentages
                discharge_null_percent = (discharge_nulls / discharge_rows * 100) if discharge_rows > 0 else 0
                non_discharge_null_percent = (non_discharge_nulls / non_discharge_rows * 100) if non_discharge_rows > 0 else 0
                
                # Get statistics from non-null values
                valid_data = df[col].dropna()
                if len(valid_data) > 0:
                    stats = {
                        'min': round(float(valid_data.min()), 3),
                        'max': round(float(valid_data.max()), 3),
                        'mean': round(float(valid_data.mean()), 3)
                    }
                else:
                    stats = {}
                
                results['discharge_columns'][col] = {
                    'total_rows': total_rows,
                    'overall_null_count': int(null_count),
                    'overall_null_percent': round(null_percent, 4),
                    'discharge_rows': discharge_rows,
                    'discharge_nulls': int(discharge_nulls),
                    'discharge_null_percent': round(discharge_null_percent, 4),
                    'discharge_data_rows': int(discharge_rows_with_data),
                    'non_discharge_rows': non_discharge_rows,
                    'non_discharge_nulls': int(non_discharge_nulls),
                    'non_discharge_null_percent': round(non_discharge_null_percent, 4),
                    'stats': stats
                }
                
                # Expected nulls (during rest/charge)
                if non_discharge_nulls > 0:
                    results['expected_nulls'][col] = {
                        'count': int(non_discharge_nulls),
                        'percent': round(non_discharge_null_percent, 4),
                        'context': 'Nulls during rest/charge are expected by design'
                    }
                
                # Unexpected nulls (during discharge)
                if discharge_nulls > 0:
                    results['unexpected_nulls'][col] = {
                        'count': int(discharge_nulls),
                        'percent': round(discharge_null_percent, 4),
                        'context': f'Column should have data during discharge ({discharge_rows} discharge rows)',
                        'missing_during_discharge': int(discharge_nulls)
                    }
            else:
                results['discharge_columns'][col] = {
                    'status': 'column_not_found',
                    'note': f'Column {col} not in file'
                }
        
        # Calculate overall statistics
        total_unexpected_nulls = sum(v['count'] for v in results['unexpected_nulls'].values())
        total_expected_nulls = sum(v['count'] for v in results['expected_nulls'].values())
        
        # Calculate data completeness for discharge periods
        if discharge_rows > 0:
            discharge_completeness = {}
            for col in DISCHARGE_ONLY_COLUMNS:
                if col in df.columns and col in results['discharge_columns']:
                    col_data = results['discharge_columns'][col]
                    if col_data['discharge_rows'] > 0:
                        completeness = 100 - col_data['discharge_null_percent']
                        discharge_completeness[col] = round(completeness, 2)
            
            results['overall_stats'] = {
                'total_unexpected_nulls': total_unexpected_nulls,
                'total_expected_nulls': total_expected_nulls,
                'discharge_rows': discharge_rows,
                'discharge_completeness': discharge_completeness,
                'avg_discharge_completeness': round(np.mean(list(discharge_completeness.values())), 2) if discharge_completeness else None
            }
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

def scan_folder(folder_path, folder_name):
    """
    Scan all CSV files in a folder for null value analysis.
    """
    print(f"\n{'='*80}")
    print(f"SCANNING FOLDER: {folder_name}")
    print(f"{'='*80}")
    
    folder_results = {
        'folder': folder_name,
        'files_checked': 0,
        'files_with_errors': 0,
        'files_with_unexpected_nulls': 0,
        'total_unexpected_nulls': 0,
        'file_details': [],
        'summary_stats': {
            'avg_continuous_completeness': 0,
            'avg_discharge_completeness': 0,
            'worst_file': None,
            'worst_completeness': 100
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
    
    continuous_completeness = []
    discharge_completeness = []
    worst_completeness = 100
    worst_file = None
    
    for csv_file in sorted(csv_files):
        print(f"  Checking: {csv_file.name}")
        results = analyze_null_values(csv_file)
        folder_results['file_details'].append(results)
        
        if results['error']:
            folder_results['files_with_errors'] += 1
            print(f"    ERROR: {results['error']}")
        else:
            # Check for unexpected nulls
            if results['unexpected_nulls']:
                folder_results['files_with_unexpected_nulls'] += 1
                folder_results['total_unexpected_nulls'] += results['overall_stats'].get('total_unexpected_nulls', 0)
                print(f"    Found {len(results['unexpected_nulls'])} columns with unexpected nulls")
            
            # Calculate continuous column completeness
            continuous_complete = []
            for col, data in results['continuous_columns'].items():
                if isinstance(data, dict) and 'null_percent' in data:
                    continuous_complete.append(100 - data['null_percent'])
            
            if continuous_complete:
                avg_cont = np.mean(continuous_complete)
                continuous_completeness.append(avg_cont)
            
            # Get discharge completeness
            if results['overall_stats'].get('avg_discharge_completeness'):
                dis_comp = results['overall_stats']['avg_discharge_completeness']
                discharge_completeness.append(dis_comp)
                
                # Track worst file
                if dis_comp < worst_completeness:
                    worst_completeness = dis_comp
                    worst_file = results['file']
            
            # Print quick summary
            if results['overall_stats'].get('avg_discharge_completeness'):
                print(f"    Discharge data completeness: {results['overall_stats']['avg_discharge_completeness']}%")
    
    # Calculate averages
    if continuous_completeness:
        folder_results['summary_stats']['avg_continuous_completeness'] = round(np.mean(continuous_completeness), 2)
    if discharge_completeness:
        folder_results['summary_stats']['avg_discharge_completeness'] = round(np.mean(discharge_completeness), 2)
    folder_results['summary_stats']['worst_file'] = worst_file
    folder_results['summary_stats']['worst_completeness'] = round(worst_completeness, 2) if worst_completeness < 100 else 100
    
    return folder_results

def print_summary(all_results):
    """
    Print comprehensive summary of null value analysis.
    """
    print("\n" + "="*100)
    print("NULL VALUE QUANTIFICATION - SUMMARY")
    print("="*100)
    print("\nDATASET DESIGN CONSTRAINTS:")
    print("  • Continuous columns (should have data always):")
    print("    - voltage_charger, temperature_battery, time, mode")
    print("  • Discharge-only columns (data only during mode=-1):")
    print("    - voltage_load, current_load, temperature_mosfet, temperature_resistor, mission_type")
    print("="*100)
    
    total_files = 0
    files_with_errors = 0
    files_with_unexpected = 0
    total_unexpected = 0
    
    all_continuous_comp = []
    all_discharge_comp = []
    
    for folder_result in all_results:
        print(f"\nFolder: {folder_result['folder']}")
        print("-" * 60)
        print(f"  Files checked: {folder_result['files_checked']}")
        print(f"  Files with errors: {folder_result['files_with_errors']}")
        print(f"  Files with unexpected nulls: {folder_result['files_with_unexpected_nulls']}")
        print(f"  Total unexpected nulls: {folder_result['total_unexpected_nulls']:,}")
        print(f"  Avg continuous completeness: {folder_result['summary_stats']['avg_continuous_completeness']}%")
        print(f"  Avg discharge completeness: {folder_result['summary_stats']['avg_discharge_completeness']}%")
        
        if folder_result['summary_stats']['worst_file']:
            print(f"  Worst file: {folder_result['summary_stats']['worst_file']} "
                  f"({folder_result['summary_stats']['worst_completeness']}% discharge completeness)")
        
        total_files += folder_result['files_checked']
        files_with_errors += folder_result['files_with_errors']
        files_with_unexpected += folder_result['files_with_unexpected_nulls']
        total_unexpected += folder_result['total_unexpected_nulls']
        
        if folder_result['summary_stats']['avg_continuous_completeness'] > 0:
            all_continuous_comp.append(folder_result['summary_stats']['avg_continuous_completeness'])
        if folder_result['summary_stats']['avg_discharge_completeness'] > 0:
            all_discharge_comp.append(folder_result['summary_stats']['avg_discharge_completeness'])
        
        # Show detailed column analysis for first few files
        print(f"\n  Sample column analysis (first file):")
        for file_result in folder_result['file_details'][:1]:
            if not file_result['error']:
                # Continuous columns
                print(f"    Continuous columns:")
                for col, data in file_result['continuous_columns'].items():
                    if isinstance(data, dict) and 'null_percent' in data:
                        status = "OK" if data['null_percent'] == 0 else f"MISSING {data['null_percent']}%"
                        print(f"      • {col}: {status}")
                
                # Discharge columns
                print(f"    Discharge-only columns:")
                for col, data in file_result['discharge_columns'].items():
                    if isinstance(data, dict) and 'discharge_null_percent' in data:
                        if data['discharge_null_percent'] == 0:
                            print(f"      • {col}: OK (data during all discharge)")
                        else:
                            print(f"      • {col}: {data['discharge_null_percent']}% missing during discharge")
    
    print("\n" + "="*100)
    print("OVERALL STATISTICS")
    print("="*100)
    print(f"Total files analyzed: {total_files}")
    print(f"Files with errors: {files_with_errors}")
    print(f"Files with unexpected nulls: {files_with_unexpected}")
    print(f"Total unexpected nulls across all files: {total_unexpected:,}")
    
    if all_continuous_comp:
        print(f"Average continuous column completeness: {np.mean(all_continuous_comp):.2f}%")
    if all_discharge_comp:
        print(f"Average discharge column completeness: {np.mean(all_discharge_comp):.2f}%")
    
    print("\n" + "="*100)
    print("FINAL QUALITY ASSESSMENT - NULL VALUE QUANTIFICATION")
    print("="*100)
    
    # Determine quality based on unexpected nulls
    if files_with_unexpected == 0 and all_discharge_comp and np.mean(all_discharge_comp) == 100:
        print("\nRESULT: EXCELLENT (++)")
        print("="*40)
        print("Perfect data completeness:")
        print("  ✓ No unexpected nulls in any file")
        print("  ✓ Continuous columns have 100% data")
        print("  ✓ Discharge columns have data for all discharge periods")
        print("\nThe dataset fully satisfies the null value criterion.")
        
    elif files_with_unexpected == 0 and all_discharge_comp and np.mean(all_discharge_comp) > 99.5:
        print("\nRESULT: GOOD (+)")
        print("="*40)
        print("Excellent data completeness with minor issues:")
        print(f"  ✓ No files with unexpected nulls")
        print(f"  ✓ {np.mean(all_discharge_comp):.2f}% discharge data completeness")
        print("\nMinor data gaps do not impact overall quality.")
        
    elif files_with_unexpected < total_files * 0.1 and np.mean(all_discharge_comp) > 98:
        print("\nRESULT: ACCEPTABLE (o)")
        print("="*40)
        print("Acceptable data completeness:")
        print(f"  • {files_with_unexpected} files have unexpected nulls ({files_with_unexpected/total_files*100:.1f}%)")
        print(f"  • {np.mean(all_discharge_comp):.2f}% average discharge completeness")
        print("\nThe dataset is usable but has some missing data during discharge periods.")
        
    else:
        print("\nRESULT: POOR (-)")
        print("="*40)
        print("Significant unexpected nulls detected:")
        print(f"  • {files_with_unexpected} files have unexpected nulls")
        print(f"  • Total unexpected nulls: {total_unexpected:,}")
        if all_discharge_comp:
            print(f"  • Average discharge completeness: {np.mean(all_discharge_comp):.2f}%")
        print("\nRecommendation: Investigate files with missing data during discharge periods.")

def save_report(all_results, output_file="null_values_report.txt"):
    """
    Save detailed report to file.
    """
    with open(output_file, 'w', encoding='ascii', errors='replace') as f:
        f.write("="*100 + "\n")
        f.write("NASA BATTERY DATASET - NULL VALUE QUANTIFICATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*100 + "\n\n")
        f.write("DESIGN CONSTRAINTS:\n")
        f.write("- Continuous columns: voltage_charger, temperature_battery, time, mode\n")
        f.write("- Discharge-only columns: voltage_load, current_load, temperature_mosfet, temperature_resistor, mission_type\n")
        f.write("- Nulls during rest/charge are EXPECTED for discharge-only columns\n")
        f.write("- Nulls during discharge are UNEXPECTED\n\n")
        
        for folder_result in all_results:
            f.write(f"\nFOLDER: {folder_result['folder']}\n")
            f.write("-"*60 + "\n")
            
            for file_result in folder_result['file_details']:
                f.write(f"\nFile: {file_result['file']} ({file_result['file_size_mb']} MB)\n")
                f.write(f"  Rows: {file_result['total_rows']:,}\n")
                
                if file_result['error']:
                    f.write(f"  ERROR: {file_result['error']}\n")
                    continue
                
                # Mode distribution
                md = file_result['mode_distribution']
                f.write(f"  Mode distribution: discharge={md['discharge']}, rest={md['rest']}, charge={md['charge']}\n")
                
                # Continuous columns
                f.write(f"\n  CONTINUOUS COLUMNS:\n")
                for col, data in file_result['continuous_columns'].items():
                    if isinstance(data, dict) and 'null_percent' in data:
                        status = "OK" if data['null_percent'] == 0 else f"MISSING {data['null_percent']}%"
                        f.write(f"    {col}: {status}\n")
                
                # Discharge columns
                f.write(f"\n  DISCHARGE-ONLY COLUMNS:\n")
                for col, data in file_result['discharge_columns'].items():
                    if isinstance(data, dict) and 'discharge_null_percent' in data:
                        f.write(f"    {col}:\n")
                        f.write(f"      Overall nulls: {data['overall_null_percent']}% ({data['overall_null_count']} rows)\n")
                        f.write(f"      During discharge: {data['discharge_null_percent']}% null ({data['discharge_nulls']} of {data['discharge_rows']} rows)\n")
                        f.write(f"      During rest/charge: {data['non_discharge_null_percent']}% null (expected)\n")
                        if data['stats']:
                            f.write(f"      Stats: min={data['stats']['min']}, max={data['stats']['max']}, mean={data['stats']['mean']}\n")
                
                # Expected nulls (by design)
                if file_result['expected_nulls']:
                    f.write(f"\n  EXPECTED NULLS (by design):\n")
                    for col, data in file_result['expected_nulls'].items():
                        f.write(f"    {col}: {data['count']} rows ({data['percent']}%) - {data['context']}\n")
                
                # Unexpected nulls (issues)
                if file_result['unexpected_nulls']:
                    f.write(f"\n  UNEXPECTED NULLS (ISSUES):\n")
                    for col, data in file_result['unexpected_nulls'].items():
                        f.write(f"    {col}: {data['count']} rows ({data['percent']}%) - {data['context']}\n")
                
                # Overall stats
                if file_result['overall_stats']:
                    os = file_result['overall_stats']
                    f.write(f"\n  OVERALL:\n")
                    f.write(f"    Discharge rows: {os['discharge_rows']}\n")
                    if os.get('discharge_completeness'):
                        f.write(f"    Discharge completeness by column:\n")
                        for col, comp in os['discharge_completeness'].items():
                            f.write(f"      {col}: {comp}%\n")
        
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
    print("NASA RANDOMIZED BATTERY DATASET - NULL VALUE QUANTIFICATION")
    print("="*100)
    print("\nAnalyzing missing values with context:")
    print("  • Continuous columns: should have 0% nulls")
    print("  • Discharge-only columns: nulls expected during rest/charge")
    print("  • Discharge-only columns: should have 0% nulls during discharge")
    
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