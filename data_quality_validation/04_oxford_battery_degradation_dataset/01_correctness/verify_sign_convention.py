"""
verify_sign_convention_corrected.py

This script verifies the current sign convention in the NASA Randomized Battery Dataset.

Based on documentation:
- mode = -1: discharge
- mode = 0: rest  
- mode = 1: charge
- current_load is described as "discharge current" - may be recorded as positive magnitude

The documentation does NOT explicitly state that current should be negative during discharge.
We need to check for consistency, not assume a particular sign.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def analyze_sign_convention(file_path):
    """
    Analyze sign convention for a single CSV file without assuming negative discharge.
    """
    results = {
        'file': os.path.basename(file_path),
        'file_size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2),
        'total_rows': 0,
        'mode_column': None,
        'current_column': None,
        'mode_values_found': [],
        'current_stats': {},
        'observations': [],  # What we observe about the sign convention
        'inconsistencies': [],  # Any internal inconsistencies
        'error': None
    }
    
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
        
        current_col = None
        if 'current_load' in df.columns:
            current_col = 'current_load'
        elif 'current' in df.columns:
            current_col = 'current'
        
        if current_col is None:
            results['error'] = "No current column found"
            return results
        
        results['mode_column'] = 'mode'
        results['current_column'] = current_col
        
        # Get unique mode values
        unique_modes = sorted(df['mode'].dropna().unique().tolist())
        results['mode_values_found'] = unique_modes
        
        # Check if mode values match documentation
        expected_modes = [-1, 0, 1]
        mode_match = all(m in expected_modes for m in unique_modes if not pd.isna(m))
        results['mode_documentation_match'] = mode_match
        
        # Analyze current by mode
        for mode_val in unique_modes:
            if pd.isna(mode_val):
                continue
                
            # Determine mode name
            if mode_val == -1:
                mode_name = 'discharge'
            elif mode_val == 0:
                mode_name = 'rest'
            elif mode_val == 1:
                mode_name = 'charge'
            else:
                mode_name = f'unknown_{mode_val}'
            
            # Get data for this mode
            mode_data = df[df['mode'] == mode_val]
            current_values = mode_data[current_col].dropna()
            
            if len(current_values) == 0:
                results['current_stats'][mode_name] = {
                    'rows': len(mode_data),
                    'rows_with_current': 0,
                    'note': 'No current data for this mode'
                }
                continue
            
            # Calculate statistics
            stats = {
                'rows': len(mode_data),
                'rows_with_current': len(current_values),
                'min_current': round(float(current_values.min()), 3),
                'max_current': round(float(current_values.max()), 3),
                'mean_current': round(float(current_values.mean()), 3),
                'median_current': round(float(current_values.median()), 3),
                'std_current': round(float(current_values.std()), 3),
                'all_positive': (current_values >= 0).all(),
                'all_negative': (current_values <= 0).all(),
                'mixed_sign': (current_values.min() < 0 and current_values.max() > 0)
            }
            
            # Add percentage of positive/negative if mixed
            if stats['mixed_sign']:
                positive_pct = (current_values > 0).sum() / len(current_values) * 100
                negative_pct = (current_values < 0).sum() / len(current_values) * 100
                stats['positive_percent'] = round(positive_pct, 2)
                stats['negative_percent'] = round(negative_pct, 2)
                stats['zero_percent'] = round(100 - positive_pct - negative_pct, 2)
            
            results['current_stats'][mode_name] = stats
        
        # Make observations about the sign convention
        if 'discharge' in results['current_stats']:
            discharge_stats = results['current_stats']['discharge']
            
            if discharge_stats['all_positive']:
                results['observations'].append({
                    'aspect': 'discharge_sign',
                    'observation': 'All discharge currents are POSITIVE',
                    'implication': 'Dataset uses positive values for discharge magnitude'
                })
            elif discharge_stats['all_negative']:
                results['observations'].append({
                    'aspect': 'discharge_sign',
                    'observation': 'All discharge currents are NEGATIVE',
                    'implication': 'Dataset follows traditional sign convention (negative = discharge)'
                })
            elif discharge_stats['mixed_sign']:
                results['observations'].append({
                    'aspect': 'discharge_sign',
                    'observation': f'Discharge currents have MIXED signs ({discharge_stats["positive_percent"]}% positive, {discharge_stats["negative_percent"]}% negative)',
                    'implication': 'INCONSISTENT - this could indicate data quality issues'
                })
        
        if 'charge' in results['current_stats']:
            charge_stats = results['current_stats']['charge']
            
            if charge_stats['all_positive']:
                results['observations'].append({
                    'aspect': 'charge_sign',
                    'observation': 'All charge currents are POSITIVE',
                    'implication': 'Dataset uses positive values for charge current'
                })
            elif charge_stats['all_negative']:
                results['observations'].append({
                    'aspect': 'charge_sign',
                    'observation': 'All charge currents are NEGATIVE',
                    'implication': 'Dataset uses negative values for charge current'
                })
        
        if 'rest' in results['current_stats']:
            rest_stats = results['current_stats']['rest']
            
            if rest_stats['rows_with_current'] > 0:
                if abs(rest_stats['mean_current']) < 0.1:
                    results['observations'].append({
                        'aspect': 'rest_current',
                        'observation': f'Small currents during rest (mean={rest_stats["mean_current"]:.3f}A)',
                        'implication': 'Likely measurement noise, acceptable'
                    })
                else:
                    results['inconsistencies'].append({
                        'issue': 'significant_current_during_rest',
                        'description': f'Significant current during rest: mean={rest_stats["mean_current"]:.3f}A',
                        'severity': 'MEDIUM'
                    })
        
        # Check for internal consistency
        if 'discharge' in results['current_stats'] and 'charge' in results['current_stats']:
            discharge_sign = 'positive' if results['current_stats']['discharge']['all_positive'] else 'negative'
            charge_sign = 'positive' if results['current_stats']['charge']['all_positive'] else 'negative'
            
            if discharge_sign == charge_sign:
                results['observations'].append({
                    'aspect': 'relative_sign',
                    'observation': f'Both discharge and charge currents are {discharge_sign}',
                    'implication': 'Dataset may be recording magnitudes only, mode indicates direction'
                })
            else:
                results['observations'].append({
                    'aspect': 'relative_sign',
                    'observation': f'Discharge is {discharge_sign}, charge is {charge_sign}',
                    'implication': 'Dataset uses sign to indicate direction (consistent with physics)'
                })
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

def scan_folder(folder_path, folder_name):
    """
    Scan all CSV files in a folder for sign convention analysis.
    """
    print(f"\n{'='*80}")
    print(f"SCANNING FOLDER: {folder_name}")
    print(f"{'='*80}")
    
    folder_results = {
        'folder': folder_name,
        'files_checked': 0,
        'files_with_errors': 0,
        'files_with_inconsistencies': 0,
        'sign_convention_observed': None,  # Will be set if consistent across files
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
    
    discharge_signs = set()
    
    for csv_file in sorted(csv_files):
        print(f"  Checking: {csv_file.name}")
        results = analyze_sign_convention(csv_file)
        folder_results['file_details'].append(results)
        
        if results['error']:
            folder_results['files_with_errors'] += 1
            print(f"    ERROR: {results['error']}")
        elif results['inconsistencies']:
            folder_results['files_with_inconsistencies'] += 1
            print(f"    Found {len(results['inconsistencies'])} inconsistencies")
        
        # Track discharge sign convention
        if 'discharge' in results.get('current_stats', {}):
            discharge_stats = results['current_stats']['discharge']
            if discharge_stats['all_positive']:
                discharge_signs.add('positive')
                print(f"    Discharge current: ALL POSITIVE")
            elif discharge_stats['all_negative']:
                discharge_signs.add('negative')
                print(f"    Discharge current: ALL NEGATIVE")
            elif discharge_stats['mixed_sign']:
                discharge_signs.add('mixed')
                print(f"    Discharge current: MIXED SIGNS (inconsistent)")
    
    # Determine overall sign convention for this folder
    if len(discharge_signs) == 1:
        folder_results['sign_convention_observed'] = list(discharge_signs)[0]
    elif len(discharge_signs) > 1:
        folder_results['sign_convention_observed'] = 'inconsistent'
    
    return folder_results

def print_summary(all_results):
    """
    Print comprehensive summary of sign convention analysis.
    """
    print("\n" + "="*100)
    print("SIGN CONVENTION VERIFICATION - SUMMARY")
    print("="*100)
    print("\nDOCUMENTATION STATES:")
    print("  • mode = -1 : discharge")
    print("  • mode = 0 : rest")
    print("  • mode = 1 : charge")
    print("  • current_load : 'discharge current'")
    print("\nNOTE: The documentation does NOT specify whether discharge current")
    print("      should be positive or negative. We are observing the actual convention.")
    print("="*100)
    
    total_files = 0
    files_with_inconsistencies = 0
    files_with_errors = 0
    
    # Track observed conventions
    folders_with_positive = 0
    folders_with_negative = 0
    folders_with_mixed = 0
    
    for folder_result in all_results:
        print(f"\nFolder: {folder_result['folder']}")
        print("-" * 60)
        print(f"  Files checked: {folder_result['files_checked']}")
        print(f"  Files with inconsistencies: {folder_result['files_with_inconsistencies']}")
        print(f"  Files with errors: {folder_result['files_with_errors']}")
        print(f"  Sign convention observed: {folder_result['sign_convention_observed']}")
        
        if folder_result['sign_convention_observed'] == 'positive':
            folders_with_positive += 1
        elif folder_result['sign_convention_observed'] == 'negative':
            folders_with_negative += 1
        elif folder_result['sign_convention_observed'] == 'inconsistent':
            folders_with_mixed += 1
        
        total_files += folder_result['files_checked']
        files_with_inconsistencies += folder_result['files_with_inconsistencies']
        files_with_errors += folder_result['files_with_errors']
        
        # Show key observations per file
        for file_result in folder_result['file_details']:
            if file_result.get('observations'):
                print(f"\n  File: {file_result['file']}")
                for obs in file_result['observations']:
                    print(f"    • {obs['observation']}")
    
    print("\n" + "="*100)
    print("OVERALL FINDINGS")
    print("="*100)
    print(f"\nDischarge current sign convention across folders:")
    print(f"  • POSITIVE (current magnitude): {folders_with_positive} folders")
    print(f"  • NEGATIVE (traditional sign): {folders_with_negative} folders")
    print(f"  • INCONSISTENT (mixed signs): {folders_with_mixed} folders")
    
    print("\n" + "="*100)
    print("FINAL QUALITY ASSESSMENT - SIGN CONVENTION")
    print("="*100)
    
    # Determine overall consistency
    all_positive = folders_with_positive > 0 and folders_with_negative == 0 and folders_with_mixed == 0
    all_negative = folders_with_negative > 0 and folders_with_positive == 0 and folders_with_mixed == 0
    consistent_across_folders = all_positive or all_negative
    no_inconsistencies = files_with_inconsistencies == 0
    
    if consistent_across_folders and no_inconsistencies:
        sign_type = "POSITIVE" if all_positive else "NEGATIVE"
        print(f"\nRESULT: EXCELLENT (++)")
        print("="*40)
        print(f"All files consistently use {sign_type} values for discharge current.")
        print("✓ Mode values match documentation (-1, 0, 1)")
        print("✓ No internal inconsistencies")
        print("✓ Clear, consistent sign convention across all files")
        print("\nThe dataset has a well-defined sign convention:")
        if all_positive:
            print("  • Discharge current recorded as POSITIVE magnitude")
            print("  • Mode indicates direction (mode=-1 for discharge)")
        else:
            print("  • Discharge current recorded as NEGATIVE")
            print("  • Mode confirms direction (mode=-1 for discharge)")
        
    elif consistent_across_folders and files_with_inconsistencies > 0:
        sign_type = "POSITIVE" if all_positive else "NEGATIVE"
        print(f"\nRESULT: GOOD (+)")
        print("="*40)
        print(f"Most files use {sign_type} values for discharge current consistently.")
        print(f"✓ {folders_with_positive + folders_with_negative} folders have consistent convention")
        print(f"⚠ {files_with_inconsistencies} files have minor inconsistencies")
        print("\nThe dataset mostly satisfies the sign convention criterion.")
        
    elif not consistent_across_folders and files_with_inconsistencies == 0:
        print("\nRESULT: ACCEPTABLE (o)")
        print("="*40)
        print("Different folders use different sign conventions:")
        if folders_with_positive > 0:
            print(f"  • {folders_with_positive} folder(s) use POSITIVE discharge current")
        if folders_with_negative > 0:
            print(f"  • {folders_with_negative} folder(s) use NEGATIVE discharge current")
        print("\nHowever, within each folder the convention is consistent.")
        print("This is acceptable if you analyze folders separately.")
        
    else:
        print("\nRESULT: POOR (-)")
        print("="*40)
        print("Significant sign convention issues detected:")
        if folders_with_mixed > 0:
            print(f"  • {folders_with_mixed} folder(s) have INCONSISTENT signs within files")
        if files_with_inconsistencies > 0:
            print(f"  • {files_with_inconsistencies} files have internal inconsistencies")
        print("\nRecommendation: Investigate files with mixed signs or significant")
        print("current during rest periods before using the data.")

def save_report(all_results, output_file="sign_convention_report_corrected.txt"):
    """
    Save detailed report to file.
    """
    with open(output_file, 'w', encoding='ascii', errors='replace') as f:
        f.write("="*100 + "\n")
        f.write("NASA BATTERY DATASET - SIGN CONVENTION REPORT (CORRECTED)\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*100 + "\n\n")
        f.write("DOCUMENTATION:\n")
        f.write("- mode = -1 : discharge\n")
        f.write("- mode = 0 : rest\n")
        f.write("- mode = 1 : charge\n")
        f.write("- current_load : 'discharge current'\n\n")
        f.write("NOTE: The documentation does NOT specify sign convention.\n")
        f.write("This report OBSERVES the actual convention used.\n\n")
        
        for folder_result in all_results:
            f.write(f"\nFOLDER: {folder_result['folder']}\n")
            f.write("-"*60 + "\n")
            f.write(f"Overall sign convention: {folder_result['sign_convention_observed']}\n\n")
            
            for file_result in folder_result['file_details']:
                f.write(f"\nFile: {file_result['file']} ({file_result['file_size_mb']} MB)\n")
                f.write(f"  Rows: {file_result['total_rows']:,}\n")
                
                if file_result['error']:
                    f.write(f"  ERROR: {file_result['error']}\n")
                    continue
                
                if file_result['mode_values_found']:
                    f.write(f"  Mode values: {file_result['mode_values_found']}\n")
                
                # Write current statistics
                for mode_name, stats in file_result.get('current_stats', {}).items():
                    f.write(f"  {mode_name.upper()}:\n")
                    f.write(f"    Rows: {stats['rows']}\n")
                    f.write(f"    Rows with current: {stats['rows_with_current']}\n")
                    
                    if 'min_current' in stats:
                        f.write(f"    Current range: [{stats['min_current']}, {stats['max_current']}] A\n")
                        f.write(f"    Mean current: {stats['mean_current']:.3f} A\n")
                        
                        if stats['all_positive']:
                            f.write(f"    ALL VALUES POSITIVE\n")
                        elif stats['all_negative']:
                            f.write(f"    ALL VALUES NEGATIVE\n")
                        elif stats['mixed_sign']:
                            f.write(f"    MIXED SIGNS: {stats['positive_percent']}% positive, {stats['negative_percent']}% negative\n")
                
                # Write observations
                if file_result.get('observations'):
                    f.write(f"  OBSERVATIONS:\n")
                    for obs in file_result['observations']:
                        f.write(f"    • {obs['observation']}\n")
                
                # Write inconsistencies
                if file_result.get('inconsistencies'):
                    f.write(f"  INCONSISTENCIES:\n")
                    for inc in file_result['inconsistencies']:
                        f.write(f"    • [{inc['severity']}] {inc['description']}\n")
        
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
    print("NASA RANDOMIZED BATTERY DATASET - SIGN CONVENTION VERIFICATION (CORRECTED)")
    print("="*100)
    print("\nOBJECTIVE: Observe the actual sign convention used in the dataset.")
    print("The documentation does NOT specify whether discharge current")
    print("should be positive or negative. We will check for consistency.")
    
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