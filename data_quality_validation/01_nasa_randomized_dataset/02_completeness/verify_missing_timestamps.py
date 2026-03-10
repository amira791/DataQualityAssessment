"""
verify_missing_timestamps.py

This script examines the time series for gaps where data logging may have been interrupted.
It identifies missing timestamps and quantifies data logging continuity.

What we check:
1. Expected vs actual data points based on time range
2. Gaps in the time sequence
3. Irregular sampling intervals
4. Data logging interruptions
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Expected logging interval (from exploration)
EXPECTED_INTERVAL = 1.0  # seconds
TOLERANCE_FACTOR = 1.5  # Allow up to 1.5x expected interval as normal variation

def analyze_timestamp_gaps(file_path):
    """
    Analyze timestamp continuity and detect gaps in data logging.
    """
    results = {
        'file': os.path.basename(file_path),
        'file_size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2),
        'total_rows': 0,
        'time_column': None,
        'time_range': {},
        'expected_points': 0,
        'actual_points': 0,
        'missing_points_estimate': 0,
        'gap_analysis': {},
        'logging_quality': {},
        'error': None
    }
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, low_memory=False)
        results['total_rows'] = len(df)
        
        if len(df) == 0:
            results['error'] = "File is empty"
            return results
        
        # Find time column
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
        if df[time_col].dtype == 'object':
            df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
        
        # Drop any rows with NaN time
        df = df.dropna(subset=[time_col])
        if len(df) == 0:
            results['error'] = "No valid time values"
            return results
        
        # Sort by time
        df = df.sort_values(by=time_col).reset_index(drop=True)
        results['actual_points'] = len(df)
        
        # Get time range
        time_min = df[time_col].min()
        time_max = df[time_col].max()
        time_span = time_max - time_min
        
        results['time_range'] = {
            'min_time': round(float(time_min), 3),
            'max_time': round(float(time_max), 3),
            'total_span_seconds': round(float(time_span), 3),
            'total_span_hours': round(float(time_span) / 3600, 2),
            'total_span_days': round(float(time_span) / 86400, 2)
        }
        
        # Calculate expected number of points if logging was perfect
        results['expected_points'] = int(time_span / EXPECTED_INTERVAL) + 1
        results['missing_points_estimate'] = results['expected_points'] - results['actual_points']
        results['data_completeness_percent'] = round(
            (results['actual_points'] / results['expected_points']) * 100, 2
        )
        
        # Calculate time differences
        time_diffs = df[time_col].diff().dropna()
        
        # Basic statistics of time differences
        results['interval_stats'] = {
            'mean': round(float(time_diffs.mean()), 6),
            'median': round(float(time_diffs.median()), 6),
            'std': round(float(time_diffs.std()), 6),
            'min': round(float(time_diffs.min()), 6),
            'max': round(float(time_diffs.max()), 6),
            'q1': round(float(time_diffs.quantile(0.25)), 6),
            'q3': round(float(time_diffs.quantile(0.75)), 6)
        }
        
        # Identify gaps (interruptions in logging)
        expected_max_interval = EXPECTED_INTERVAL * TOLERANCE_FACTOR
        gaps = time_diffs[time_diffs > expected_max_interval]
        
        # Analyze gaps
        if len(gaps) > 0:
            gap_durations = gaps.values
            total_gap_time = gap_durations.sum() - (len(gaps) * EXPECTED_INTERVAL)
            
            results['gap_analysis'] = {
                'total_gaps': int(len(gaps)),
                'total_gap_time_seconds': round(float(total_gap_time), 3),
                'total_gap_time_hours': round(float(total_gap_time) / 3600, 2),
                'max_gap_seconds': round(float(gap_durations.max()), 3),
                'max_gap_hours': round(float(gap_durations.max()) / 3600, 2),
                'min_gap_seconds': round(float(gap_durations.min()), 3),
                'mean_gap_seconds': round(float(gap_durations.mean()), 3),
                'median_gap_seconds': round(float(np.median(gap_durations)), 3),
                
                # Distribution of gaps
                'gaps_1_10s': int(np.sum((gap_durations > 1) & (gap_durations <= 10))),
                'gaps_10_60s': int(np.sum((gap_durations > 10) & (gap_durations <= 60))),
                'gaps_1_5m': int(np.sum((gap_durations > 60) & (gap_durations <= 300))),
                'gaps_5_30m': int(np.sum((gap_durations > 300) & (gap_durations <= 1800))),
                'gaps_30m_plus': int(np.sum(gap_durations > 1800))
            }
            
            # Calculate percentage of time missing due to gaps
            missing_percent = (total_gap_time / time_span) * 100
            results['gap_analysis']['missing_time_percent'] = round(float(missing_percent), 4)
        else:
            results['gap_analysis'] = {
                'total_gaps': 0,
                'note': 'No significant gaps detected'
            }
        
        # Assess logging quality
        quality_indicators = {
            'interval_consistency': time_diffs.std() / time_diffs.mean() if time_diffs.mean() > 0 else 0,
            'points_completeness': results['data_completeness_percent'],
            'has_gaps': len(gaps) > 0,
            'max_gap_relative': (gap_durations.max() / time_span * 100) if len(gaps) > 0 else 0
        }
        
        results['logging_quality'] = quality_indicators
        
        # Check for time reversals (non-monotonic)
        if not (df[time_col].is_monotonic_increasing):
            reversals = (df[time_col].diff() < 0).sum()
            results['time_reversals'] = int(reversals)
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

def scan_folder(folder_path, folder_name):
    """
    Scan all CSV files in a folder for timestamp gaps.
    """
    print(f"\n{'='*80}")
    print(f"SCANNING FOLDER: {folder_name}")
    print(f"{'='*80}")
    
    folder_results = {
        'folder': folder_name,
        'files_checked': 0,
        'files_with_errors': 0,
        'total_gaps_across_files': 0,
        'total_missing_points': 0,
        'file_details': [],
        'summary_stats': {
            'avg_completeness': 0,
            'total_logging_days': 0,
            'worst_gap_file': None,
            'worst_gap_seconds': 0
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
    
    total_completeness = 0
    total_logging_days = 0
    worst_gap = 0
    worst_gap_file = None
    
    for csv_file in sorted(csv_files):
        print(f"  Checking: {csv_file.name}")
        results = analyze_timestamp_gaps(csv_file)
        folder_results['file_details'].append(results)
        
        if results['error']:
            folder_results['files_with_errors'] += 1
            print(f"    ERROR: {results['error']}")
        else:
            # Accumulate statistics
            completeness = results.get('data_completeness_percent', 0)
            total_completeness += completeness
            
            days = results.get('time_range', {}).get('total_span_days', 0)
            total_logging_days += days
            
            gaps = results.get('gap_analysis', {})
            if gaps.get('total_gaps', 0) > 0:
                folder_results['total_gaps_across_files'] += gaps['total_gaps']
                folder_results['total_missing_points'] += results.get('missing_points_estimate', 0)
                
                max_gap = gaps.get('max_gap_seconds', 0)
                if max_gap > worst_gap:
                    worst_gap = max_gap
                    worst_gap_file = results['file']
            
            # Print summary for this file
            print(f"    Time span: {results['time_range']['total_span_days']:.1f} days")
            print(f"    Completeness: {results['data_completeness_percent']:.2f}%")
            if results['gap_analysis']['total_gaps'] > 0:
                print(f"    Gaps: {results['gap_analysis']['total_gaps']} "
                      f"(missing {results['gap_analysis']['missing_time_percent']:.4f}% of time)")
            else:
                print(f"    Gaps: None detected")
    
    # Calculate averages
    valid_files = folder_results['files_checked'] - folder_results['files_with_errors']
    if valid_files > 0:
        folder_results['summary_stats']['avg_completeness'] = round(total_completeness / valid_files, 2)
        folder_results['summary_stats']['total_logging_days'] = round(total_logging_days, 1)
        folder_results['summary_stats']['worst_gap_file'] = worst_gap_file
        folder_results['summary_stats']['worst_gap_seconds'] = round(worst_gap, 1)
        folder_results['summary_stats']['worst_gap_hours'] = round(worst_gap / 3600, 2)
    
    return folder_results

def print_summary(all_results):
    """
    Print comprehensive summary of timestamp gap analysis.
    """
    print("\n" + "="*100)
    print("MISSING TIMESTAMP DETECTION - SUMMARY")
    print("="*100)
    print(f"\nExpected logging interval: {EXPECTED_INTERVAL} second")
    print(f"Gap definition: > {EXPECTED_INTERVAL * TOLERANCE_FACTOR} seconds")
    print("="*100)
    
    total_files = 0
    files_with_errors = 0
    files_with_gaps = 0
    total_gaps = 0
    total_missing_points = 0
    total_logging_days = 0
    completeness_scores = []
    
    for folder_result in all_results:
        print(f"\nFolder: {folder_result['folder']}")
        print("-" * 60)
        print(f"  Files checked: {folder_result['files_checked']}")
        print(f"  Files with errors: {folder_result['files_with_errors']}")
        print(f"  Files with gaps: {folder_result['files_with_gaps_detected'] if 'files_with_gaps_detected' in folder_result else 'N/A'}")
        print(f"  Total gaps across files: {folder_result['total_gaps_across_files']}")
        print(f"  Total missing points estimate: {folder_result['total_missing_points']:,}")
        print(f"  Average completeness: {folder_result['summary_stats']['avg_completeness']}%")
        print(f"  Total logging days: {folder_result['summary_stats']['total_logging_days']}")
        
        if folder_result['summary_stats']['worst_gap_file']:
            print(f"  Worst gap: {folder_result['summary_stats']['worst_gap_hours']} hours "
                  f"in {folder_result['summary_stats']['worst_gap_file']}")
        
        total_files += folder_result['files_checked']
        files_with_errors += folder_result['files_with_errors']
        total_gaps += folder_result['total_gaps_across_files']
        total_missing_points += folder_result['total_missing_points']
        total_logging_days += folder_result['summary_stats']['total_logging_days']
        completeness_scores.append(folder_result['summary_stats']['avg_completeness'])
        
        # Show detailed file info
        print(f"\n  Detailed file analysis:")
        for file_result in folder_result['file_details']:
            if file_result['error']:
                print(f"    • {file_result['file']}: ERROR - {file_result['error']}")
            else:
                gap_count = file_result['gap_analysis']['total_gaps']
                completeness = file_result['data_completeness_percent']
                days = file_result['time_range']['total_span_days']
                
                if gap_count > 0:
                    missing_pct = file_result['gap_analysis']['missing_time_percent']
                    print(f"    • {file_result['file']}: {days:.1f} days, "
                          f"{completeness:.2f}% complete, "
                          f"{gap_count} gaps (missing {missing_pct:.4f}% time)")
                    files_with_gaps += 1
                else:
                    print(f"    • {file_result['file']}: {days:.1f} days, "
                          f"{completeness:.2f}% complete, NO GAPS")
    
    print("\n" + "="*100)
    print("OVERALL STATISTICS")
    print("="*100)
    print(f"Total files analyzed: {total_files}")
    print(f"Files with errors: {files_with_errors}")
    print(f"Files with gaps detected: {files_with_gaps}")
    print(f"Total gaps across all files: {total_gaps}")
    print(f"Total missing points estimate: {total_missing_points:,}")
    print(f"Total logging days across all files: {total_logging_days:.1f} days")
    print(f"Average completeness: {np.mean(completeness_scores):.2f}%")
    
    print("\n" + "="*100)
    print("FINAL QUALITY ASSESSMENT - MISSING TIMESTAMPS")
    print("="*100)
    
    # Calculate overall quality indicators
    avg_completeness = np.mean(completeness_scores) if completeness_scores else 0
    files_with_issues_percent = (files_with_gaps / (total_files - files_with_errors)) * 100 if total_files > files_with_errors else 0
    
    if avg_completeness > 99.9 and files_with_gaps == 0:
        print("\nRESULT: EXCELLENT (++)")
        print("="*40)
        print("Exceptional data continuity:")
        print(f"  ✓ {avg_completeness:.2f}% data completeness")
        print("  ✓ No gaps detected in any file")
        print("  ✓ Perfect logging continuity")
        print("\nThe dataset fully satisfies the missing timestamp criterion.")
        
    elif avg_completeness > 99.5 and files_with_issues_percent < 20:
        print("\nRESULT: GOOD (+)")
        print("="*40)
        print("Very good data continuity with minor gaps:")
        print(f"  ✓ {avg_completeness:.2f}% average completeness")
        print(f"  ⚠ {files_with_gaps} files ({files_with_issues_percent:.1f}%) have small gaps")
        print("\nMinor gaps do not significantly impact data quality.")
        
    elif avg_completeness > 98 and files_with_issues_percent < 50:
        print("\nRESULT: ACCEPTABLE (o)")
        print("="*40)
        print("Acceptable data continuity with some gaps:")
        print(f"  • {avg_completeness:.2f}% average completeness")
        print(f"  • {files_with_gaps} files ({files_with_issues_percent:.1f}%) have gaps")
        print("\nThe dataset is usable but be aware of gaps in some files.")
        
    else:
        print("\nRESULT: POOR (-)")
        print("="*40)
        print("Significant data logging interruptions detected:")
        print(f"  • {avg_completeness:.2f}% average completeness")
        print(f"  • {files_with_gaps} files have gaps")
        print(f"  • Total missing points: {total_missing_points:,}")
        print("\nRecommendation: Review files with large gaps before analysis.")

def save_report(all_results, output_file="timestamp_gaps_report.txt"):
    """
    Save detailed report to file.
    """
    with open(output_file, 'w', encoding='ascii', errors='replace') as f:
        f.write("="*100 + "\n")
        f.write("NASA BATTERY DATASET - MISSING TIMESTAMP DETECTION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*100 + "\n\n")
        f.write(f"Expected logging interval: {EXPECTED_INTERVAL} second\n")
        f.write(f"Gap definition: > {EXPECTED_INTERVAL * TOLERANCE_FACTOR} seconds\n\n")
        
        for folder_result in all_results:
            f.write(f"\nFOLDER: {folder_result['folder']}\n")
            f.write("-"*60 + "\n")
            
            for file_result in folder_result['file_details']:
                f.write(f"\nFile: {file_result['file']} ({file_result['file_size_mb']} MB)\n")
                f.write(f"  Rows: {file_result['total_rows']:,}\n")
                
                if file_result['error']:
                    f.write(f"  ERROR: {file_result['error']}\n")
                    continue
                
                # Time range
                tr = file_result['time_range']
                f.write(f"  Time range: {tr['min_time']:.0f} to {tr['max_time']:.0f} seconds\n")
                f.write(f"  Duration: {tr['total_span_days']:.2f} days ({tr['total_span_hours']:.1f} hours)\n")
                
                # Completeness
                f.write(f"  Data completeness: {file_result['data_completeness_percent']:.4f}%\n")
                f.write(f"  Expected points: {file_result['expected_points']:,}\n")
                f.write(f"  Actual points: {file_result['actual_points']:,}\n")
                f.write(f"  Missing points estimate: {file_result['missing_points_estimate']:,}\n")
                
                # Interval statistics
                iv = file_result['interval_stats']
                f.write(f"  Interval stats: mean={iv['mean']:.3f}s, median={iv['median']:.3f}s, std={iv['std']:.3f}s\n")
                f.write(f"  Interval range: [{iv['min']:.3f}, {iv['max']:.3f}] seconds\n")
                
                # Gap analysis
                gaps = file_result['gap_analysis']
                if gaps['total_gaps'] > 0:
                    f.write(f"  GAP ANALYSIS:\n")
                    f.write(f"    Total gaps: {gaps['total_gaps']}\n")
                    f.write(f"    Total gap time: {gaps['total_gap_time_hours']:.3f} hours\n")
                    f.write(f"    Missing time percent: {gaps['missing_time_percent']:.6f}%\n")
                    f.write(f"    Max gap: {gaps['max_gap_hours']:.3f} hours ({gaps['max_gap_seconds']:.0f} seconds)\n")
                    f.write(f"    Mean gap: {gaps['mean_gap_seconds']:.1f} seconds\n")
                    f.write(f"    Gap distribution:\n")
                    f.write(f"      1-10s: {gaps['gaps_1_10s']}\n")
                    f.write(f"      10-60s: {gaps['gaps_10_60s']}\n")
                    f.write(f"      1-5m: {gaps['gaps_1_5m']}\n")
                    f.write(f"      5-30m: {gaps['gaps_5_30m']}\n")
                    f.write(f"      30m+: {gaps['gaps_30m_plus']}\n")
                else:
                    f.write(f"  No significant gaps detected\n")
        
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
    print("NASA RANDOMIZED BATTERY DATASET - MISSING TIMESTAMP DETECTION")
    print("="*100)
    print("\nAnalyzing time series for gaps and logging interruptions:")
    print(f"  • Expected logging interval: {EXPECTED_INTERVAL} second")
    print(f"  • Gap threshold: > {EXPECTED_INTERVAL * TOLERANCE_FACTOR} seconds")
    print("  • Will estimate missing data points")
    print("  • Will characterize gap sizes and distribution")
    
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