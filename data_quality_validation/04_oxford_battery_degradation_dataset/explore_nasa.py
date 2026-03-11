"""
explore_dataset.py

This script explores the NASA Randomized Battery Dataset to understand:
- Actual column names and their formats
- Data types of each column
- Basic statistics (min, max, mean) for numerical columns
- Presence of null values
- Sample rows to understand the data structure
- Differences between files in different folders

This exploration will inform the verification scripts that follow.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tabulate import tabulate

# Define the base path to the dataset
base_path = r"C:\Users\admin\Desktop\DR2\11 All Datasets\02 NASA Randomized Battery Dataset\battery_alt_dataset"

# Define the three folders to explore
folders = [
    ('regular_alt_batteries', os.path.join(base_path, 'regular_alt_batteries')),
    ('recommissioned_batteries', os.path.join(base_path, 'recommissioned_batteries')),
    ('second_life_batteries', os.path.join(base_path, 'second_life_batteries'))
]

def explore_file(file_path, sample_size=5):
    """
    Explore a single CSV file and return key information.
    """
    file_info = {
        'file_name': os.path.basename(file_path),
        'file_size_kb': round(os.path.getsize(file_path) / 1024, 2),
        'exists': True,
        'error': None,
        'row_count': 0,
        'column_count': 0,
        'columns': [],
        'column_names_raw': [],  # Original column names as they appear in the file
        'column_names_cleaned': [],  # Column names with spaces replaced
        'dtypes': {},
        'null_counts': {},
        'null_percentages': {},
        'numeric_summary': {},
        'sample_rows': None,
        'mode_values': None,  # Check the 'mode' column values if it exists
        'mission_type_values': None  # Check mission_type values if it exists
    }
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        file_info['row_count'] = len(df)
        file_info['column_count'] = len(df.columns)
        file_info['columns'] = list(df.columns)
        file_info['column_names_raw'] = list(df.columns)
        
        # Clean column names (replace spaces with underscores)
        cleaned_columns = [col.replace(' ', '_') for col in df.columns]
        file_info['column_names_cleaned'] = cleaned_columns
        
        # Get data types
        file_info['dtypes'] = df.dtypes.to_dict()
        
        # Get null counts and percentages
        null_counts = df.isnull().sum()
        file_info['null_counts'] = null_counts.to_dict()
        file_info['null_percentages'] = (null_counts / len(df) * 100).to_dict()
        
        # Get sample rows
        file_info['sample_rows'] = df.head(sample_size).to_dict('records')
        
        # Check specific columns if they exist
        if 'mode' in df.columns:
            file_info['mode_values'] = sorted(df['mode'].dropna().unique().tolist())
        
        if 'mission type' in df.columns:
            file_info['mission_type_values'] = sorted(df['mission type'].dropna().unique().tolist())
        elif 'mission_type' in df.columns:
            file_info['mission_type_values'] = sorted(df['mission_type'].dropna().unique().tolist())
        
        # Get numeric summary for key columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ['voltage charger', 'voltage load', 'current load', 
                       'temperature battery', 'temperature mosfet', 'temperature resistor',
                       'voltage_charger', 'voltage_load', 'current_load',
                       'temperature_battery', 'temperature_mosfet', 'temperature_resistor']:
                file_info['numeric_summary'][col] = {
                    'min': float(df[col].min()) if not df[col].isnull().all() else None,
                    'max': float(df[col].max()) if not df[col].isnull().all() else None,
                    'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
                    'std': float(df[col].std()) if not df[col].isnull().all() else None
                }
        
    except Exception as e:
        file_info['error'] = str(e)
        file_info['exists'] = False
    
    return file_info

def scan_folder(folder_path, folder_name):
    """
    Scan all CSV files in a folder and collect exploration data.
    """
    print(f"\n{'='*80}")
    print(f"EXPLORING FOLDER: {folder_name}")
    print(f"{'='*80}")
    
    folder_info = {
        'folder_name': folder_name,
        'folder_path': folder_path,
        'exists': os.path.exists(folder_path),
        'file_count': 0,
        'files': [],
        'common_columns': set(),
        'all_columns': set(),
        'column_frequency': {},
        'summary': {}
    }
    
    if not os.path.exists(folder_path):
        print(f" Folder does not exist: {folder_path}")
        return folder_info
    
    # Find all CSV files
    csv_files = list(Path(folder_path).glob("*.csv"))
    folder_info['file_count'] = len(csv_files)
    
    print(f"Found {len(csv_files)} CSV files")
    
    if not csv_files:
        return folder_info
    
    # Explore each file
    for i, csv_file in enumerate(sorted(csv_files), 1):
        print(f"\n[{i}/{len(csv_files)}] Exploring: {csv_file.name}")
        file_info = explore_file(csv_file)
        folder_info['files'].append(file_info)
        
        # Track column names across files
        for col in file_info['column_names_raw']:
            folder_info['all_columns'].add(col)
            if col not in folder_info['column_frequency']:
                folder_info['column_frequency'][col] = 0
            folder_info['column_frequency'][col] += 1
        
        # Print basic info about this file
        if file_info['error']:
            print(f"  ⚠ Error: {file_info['error']}")
        else:
            print(f"  Rows: {file_info['row_count']:,}, Columns: {file_info['column_count']}")
            print(f"  Columns: {', '.join(file_info['column_names_raw'][:5])}...")
            
            # Show mode values if available
            if file_info['mode_values']:
                print(f"  Mode values: {file_info['mode_values']}")
            
            # Show mission type if available
            if file_info['mission_type_values']:
                print(f"  Mission type values: {file_info['mission_type_values']}")
    
    # Find common columns (present in all files)
    if folder_info['files']:
        common_columns = set(folder_info['files'][0]['column_names_raw'])
        for file_info in folder_info['files'][1:]:
            common_columns = common_columns.intersection(set(file_info['column_names_raw']))
        folder_info['common_columns'] = common_columns
    
    return folder_info

def print_folder_summary(folder_info):
    """
    Print a summary of the folder exploration.
    """
    if not folder_info['exists']:
        return
    
    print(f"\n SUMMARY FOR: {folder_info['folder_name']}")
    print(f"{'-'*60}")
    print(f"Files found: {folder_info['file_count']}")
    
    if folder_info['file_count'] == 0:
        return
    
    # Column analysis
    print(f"\nColumn Analysis:")
    print(f"  Total unique columns across all files: {len(folder_info['all_columns'])}")
    print(f"  Columns present in ALL files: {len(folder_info['common_columns'])}")
    
    if folder_info['common_columns']:
        print(f"  Common columns: {', '.join(sorted(folder_info['common_columns']))}")
    
    print(f"\nColumn frequency (present in # of files):")
    for col, freq in sorted(folder_info['column_frequency'].items(), key=lambda x: x[1], reverse=True):
        percentage = (freq / folder_info['file_count']) * 100
        print(f"  • {col}: {freq}/{folder_info['file_count']} files ({percentage:.1f}%)")
    
    # Data summary from first file
    if folder_info['files'] and not folder_info['files'][0]['error']:
        first_file = folder_info['files'][0]
        print(f"\nData Sample (first file: {first_file['file_name']}):")
        if first_file['sample_rows']:
            sample_df = pd.DataFrame(first_file['sample_rows'])
            print(tabulate(sample_df.head(3), headers='keys', tablefmt='psql', showindex=False))
        
        # Numeric ranges
        if first_file['numeric_summary']:
            print(f"\nNumeric ranges (first file):")
            for col, stats in first_file['numeric_summary'].items():
                if stats['min'] is not None:
                    print(f"  • {col}: min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}")

def print_cross_folder_comparison(all_folder_infos):
    """
    Compare column structures across different folders.
    """
    print(f"\n{'='*80}")
    print("CROSS-FOLDER COMPARISON")
    print('='*80)
    
    # Compare columns across folders
    all_columns_by_folder = {}
    for folder_info in all_folder_infos:
        if folder_info['exists'] and folder_info['file_count'] > 0:
            all_columns_by_folder[folder_info['folder_name']] = folder_info['all_columns']
    
    if not all_columns_by_folder:
        print("No folders with data to compare.")
        return
    
    # Find columns common to all folders
    common_across_folders = set.intersection(*all_columns_by_folder.values()) if all_columns_by_folder else set()
    
    print(f"\nColumns present in ALL folders:")
    if common_across_folders:
        for col in sorted(common_across_folders):
            print(f"  • {col}")
    else:
        print("  No columns are common across all folders")
    
    # Find columns unique to each folder
    print(f"\nColumns unique to each folder:")
    for folder_name, columns in all_columns_by_folder.items():
        other_columns = set.union(*[v for k, v in all_columns_by_folder.items() if k != folder_name])
        unique_cols = columns - other_columns
        if unique_cols:
            print(f"  • {folder_name}: {', '.join(sorted(unique_cols))}")
        else:
            print(f"  • {folder_name}: No unique columns")
    
    # Check for column name variations
    print(f"\nPotential column name variations:")
    variations_found = []
    
    # Look for common patterns like spaces vs underscores
    for folder_name, columns in all_columns_by_folder.items():
        for col in columns:
            if ' ' in col:
                alt_name = col.replace(' ', '_')
                if alt_name in columns:
                    variations_found.append((col, alt_name, folder_name))
    
    if variations_found:
        for orig, alt, folder in variations_found:
            print(f"  • In {folder}: '{orig}' and '{alt}' both exist")
    else:
        print("  No obvious column name variations detected")

def main():
    print("="*80)
    print("NASA RANDOMIZED BATTERY DATASET - EXPLORATION")
    print("="*80)
    print(f"Base path: {base_path}")
    
    # Check if base path exists
    if not os.path.exists(base_path):
        print(f"\n ERROR: Base path does not exist: {base_path}")
        print("Please update the base_path variable to point to your dataset location.")
        return
    
    # Explore each folder
    all_folder_infos = []
    for folder_name, folder_path in folders:
        folder_info = scan_folder(folder_path, folder_name)
        all_folder_infos.append(folder_info)
        print_folder_summary(folder_info)
    
    # Cross-folder comparison
    print_cross_folder_comparison(all_folder_infos)
    
    print(f"\n{'='*80}")
    print("EXPLORATION COMPLETE")
    print("="*80)
    print("\nKey findings to note for verification scripts:")
    print("1. Note the exact column names (with spaces or underscores)")
    print("2. Check if all expected columns are present in all files")
    print("3. Observe the actual ranges of values to inform plausibility bounds")
    print("4. Note any patterns in null values")
    print("5. Verify the mode and mission type values match documentation")

if __name__ == "__main__":
    # Install tabulate if not available
    try:
        from tabulate import tabulate
    except ImportError:
        print("Installing required package: tabulate")
        import subprocess
        subprocess.check_call(['pip', 'install', 'tabulate'])
        from tabulate import tabulate
    
    main()