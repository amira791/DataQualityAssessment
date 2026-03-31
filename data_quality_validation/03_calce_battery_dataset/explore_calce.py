"""
CALCE Battery Dataset Explorer
================================
Purpose: Understand the structure, format, and content of the CALCE dataset
before performing quality assessment.

Dataset location: C:/Users/admin/Desktop/DR2/11 All Datasets/03 CALCE Battery Dataset/Kaggle Dataset/calce_dataset
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up paths
from pathlib import Path

DATASET_PATH = Path(r"C:\Users\admin\Desktop\DR2\11 All Datasets\03 CALCE Battery Dataset\Kaggle Dataset\calce_dataset")
OUTPUT_DIR = Path(r"C:\Users\admin\Desktop\CALCE_exploration_output")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("CALCE BATTERY DATASET EXPLORER")
print("=" * 80)
print(f"Dataset path: {DATASET_PATH}")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 80)

# ============================================================================
# 1. UNDERSTAND DIRECTORY STRUCTURE
# ============================================================================
print("\n" + "=" * 80)
print("1. DIRECTORY STRUCTURE ANALYSIS")
print("=" * 80)

def analyze_directory_structure(path, max_depth=3):
    """Analyze and print directory structure"""
    structure = []
    
    def walk_dir(p, depth=0, max_depth=3):
        if depth > max_depth:
            return
        try:
            items = sorted([item for item in p.iterdir() if not item.name.startswith('.')])
            for item in items:
                indent = "  " * depth
                if item.is_dir():
                    structure.append(f"{indent}📁 {item.name}/")
                    walk_dir(item, depth + 1, max_depth)
                else:
                    # Get file extension and size
                    size = item.stat().st_size / 1024  # KB
                    structure.append(f"{indent}📄 {item.name} ({size:.1f} KB)")
        except PermissionError:
            pass
    
    walk_dir(path, 0, max_depth)
    return structure

print("\nDirectory structure (first 3 levels):")
structure = analyze_directory_structure(DATASET_PATH, max_depth=3)
for line in structure[:100]:  # Show first 100 lines
    print(line)
if len(structure) > 100:
    print(f"... and {len(structure) - 100} more items")

# ============================================================================
# 2. IDENTIFY ALL DATA FILES
# ============================================================================
print("\n" + "=" * 80)
print("2. DATA FILE INVENTORY")
print("=" * 80)

# Find all data files
all_files = []
extensions = ['*.txt', '*.xlsx', '*.csv', '*.xls']

for ext in extensions:
    all_files.extend(glob.glob(str(DATASET_PATH / '**' / ext), recursive=True))

print(f"\nTotal data files found: {len(all_files)}")

# Categorize by type
txt_files = [f for f in all_files if f.endswith('.txt')]
xlsx_files = [f for f in all_files if f.endswith('.xlsx')]
xls_files = [f for f in all_files if f.endswith('.xls')]
csv_files = [f for f in all_files if f.endswith('.csv')]

print(f"  - TXT files: {len(txt_files)}")
print(f"  - XLSX files: {len(xlsx_files)}")
print(f"  - XLS files: {len(xls_files)}")
print(f"  - CSV files: {len(csv_files)}")

# Group by cell ID
cell_ids = set()
for file_path in all_files:
    # Extract cell ID from path
    path_parts = Path(file_path).parts
    for part in path_parts:
        if part.startswith(('CS2_', 'CX2_')):
            cell_ids.add(part)
            break

print(f"\nUnique cell IDs found: {len(cell_ids)}")
print(f"Cell IDs: {sorted(cell_ids)}")

# ============================================================================
# 3. SAMPLE DATA FILES - UNDERSTAND FORMAT
# ============================================================================
print("\n" + "=" * 80)
print("3. DATA FILE FORMAT ANALYSIS")
print("=" * 80)

def analyze_file_format(file_path):
    """Determine file format and structure"""
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    
    result = {
        'path': str(file_path),
        'name': file_path.name,
        'size_kb': file_path.stat().st_size / 1024,
        'extension': ext,
        'encoding': None,
        'has_header': None,
        'n_rows': None,
        'n_cols': None,
        'columns': None,
        'sample_data': None
    }
    
    try:
        if ext == '.txt':
            # Try reading with different delimiters
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
                second_line = f.readline().strip() if f.readline() else ""
            
            # Detect delimiter
            if '\t' in first_line:
                delimiter = '\t'
            elif ',' in first_line:
                delimiter = ','
            elif ' ' in first_line and len(first_line.split()) > 1:
                delimiter = ' '
            else:
                delimiter = None
            
            # Try to read
            if delimiter:
                df_sample = pd.read_csv(file_path, delimiter=delimiter, nrows=10, 
                                       encoding='utf-8', engine='python', on_bad_lines='skip')
                result['has_header'] = True
            else:
                # Try reading as fixed width or space-separated
                df_sample = pd.read_csv(file_path, delim_whitespace=True, nrows=10,
                                       encoding='utf-8', engine='python', on_bad_lines='skip')
                result['has_header'] = True
            
            result['n_rows'] = len(df_sample)
            result['n_cols'] = len(df_sample.columns)
            result['columns'] = list(df_sample.columns)
            result['sample_data'] = df_sample.head(3).to_string()
            result['delimiter'] = delimiter if delimiter else 'whitespace'
            
        elif ext in ['.xlsx', '.xls']:
            # Read Excel file
            df_sample = pd.read_excel(file_path, nrows=10)
            result['has_header'] = True
            result['n_rows'] = len(df_sample)
            result['n_cols'] = len(df_sample.columns)
            result['columns'] = list(df_sample.columns)
            result['sample_data'] = df_sample.head(3).to_string()
            result['sheet_names'] = pd.ExcelFile(file_path).sheet_names
            
    except Exception as e:
        result['error'] = str(e)
    
    return result

# Sample one file from each category
print("\nAnalyzing sample files...")

# Sample TXT file (CS2_8)
txt_sample = next((f for f in txt_files if 'CS2_8' in f and '1_19_10' in f), txt_files[0] if txt_files else None)
if txt_sample:
    print(f"\n--- TXT Sample: {Path(txt_sample).name} ---")
    analysis = analyze_file_format(txt_sample)
    print(f"Size: {analysis['size_kb']:.1f} KB")
    print(f"Columns: {analysis['columns']}")
    print(f"First few rows:")
    print(analysis['sample_data'])

# Sample XLSX file (CS2_33)
xlsx_sample = next((f for f in xlsx_files if 'CS2_33' in f and '10_04_10' in f), xlsx_files[0] if xlsx_files else None)
if xlsx_sample:
    print(f"\n--- XLSX Sample: {Path(xlsx_sample).name} ---")
    analysis = analyze_file_format(xlsx_sample)
    print(f"Size: {analysis['size_kb']:.1f} KB")
    print(f"Sheets: {analysis.get('sheet_names', ['Unknown'])}")
    print(f"Columns: {analysis['columns']}")
    print(f"First few rows:")
    print(analysis['sample_data'])

# ============================================================================
# 4. UNDERSTAND CELL TYPES AND TEST CONDITIONS
# ============================================================================
print("\n" + "=" * 80)
print("4. CELL TYPE AND TEST CONDITION MAPPING")
print("=" * 80)

# Based on directory structure
cell_info = {
    'CS2': {
        'type_1': ['CS2_8', 'CS2_21', 'CS2_33', 'CS2_34'],
        'type_2': ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
    },
    'CX2': {
        'type_1': ['CX2_16', 'CX2_31', 'CX2_33', 'CX2_35'],
        'type_2': ['CX2_34', 'CX2_36', 'CX2_37', 'CX2_38']
    }
}

print("\nCell grouping by type:")
for family, types in cell_info.items():
    print(f"\n{family} cells:")
    for test_type, cells in types.items():
        print(f"  {test_type}: {', '.join(cells)}")

# ============================================================================
# 5. LOAD AND UNDERSTAND A COMPLETE CELL'S DATA
# ============================================================================
print("\n" + "=" * 80)
print("5. COMPLETE CELL DATA ANALYSIS")
print("=" * 80)

def load_complete_cell_data(cell_id, dataset_path):
    """Load all data files for a specific cell"""
    cell_files = []
    cell_path = None
    
    # Find all files for this cell
    for root, dirs, files in os.walk(dataset_path):
        if cell_id in root:
            cell_path = Path(root)
            for file in files:
                if file.endswith(('.txt', '.xlsx', '.xls')):
                    cell_files.append(cell_path / file)
    
    if not cell_files:
        return None
    
    # Sort by date (from filename)
    def extract_date(filename):
        # Extract date from filename (e.g., CS2_8_1_19_10.txt -> Jan 19, 2010)
        parts = filename.stem.split('_')
        if len(parts) >= 4:
            try:
                month = int(parts[-3])
                day = int(parts[-2])
                year = 2000 + int(parts[-1])
                return datetime(year, month, day)
            except:
                pass
        return datetime.min
    
    cell_files.sort(key=lambda x: extract_date(x))
    
    # Load each file
    all_data = []
    for file_path in cell_files:
        try:
            if file_path.suffix == '.txt':
                df = pd.read_csv(file_path, delim_whitespace=True, 
                                encoding='utf-8', engine='python', on_bad_lines='skip')
            else:  # Excel
                df = pd.read_excel(file_path)
            
            # Add metadata
            df['file_name'] = file_path.name
            df['cell_id'] = cell_id
            all_data.append(df)
        except Exception as e:
            print(f"  Error loading {file_path.name}: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None

# Load one cell from each type for comparison
print("\nLoading CS2_8 (CS2 type 1)...")
cs2_8_data = load_complete_cell_data('CS2_8', DATASET_PATH)
if cs2_8_data is not None:
    print(f"  Total rows: {len(cs2_8_data):,}")
    print(f"  Columns: {list(cs2_8_data.columns)}")
    print(f"  Date range: {cs2_8_data['file_name'].iloc[0]} to {cs2_8_data['file_name'].iloc[-1]}")
    print(f"  Number of files: {cs2_8_data['file_name'].nunique()}")
    print(f"\n  Data sample:")
    print(cs2_8_data.head(10).to_string())
    print(f"\n  Data types:")
    print(cs2_8_data.dtypes)

print("\nLoading CS2_35 (CS2 type 2)...")
cs2_35_data = load_complete_cell_data('CS2_35', DATASET_PATH)
if cs2_35_data is not None:
    print(f"  Total rows: {len(cs2_35_data):,}")
    print(f"  Columns: {list(cs2_35_data.columns)}")
    print(f"  Number of files: {cs2_35_data['file_name'].nunique()}")

# ============================================================================
# 6. UNDERSTAND DATA SCHEMA
# ============================================================================
print("\n" + "=" * 80)
print("6. DATA SCHEMA ANALYSIS")
print("=" * 80)

# Analyze column names across all files
all_columns = set()
column_frequency = {}

for file_path in all_files[:50]:  # Sample first 50 files
    try:
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
                if first_line and not first_line[0].isdigit():
                    cols = first_line.split()
                    for col in cols:
                        all_columns.add(col)
                        column_frequency[col] = column_frequency.get(col, 0) + 1
    except:
        pass

print("\nCommon column names found:")
for col, freq in sorted(column_frequency.items(), key=lambda x: x[1], reverse=True)[:20]:
    print(f"  {col}: appears in {freq} files")

# ============================================================================
# 7. DATA CHARACTERISTICS SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("7. DATA CHARACTERISTICS SUMMARY")
print("=" * 80)

summary = {
    'total_files': len(all_files),
    'txt_files': len(txt_files),
    'xlsx_files': len(xlsx_files),
    'unique_cells': len(cell_ids),
    'file_size_range': None,
    'sample_rate': None,
    'voltage_range': None,
    'current_range': None,
    'temperature_range': None
}

# Get file size statistics
file_sizes = [Path(f).stat().st_size for f in all_files]
summary['file_size_range'] = (min(file_sizes) / 1024, max(file_sizes) / 1024, 
                               np.mean(file_sizes) / 1024)

# Analyze one data file for value ranges
if cs2_8_data is not None:
    # Look for voltage, current, temperature columns
    possible_voltage_cols = [col for col in cs2_8_data.columns if 'volt' in col.lower()]
    possible_current_cols = [col for col in cs2_8_data.columns if 'curr' in col.lower()]
    possible_temp_cols = [col for col in cs2_8_data.columns if 'temp' in col.lower()]
    
    if possible_voltage_cols:
        v_col = possible_voltage_cols[0]
        summary['voltage_range'] = (cs2_8_data[v_col].min(), cs2_8_data[v_col].max())
    
    if possible_current_cols:
        i_col = possible_current_cols[0]
        summary['current_range'] = (cs2_8_data[i_col].min(), cs2_8_data[i_col].max())
    
    if possible_temp_cols:
        t_col = possible_temp_cols[0]
        summary['temperature_range'] = (cs2_8_data[t_col].min(), cs2_8_data[t_col].max())

print("\nDataset Summary:")
for key, value in summary.items():
    if isinstance(value, tuple):
        print(f"  {key}: {value}")
    else:
        print(f"  {key}: {value}")

# ============================================================================
# 8. GENERATE EXPLORATION REPORT
# ============================================================================
print("\n" + "=" * 80)
print("8. GENERATING EXPLORATION REPORT")
print("=" * 80)

# Create a markdown report
report_path = OUTPUT_DIR / "CALCE_dataset_exploration_report.md"

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# CALCE Battery Dataset Exploration Report\n\n")
    f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("## 1. Dataset Overview\n\n")
    f.write(f"- **Total files**: {summary['total_files']}\n")
    f.write(f"- **TXT files**: {summary['txt_files']}\n")
    f.write(f"- **XLSX files**: {summary['xlsx_files']}\n")
    f.write(f"- **Unique cells**: {summary['unique_cells']}\n")
    f.write(f"- **File size range**: {summary['file_size_range'][0]:.1f} KB - {summary['file_size_range'][1]:.1f} KB (avg: {summary['file_size_range'][2]:.1f} KB)\n\n")
    
    f.write("## 2. Cell Types and Test Conditions\n\n")
    f.write("### CS2 Cells (CS2_xxx format)\n")
    f.write("- **Type 1**: CS2_8, CS2_21, CS2_33, CS2_34\n")
    f.write("- **Type 2**: CS2_35, CS2_36, CS2_37, CS2_38\n\n")
    f.write("### CX2 Cells (CX2_xxx format)\n")
    f.write("- **Type 1**: CX2_16, CX2_31, CX2_33, CX2_35\n")
    f.write("- **Type 2**: CX2_34, CX2_36, CX2_37, CX2_38\n\n")
    
    f.write("## 3. Data Format\n\n")
    f.write("- **TXT files**: Space-separated or tab-separated values\n")
    f.write("- **XLSX files**: Excel format with multiple sheets possible\n")
    f.write("- **File naming**: `CELLID_MM_DD_YY.ext` (e.g., CS2_8_1_19_10.txt)\n\n")
    
    if cs2_8_data is not None:
        f.write("## 4. Sample Data (CS2_8)\n\n")
        f.write(f"- **Total rows**: {len(cs2_8_data):,}\n")
        f.write(f"- **Number of files**: {cs2_8_data['file_name'].nunique()}\n")
        f.write(f"- **Columns**: {', '.join(cs2_8_data.columns)}\n\n")
        
        if summary['voltage_range']:
            f.write(f"## 5. Measurement Ranges\n\n")
            f.write(f"- **Voltage range**: {summary['voltage_range'][0]:.3f} V - {summary['voltage_range'][1]:.3f} V\n")
        if summary['current_range']:
            f.write(f"- **Current range**: {summary['current_range'][0]:.3f} A - {summary['current_range'][1]:.3f} A\n")
        if summary['temperature_range']:
            f.write(f"- **Temperature range**: {summary['temperature_range'][0]:.1f}°C - {summary['temperature_range'][1]:.1f}°C\n\n")
    
    f.write("## 6. Next Steps\n\n")
    f.write("1. Identify consistent column names across all files\n")
    f.write("2. Determine cycle numbers and SOH calculation method\n")
    f.write("3. Create unified data loading function\n")
    f.write("4. Perform quality assessment per defined criteria\n")

print(f"\nReport saved to: {report_path}")

# ============================================================================
# 9. VISUALIZATION (if data available)
# ============================================================================
print("\n" + "=" * 80)
print("9. CREATING VISUALIZATIONS")
print("=" * 80)

if cs2_8_data is not None and possible_voltage_cols and possible_current_cols:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Voltage over time (first 1000 points)
    v_col = possible_voltage_cols[0]
    axes[0, 0].plot(cs2_8_data[v_col].head(1000))
    axes[0, 0].set_title('Voltage (first 1000 samples)')
    axes[0, 0].set_ylabel('Voltage (V)')
    axes[0, 0].set_xlabel('Sample index')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Current over time
    i_col = possible_current_cols[0]
    axes[0, 1].plot(cs2_8_data[i_col].head(1000))
    axes[0, 1].set_title('Current (first 1000 samples)')
    axes[0, 1].set_ylabel('Current (A)')
    axes[0, 1].set_xlabel('Sample index')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Voltage vs Current
    axes[1, 0].scatter(cs2_8_data[v_col].head(5000), cs2_8_data[i_col].head(5000), 
                       s=1, alpha=0.5)
    axes[1, 0].set_title('Voltage vs Current')
    axes[1, 0].set_xlabel('Voltage (V)')
    axes[1, 0].set_ylabel('Current (A)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Histogram of voltage
    axes[1, 1].hist(cs2_8_data[v_col].dropna(), bins=50, alpha=0.7)
    axes[1, 1].set_title('Voltage Distribution')
    axes[1, 1].set_xlabel('Voltage (V)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'calce_sample_visualization.png', dpi=150)
    plt.close()
    print(f"Visualization saved to: {OUTPUT_DIR / 'calce_sample_visualization.png'}")

print("\n" + "=" * 80)
print("EXPLORATION COMPLETE!")
print("=" * 80)
print(f"\nOutput files saved to: {OUTPUT_DIR}")
print("\nKey findings:")
print("1. The dataset contains both TXT and XLSX files")
print("2. Files are organized by cell ID and test type")
print("3. CS2_8 appears to be a representative cell with complete data")
print("4. Need to identify column names (they vary between file formats)")
print("5. Ready for quality assessment phase")