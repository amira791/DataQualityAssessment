import pandas as pd
import os
from pathlib import Path

base_path = r"C:\Users\admin\Desktop\DR2\11 All Datasets\03 CALCE Battery Dataset"

def find_aging_data():
    """Search for files that might contain multiple cycles"""
    
    print("🔍 SEARCHING FOR AGING DATA (multiple cycles)")
    print("="*70)
    
    for sample in ['Sample 01', 'Sample 02']:
        print(f"\n📁 {sample}:")
        print("-"*40)
        
        sample_path = Path(base_path) / sample
        
        # Look for cycling data in all subfolders
        for folder in ['Initial Capacity Data', 'Data for 0°C', 'Data for 25°C', 'Data for 45°C']:
            folder_path = sample_path / folder
            if folder_path.exists():
                print(f"\n  📂 {folder}:")
                
                # Check each Excel file
                for file in list(folder_path.glob("*.xlsx")) + list(folder_path.glob("*.xls")):
                    if file.name.startswith('~$'):  # Skip temp files
                        continue
                        
                    try:
                        # Try to read the file and check cycles
                        if file.suffix == '.xlsx':
                            df = pd.read_excel(file, nrows=1000)  # Read just first 1000 rows for speed
                        else:
                            # For .xls files, try different header positions
                            for header_row in [0, 5, 6, 7, 8, 9]:
                                try:
                                    df = pd.read_excel(file, header=header_row, nrows=1000)
                                    if 'Cycle' in str(df.columns) or 'Cycle_Index' in str(df.columns):
                                        break
                                except:
                                    continue
                        
                        # Check if this file has cycle information
                        cycle_col = None
                        for col in df.columns:
                            if 'Cycle' in str(col):
                                cycle_col = col
                                break
                        
                        if cycle_col:
                            unique_cycles = df[cycle_col].nunique()
                            print(f"    📊 {file.name}: {unique_cycles} unique cycles")
                            
                            # If multiple cycles found, show more details
                            if unique_cycles > 5:
                                print(f"      ✅ POTENTIAL AGING DATA - {unique_cycles} cycles")
                                # Show first few and last few cycles
                                cycle_values = sorted(df[cycle_col].unique())
                                if len(cycle_values) > 10:
                                    print(f"      Cycles: {cycle_values[:5]} ... {cycle_values[-5:]}")
                                else:
                                    print(f"      Cycles: {cycle_values}")
                        else:
                            print(f"    📄 {file.name}: No cycle column found")
                            
                    except Exception as e:
                        print(f"    ❌ Error reading {file.name}: {str(e)[:50]}")

# Run the search
find_aging_data()