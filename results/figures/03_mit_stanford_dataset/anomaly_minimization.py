import pandas as pd
import numpy as np
from pathlib import Path

# Set the project path
project_path = Path(r"C:\Users\admin\Desktop\DR2\11 All Datasets\02 NASA Randomized Battery Dataset\battery_alt_dataset")

def analyze_missing_values(df, battery_name):
    """
    Simple analysis of missing values in the battery dataset
    """
    results = {
        'battery': battery_name,
        'total_rows': len(df)
    }
    
    # Check missing values in each column
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    # Store missing data for each column
    for col in df.columns:
        results[f'{col}_missing'] = missing_counts[col]
        results[f'{col}_missing_pct'] = round(missing_percentages[col], 2)
    
    return results

# Analyze all batteries
all_results = []

print("="*100)
print("MISSING VALUES ANALYSIS BY BATTERY")
print("="*100)

for folder in ['regular_alt_batteries', 'recommissioned_batteries', 'second_life_batteries']:
    folder_path = project_path / folder
    
    if folder_path.exists():
        print(f"\n📁 {folder.upper()}")
        print("-" * 80)
        
        csv_files = sorted(list(folder_path.glob('*.csv')))
        
        for csv_file in csv_files:
            try:
                # Read CSV
                df = pd.read_csv(csv_file)
                df.columns = df.columns.str.strip()
                
                battery_name = f"{folder}/{csv_file.name}"
                print(f"\n  📄 {csv_file.name}:")
                
                # Analyze missing values
                results = analyze_missing_values(df, battery_name)
                all_results.append(results)
                
                # Print missing values summary for this battery
                missing_cols = []
                for col in df.columns:
                    missing_count = results[f'{col}_missing']
                    missing_pct = results[f'{col}_missing_pct']
                    
                    if missing_count > 0:
                        missing_cols.append(f"{col}: {missing_count} rows ({missing_pct}%)")
                
                if missing_cols:
                    print(f"    Total rows: {len(df)}")
                    print(f"    ⚠️  Columns with missing values:")
                    for col_info in missing_cols:
                        print(f"      - {col_info}")
                else:
                    print(f"    ✅ No missing values in any column")
                    
            except Exception as e:
                print(f"  Error reading {csv_file.name}: {str(e)}")

# Create summary dataframe
print("\n" + "="*100)
print("SUMMARY - BATTERIES WITH MISSING VALUES")
print("="*100)

summary_df = pd.DataFrame(all_results)

# Display which columns commonly have missing values
columns_of_interest = ['voltage_load', 'current_load', 'temperature_mosfet', 
                       'temperature_resistor', 'voltage_charger', 'temperature_battery']

print("\n📊 Missing Values Summary by Column:")
print("-" * 80)

for col in columns_of_interest:
    if f'{col}_missing' in summary_df.columns:
        batteries_with_missing = summary_df[summary_df[f'{col}_missing'] > 0]
        if len(batteries_with_missing) > 0:
            avg_missing = batteries_with_missing[f'{col}_missing_pct'].mean()
            print(f"\n{col}:")
            print(f"  • {len(batteries_with_missing)} batteries have missing values")
            print(f"  • Average missing: {avg_missing:.1f}%")
            print(f"  • Range: {batteries_with_missing[f'{col}_missing'].min()} to {batteries_with_missing[f'{col}_missing'].max()} rows")

# Show batteries with most missing values
print("\n" + "="*100)
print("TOP 10 BATTERIES WITH MOST MISSING VALUES")
print("="*100)

# Calculate total missing per battery
missing_totals = []
for idx, row in summary_df.iterrows():
    total_missing = 0
    for col in columns_of_interest:
        total_missing += row[f'{col}_missing']
    missing_totals.append({'battery': row['battery'], 'total_missing': total_missing})

missing_totals_df = pd.DataFrame(missing_totals)
missing_totals_df = missing_totals_df.sort_values('total_missing', ascending=False)

for idx, row in missing_totals_df.head(10).iterrows():
    print(f"\n{row['battery']}:")
    print(f"  Total missing values: {row['total_missing']} rows")
    
    # Show breakdown by column for this battery
    battery_data = summary_df[summary_df['battery'] == row['battery']].iloc[0]
    for col in columns_of_interest:
        missing = battery_data[f'{col}_missing']
        if missing > 0:
            pct = battery_data[f'{col}_missing_pct']
            print(f"    • {col}: {missing} rows ({pct}%)")

# Save detailed results
summary_df.to_csv('missing_values_detailed.csv', index=False)
print(f"\n✅ Detailed results saved to 'missing_values_detailed.csv'")

# Quick stats by folder
print("\n" + "="*100)
print("SUMMARY BY BATTERY CATEGORY")
print("="*100)

for folder in summary_df['battery'].str.split('/').str[0].unique():
    folder_batteries = summary_df[summary_df['battery'].str.contains(folder)]
    print(f"\n{folder}:")
    print(f"  Total batteries: {len(folder_batteries)}")
    
    # Check which columns have missing values in this folder
    for col in columns_of_interest:
        col_missing = folder_batteries[f'{col}_missing'].sum()
        if col_missing > 0:
            batteries_with = len(folder_batteries[folder_batteries[f'{col}_missing'] > 0])
            print(f"  • {col}: missing in {batteries_with}/{len(folder_batteries)} batteries")