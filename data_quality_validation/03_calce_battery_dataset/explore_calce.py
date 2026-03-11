"""
explore.py - Comprehensive exploration script for CALCE Battery Dataset
Dataset path: C:\Users\admin\Desktop\DR2\11 All Datasets\10 Battery Archive Datasets\Battery Archive Data\CALCE\CALCE
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class CALCEExplorer:
    def __init__(self, data_path):
        """
        Initialize the explorer with the path to CALCE dataset
        
        Parameters:
        data_path (str): Full path to the CALCE dataset folder
        """
        self.data_path = Path(data_path)
        self.timeseries_files = []
        self.cycle_data_files = []
        self.summary_df = None
        
        print(f" Initializing CALCE Dataset Explorer")
        print(f" Dataset path: {self.data_path}")
        print("-" * 60)
        
    def scan_dataset(self):
        """Scan the dataset and categorize all CSV files"""
        print("\n Scanning dataset for CSV files...")
        
        # Find all CSV files
        all_csv_files = list(self.data_path.glob("*.csv"))
        
        # Separate timeseries and cycle data files
        self.timeseries_files = [f for f in all_csv_files if 'timeseries' in f.name.lower()]
        self.cycle_data_files = [f for f in all_csv_files if 'cycle_data' in f.name.lower()]
        
        print(f"   Found {len(all_csv_files)} total CSV files")
        print(f"    Timeseries files: {len(self.timeseries_files)}")
        print(f"    Cycle data files: {len(self.cycle_data_files)}")
        
        if len(all_csv_files) == 0:
            print("    No CSV files found! Check the path.")
            return False
        return True
    
    def parse_filename(self, filename):
        """Parse the CALCE filename convention to extract metadata"""
        try:
            # Example: CALCE_CX2-16_prism_LCO_25C_0-100_0.5-0.5C_a_timeseries.csv
            name_parts = filename.stem.split('_')
            
            metadata = {
                'cell_id': name_parts[1] if len(name_parts) > 1 else 'unknown',
                'form_factor': name_parts[2] if len(name_parts) > 2 else 'unknown',
                'chemistry': name_parts[3] if len(name_parts) > 3 else 'unknown',
                'temperature': name_parts[4] if len(name_parts) > 4 else 'unknown',
                'soc_range': name_parts[5] if len(name_parts) > 5 else 'unknown',
                'c_rate': name_parts[6] if len(name_parts) > 6 else 'unknown',
                'cell_suffix': name_parts[7] if len(name_parts) > 7 else 'unknown',
                'data_type': name_parts[8] if len(name_parts) > 8 else 'unknown'
            }
            return metadata
        except Exception as e:
            print(f"     Could not parse filename {filename.name}: {e}")
            return {}
    
    def create_dataset_summary(self):
        """Create a summary DataFrame of all files in the dataset"""
        print("\n Creating dataset summary...")
        
        summary_data = []
        
        # Process timeseries files
        for file_path in self.timeseries_files:
            metadata = self.parse_filename(file_path)
            metadata.update({
                'file_name': file_path.name,
                'file_path': str(file_path),
                'file_size_kb': round(file_path.stat().st_size / 1024, 2),
                'file_type': 'timeseries'
            })
            summary_data.append(metadata)
        
        # Process cycle data files
        for file_path in self.cycle_data_files:
            metadata = self.parse_filename(file_path)
            metadata.update({
                'file_name': file_path.name,
                'file_path': str(file_path),
                'file_size_kb': round(file_path.stat().st_size / 1024, 2),
                'file_type': 'cycle_data'
            })
            summary_data.append(metadata)
        
        self.summary_df = pd.DataFrame(summary_data)
        
        if len(self.summary_df) > 0:
            print(f"\n Dataset Summary:")
            print(f"   Total files: {len(self.summary_df)}")
            print(f"   Unique cells: {self.summary_df['cell_id'].nunique()}")
            print(f"   Temperature conditions: {self.summary_df['temperature'].unique().tolist()}")
            print(f"   SOC ranges: {self.summary_df['soc_range'].unique().tolist()}")
            print(f"   C-rates: {self.summary_df['c_rate'].unique().tolist()}")
        
        return self.summary_df
    
    def explore_file_structure(self, sample_file=None):
        """Explore the structure of a sample CSV file"""
        print("\n Exploring file structure...")
        
        if sample_file is None and len(self.timeseries_files) > 0:
            sample_file = self.timeseries_files[0]
        elif sample_file is None and len(self.cycle_data_files) > 0:
            sample_file = self.cycle_data_files[0]
        
        if sample_file:
            print(f"\n   Sample file: {sample_file.name}")
            print(f"   Full path: {sample_file}")
            
            try:
                # Read the CSV file
                df = pd.read_csv(sample_file, nrows=5)
                
                print(f"\n   File Structure:")
                print(f"   Shape: {df.shape}")
                print(f"\n   Columns:")
                for i, col in enumerate(df.columns, 1):
                    print(f"     {i}. {col}")
                
                print(f"\n   Data Types:")
                print(df.dtypes.to_string())
                
                print(f"\n   First 5 rows:")
                print(df.to_string())
                
                return df.columns.tolist()
                
            except Exception as e:
                print(f"    Error reading file: {e}")
        else:
            print("    No sample files found!")
        return None
    
    def analyze_timeseries_file(self, file_path):
        """Analyze a timeseries file in detail"""
        print(f"\n Analyzing timeseries file: {Path(file_path).name}")
        
        try:
            # Load the full timeseries data
            df = pd.read_csv(file_path)
            
            print(f"\n   Basic Statistics:")
            print(f"   Total data points: {len(df):,}")
            print(f"   Time range: {df['Test_Time(s)'].min():.2f} to {df['Test_Time(s)'].max():.2f} seconds")
            print(f"   Duration: {(df['Test_Time(s)'].max() - df['Test_Time(s)'].min()) / 3600:.2f} hours")
            
            # Voltage statistics
            if 'Voltage(V)' in df.columns:
                print(f"\n   Voltage (V):")
                print(f"     Min: {df['Voltage(V)'].min():.3f}")
                print(f"     Max: {df['Voltage(V)'].max():.3f}")
                print(f"     Mean: {df['Voltage(V)'].mean():.3f}")
                print(f"     Std: {df['Voltage(V)'].std():.3f}")
            
            # Current statistics
            if 'Current(A)' in df.columns:
                print(f"\n   Current (A):")
                print(f"     Min: {df['Current(A)'].min():.3f}")
                print(f"     Max: {df['Current(A)'].max():.3f}")
                print(f"     Mean: {df['Current(A)'].mean():.3f}")
                
                # Charging/Discharging analysis
                charging = df[df['Current(A)'] > 0]
                discharging = df[df['Current(A)'] < 0]
                resting = df[df['Current(A)'] == 0]
                
                print(f"\n   Operating Modes:")
                print(f"     Charging: {len(charging)} points ({len(charging)/len(df)*100:.1f}%)")
                print(f"     Discharging: {len(discharging)} points ({len(discharging)/len(df)*100:.1f}%)")
                print(f"     Resting: {len(resting)} points ({len(resting)/len(df)*100:.1f}%)")
            
            # Step analysis
            if 'Step_Index' in df.columns:
                n_steps = df['Step_Index'].nunique()
                print(f"\n   Test Steps: {n_steps} unique steps")
                step_durations = df.groupby('Step_Index')['Test_Time(s)'].agg(['min', 'max'])
                step_durations['duration'] = step_durations['max'] - step_durations['min']
                print(f"     Avg step duration: {step_durations['duration'].mean()/60:.2f} minutes")
            
            return df
            
        except Exception as e:
            print(f"    Error analyzing file: {e}")
            return None
    
    def analyze_cycle_data_file(self, file_path):
        """Analyze a cycle data file in detail"""
        print(f"\n Analyzing cycle data file: {Path(file_path).name}")
        
        try:
            # Load the cycle data
            df = pd.read_csv(file_path)
            
            print(f"\n   Basic Statistics:")
            print(f"   Total cycles: {len(df)}")
            print(f"   Cycle range: {df['Cycle_Index'].min()} to {df['Cycle_Index'].max()}")
            
            # Discharge capacity analysis (key degradation metric)
            if 'Discharge_Capacity(Ah)' in df.columns:
                cap_col = 'Discharge_Capacity(Ah)'
                
                print(f"\n   Discharge Capacity (Ah):")
                print(f"     Initial: {df[cap_col].iloc[0]:.4f}")
                print(f"     Final: {df[cap_col].iloc[-1]:.4f}")
                print(f"     Max: {df[cap_col].max():.4f}")
                print(f"     Min: {df[cap_col].min():.4f}")
                print(f"     Mean: {df[cap_col].mean():.4f}")
                
                # Calculate degradation if enough cycles
                if len(df) > 10:
                    total_degradation = df[cap_col].iloc[0] - df[cap_col].iloc[-1]
                    degradation_percent = (total_degradation / df[cap_col].iloc[0]) * 100
                    print(f"\n   Degradation Analysis:")
                    print(f"     Total capacity loss: {total_degradation:.4f} Ah")
                    print(f"     Degradation: {degradation_percent:.2f}%")
                    
                    # Calculate degradation rate
                    cycles = len(df)
                    deg_per_cycle = total_degradation / cycles
                    deg_per_100_cycles = deg_per_cycle * 100
                    print(f"     Degradation rate: {deg_per_cycle:.6f} Ah/cycle")
                    print(f"     Degradation per 100 cycles: {deg_per_100_cycles:.4f} Ah")
            
            # Coulombic efficiency analysis
            if 'Coulombic_Efficiency(%)' in df.columns:
                print(f"\n   Coulombic Efficiency (%):")
                print(f"     Min: {df['Coulombic_Efficiency(%)'].min():.2f}")
                print(f"     Max: {df['Coulombic_Efficiency(%)'].max():.2f}")
                print(f"     Mean: {df['Coulombic_Efficiency(%)'].mean():.2f}")
                print(f"     Std: {df['Coulombic_Efficiency(%)'].std():.2f}")
            
            return df
            
        except Exception as e:
            print(f"    Error analyzing file: {e}")
            return None
    
    def plot_capacity_degradation(self, cell_id=None):
        """Plot capacity degradation for one or all cells"""
        print("\n Generating capacity degradation plots...")
        
        # Find cycle data files
        if cell_id:
            files = [f for f in self.cycle_data_files if cell_id.lower() in f.name.lower()]
        else:
            files = self.cycle_data_files
        
        if not files:
            print("    No cycle data files found!")
            return
        
        plt.figure(figsize=(14, 8))
        
        for file_path in files[:5]:  # Limit to 5 cells to avoid overcrowding
            try:
                df = pd.read_csv(file_path)
                if 'Discharge_Capacity(Ah)' in df.columns and 'Cycle_Index' in df.columns:
                    cell_name = self.parse_filename(Path(file_path)).get('cell_id', 'unknown')
                    plt.plot(df['Cycle_Index'], df['Discharge_Capacity(Ah)'], 
                            marker='.', markersize=3, linewidth=1.5, 
                            label=f'{cell_name}')
            except Exception as e:
                print(f"    Could not plot {file_path.name}: {e}")
        
        plt.xlabel('Cycle Number', fontsize=12)
        plt.ylabel('Discharge Capacity (Ah)', fontsize=12)
        plt.title('Battery Capacity Degradation Over Cycles', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        output_path = self.data_path / 'capacity_degradation_plot.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"    Plot saved to: {output_path}")
        plt.show()
    
    def plot_voltage_profile(self, sample_file=None):
        """Plot voltage profile from a sample timeseries file"""
        print("\n Generating voltage profile plot...")
        
        if sample_file is None and len(self.timeseries_files) > 0:
            sample_file = self.timeseries_files[0]
        
        if sample_file:
            try:
                df = pd.read_csv(sample_file)
                
                if 'Voltage(V)' in df.columns and 'Test_Time(s)' in df.columns:
                    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
                    
                    # Plot 1: Voltage vs Time
                    axes[0].plot(df['Test_Time(s)']/3600, df['Voltage(V)'], 
                               linewidth=1, color='blue')
                    axes[0].set_xlabel('Time (hours)', fontsize=12)
                    axes[0].set_ylabel('Voltage (V)', fontsize=12)
                    axes[0].set_title(f'Voltage Profile - {Path(sample_file).name}', 
                                     fontsize=14, fontweight='bold')
                    axes[0].grid(True, alpha=0.3)
                    
                    # Plot 2: Current vs Time (if available)
                    if 'Current(A)' in df.columns:
                        axes[1].plot(df['Test_Time(s)']/3600, df['Current(A)'], 
                                   linewidth=1, color='red')
                        axes[1].set_xlabel('Time (hours)', fontsize=12)
                        axes[1].set_ylabel('Current (A)', fontsize=12)
                        axes[1].set_title('Current Profile', fontsize=14, fontweight='bold')
                        axes[1].grid(True, alpha=0.3)
                        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                    
                    plt.tight_layout()
                    
                    # Save the plot
                    output_path = self.data_path / 'voltage_profile_plot.png'
                    plt.savefig(output_path, dpi=150, bbox_inches='tight')
                    print(f"   Plot saved to: {output_path}")
                    plt.show()
                else:
                    print("    Required columns not found in file")
                    
            except Exception as e:
                print(f"   Error creating plot: {e}")
        else:
            print("   No timeseries files found!")
    
    def generate_full_report(self):
        """Generate a comprehensive report of the entire dataset"""
        print("\n" + "="*60)
        print(" GENERATING COMPREHENSIVE DATASET REPORT")
        print("="*60)
        
        # Step 1: Scan dataset
        if not self.scan_dataset():
            return
        
        # Step 2: Create summary
        self.create_dataset_summary()
        
        # Step 3: Display dataset overview
        print("\n" + "-"*40)
        print("DATASET OVERVIEW")
        print("-"*40)
        print(f"\nTotal Files: {len(self.summary_df)}")
        print(f"\nFiles by Type:")
        print(self.summary_df['file_type'].value_counts().to_string())
        
        print(f"\nCells in Dataset:")
        print(self.summary_df['cell_id'].value_counts().to_string())
        
        print(f"\nTemperature Conditions:")
        print(self.summary_df['temperature'].value_counts().to_string())
        
        print(f"\nC-Rates:")
        print(self.summary_df['c_rate'].value_counts().to_string())
        
        # Step 4: Explore file structure
        self.explore_file_structure()
        
        # Step 5: Analyze a sample timeseries file
        if self.timeseries_files:
            print("\n" + "-"*40)
            print("SAMPLE TIMESERIES ANALYSIS")
            print("-"*40)
            self.analyze_timeseries_file(self.timeseries_files[0])
        
        # Step 6: Analyze a sample cycle data file
        if self.cycle_data_files:
            print("\n" + "-"*40)
            print("SAMPLE CYCLE DATA ANALYSIS")
            print("-"*40)
            self.analyze_cycle_data_file(self.cycle_data_files[0])
        
        # Step 7: Generate visualizations
        print("\n" + "-"*40)
        print("GENERATING VISUALIZATIONS")
        print("-"*40)
        self.plot_capacity_degradation()
        self.plot_voltage_profile()
        
        # Step 8: Save summary to CSV
        if self.summary_df is not None:
            output_summary = self.data_path / 'dataset_summary.csv'
            self.summary_df.to_csv(output_summary, index=False)
            print(f"\n✅ Dataset summary saved to: {output_summary}")
        
        print("\n" + "="*60)
        print("✅ REPORT COMPLETE")
        print("="*60)

def main():
    """Main function to run the explorer"""
    
    # Set your dataset path here
    dataset_path = r"C:\Users\admin\Desktop\DR2\11 All Datasets\10 Battery Archive Datasets\Battery Archive Data\CALCE\CALCE"
    
    # Create explorer instance
    explorer = CALCEExplorer(dataset_path)
    
    # Menu for interactive exploration
    while True:
        print("\n" + "="*50)
        print("CALCE DATASET EXPLORER MENU")
        print("="*50)
        print("1. Scan dataset and show summary")
        print("2. Explore file structure")
        print("3. Analyze a specific timeseries file")
        print("4. Analyze a specific cycle data file")
        print("5. Plot capacity degradation")
        print("6. Plot voltage profile")
        print("7. Generate full report")
        print("8. Exit")
        print("-"*50)
        
        choice = input("Enter your choice (1-8): ").strip()
        
        if choice == '1':
            explorer.scan_dataset()
            explorer.create_dataset_summary()
            
        elif choice == '2':
            explorer.explore_file_structure()
            
        elif choice == '3':
            if not explorer.timeseries_files:
                print("No timeseries files found. Scan dataset first.")
                continue
            
            print("\nAvailable timeseries files:")
            for i, file in enumerate(explorer.timeseries_files[:10], 1):
                print(f"{i}. {file.name}")
            
            if len(explorer.timeseries_files) > 10:
                print(f"... and {len(explorer.timeseries_files)-10} more")
            
            try:
                idx = int(input("\nEnter file number: ")) - 1
                if 0 <= idx < len(explorer.timeseries_files):
                    explorer.analyze_timeseries_file(explorer.timeseries_files[idx])
                else:
                    print("Invalid file number")
            except ValueError:
                print("Invalid input")
                
        elif choice == '4':
            if not explorer.cycle_data_files:
                print("No cycle data files found. Scan dataset first.")
                continue
            
            print("\nAvailable cycle data files:")
            for i, file in enumerate(explorer.cycle_data_files[:10], 1):
                print(f"{i}. {file.name}")
            
            if len(explorer.cycle_data_files) > 10:
                print(f"... and {len(explorer.cycle_data_files)-10} more")
            
            try:
                idx = int(input("\nEnter file number: ")) - 1
                if 0 <= idx < len(explorer.cycle_data_files):
                    explorer.analyze_cycle_data_file(explorer.cycle_data_files[idx])
                else:
                    print("Invalid file number")
            except ValueError:
                print("Invalid input")
                
        elif choice == '5':
            explorer.plot_capacity_degradation()
            
        elif choice == '6':
            explorer.plot_voltage_profile()
            
        elif choice == '7':
            explorer.generate_full_report()
            
        elif choice == '8':
            print("\n👋 Exiting explorer. Goodbye!")
            break
            
        else:
            print("Invalid choice. Please enter 1-8")

if __name__ == "__main__":
    main()