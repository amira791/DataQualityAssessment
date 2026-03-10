"""
Battery Dataset Explorer for MIT-Stanford-TRI Fast-Charging Dataset
This script helps explore and understand the structure of the JSON files
"""

import json
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from collections import Counter

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class BatteryDatasetExplorer:
    def __init__(self, base_path):
        """
        Initialize the explorer with the path to the dataset
        
        Parameters:
        base_path (str): Path to the main dataset directory
        """
        self.base_path = Path(base_path)
        self.dataset_info = {}
        
    def explore_directory_structure(self):
        """Explore and print the directory structure"""
        print("\n" + "="*80)
        print("DATASET DIRECTORY STRUCTURE")
        print("="*80)
        
        # Find all JSON files
        json_files = list(self.base_path.rglob("*_structure.json"))
        
        # Group by parent directory
        dir_structure = {}
        for file in json_files:
            parent = file.parent.name
            if parent not in dir_structure:
                dir_structure[parent] = []
            dir_structure[parent].append(file.name)
        
        # Print summary
        print(f"\nTotal JSON files found: {len(json_files)}")
        print("\nFiles per directory:")
        for directory, files in sorted(dir_structure.items()):
            print(f"   {directory}: {len(files)} files")
            
        return dir_structure
    
    def examine_json_structure(self, sample_file):
        """
        Examine the structure of a sample JSON file
        
        Parameters:
        sample_file (Path): Path to a sample JSON file
        """
        print("\n" + "="*80)
        print(f"EXAMINING SAMPLE FILE: {sample_file.name}")
        print("="*80)
        
        with open(sample_file, 'r') as f:
            data = json.load(f)
        
        # Print top-level keys
        print(f"\n📊 Top-level structure:")
        for key in data.keys():
            value_type = type(data[key]).__name__
            if isinstance(data[key], dict):
                size = len(data[key])
                print(f"  • {key}: {value_type} with {size} sub-keys")
            elif isinstance(data[key], list):
                print(f"  • {key}: {value_type} with {len(data[key])} items")
            else:
                print(f"  • {key}: {value_type}")
        
        return data
    
    def extract_battery_metadata(self, data):
        """Extract metadata about the battery and experiment"""
        print("\n" + "="*80)
        print("BATTERY METADATA")
        print("="*80)
        
        metadata = {}
        
        # Look for common metadata fields
        if 'barcode' in data:
            print(f"  • Barcode: {data['barcode']}")
            metadata['barcode'] = data['barcode']
        
        if 'channel_id' in data:
            print(f"  • Channel ID: {data['channel_id']}")
            metadata['channel_id'] = data['channel_id']
        
        if 'cycle_stats' in data:
            print(f"  • Total cycles: {data['cycle_stats'].get('total_cycles', 'N/A')}")
            metadata['total_cycles'] = data['cycle_stats'].get('total_cycles', None)
        
        # Try to find protocol information
        if 'protocol' in data:
            print(f"  • Protocol: {data['protocol']}")
            metadata['protocol'] = data['protocol']
        
        return metadata
    
    def analyze_cycle_data(self, data):
        """
        Analyze the cycling data structure
        
        Returns DataFrame with cycle summaries
        """
        print("\n" + "="*80)
        print("CYCLE DATA ANALYSIS")
        print("="*80)
        
        cycle_data = []
        
        # Look for cycles in different possible locations
        if 'cycles' in data:
            cycles = data['cycles']
            print(f"\n Found {len(cycles)} cycles")
            
            for i, cycle in enumerate(cycles[:5]):  # Show first 5 cycles
                print(f"\n  Cycle {i+1}:")
                
                # Check what's in each cycle
                if isinstance(cycle, dict):
                    for key in cycle.keys():
                        if isinstance(cycle[key], list):
                            print(f"    • {key}: {len(cycle[key])} data points")
                        else:
                            print(f"    • {key}: {type(cycle[key]).__name__}")
                    
                    # Try to extract basic cycle info
                    cycle_info = {
                        'cycle_number': i+1,
                        'has_voltage': 'voltage' in cycle or 'V' in cycle,
                        'has_current': 'current' in cycle or 'I' in cycle,
                        'has_temperature': 'temperature' in cycle or 'T' in cycle,
                        'has_time': 'time' in cycle or 't' in cycle,
                    }
                    
                    # If it's a list of measurements, check first item
                    for key in ['voltage', 'V', 'current', 'I']:
                        if key in cycle and len(cycle[key]) > 0:
                            if isinstance(cycle[key], list):
                                cycle_info[f'first_{key}'] = cycle[key][0]
                            else:
                                cycle_info[f'{key}_type'] = type(cycle[key]).__name__
                    
                    cycle_data.append(cycle_info)
        
        return pd.DataFrame(cycle_data)
    
    def compare_datasets(self):
        """Compare the two main datasets"""
        print("\n" + "="*80)
        print("COMPARING THE TWO MAIN DATASETS")
        print("="*80)
        
        # Find the two main directories
        fast_charge_dir = self.base_path / "Data-driven prediction of battery cycle life before capacity degradation" / "FastCharge"
        optimization_dir = self.base_path / "Closed-loop optimization of extreme fast charging for batteries using machine learning"
        
        if fast_charge_dir.exists():
            fc_files = list(fast_charge_dir.glob("*_structure.json"))
            print(f"\n FastCharge Dataset: {len(fc_files)} files")
            
            # Sample a few files to understand naming
            print("\n  Sample filenames:")
            for f in sorted(fc_files)[:5]:
                print(f"    • {f.name}")
        
        if optimization_dir.exists():
            oed_files = list(optimization_dir.rglob("*_structure.json"))
            print(f"\n Optimization Dataset: {len(oed_files)} files")
            print("\n  Batches found:")
            for batch in sorted(optimization_dir.iterdir()):
                if batch.is_dir():
                    batch_files = len(list(batch.glob("*_structure.json")))
                    print(f"    • {batch.name}: {batch_files} files")
    
    def visualize_sample_data(self, data, output_dir="plots"):
        """
        Create sample visualizations from the data
        
        Parameters:
        data (dict): Loaded JSON data
        output_dir (str): Directory to save plots
        """
        print("\n" + "="*80)
        print("CREATING SAMPLE VISUALIZATIONS")
        print("="*80)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Look for cycle data
        if 'cycles' in data and len(data['cycles']) > 0:
            cycles = data['cycles']
            
            # Extract first few cycles for plotting
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            for idx, cycle_num in enumerate([0, min(5, len(cycles)-1), 
                                            min(10, len(cycles)-1), 
                                            min(50, len(cycles)-1)]):
                if cycle_num < len(cycles):
                    cycle = cycles[cycle_num]
                    ax = axes[idx // 2, idx % 2]
                    
                    # Try to plot voltage vs time
                    if 'voltage' in cycle and 'time' in cycle:
                        ax.plot(cycle['time'], cycle['voltage'])
                        ax.set_xlabel('Time (s)')
                        ax.set_ylabel('Voltage (V)')
                        ax.set_title(f'Cycle {cycle_num + 1}')
                        ax.grid(True, alpha=0.3)
                    
                    elif 'V' in cycle and 't' in cycle:
                        ax.plot(cycle['t'], cycle['V'])
                        ax.set_xlabel('Time (s)')
                        ax.set_ylabel('Voltage (V)')
                        ax.set_title(f'Cycle {cycle_num + 1}')
                        ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/sample_cycles.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   Saved cycle plots to {output_dir}/sample_cycles.png")
        
        # Try to extract and plot capacity fade if available
        if 'capacity' in data:
            plt.figure(figsize=(10, 6))
            
            if isinstance(data['capacity'], dict) and 'cycles' in data['capacity']:
                cycles = range(1, len(data['capacity']['cycles']) + 1)
                capacities = data['capacity']['cycles']
                plt.plot(cycles, capacities, 'o-', markersize=3)
                plt.xlabel('Cycle Number')
                plt.ylabel('Capacity (Ah)')
                plt.title('Capacity Fade Curve')
                plt.grid(True, alpha=0.3)
                plt.savefig(f'{output_dir}/capacity_fade.png', dpi=150, bbox_inches='tight')
                print(f"   Saved capacity fade plot to {output_dir}/capacity_fade.png")
            
            plt.close()
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*80)
        print("DATASET SUMMARY REPORT")
        print("="*80)
        
        # Count total files
        all_files = list(self.base_path.rglob("*_structure.json"))
        
        # Group by type
        fastcharge_files = [f for f in all_files if 'FastCharge' in str(f)]
        optimization_files = [f for f in all_files if 'oed' in str(f) or 'batch9' in str(f)]
        
        print(f"\n Dataset Overview:")
        print(f"  • Total files: {len(all_files)}")
        print(f"  • FastCharge dataset: {len(fastcharge_files)} files")
        print(f"  • Optimization dataset: {len(optimization_files)} files")
        
        # Analyze file sizes
        file_sizes = [f.stat().st_size / (1024*1024) for f in all_files]  # Size in MB
        
        print(f"\n File Size Statistics:")
        print(f"  • Average file size: {np.mean(file_sizes):.2f} MB")
        print(f"  • Min file size: {np.min(file_sizes):.2f} MB")
        print(f"  • Max file size: {np.max(file_sizes):.2f} MB")
        print(f"  • Total dataset size: {np.sum(file_sizes):.2f} MB ({np.sum(file_sizes)/1024:.2f} GB)")
        
        # Channel distribution
        channels = []
        for f in all_files:
            match = re.search(r'CH(\d+)', f.name)
            if match:
                channels.append(int(match.group(1)))
        
        if channels:
            channel_counts = Counter(channels)
            print(f"\n🔌 Channel Usage:")
            print(f"  • Channels used: {len(channel_counts)} out of 48")
            print(f"  • Most used channels: {channel_counts.most_common(5)}")
        
        return {
            'total_files': len(all_files),
            'fastcharge_files': len(fastcharge_files),
            'optimization_files': len(optimization_files),
            'avg_file_size_mb': np.mean(file_sizes),
            'total_size_gb': np.sum(file_sizes)/1024
        }

def main():
    """Main function to run the exploration"""
    
    # Set the path to your dataset
    # Update this path to match your actual dataset location
    dataset_path = r"C:\Users\admin\Desktop\DR2\11 All Datasets\04 MIT–Stanford–TRI Fast-Charging Dataset\Main Website"
    
    # Check if path exists
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Path not found: {dataset_path}")
        print("Please update the dataset_path variable with the correct path.")
        return
    
    # Initialize explorer
    explorer = BatteryDatasetExplorer(dataset_path)
    
    # Step 1: Explore directory structure
    dir_structure = explorer.explore_directory_structure()
    
    # Step 2: Find a sample file to examine
    sample_files = list(Path(dataset_path).rglob("*_structure.json"))
    if sample_files:
        # Try to get one from each dataset for comparison
        fastcharge_sample = None
        optimization_sample = None
        
        for f in sample_files:
            if 'FastCharge' in str(f) and not fastcharge_sample:
                fastcharge_sample = f
            elif ('oed' in str(f) or 'batch9' in str(f)) and not optimization_sample:
                optimization_sample = f
            
            if fastcharge_sample and optimization_sample:
                break
        
        # Examine FastCharge sample
        if fastcharge_sample:
            print("\n" + "🔬"*40)
            print("ANALYZING FASTCHARGE DATASET SAMPLE")
            print("🔬"*40)
            data = explorer.examine_json_structure(fastcharge_sample)
            explorer.extract_battery_metadata(data)
            explorer.analyze_cycle_data(data)
            explorer.visualize_sample_data(data, "fastcharge_plots")
        
        # Examine Optimization sample
        if optimization_sample:
            print("\n" + "🤖"*40)
            print("ANALYZING OPTIMIZATION DATASET SAMPLE")
            print("🤖"*40)
            data = explorer.examine_json_structure(optimization_sample)
            explorer.extract_battery_metadata(data)
            explorer.analyze_cycle_data(data)
            explorer.visualize_sample_data(data, "optimization_plots")
        
        # Compare datasets
        explorer.compare_datasets()
        
        # Generate summary report
        summary = explorer.generate_summary_report()
        
        print("\n" + "-"*40)
        print("EXPLORATION COMPLETE!")
        print("-"*40)
        print("\nNext steps you might want to take:")
        print("  1. Load multiple JSON files to create a comprehensive dataset")
        print("  2. Extract features for machine learning (cycle life, capacity fade, etc.)")
        print("  3. Compare charging protocols between different batches")
        print("  4. Analyze degradation patterns across different batteries")
        print("  5. Build predictive models for battery lifetime")
        
    else:
        print(" No JSON files found in the specified directory.")

if __name__ == "__main__":
    # Import re here to avoid issues
    import re
    main()