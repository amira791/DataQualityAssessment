"""
Advanced Battery Data Analysis
Extract and visualize actual cycling data
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from tqdm import tqdm  # for progress bars

class AdvancedBatteryAnalyzer:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        
    def load_and_analyze_battery(self, json_path):
        """Load a single battery file and extract key metrics"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract basic info
        battery_info = {
            'file': json_path.name,
            'barcode': data.get('barcode', 'Unknown'),
            'channel': data.get('channel_id', -1),
            'protocol': data.get('protocol', 'Unknown'),
            'total_cycles': 0,
            'initial_capacity': None,
            'final_capacity': None,
            'cycle_life': None  # Cycles to 80% capacity
        }
        
        # Try to get cycle data
        if 'cycles_interpolated' in data:
            cycles_data = data['cycles_interpolated']
            
            # Look for capacity information
            if 'capacity' in cycles_data:
                capacities = cycles_data['capacity']
                if isinstance(capacities, list) and len(capacities) > 0:
                    battery_info['total_cycles'] = len(capacities)
                    battery_info['initial_capacity'] = capacities[0]
                    battery_info['final_capacity'] = capacities[-1]
                    
                    # Find cycle life (when capacity drops below 80% of initial)
                    if capacities[0] > 0:
                        threshold = 0.8 * capacities[0]
                        for i, cap in enumerate(capacities):
                            if cap <= threshold:
                                battery_info['cycle_life'] = i + 1
                                break
        
        return battery_info, data
    
    def extract_charging_protocol(self, protocol_string):
        """Parse the charging protocol to extract rates"""
        # Example: "20170630-4_4C_55per_6C.sdu" -> 4C, 6C rates
        import re
        rates = re.findall(r'(\d+\.?\d*)C', protocol_string)
        return rates
    
    def analyze_all_batteries(self):
        """Analyze all batteries in the dataset"""
        all_files = list(self.base_path.rglob("*_structure.json"))
        
        results = []
        fastcharge_data = []
        optimization_data = []
        
        print(f"Analyzing {len(all_files)} battery files...")
        
        for file_path in tqdm(all_files):
            try:
                info, raw_data = self.load_and_analyze_battery(file_path)
                
                # Categorize
                if 'FastCharge' in str(file_path):
                    info['dataset'] = 'FastCharge'
                    fastcharge_data.append((info, raw_data))
                else:
                    info['dataset'] = 'Optimization'
                    optimization_data.append((info, raw_data))
                
                # Extract charging rates
                if info['protocol'] != 'Unknown':
                    info['charging_rates'] = self.extract_charging_protocol(info['protocol'])
                
                results.append(info)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        return pd.DataFrame(results), fastcharge_data, optimization_data
    
    def plot_capacity_fade_comparison(self, fastcharge_data, optimization_data, n_samples=5):
        """Plot capacity fade curves for sample batteries"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot FastCharge samples
        ax = axes[0]
        for i, (info, data) in enumerate(fastcharge_data[:n_samples]):
            if 'cycles_interpolated' in data and 'capacity' in data['cycles_interpolated']:
                capacities = data['cycles_interpolated']['capacity']
                cycles = range(1, len(capacities) + 1)
                ax.plot(cycles, capacities, label=f"CH{info['channel']}", alpha=0.7)
        
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Capacity (Ah)')
        ax.set_title('FastCharge Dataset - Capacity Fade')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot Optimization samples
        ax = axes[1]
        for i, (info, data) in enumerate(optimization_data[:n_samples]):
            if 'cycles_interpolated' in data and 'capacity' in data['cycles_interpolated']:
                capacities = data['cycles_interpolated']['capacity']
                cycles = range(1, len(capacities) + 1)
                ax.plot(cycles, capacities, label=f"CH{info['channel']}", alpha=0.7)
        
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Capacity (Ah)')
        ax.set_title('Optimization Dataset - Capacity Fade')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('capacity_fade_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def analyze_cycle_life_distribution(self, df):
        """Analyze the distribution of battery cycle life"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Cycle life distribution
        ax = axes[0, 0]
        df_valid = df[df['cycle_life'].notna()]
        ax.hist(df_valid['cycle_life'], bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Cycle Life (cycles to 80% capacity)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Battery Cycle Life')
        ax.axvline(df_valid['cycle_life'].median(), color='r', linestyle='--', 
                   label=f"Median: {df_valid['cycle_life'].median():.0f}")
        ax.legend()
        
        # 2. Cycle life by dataset
        ax = axes[0, 1]
        df.boxplot(column='cycle_life', by='dataset', ax=ax)
        ax.set_title('Cycle Life by Dataset')
        ax.set_ylabel('Cycle Life')
        
        # 3. Initial vs Final Capacity
        ax = axes[1, 0]
        ax.scatter(df['initial_capacity'], df['final_capacity'], alpha=0.5)
        ax.set_xlabel('Initial Capacity (Ah)')
        ax.set_ylabel('Final Capacity (Ah)')
        ax.set_title('Initial vs Final Capacity')
        
        # Add diagonal line for reference
        max_cap = max(df['initial_capacity'].max(), df['final_capacity'].max())
        ax.plot([0, max_cap], [0, max_cap], 'r--', alpha=0.5, label='No degradation')
        ax.legend()
        
        # 4. Capacity loss
        ax = axes[1, 1]
        df['capacity_loss'] = df['initial_capacity'] - df['final_capacity']
        df['capacity_loss_pct'] = (df['capacity_loss'] / df['initial_capacity']) * 100
        ax.hist(df['capacity_loss_pct'].dropna(), bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Capacity Loss (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Capacity Loss')
        
        plt.tight_layout()
        plt.savefig('cycle_life_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return df
    
    def generate_detailed_report(self, df):
        """Generate a detailed statistical report"""
        print("\n" + "="*80)
        print("DETAILED BATTERY DATASET REPORT")
        print("="*80)
        
        # Overall statistics
        print(f"\n Overall Statistics:")
        print(f"  • Total batteries: {len(df)}")
        print(f"  • FastCharge batteries: {len(df[df['dataset']=='FastCharge'])}")
        print(f"  • Optimization batteries: {len(df[df['dataset']=='Optimization'])}")
        
        # Cycle life statistics
        valid_cycle_life = df[df['cycle_life'].notna()]
        print(f"\n Cycle Life Statistics (cycles to 80% capacity):")
        print(f"  • Mean: {valid_cycle_life['cycle_life'].mean():.1f}")
        print(f"  • Median: {valid_cycle_life['cycle_life'].median():.1f}")
        print(f"  • Std: {valid_cycle_life['cycle_life'].std():.1f}")
        print(f"  • Min: {valid_cycle_life['cycle_life'].min():.0f}")
        print(f"  • Max: {valid_cycle_life['cycle_life'].max():.0f}")
        
        # Capacity statistics
        print(f"\n Capacity Statistics:")
        print(f"  • Avg Initial Capacity: {df['initial_capacity'].mean():.3f} Ah")
        print(f"  • Avg Final Capacity: {df['final_capacity'].mean():.3f} Ah")
        print(f"  • Avg Capacity Loss: {(df['initial_capacity'] - df['final_capacity']).mean():.3f} Ah")
        
        # Protocol analysis
        all_rates = []
        for rates in df['charging_rates'].dropna():
            all_rates.extend([float(r) for r in rates])
        
        if all_rates:
            print(f"\n⚡ Charging Rates Analysis:")
            print(f"  • Unique rates used: {sorted(set(all_rates))}")
            print(f"  • Most common rates: {pd.Series(all_rates).value_counts().head().to_dict()}")
        
        # Dataset comparison
        print(f"\n Dataset Comparison (Cycle Life):")
        for dataset in ['FastCharge', 'Optimization']:
            subset = valid_cycle_life[valid_cycle_life['dataset'] == dataset]
            if len(subset) > 0:
                print(f"\n  {dataset}:")
                print(f"    • Count: {len(subset)}")
                print(f"    • Mean cycle life: {subset['cycle_life'].mean():.1f}")
                print(f"    • Median cycle life: {subset['cycle_life'].median():.1f}")
        
        return df

def main():
    # Set your path
    dataset_path = r"C:\Users\admin\Desktop\DR2\11 All Datasets\04 MIT–Stanford–TRI Fast-Charging Dataset\Main Website"
    
    # Initialize analyzer
    analyzer = AdvancedBatteryAnalyzer(dataset_path)
    
    # Analyze all batteries
    df, fastcharge_data, optimization_data = analyzer.analyze_all_batteries()
    
    # Save summary to CSV
    df.to_csv('battery_summary.csv', index=False)
    print(f"\n Saved summary to battery_summary.csv")
    
    # Plot capacity fade comparison
    analyzer.plot_capacity_fade_comparison(fastcharge_data, optimization_data)
    
    # Analyze cycle life distribution
    df = analyzer.analyze_cycle_life_distribution(df)
    
    # Generate detailed report
    analyzer.generate_detailed_report(df)
    
    # Additional analysis: Find best and worst performing batteries
    print("\n" + "="*80)
    print("TOP PERFORMING BATTERIES")
    print("="*80)
    
    top_batteries = df.nlargest(5, 'cycle_life')[['barcode', 'dataset', 'cycle_life', 'initial_capacity']]
    print(top_batteries.to_string())
    
    print("\n" + "="*80)
    print("WORST PERFORMING BATTERIES")
    print("="*80)
    
    worst_batteries = df.nsmallest(5, 'cycle_life')[['barcode', 'dataset', 'cycle_life', 'initial_capacity']]
    print(worst_batteries.to_string())

if __name__ == "__main__":
    # Install tqdm if needed: pip install tqdm
    main()