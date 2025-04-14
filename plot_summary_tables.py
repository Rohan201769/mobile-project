import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from itertools import product

def plot_summary_tables(input_dir, output_dir):
    """Plot visualizations from summary tables"""
    summary_dir = input_dir
    plots_dir = output_dir
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot the overall BER summary table
    try:
        overall_summary = pd.read_csv(os.path.join(summary_dir, "ber_summary_table.csv"))
        
        # Create heatmap of BER values by SNR and model
        plt.figure(figsize=(14, 10))
        pivot_table = overall_summary.pivot_table(
            values='BER', 
            index=['Model', 'Config'], 
            columns='SNR'
        )
        
        # Use log scale for better visualization of BER values
        sns.heatmap(pivot_table, annot=True, fmt='.2e', cmap='viridis_r', 
                   cbar_kws={'label': 'BER (log scale)'})
        plt.title('BER Performance Summary Across Models and Configurations', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "overall_ber_heatmap.png"))
        plt.close()
        print("Created overall BER heatmap")
        
        # Create grouped bar plots for each SNR value
        for snr in overall_summary['SNR'].unique():
            snr_data = overall_summary[overall_summary['SNR'] == snr]
            
            plt.figure(figsize=(14, 8))
            sns.barplot(x='Config', y='BER', hue='Model', data=snr_data)
            plt.yscale('log')  # Use log scale for BER
            plt.title(f'BER Comparison at SNR = {snr} dB', fontsize=16)
            plt.xlabel('MIMO Configuration', fontsize=14)
            plt.ylabel('BER (log scale)', fontsize=14)
            plt.legend(title='Model', fontsize=12)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"ber_comparison_snr_{snr}.png"))
            plt.close()
            print(f"Created BER comparison plot for SNR = {snr} dB")
    except Exception as e:
        print(f"Error plotting overall summary table: {e}")
    
    # Plot modulation-specific summary tables
    for modulation in ['BPSK', 'QPSK']:
        for channel in ['nonideal']:  # Add 'ideal' if you have that data
            try:
                filename = f"summary_{modulation}_{channel}.csv"
                filepath = os.path.join(summary_dir, filename)
                
                if not os.path.exists(filepath):
                    print(f"File not found: {filepath}")
                    continue
                    
                summary_df = pd.read_csv(filepath)
                
                # Extract SNR columns
                snr_columns = [col for col in summary_df.columns if col.startswith('SNR_')]
                
                # Create a melted dataframe for easier plotting
                melted_df = pd.melt(
                    summary_df, 
                    id_vars=['Model', 'Config'], 
                    value_vars=snr_columns,
                    var_name='SNR', 
                    value_name='BER'
                )
                
                # Clean up SNR column to show only the dB value
                melted_df['SNR'] = melted_df['SNR'].str.replace('SNR_', '').str.replace('dB', '')
                melted_df['SNR'] = melted_df['SNR'].astype(float)
                
                # Line plot showing BER vs SNR for each model and configuration
                plt.figure(figsize=(14, 10))
                for model in melted_df['Model'].unique():
                    for config in melted_df['Config'].unique():
                        data = melted_df[(melted_df['Model'] == model) & 
                                         (melted_df['Config'] == config)]
                        if not data.empty:
                            plt.semilogy(data['SNR'], data['BER'], 
                                       marker='o', linestyle='-', 
                                       label=f"{model} - {config}")
                
                plt.grid(True, which="both", ls="--")
                plt.xlabel('Eb/N0 (dB)', fontsize=14)
                plt.ylabel('BER (log scale)', fontsize=14)
                plt.title(f'BER vs SNR - {modulation}, {channel} channel', fontsize=16)
                plt.legend(fontsize=10)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"ber_vs_snr_{modulation}_{channel}.png"))
                plt.close()
                print(f"Created BER vs SNR plot for {modulation}, {channel}")
                
                # Grouped bar chart for each SNR point
                for snr in melted_df['SNR'].unique():
                    plt.figure(figsize=(12, 8))
                    snr_data = melted_df[melted_df['SNR'] == snr]
                    
                    # Sort by model and config for better visualization
                    snr_data = snr_data.sort_values(['Model', 'Config'])
                    
                    # Create group labels
                    snr_data['Group'] = snr_data['Model'] + ' - ' + snr_data['Config']
                    
                    # Plot
                    ax = sns.barplot(x='Group', y='BER', data=snr_data)
                    plt.yscale('log')
                    plt.xticks(rotation=45, ha='right')
                    plt.title(f'BER Comparison at {snr} dB - {modulation}, {channel} channel', fontsize=16)
                    plt.ylabel('BER (log scale)', fontsize=14)
                    plt.xlabel('')
                    plt.grid(True, which='both', linestyle='--', axis='y')
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, f"ber_comparison_{modulation}_{channel}_snr_{snr}.png"))
                    plt.close()
                    print(f"Created BER comparison plot for {modulation}, {channel} at SNR = {snr}")
                
            except Exception as e:
                print(f"Error plotting {modulation}_{channel} summary table: {e}")

def main():
    parser = argparse.ArgumentParser(description='Plot visualizations from MIMO summary tables')
    parser.add_argument('--input_dir', type=str, default='./analysis_results/summary_tables',
                        help='Directory containing the summary tables')
    parser.add_argument('--output_dir', type=str, default='./analysis_results/summary_plots',
                        help='Directory to save visualization results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot summary tables
    print("Creating visualizations from summary tables...")
    plot_summary_tables(args.input_dir, args.output_dir)
    
    print("Visualization complete!")

if __name__ == "__main__":
    main()