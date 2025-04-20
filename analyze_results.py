import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from itertools import product

def load_ber_data(result_dir):
    """Load BER data from result files"""
    ber_files = glob.glob(os.path.join(result_dir, "*_ber.txt"))
    data = []
    
    for file_path in ber_files:
        
        filename = os.path.basename(file_path)
        parts = filename.replace("_ber.txt", "").split("_")
        
        if len(parts) < 4:
            print(f"Skipping file with unexpected format: {filename}")
            continue
        
        model = parts[0]
        config = parts[1]
        modulation = parts[2]
        channel = parts[3]
        
        
        try:
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                data.append({
                    "Model": model,
                    "Config": config,
                    "Modulation": modulation,
                    "Channel": channel,
                    "SNR": row["SNR"],
                    "BER": row["BER"]
                })
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    return pd.DataFrame(data)

def load_comparison_data(result_dir):
    """Load comparison data between traditional and intelligent receivers"""
    comparison_files = glob.glob(os.path.join(result_dir, "comparison_*_ber.txt"))
    data = []
    
    for file_path in comparison_files:
        
        filename = os.path.basename(file_path)
        parts = filename.replace("comparison_", "").replace("_ber.txt", "").split("_")
        
        if len(parts) < 4:
            print(f"Skipping file with unexpected format: {filename}")
            continue
        
        model = parts[0]
        config = parts[1]
        modulation = parts[2]
        channel = parts[3]
        
        
        try:
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                
                data.append({
                    "Model": "Traditional",
                    "Config": config,
                    "Modulation": modulation,
                    "Channel": channel,
                    "SNR": row["SNR"],
                    "BER": row["Traditional_BER"]
                })
                
                
                data.append({
                    "Model": model,
                    "Config": config,
                    "Modulation": modulation,
                    "Channel": channel,
                    "SNR": row["SNR"],
                    "BER": row["Intelligent_BER"]
                })
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    return pd.DataFrame(data)

def plot_model_comparison(df, output_dir):
    """Compare all models for each configuration"""
    configs = df["Config"].unique()
    modulations = df["Modulation"].unique()
    channels = df["Channel"].unique()
    
    os.makedirs(output_dir, exist_ok=True)
    
    for config, modulation, channel in product(configs, modulations, channels):
        subset = df[(df["Config"] == config) & 
                    (df["Modulation"] == modulation) & 
                    (df["Channel"] == channel)]
        
        if subset.empty:
            continue
        
        plt.figure(figsize=(12, 8))
        
        models = [model for model in subset["Model"].unique() if model != "Traditional"]
        for model in models:
            model_data = subset[subset["Model"] == model]
            plt.semilogy(model_data["SNR"], model_data["BER"], marker='o', linestyle='-', label=model)
        
        
        trad_data = subset[subset["Model"] == "Traditional"]
        if not trad_data.empty:
            plt.semilogy(trad_data["SNR"], trad_data["BER"], marker='s', linestyle='--', label="Traditional", color='black')
        
        plt.grid(True, which="both", ls="--")
        plt.xlabel('Eb/N0 (dB)', fontsize=14)
        plt.ylabel('BER', fontsize=14)
        plt.title(f'Model Comparison - {config} MIMO, {modulation}, {channel} channel', fontsize=16)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        filename = f"model_comparison_{config}_{modulation}_{channel}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f"Saved {filename}")

def plot_config_comparison(df, output_dir):
    """Compare different antenna configurations for each model"""
    models = [model for model in df["Model"].unique() if model != "Traditional"]
    modulations = df["Modulation"].unique()
    channels = df["Channel"].unique()
    
    os.makedirs(output_dir, exist_ok=True)
    
    for model, modulation, channel in product(models, modulations, channels):
        subset = df[(df["Model"] == model) & 
                    (df["Modulation"] == modulation) & 
                    (df["Channel"] == channel)]
        
        if subset.empty:
            continue
        
        plt.figure(figsize=(12, 8))
        
        configs = subset["Config"].unique()
        for config in configs:
            config_data = subset[subset["Config"] == config]
            plt.semilogy(config_data["SNR"], config_data["BER"], marker='o', linestyle='-', label=config)
        
        plt.grid(True, which="both", ls="--")
        plt.xlabel('Eb/N0 (dB)', fontsize=14)
        plt.ylabel('BER', fontsize=14)
        plt.title(f'{model} - Antenna Configuration Comparison\n{modulation}, {channel} channel', fontsize=16)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        filename = f"config_comparison_{model}_{modulation}_{channel}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f"Saved {filename}")

def plot_channel_comparison(df, output_dir):
    """Compare performance between ideal and nonideal channels"""
    models = df["Model"].unique()
    configs = df["Config"].unique()
    modulations = df["Modulation"].unique()
    
    os.makedirs(output_dir, exist_ok=True)
    
    for model, config, modulation in product(models, configs, modulations):
        subset = df[(df["Model"] == model) & 
                    (df["Config"] == config) & 
                    (df["Modulation"] == modulation)]
        
        if subset.empty:
            continue
        
        plt.figure(figsize=(12, 8))
        
        channels = subset["Channel"].unique()
        if len(channels) < 2:
            plt.close()
            continue
            
        for channel in channels:
            channel_data = subset[subset["Channel"] == channel]
            plt.semilogy(channel_data["SNR"], channel_data["BER"], marker='o', linestyle='-', label=f"{channel} channel")
        
        plt.grid(True, which="both", ls="--")
        plt.xlabel('Eb/N0 (dB)', fontsize=14)
        plt.ylabel('BER', fontsize=14)
        plt.title(f'{model} - {config} MIMO, {modulation}\nChannel Comparison', fontsize=16)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        filename = f"channel_comparison_{model}_{config}_{modulation}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f"Saved {filename}")

def plot_modulation_comparison(df, output_dir):
    """Compare performance between BPSK and QPSK modulation"""
    models = df["Model"].unique()
    configs = df["Config"].unique()
    channels = df["Channel"].unique()
    
    os.makedirs(output_dir, exist_ok=True)
    
    for model, config, channel in product(models, configs, channels):
        subset = df[(df["Model"] == model) & 
                    (df["Config"] == config) & 
                    (df["Channel"] == channel)]
        
        if subset.empty:
            continue
        
        plt.figure(figsize=(12, 8))
        
        modulations = subset["Modulation"].unique()
        if len(modulations) < 2:
            plt.close()
            continue
            
        for modulation in modulations:
            mod_data = subset[subset["Modulation"] == modulation]
            plt.semilogy(mod_data["SNR"], mod_data["BER"], marker='o', linestyle='-', label=f"{modulation}")
        
        plt.grid(True, which="both", ls="--")
        plt.xlabel('Eb/N0 (dB)', fontsize=14)
        plt.ylabel('BER', fontsize=14)
        plt.title(f'{model} - {config} MIMO, {channel} channel\nModulation Comparison', fontsize=16)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        filename = f"modulation_comparison_{model}_{config}_{channel}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f"Saved {filename}")

def create_summary_table(df, output_dir):
    """Create a summary table of BER values at specific SNR points"""
    
    snr_points = [0, 2, 4, 6, 8]
    summary_data = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    for snr in snr_points:
        
        closest_snr = df["SNR"].unique()[np.abs(df["SNR"].unique() - snr).argmin()]
        
        snr_df = df[df["SNR"] == closest_snr]
        
        for _, row in snr_df.iterrows():
            summary_data.append({
                "SNR": closest_snr,
                "Model": row["Model"],
                "Config": row["Config"],
                "Modulation": row["Modulation"],
                "Channel": row["Channel"],
                "BER": row["BER"]
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    
    summary_df.to_csv(os.path.join(output_dir, "ber_summary_table.csv"), index=False)
    
    
    configs = summary_df["Config"].unique()
    modulations = summary_df["Modulation"].unique()
    channels = summary_df["Channel"].unique()
    
    for modulation, channel in product(modulations, channels):
        table_data = []
        
        for config in configs:
            config_data = summary_df[(summary_df["Config"] == config) &
                                     (summary_df["Modulation"] == modulation) &
                                     (summary_df["Channel"] == channel)]
            
            if config_data.empty:
                continue
                
            for model in config_data["Model"].unique():
                model_data = config_data[config_data["Model"] == model]
                
                row_data = {"Model": model, "Config": config}
                for snr in snr_points:
                    closest_snr = model_data["SNR"].unique()[np.abs(model_data["SNR"].unique() - snr).argmin()]
                    snr_value = model_data[model_data["SNR"] == closest_snr]["BER"].values
                    
                    if len(snr_value) > 0:
                        row_data[f"SNR_{snr}dB"] = snr_value[0]
                    else:
                        row_data[f"SNR_{snr}dB"] = np.nan
                
                table_data.append(row_data)
        
        if table_data:
            table_df = pd.DataFrame(table_data)
            table_df.to_csv(os.path.join(output_dir, f"summary_{modulation}_{channel}.csv"), index=False)
            print(f"Saved summary table for {modulation}, {channel}")

def main():
    parser = argparse.ArgumentParser(description='Analyze MIMO results')
    parser.add_argument('--result_dir', type=str, default='./comprehensive_results',
                        help='Directory containing the results')
    parser.add_argument('--output_dir', type=str, default='./analysis_results',
                        help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    
    print("Loading BER data...")
    ber_df = load_ber_data(args.result_dir)
    
    print("Loading comparison data...")
    comparison_df = load_comparison_data(args.result_dir)
    
    
    combined_df = pd.concat([ber_df, comparison_df]).drop_duplicates()
    
    
    print("Creating model comparison plots...")
    plot_model_comparison(combined_df, os.path.join(args.output_dir, "model_comparison"))
    
    print("Creating antenna configuration comparison plots...")
    plot_config_comparison(combined_df, os.path.join(args.output_dir, "config_comparison"))
    
    print("Creating channel comparison plots...")
    plot_channel_comparison(combined_df, os.path.join(args.output_dir, "channel_comparison"))
    
    print("Creating modulation comparison plots...")
    plot_modulation_comparison(combined_df, os.path.join(args.output_dir, "modulation_comparison"))
    
    print("Creating summary tables...")
    create_summary_table(combined_df, os.path.join(args.output_dir, "summary_tables"))
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()