import os
import subprocess
import argparse
import time
from itertools import product

def run_command(command):
    """Run a command and print its output"""
    print(f"Running: {command}")
    start_time = time.time()
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

    
    for line in process.stdout:
        line = line.rstrip()
        print(line)
    
    process.wait()
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Command completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("-" * 80)
    
    return process.returncode

def main():
    parser = argparse.ArgumentParser(description='Batch training for MIMO models')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'ber', 'compare', 'all'], default='all',
                    help='Mode to run: train, test, ber, compare, or all')
    parser.add_argument('--epochs', type=int, default=8, help='Number of epochs for training')
    parser.add_argument('--channel', type=str, choices=['ideal', 'nonideal', 'both'], default='nonideal',
                        help='Channel type to use')
    parser.add_argument('--modulation', type=str, choices=['BPSK', 'QPSK', 'both'], default='both',
                        help='Modulation scheme to use')
    parser.add_argument('--models', type=str, default='all',
                        help='Models to train (comma separated): mobilenetv2,densenet,resnet50,vgg,squeezenet or all')
    parser.add_argument('--configs', type=str, default='all',
                        help='Antenna configurations (comma separated): 2x1,2x2,3x1,4x1,4x2,4x4 or all')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--train_samples', type=int, default=40000, help='Number of training samples')
    parser.add_argument('--val_samples', type=int, default=20000, help='Number of validation samples')
    parser.add_argument('--test_samples', type=int, default=40000, help='Number of test samples')
    parser.add_argument('--result_dir', type=str, default='./comprehensive_results', help='Directory to save results')
    
    args = parser.parse_args()
    
    
    os.makedirs(args.result_dir, exist_ok=True)
    
    
    
    if args.models.lower() == 'all':
        models = ['mobilenetv2', 'densenet', 'resnet50', 'vgg', 'squeezenet']  
    else:
        models = [model.strip().lower() for model in args.models.split(',')]
    
    
    if args.configs.lower() == 'all':
        configs = ['2x1', '2x2', '3x1', '4x1', '4x2', '4x4']
    else:
        configs = [config.strip().lower() for config in args.configs.split(',')]
    
    
    if args.modulation.lower() == 'both':
        modulations = ['BPSK', 'QPSK']
    else:
        modulations = [args.modulation]
    
    
    if args.channel.lower() == 'both':
        channels = ['ideal', 'nonideal']
    else:
        channels = [args.channel]
    
    
    combinations = list(product(models, configs, modulations, channels))
    total_combinations = len(combinations)
    
    print(f"Will process {total_combinations} combinations in total.")
    
    
    for i, (model, config, modulation, channel) in enumerate(combinations):
        print(f"Processing combination {i+1}/{total_combinations}:")
        print(f"Model: {model}, Config: {config}, Modulation: {modulation}, Channel: {channel}")
        
        
        tx, rx = map(int, config.split('x'))
        
        
        base_params = (
            f"--model {model} "
            f"--tx {tx} "
            f"--rx {rx} "
            f"--modulation {modulation} "
            f"--channel {channel} "
            f"--batch_size {args.batch_size} "
            f"--train_samples {args.train_samples} "
            f"--val_samples {args.val_samples} "
            f"--test_samples {args.test_samples} "
            f"--save_dir {args.result_dir}"
        )
        
        
        model_path = f"{args.result_dir}/{model}_{tx}x{rx}_{modulation}_{channel}_model.pth"
        
        
        if args.mode in ['train', 'all']:
            train_cmd = f"python train.py {base_params} --run_mode train --epochs {args.epochs}"
            run_command(train_cmd)
            
        
        if args.mode in ['test', 'all']:
            if os.path.exists(model_path):
                test_cmd = f"python train.py {base_params} --run_mode test --load_model {model_path}"
                run_command(test_cmd)
            else:
                print(f"Warning: Model file {model_path} not found. Skipping test step.")

        
        if os.path.exists(model_path):
            
            if args.mode in ['ber', 'all']:
                ber_cmd = f"python train.py {base_params} --run_mode ber --load_model {model_path}"
                run_command(ber_cmd)
            
            
            if args.mode in ['compare', 'all']:
                compare_cmd = f"python compare_receivers.py {base_params} --model_path {model_path}"
                run_command(compare_cmd)
        else:
            print(f"Warning: Model file {model_path} not found. Skipping BER and comparison steps.")
    
    print("All combinations processed!")

if __name__ == "__main__":
    main()

    
    
