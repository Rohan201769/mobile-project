import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import argparse
import os


from mimo_models import DenseNetForMIMO, ResNet50ForMIMO, MobileNetV2ForMIMO, VGGForMIMO, SqueezeNetForMIMO
from mimo_data import MIMODataset, create_mimo_dataloaders

class TraditionalReceiver:
    """
    Implementation of the traditional MIMO receiver with ML decoding
    """
    def __init__(self, tx_antennas, rx_antennas, modulation='BPSK'):
        self.tx_antennas = tx_antennas
        self.rx_antennas = rx_antennas
        self.modulation = modulation
    
    def decode(self, received_signal, channel, noise_var=1.0):
        """
        ML decoding for MIMO system
        Args:
            received_signal: The received signal (complex)
            channel: The channel matrix
            noise_var: Noise variance
        
        Returns:
            decoded_bits: The decoded information bits
        """
        
        
        if self.modulation == 'BPSK':
            
            possible_symbols = np.array([-1, 1])
            
            
            min_distance = float('inf')
            best_symbol = None
            
            for symbol in possible_symbols:
                
                
                symbol_vector = np.ones(self.tx_antennas) * symbol
                
                
                expected_signal = np.dot(channel, symbol_vector)
                
                
                
                min_length = min(received_signal.size, expected_signal.size)
                
                
                received_flat = received_signal.flatten()[:min_length]
                expected_flat = expected_signal.flatten()[:min_length]
                
                
                distance = np.sum(np.abs(received_flat - expected_flat) ** 2)
                
                if distance < min_distance:
                    min_distance = distance
                    best_symbol = symbol
            
            
            decoded_bit = (best_symbol + 1) // 2
            return decoded_bit
        
        elif self.modulation == 'QPSK':
            
            possible_symbols = np.array([
                -1-1j, -1+1j, 1-1j, 1+1j
            ])
            
            
            min_distance = float('inf')
            best_symbol = None
            
            for symbol in possible_symbols:
                
                symbol_vector = np.ones(self.tx_antennas) * symbol
                
                
                expected_signal = np.dot(channel, symbol_vector)
                
                
                min_length = min(received_signal.size, expected_signal.size)
                
                
                received_flat = received_signal.flatten()[:min_length]
                expected_flat = expected_signal.flatten()[:min_length]
                
                
                distance = np.sum(np.abs(received_flat - expected_flat) ** 2)
                
                if distance < min_distance:
                    min_distance = distance
                    best_symbol = symbol
            
            
            real_bit = (np.real(best_symbol) + 1) // 2
            imag_bit = (np.imag(best_symbol) + 1) // 2
            return np.array([real_bit, imag_bit])
    
    def compute_ber(self, dataset, num_samples=1000):
        """
        Compute BER for the traditional receiver
        """
        total_bits = 0
        bit_errors = 0
        
        
        if hasattr(dataset, 'channel_type') and dataset.channel_type == 'ideal':
            channel = np.eye(self.rx_antennas, self.tx_antennas)
        else:
            
            h_real = np.random.normal(0, 1/np.sqrt(2), (self.rx_antennas, self.tx_antennas))
            h_imag = np.random.normal(0, 1/np.sqrt(2), (self.rx_antennas, self.tx_antennas))
            channel = h_real + 1j * h_imag
        
        for i in tqdm(range(min(num_samples, len(dataset))), desc="Traditional Receiver"):
            
            inputs, targets = dataset[i]
            
            
            received_signal = inputs[:, 0] + 1j * inputs[:, 1]
            
            
            decoded_bits = self.decode(received_signal, channel)
            
            
            if isinstance(targets, torch.Tensor):
                targets = targets.numpy()
            
            
            if np.isscalar(decoded_bits):
                
                bit_errors += (decoded_bits != targets.flatten()[0])
                total_bits += 1
            else:
                
                decoded_bits_array = np.atleast_1d(decoded_bits)  
                targets_flat = targets.flatten()
                min_len = min(len(decoded_bits_array), len(targets_flat))
                
                
                decoded_bits_array = decoded_bits_array[:min_len]
                targets_flat = targets_flat[:min_len]
                
                
                bit_errors += np.sum(decoded_bits_array != targets_flat)
                total_bits += min_len
        
        if total_bits == 0:
            return 1.0  
        return bit_errors / total_bits


def compare_receivers(snr_range, tx_antennas, rx_antennas, modulation, channel_type,
                     intelligent_model_path, model_type, test_samples=1000, device='cuda'):
    """
    Compare traditional and intelligent receivers
    """
    snr_values = np.array(snr_range)
    traditional_ber = []
    intelligent_ber = []
    
    
    
    if model_type == 'densenet':
        model = DenseNetForMIMO(in_channels=2, rx_antennas=rx_antennas, tx_antennas=tx_antennas, num_classes=8)
    elif model_type == 'resnet50':
        model = ResNet50ForMIMO(in_channels=2, rx_antennas=rx_antennas, tx_antennas=tx_antennas, num_classes=8)
    elif model_type == 'vgg':
        model = VGGForMIMO(in_channels=2, rx_antennas=rx_antennas, tx_antennas=tx_antennas, num_classes=8)
    elif model_type == 'squeezenet':  
        model = SqueezeNetForMIMO(in_channels=2, rx_antennas=rx_antennas, tx_antennas=tx_antennas, num_classes=8)
    else:  
        model = MobileNetV2ForMIMO(in_channels=2, rx_antennas=rx_antennas, tx_antennas=tx_antennas, num_classes=8)
    
    
    original_rx_antennas = rx_antennas  
    try:
        checkpoint = torch.load(intelligent_model_path, weights_only=True)
        
        if model_type == 'mobilenetv2' and 'shallow_features.0.weight' in checkpoint:
            saved_in_channels = checkpoint['shallow_features.0.weight'].shape[1]
            detected_rx_antennas = saved_in_channels // 2
            if detected_rx_antennas != rx_antennas:
                print(f"Mismatch detected: Model was trained with {detected_rx_antennas} rx antennas but parameter is {rx_antennas}")
                print(f"Recreating model with {detected_rx_antennas} rx antennas")
                rx_antennas = detected_rx_antennas
                if model_type == 'mobilenetv2':
                    model = MobileNetV2ForMIMO(in_channels=2, rx_antennas=rx_antennas, tx_antennas=tx_antennas, num_classes=8)
        
        
        elif model_type == 'resnet50' and 'conv1.weight' in checkpoint:
            saved_in_channels = checkpoint['conv1.weight'].shape[1]
            detected_rx_antennas = saved_in_channels // 2
            if detected_rx_antennas != rx_antennas:
                print(f"Mismatch detected: Model was trained with {detected_rx_antennas} rx antennas but parameter is {rx_antennas}")
                print(f"Recreating model with {detected_rx_antennas} rx antennas")
                rx_antennas = detected_rx_antennas
                model = ResNet50ForMIMO(in_channels=2, rx_antennas=rx_antennas, tx_antennas=tx_antennas, num_classes=8)
            
        elif model_type == 'densenet' and 'conv1.weight' in checkpoint:
            saved_in_channels = checkpoint['conv1.weight'].shape[1]
            detected_rx_antennas = saved_in_channels // 2
            if detected_rx_antennas != rx_antennas:
                print(f"Mismatch detected: Model was trained with {detected_rx_antennas} rx antennas but parameter is {rx_antennas}")
                print(f"Recreating model with {detected_rx_antennas} rx antennas")
                rx_antennas = detected_rx_antennas
                model = DenseNetForMIMO(in_channels=2, rx_antennas=rx_antennas, tx_antennas=tx_antennas, num_classes=8)

        
        elif model_type == 'squeezenet' and 'features.0.weight' in checkpoint:
            saved_in_channels = checkpoint['features.0.weight'].shape[1]
            detected_rx_antennas = saved_in_channels // 2
            if detected_rx_antennas != rx_antennas:
                print(f"Mismatch detected: Model was trained with {detected_rx_antennas} rx antennas but parameter is {rx_antennas}")
                print(f"Recreating model with {detected_rx_antennas} rx antennas")
                rx_antennas = detected_rx_antennas
                model = SqueezeNetForMIMO(in_channels=2, rx_antennas=rx_antennas, tx_antennas=tx_antennas, num_classes=8)

        elif model_type == 'vgg' and 'features.0.weight' in checkpoint:
            saved_in_channels = checkpoint['features.0.weight'].shape[1]
            detected_rx_antennas = saved_in_channels // 2
            if detected_rx_antennas != rx_antennas:
                print(f"Mismatch detected: Model was trained with {detected_rx_antennas} rx antennas but parameter is {rx_antennas}")
                print(f"Recreating model with {detected_rx_antennas} rx antennas")
                rx_antennas = detected_rx_antennas
                model = VGGForMIMO(in_channels=2, rx_antennas=rx_antennas, tx_antennas=tx_antennas, num_classes=8)
    except Exception as e:
        print(f"Warning: Could not automatically detect receiving antennas: {e}")
    model.load_state_dict(torch.load(intelligent_model_path, map_location=torch.device('cpu')))

    
    model = model.to(device)
    model.eval()
    
    
    traditional_receiver = TraditionalReceiver(tx_antennas, rx_antennas, modulation)
    
    for snr in snr_values:
        
        test_dataset = MIMODataset(
            num_samples=test_samples,
            tx_antennas=tx_antennas,
            rx_antennas=rx_antennas,  
            modulation=modulation,
            Eb_N0_dB=[snr],
            channel_type=channel_type
        )
        
        
        trad_ber = traditional_receiver.compute_ber(test_dataset)
        traditional_ber.append(trad_ber)
        
        
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        total_bits = 0
        bit_errors = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Intelligent Receiver"):
                
                if not isinstance(inputs, torch.Tensor):
                    inputs = torch.tensor(inputs, dtype=torch.float32)
                if not isinstance(targets, torch.Tensor):
                    targets = torch.tensor(targets, dtype=torch.float32)
                
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                
                outputs = model(inputs)
                
                
                predictions = (outputs > 0.5).float()
                total_bits += targets.numel()
                bit_errors += torch.sum(predictions != targets).item()
        
        intel_ber = bit_errors / total_bits
        intelligent_ber.append(intel_ber)
        
        print(f"SNR: {snr} dB - Traditional BER: {trad_ber:.6f}, Intelligent BER: {intel_ber:.6f}")
    
    return snr_values, traditional_ber, intelligent_ber


def plot_comparison(snr_values, traditional_ber, intelligent_ber, model_type, tx_rx_config, save_path=None):
    """
    Plot comparison between traditional and intelligent receivers
    """
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_values, traditional_ber, 'o-', label='Traditional Receiver')
    plt.semilogy(snr_values, intelligent_ber, 's-', label=f'Intelligent Receiver ({model_type})')
    plt.grid(True, which="both", ls="--")
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title(f'BER Comparison ({tx_rx_config})')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def main():
    
    parser = argparse.ArgumentParser(description='Compare traditional and intelligent MIMO receivers')
    parser.add_argument('--model', type=str, default='mobilenetv2', 
                   choices=['densenet', 'resnet50', 'mobilenetv2', 'vgg', 'squeezenet'],  
                   help='Intelligent receiver model architecture')
    parser.add_argument('--tx', type=int, default=2, choices=[2, 3, 4,8],
                      help='Number of transmit antennas')
    parser.add_argument('--rx', type=int, default=1, choices=[1, 2, 3, 4,8],
                      help='Number of receive antennas')
    parser.add_argument('--modulation', type=str, default='BPSK', choices=['BPSK', 'QPSK'],
                      help='Modulation scheme')
    parser.add_argument('--channel', type=str, default='nonideal', choices=['ideal', 'nonideal'],
                      help='Channel type')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained intelligent receiver model')
    parser.add_argument('--save_dir', type=str, default='./results',
                      help='Directory to save results')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use (cuda or cpu), default will auto-detect')
    parser.add_argument('--batch_size', type=int, default=256,
                      help='Batch size for testing')
    parser.add_argument('--train_samples', type=int, default=40000,
                      help='Number of training samples (not used in compare)')
    parser.add_argument('--val_samples', type=int, default=20000,
                      help='Number of validation samples (not used in compare)')
    parser.add_argument('--test_samples', type=int, default=40000,
                      help='Number of test samples')
    
    args = parser.parse_args()
    
    
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    
    snr_range = np.arange(0, 8.5, 0.5)
    
    
    tx_rx_config = f"{args.tx}x{args.rx} MIMO ({args.modulation}, {args.channel} channel)"
    
    
    snr_values, traditional_ber, intelligent_ber = compare_receivers(
        snr_range=snr_range,
        tx_antennas=args.tx,
        rx_antennas=args.rx,
        modulation=args.modulation,
        channel_type=args.channel,
        intelligent_model_path=args.model_path,
        model_type=args.model,
        test_samples=args.test_samples,
        device=device
    )
    
    
    comparison_save_path = os.path.join(
        args.save_dir, 
        f"comparison_{args.model}_{args.tx}x{args.rx}_{args.modulation}_{args.channel}.png"
    )
    plot_comparison(
        snr_values=snr_values,
        traditional_ber=traditional_ber,
        intelligent_ber=intelligent_ber,
        model_type=args.model,
        tx_rx_config=tx_rx_config,
        save_path=comparison_save_path
    )
    
    
    comparison_data_path = os.path.join(
        args.save_dir, 
        f"comparison_{args.model}_{args.tx}x{args.rx}_{args.modulation}_{args.channel}.txt"
    )
    with open(comparison_data_path, 'w') as f:
        f.write("SNR,Traditional_BER,Intelligent_BER\n")
        for snr, trad_ber, intel_ber in zip(snr_values, traditional_ber, intelligent_ber):
            f.write(f"{snr},{trad_ber},{intel_ber}\n")
    
    print(f"Comparison plot saved to {comparison_save_path}")
    print(f"Comparison data saved to {comparison_data_path}")


if __name__ == "__main__":
    main()