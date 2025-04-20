import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
import argparse


from mimo_models import DenseNetForMIMO, ResNet50ForMIMO, MobileNetV2ForMIMO, VGGForMIMO, SqueezeNetForMIMO
from mimo_data import create_mimo_dataloaders

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=8, device='cuda'):
    """
    Train the model
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train
        device: Device to train on ('cuda' or 'cpu')
    
    Returns:
        model: Trained model
        history: Dictionary of training history
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for inputs, targets in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            
            optimizer.zero_grad()
            
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            
            loss.backward()
            optimizer.step()
            
            
            running_loss += loss.item() * inputs.size(0)
            
            
            predictions = (outputs > 0.5).float()
            correct_predictions = torch.sum(predictions == targets).item()
            running_corrects += correct_predictions
            total_samples += (inputs.size(0) * targets.size(1))
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / total_samples
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                
                running_loss += loss.item() * inputs.size(0)
                
                
                predictions = (outputs > 0.5).float()
                correct_predictions = torch.sum(predictions == targets).item()
                running_corrects += correct_predictions
                total_samples += (inputs.size(0) * targets.size(1))
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects / total_samples
        
        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc)
        
        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        
        scheduler.step()
    
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    return model, history


def evaluate_model(model, test_loader, criterion, device='cuda'):
    """
    Evaluate the model on the test set
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to evaluate on ('cuda' or 'cpu')
    
    Returns:
        test_loss: Test loss
        test_acc: Test accuracy
        ber: Bit error rate
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    total_bits = 0
    bit_errors = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            
            running_loss += loss.item() * inputs.size(0)
            
            
            predictions = (outputs > 0.5).float()
            correct_predictions = torch.sum(predictions == targets).item()
            running_corrects += correct_predictions
            total_samples += (inputs.size(0) * targets.size(1))
            
            
            total_bits += targets.numel()
            bit_errors += torch.sum(predictions != targets).item()
    
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects / total_samples
    ber = bit_errors / total_bits
    
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} BER: {ber:.6f}')
    
    return test_loss, test_acc, ber


def compute_ber_vs_snr(model, tx_antennas, rx_antennas, modulation='BPSK', 
                     channel_type='nonideal', test_samples=40000, batch_size=256, device='cuda'):
    """
    Compute BER vs SNR curve
    
    Args:
        model: PyTorch model
        tx_antennas: Number of transmit antennas
        rx_antennas: Number of receive antennas
        modulation: 'BPSK' or 'QPSK'
        channel_type: 'ideal' or 'nonideal'
        test_samples: Number of test samples
        batch_size: Batch size
        device: Device to evaluate on ('cuda' or 'cpu')
    
    Returns:
        snr_values: SNR values
        ber_values: BER values
    """
    
    snr_values = np.arange(0, 8.5, 0.5)
    ber_values = []
    
    model.eval()
    
    for snr in tqdm(snr_values, desc="Computing BER vs SNR"):
        
        _, _, test_loader = create_mimo_dataloaders(
            tx_antennas=tx_antennas,
            rx_antennas=rx_antennas,
            modulation=modulation,
            train_samples=1000,  
            val_samples=1000,
            test_samples=test_samples,
            batch_size=batch_size,
            Eb_N0_dB_train=[0],  
            Eb_N0_dB_test=[snr],  
            channel_type=channel_type
        )
        
        total_bits = 0
        bit_errors = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                
                outputs = model(inputs)
                
                
                predictions = (outputs > 0.5).float()
                total_bits += targets.numel()
                bit_errors += torch.sum(predictions != targets).item()
        
        ber = bit_errors / total_bits
        ber_values.append(ber)
        print(f"SNR: {snr} dB, BER: {ber:.6f}")
    
    return snr_values, ber_values


def save_model(model, filepath):
    """Save model to file"""
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model_class, filepath, **kwargs):
    """Load model from file"""
    
    try:
        checkpoint = torch.load(filepath)
        
        
        
        if model_class.__name__ == 'MobileNetV2ForMIMO' and 'shallow_features.0.weight' in checkpoint:
            
            saved_in_channels = checkpoint['shallow_features.0.weight'].shape[1]
            
            rx_antennas = kwargs.get('rx_antennas', 1)
            tx_antennas = kwargs.get('tx_antennas', 2)
            
            
            expected_channels = 0
            if tx_antennas == 2:
                expected_channels = 2 * rx_antennas
            elif tx_antennas == 3:
                expected_channels = 8 * rx_antennas
            elif tx_antennas == 4:
                expected_channels = 4 * rx_antennas
            
            
            if saved_in_channels != expected_channels:
                print(f"Warning: Model was trained with different channel configuration.")
                print(f"Using {saved_in_channels} input channels from saved model instead of {expected_channels}.")
                if tx_antennas == 2:
                    kwargs['rx_antennas'] = saved_in_channels // 2
                elif tx_antennas == 3:
                    kwargs['rx_antennas'] = saved_in_channels // 8
                elif tx_antennas == 4:
                    kwargs['rx_antennas'] = saved_in_channels // 4
                
                print(f"Setting rx_antennas to {kwargs['rx_antennas']}")
        
        
    except Exception as e:
        print(f"Warning: Could not analyze saved model: {e}")
    
    
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(filepath))
    print(f"Model loaded from {filepath}")
    return model


def plot_training_history(history, save_path=None):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_ber_vs_snr(snr_values, ber_values, model_name, save_path=None):
    """Plot BER vs SNR curve"""
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_values, ber_values, 'o-', label=model_name)
    plt.grid(True, which="both", ls="--")
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title('BER vs Eb/N0')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def main():
    
    parser = argparse.ArgumentParser(description='Train MIMO receiver models')
    parser.add_argument('--model', type=str, default='mobilenetv2', 
                   choices=['densenet', 'resnet50', 'mobilenetv2', 'vgg', 'squeezenet'],  
                   help='Model architecture to use')
    parser.add_argument('--tx', type=int, default=2, choices=[2, 3, 4, 8],
                        help='Number of transmit antennas')
    parser.add_argument('--rx', type=int, default=1, choices=[1, 2, 3, 4, 8],
                        help='Number of receive antennas')
    parser.add_argument('--modulation', type=str, default='BPSK', choices=['BPSK', 'QPSK'],
                        help='Modulation scheme')
    parser.add_argument('--channel', type=str, default='nonideal', choices=['ideal', 'nonideal'],
                        help='Channel type')
    parser.add_argument('--epochs', type=int, default=8,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--train_samples', type=int, default=40000,
                        help='Number of training samples')
    parser.add_argument('--val_samples', type=int, default=20000,
                        help='Number of validation samples')
    parser.add_argument('--test_samples', type=int, default=40000,
                        help='Number of test samples')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--run_mode', type=str, default='train', choices=['train', 'test', 'ber'],
                        help='Mode: train, test, or compute BER vs SNR')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to load pretrained model for testing')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu), default will auto-detect')
    
    args = parser.parse_args()
    
    
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    
    config_str = f"{args.model}_{args.tx}x{args.rx}_{args.modulation}_{args.channel}"
    model_save_path = os.path.join(args.save_dir, f"{config_str}_model.pth")
    history_save_path = os.path.join(args.save_dir, f"{config_str}_history.png")
    ber_save_path = os.path.join(args.save_dir, f"{config_str}_ber.png")
    
    
    
    if args.model == 'densenet':
        model = DenseNetForMIMO(in_channels=2, rx_antennas=args.rx, tx_antennas=args.tx, num_classes=8)
    elif args.model == 'resnet50':
        model = ResNet50ForMIMO(in_channels=2, rx_antennas=args.rx, tx_antennas=args.tx, num_classes=8)
    elif args.model == 'vgg':
        model = VGGForMIMO(in_channels=2, rx_antennas=args.rx, tx_antennas=args.tx, num_classes=8)
    elif args.model == 'squeezenet':  
        model = SqueezeNetForMIMO(in_channels=2, rx_antennas=args.rx, tx_antennas=args.tx, num_classes=8)
    else:  
        model = MobileNetV2ForMIMO(in_channels=2, rx_antennas=args.rx, tx_antennas=args.tx, num_classes=8)
    
    model = model.to(device)
    
    
    if args.run_mode == 'train':
        
        train_loader, val_loader, test_loader = create_mimo_dataloaders(
            tx_antennas=args.tx,
            rx_antennas=args.rx,
            modulation=args.modulation,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            test_samples=args.test_samples,
            batch_size=args.batch_size,
            Eb_N0_dB_train=range(0, 9),
            Eb_N0_dB_test=range(0, 9),
            channel_type=args.channel
        )

        for inputs, targets in train_loader:
            print(f"Input shape: {inputs.shape}")
            break

        print(f"Model expects input channels: {model.actual_in_channels}")
        
        
        criterion = nn.BCELoss()  
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        
        
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
        
        
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=args.epochs,
            device=device
        )
        
        
        save_model(model, model_save_path)
        
        
        plot_training_history(history, save_path=history_save_path)
        
        
        test_loss, test_acc, ber = evaluate_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device
        )
        
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, BER: {ber:.6f}")
    
    
    elif args.run_mode == 'test':
        
        
        if args.load_model:
            if args.model == 'densenet':
                model = load_model(DenseNetForMIMO, args.load_model, in_channels=2, rx_antennas=args.rx, tx_antennas=args.tx, num_classes=8)
            elif args.model == 'resnet50':
                model = load_model(ResNet50ForMIMO, args.load_model, in_channels=2, rx_antennas=args.rx, tx_antennas=args.tx, num_classes=8)
            elif args.model == 'vgg':
                model = load_model(VGGForMIMO, args.load_model, in_channels=2, rx_antennas=args.rx, tx_antennas=args.tx, num_classes=8)
            elif args.model == 'squeezenet':  
                model = load_model(SqueezeNetForMIMO, args.load_model, in_channels=2, rx_antennas=args.rx, tx_antennas=args.tx, num_classes=8)
            else:  
                model = load_model(MobileNetV2ForMIMO, args.load_model, in_channels=2, rx_antennas=args.rx, tx_antennas=args.tx, num_classes=8)
            model = model.to(device)
        
        
        _, _, test_loader = create_mimo_dataloaders(
            tx_antennas=args.tx,
            rx_antennas=args.rx,
            modulation=args.modulation,
            train_samples=1000,  
            val_samples=1000,
            test_samples=args.test_samples,
            batch_size=args.batch_size,
            Eb_N0_dB_train=range(0, 9),
            Eb_N0_dB_test=range(0, 9),
            channel_type=args.channel
        )
        
        
        criterion = nn.BCELoss()
        
        
        test_loss, test_acc, ber = evaluate_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device
        )
        
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, BER: {ber:.6f}")
    
    
    elif args.run_mode == 'ber':
        
        
        if args.load_model:
            if args.model == 'densenet':
                model = load_model(DenseNetForMIMO, args.load_model, in_channels=2, rx_antennas=args.rx, tx_antennas=args.tx, num_classes=8)
            elif args.model == 'resnet50':
                model = load_model(ResNet50ForMIMO, args.load_model, in_channels=2, rx_antennas=args.rx, tx_antennas=args.tx, num_classes=8)
            elif args.model == 'vgg':
                model = load_model(VGGForMIMO, args.load_model, in_channels=2, rx_antennas=args.rx, tx_antennas=args.tx, num_classes=8)
            elif args.model == 'squeezenet':  
                model = load_model(SqueezeNetForMIMO, args.load_model, in_channels=2, rx_antennas=args.rx, tx_antennas=args.tx, num_classes=8)
            else:  
                model = load_model(MobileNetV2ForMIMO, args.load_model, in_channels=2, rx_antennas=args.rx, tx_antennas=args.tx, num_classes=8)
            model = model.to(device)
        
        
        snr_values, ber_values = compute_ber_vs_snr(
            model=model,
            tx_antennas=args.tx,
            rx_antennas=args.rx,
            modulation=args.modulation,
            channel_type=args.channel,
            test_samples=args.test_samples,
            batch_size=args.batch_size,
            device=device
        )
        
        
        plot_ber_vs_snr(snr_values, ber_values, args.model, save_path=ber_save_path)
        
        
        ber_save_file = os.path.join(args.save_dir, f"{config_str}_ber.txt")
        with open(ber_save_file, 'w') as f:
            f.write("SNR,BER\n")
            for snr, ber in zip(snr_values, ber_values):
                f.write(f"{snr},{ber}\n")
        
        print(f"BER vs SNR curve saved to {ber_save_path}")
        print(f"BER values saved to {ber_save_file}")


if __name__ == "__main__":
    main()