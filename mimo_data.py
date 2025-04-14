import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MIMODataset(Dataset):
    def __init__(self, num_samples, tx_antennas, rx_antennas, modulation='BPSK', Eb_N0_dB=range(0, 9), channel_type='ideal'):
        """
        Dataset for MIMO communication system
        
        Args:
            num_samples: Number of samples to generate
            tx_antennas: Number of transmit antennas
            rx_antennas: Number of receive antennas
            modulation: 'BPSK' or 'QPSK'
            Eb_N0_dB: Range of Eb/N0 values in dB
            channel_type: 'ideal' or 'nonideal'
        """
        self.num_samples = num_samples
        self.tx_antennas = tx_antennas
        self.rx_antennas = rx_antennas
        self.modulation = modulation
        self.Eb_N0_dB = Eb_N0_dB if isinstance(Eb_N0_dB, list) else list(Eb_N0_dB)
        self.channel_type = channel_type
        
        # Information bit stream length (as per paper)
        self.info_bits_len = 8
        
        # Generate data
        self.X, self.y = self._generate_data()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def _generate_data(self):
        """Generate input-output pairs for MIMO communication"""
        X = []  # IQ components of received signal
        y = []  # Original information bits
        
        for _ in range(self.num_samples):
            # 1. Generate random information bits
            info_bits = np.random.randint(0, 2, self.info_bits_len)
            
            # 2. Apply channel coding (Hamming (7,4) code)
            coded_bits = self._apply_hamming_coding(info_bits)
            
            # 3. Modulation
            if self.modulation == 'BPSK':
                symbols = 2 * coded_bits - 1  # Map 0 -> -1, 1 -> 1
            elif self.modulation == 'QPSK':
                # Group bits in pairs for QPSK
                bits_reshaped = coded_bits.reshape(-1, 2)
                symbols_real = 2 * bits_reshaped[:, 0] - 1
                symbols_imag = 2 * bits_reshaped[:, 1] - 1
                symbols = symbols_real + 1j * symbols_imag
            
            # 4. Space-Time Block Coding
            stbc_symbols = self._apply_stbc(symbols)
            
            # 5. Apply channel effect
            received_signal = self._apply_channel(stbc_symbols)
            
            # 6. Add noise
            Eb_N0_dB = np.random.choice(self.Eb_N0_dB)
            noisy_signal = self._add_noise(received_signal, Eb_N0_dB)
            
            # 7. Format the input for the network
            # Stack real and imaginary parts as per the paper
            signal_real = np.real(noisy_signal).flatten()
            signal_imag = np.imag(noisy_signal).flatten()
            input_data = np.vstack([signal_real, signal_imag]).T
            
            X.append(input_data)
            y.append(info_bits)
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def _apply_hamming_coding(self, info_bits):
        """Apply (7,4) Hamming coding to information bits"""
        # Simple implementation of (7,4) Hamming code
        # In real-world, use a library like numpy for this
        result = []
        
        # Process bits in groups of 4 (as it's a (7,4) code)
        for i in range(0, len(info_bits), 4):
            block = info_bits[i:i+4]
            if len(block) < 4:  # Pad the last block if needed
                block = np.pad(block, (0, 4 - len(block)))
            
            # Generate parity bits (simple implementation)
            p1 = (block[0] + block[1] + block[3]) % 2
            p2 = (block[0] + block[2] + block[3]) % 2
            p3 = (block[1] + block[2] + block[3]) % 2
            
            # Combine information and parity bits
            coded_block = np.concatenate([block, [p1, p2, p3]])
            result.extend(coded_block)
            
        return np.array(result)
    
    def _apply_stbc(self, symbols):
        """Apply Space-Time Block Coding"""
        if self.tx_antennas == 2:
            # Implement Alamouti scheme as per the paper
            stbc_matrix = np.zeros((2, 2), dtype=complex)
            for i in range(0, len(symbols), 2):
                if i+1 < len(symbols):
                    s1, s2 = symbols[i], symbols[i+1]
                    stbc_matrix[0, 0] = s1
                    stbc_matrix[0, 1] = -np.conj(s2)
                    stbc_matrix[1, 0] = s2
                    stbc_matrix[1, 1] = np.conj(s1)
            return stbc_matrix
        
        elif self.tx_antennas == 3:
            # Implement 3 antenna STBC as per the paper
            stbc_matrix = np.zeros((3, 8), dtype=complex)
            for i in range(0, len(symbols), 4):
                if i+3 < len(symbols):
                    s1, s2, s3, s4 = symbols[i:i+4]
                    # First row
                    stbc_matrix[0, 0] = s1
                    stbc_matrix[0, 1] = s2
                    stbc_matrix[0, 2] = -s3
                    stbc_matrix[0, 3] = -s4
                    stbc_matrix[0, 4] = np.conj(s1)
                    stbc_matrix[0, 5] = -np.conj(s2)
                    stbc_matrix[0, 6] = -np.conj(s3)
                    stbc_matrix[0, 7] = -np.conj(s4)
                    # Second row
                    stbc_matrix[1, 0] = s2
                    stbc_matrix[1, 1] = s1
                    stbc_matrix[1, 2] = s4
                    stbc_matrix[1, 3] = -s3
                    stbc_matrix[1, 4] = np.conj(s2)
                    stbc_matrix[1, 5] = np.conj(s1)
                    stbc_matrix[1, 6] = np.conj(s4)
                    stbc_matrix[1, 7] = -np.conj(s3)
                    # Third row
                    stbc_matrix[2, 0] = s3
                    stbc_matrix[2, 1] = -s4
                    stbc_matrix[2, 2] = s1
                    stbc_matrix[2, 3] = s2
                    stbc_matrix[2, 4] = -np.conj(s3)
                    stbc_matrix[2, 5] = -np.conj(s4)
                    stbc_matrix[2, 6] = np.conj(s1)
                    stbc_matrix[2, 7] = np.conj(s2)
            return stbc_matrix
        
        elif self.tx_antennas == 4:
            # Implement 4 antenna STBC as per the paper
            stbc_matrix = np.zeros((4, 4), dtype=complex)
            for i in range(0, len(symbols), 4):
                if i+3 < len(symbols):
                    s1, s2, s3, s4 = symbols[i:i+4]
                    # First row
                    stbc_matrix[0, 0] = s1
                    stbc_matrix[0, 1] = s2
                    stbc_matrix[0, 2] = s3
                    stbc_matrix[0, 3] = np.conj(s4)
                    # Second row
                    stbc_matrix[1, 0] = -np.conj(s2)
                    stbc_matrix[1, 1] = np.conj(s1)
                    stbc_matrix[1, 2] = -np.conj(s4)
                    stbc_matrix[1, 3] = np.conj(s3)
                    # Third row
                    stbc_matrix[2, 0] = -np.conj(s3)
                    stbc_matrix[2, 1] = s4
                    stbc_matrix[2, 2] = np.conj(s1)
                    stbc_matrix[2, 3] = -np.conj(s2)
                    # Fourth row
                    stbc_matrix[3, 0] = np.conj(s4)
                    stbc_matrix[3, 1] = -s3
                    stbc_matrix[3, 2] = s2
                    stbc_matrix[3, 3] = s1
            return stbc_matrix
    
    def _apply_channel(self, stbc_symbols):
        """Apply channel effects to the signal"""
        if self.channel_type == 'ideal':
            # Ideal channel: Fixed channel coefficients (identity matrix)
            H = np.eye(self.rx_antennas, self.tx_antennas)
        else:
            # Non-ideal channel: Nakagami-m fading with m=1 (Rayleigh fading)
            # Generate complex Gaussian random variables
            H_real = np.random.normal(0, 1/np.sqrt(2), (self.rx_antennas, self.tx_antennas))
            H_imag = np.random.normal(0, 1/np.sqrt(2), (self.rx_antennas, self.tx_antennas))
            H = H_real + 1j * H_imag
            
            # Add shadow fading component
            # In the paper, variance is 2.1 dB
            shadow_fading = np.random.normal(0, np.sqrt(2.1), (self.rx_antennas, self.tx_antennas))
            # Convert from dB to linear
            shadow_fading_linear = 10 ** (shadow_fading / 10)
            H = H * np.sqrt(shadow_fading_linear)
        
        # Apply channel to the STBC symbols
        # For simplicity, we'll use matrix multiplication
        # In reality, this would involve more complex convolution for frequency-selective channels
        return np.dot(H, stbc_symbols)
    
    def _add_noise(self, signal, Eb_N0_dB):
        """Add AWGN noise to the signal"""
        # Calculate noise power based on Eb/N0
        Eb_N0_linear = 10 ** (Eb_N0_dB / 10)
        # Calculate Eb (energy per bit)
        # For BPSK, Eb = 1
        Eb = 1.0
        # Calculate N0 (noise spectral density)
        N0 = Eb / Eb_N0_linear
        # Generate complex Gaussian noise
        noise_real = np.random.normal(0, np.sqrt(N0/2), signal.shape)
        noise_imag = np.random.normal(0, np.sqrt(N0/2), signal.shape)
        noise = noise_real + 1j * noise_imag
        return signal + noise

def create_mimo_dataloaders(tx_antennas, rx_antennas, modulation='BPSK', 
                          train_samples=40000, val_samples=20000, test_samples=40000,
                          batch_size=256, Eb_N0_dB_train=range(0, 9), 
                          Eb_N0_dB_test=range(0, 9), channel_type='nonideal'):
    """
    Create train, validation, and test dataloaders for MIMO communication
    
    Args:
        tx_antennas: Number of transmit antennas
        rx_antennas: Number of receive antennas
        modulation: 'BPSK' or 'QPSK'
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
        batch_size: Batch size for training
        Eb_N0_dB_train: Range of Eb/N0 values in dB for training
        Eb_N0_dB_test: Range of Eb/N0 values in dB for testing
        channel_type: 'ideal' or 'nonideal'
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = MIMODataset(train_samples, tx_antennas, rx_antennas, 
                               modulation, Eb_N0_dB_train, channel_type)
    val_dataset = MIMODataset(val_samples, tx_antennas, rx_antennas, 
                             modulation, Eb_N0_dB_train, channel_type)
    test_dataset = MIMODataset(test_samples, tx_antennas, rx_antennas, 
                              modulation, Eb_N0_dB_test, channel_type)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader