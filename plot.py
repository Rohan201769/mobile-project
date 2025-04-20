
import os
import matplotlib.pyplot as plt
import numpy as np
import re

def create_final_folder():
    """Create the final folder if it doesn't exist"""
    os.makedirs('final', exist_ok=True)

def load_image_paths_from_file(filepath='png_files_list.txt'):
    """
    Load image paths from the text file
    Filter for PNG files and existing paths
    """
    with open(filepath, 'r') as f:
        paths = f.read().splitlines()
    
    
    png_paths = [path for path in paths if path.lower().endswith('.png') and os.path.exists(path)]
    return png_paths

def extract_image_metadata(filepath):
    """
    Extract metadata from filepath
    """
    filename = os.path.basename(filepath)
    
    
    patterns = [
        
        r'(model_comparison|config_comparison|modulation_comparison|channel_comparison)_(?P<config>[\dx]+)_(?P<modulation>BPSK|QPSK)_(?P<channel>ideal|nonideal)',
        
        
        r'(densenet|mobilenetv2|resnet50|vgg|squeezenet)_(?P<config>[\dx]+)_(?P<modulation>BPSK|QPSK)_(?P<channel>ideal|nonideal)',
        
        
        r'(ber_comparison|ber_vs_snr)_(?P<modulation>BPSK|QPSK)_(?P<channel>ideal|nonideal)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return match.groupdict()
    
    return {}

def group_images(image_paths):
    """
    Group images by different criteria
    """
    grouped_images = {
        'Modulation': {'BPSK': [], 'QPSK': []},
        'Models': {
            'DenseNet': [], 
            'MobileNetV2': [], 
            'ResNet50': [], 
            'VGG': [],
            'SqueezeNet': []  
        },
        'Antenna_Configs': {},
        'Channel_Types': {'Ideal': [], 'Nonideal': []},
        'Comparison_Types': {
            'Model_Comparison': [],
            'Config_Comparison': [],
            'Modulation_Comparison': [],
            'Channel_Comparison': [],
            'BER_Comparison': []
        }
    }
    
    for path in image_paths:
        metadata = extract_image_metadata(path)
        
        
        if metadata.get('modulation') == 'BPSK':
            grouped_images['Modulation']['BPSK'].append(path)
        elif metadata.get('modulation') == 'QPSK':
            grouped_images['Modulation']['QPSK'].append(path)
        
        
        model_mapping = {
            'densenet': 'DenseNet',
            'mobilenetv2': 'MobileNetV2',
            'resnet50': 'ResNet50',
            'vgg': 'VGG',
            'squeezenet': 'SqueezeNet'
        }
        
        for model_key, model_name in model_mapping.items():
            if model_key in path.lower():
                grouped_images['Models'][model_name].append(path)
        
        
        config_match = re.search(r'(\dx\d+)', path)
        if config_match:
            config = config_match.group(1)
            if config not in grouped_images['Antenna_Configs']:
                grouped_images['Antenna_Configs'][config] = []
            grouped_images['Antenna_Configs'][config].append(path)
        
        
        if 'ideal' in path.lower():
            grouped_images['Channel_Types']['Ideal'].append(path)
        elif 'nonideal' in path.lower():
            grouped_images['Channel_Types']['Nonideal'].append(path)
        
        
        comparison_types = {
            'Model_Comparison': 'model_comparison',
            'Config_Comparison': 'config_comparison',
            'Modulation_Comparison': 'modulation_comparison',
            'Channel_Comparison': 'channel_comparison',
            'BER_Comparison': 'ber_comparison'
        }
        
        for key, pattern in comparison_types.items():
            if pattern in path.lower():
                grouped_images['Comparison_Types'][key].append(path)
    
    return grouped_images

def create_combined_plots(grouped_images):
    """
    Create combined plots for different groupings
    """
    create_final_folder()
    
    def plot_image_grid(images, title, filename):
        """Plot images in a grid"""
        if not images:
            return
        
        
        n_images = len(images)
        rows = int(np.ceil(np.sqrt(n_images)))
        cols = int(np.ceil(n_images / rows))
        
        plt.figure(figsize=(15, 10))
        plt.suptitle(title, fontsize=16)
        
        for i, img_path in enumerate(images):
            plt.subplot(rows, cols, i+1)
            img = plt.imread(img_path)
            plt.imshow(img)
            plt.axis('off')
            
            
            plt.title(os.path.basename(img_path), fontsize=8)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'final/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
    
    
    for group_name, group_data in grouped_images.items():
        for subgroup_name, images in group_data.items():
            if images:
                filename = f'{group_name}_{subgroup_name}_combined.png'
                plot_image_grid(
                    images, 
                    f'{group_name}: {subgroup_name}', 
                    filename
                )
                print(f"Created combined plot: {filename}")

def main():
    
    image_paths = load_image_paths_from_file()
    
    
    grouped_images = group_images(image_paths)
    
    
    create_combined_plots(grouped_images)

if __name__ == "__main__":
    main()
