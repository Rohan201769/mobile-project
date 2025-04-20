import os

def find_png_files(directory):
    """Find all PNG files in the given directory and its subdirectories"""
    png_files = []
    
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            
            if file.lower().endswith('.png'):
                
                relative_path = os.path.join(root, file)
                png_files.append(relative_path)
    
    return png_files

def main():
    
    project_directory = "."  
    
    
    png_files = find_png_files(project_directory)
    
    
    png_files.sort()
    
    
    with open("png_files_list.txt", "w") as f:
        for file_path in png_files:
            f.write(f"{file_path}\n")
    
    print(f"Found {len(png_files)} PNG files. List saved to png_files_list.txt")

if __name__ == "__main__":
    main()