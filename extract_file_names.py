import os

def find_png_files(directory):
    """Find all PNG files in the given directory and its subdirectories"""
    png_files = []
    
    # Walk through all directories and files
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file is a PNG
            if file.lower().endswith('.png'):
                # Get the relative path
                relative_path = os.path.join(root, file)
                png_files.append(relative_path)
    
    return png_files

def main():
    # Set the directory to search (current directory by default)
    project_directory = "."  # You can change this to a specific path
    
    # Find all PNG files
    png_files = find_png_files(project_directory)
    
    # Sort the files alphabetically
    png_files.sort()
    
    # Write the file names to a text file
    with open("png_files_list.txt", "w") as f:
        for file_path in png_files:
            f.write(f"{file_path}\n")
    
    print(f"Found {len(png_files)} PNG files. List saved to png_files_list.txt")

if __name__ == "__main__":
    main()