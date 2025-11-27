import os
from PIL import Image
import glob

def crop_spectrogram_images(input_folder, output_folder, crop_box=None):
    """
    Crop mel spectrogram images to remove axes and labels
    
    Parameters:
    input_folder: Path to folder containing subfolders with images
    output_folder: Path where cropped images will be saved
    crop_box: Tuple (left, upper, right, lower) for cropping. If None, uses default.
    """
    
    # Default crop box - you may need to adjust these values based on your images
    if crop_box is None:
        crop_box = (50, 30, 200, 120)  # (left, upper, right, lower)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each animal folder
    animal_folders = ['Fox', 'Lion', 'Puma', 'Wolf']
    
    for animal in animal_folders:
        animal_input_path = os.path.join(input_folder, animal)
        animal_output_path = os.path.join(output_folder, animal)
        os.makedirs(animal_output_path, exist_ok=True)
        
        # Get all image files in the animal folder
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        image_files = []
        
        for extension in image_extensions:
            image_files.extend(glob.glob(os.path.join(animal_input_path, extension)))
        
        print(f"Processing {len(image_files)} images in {animal} folder...")
        
        for image_path in image_files:
            try:
                # Open image
                with Image.open(image_path) as img:
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Crop the image
                    cropped_img = img.crop(crop_box)
                    
                    # Save cropped image
                    filename = os.path.basename(image_path)
                    output_path = os.path.join(animal_output_path, filename)
                    cropped_img.save(output_path)
                    
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
        
        print(f"Completed processing {animal} folder")

# Alternative function to automatically detect crop boundaries
def auto_crop_spectrogram_images(input_folder, output_folder, margin=10):
    """
    Automatically crop images by detecting content boundaries
    """
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each animal folder
    animal_folders = ['fox', 'lion', 'puma', 'wolf']
    
    for animal in animal_folders:
        animal_input_path = os.path.join(input_folder, animal)
        animal_output_path = os.path.join(output_folder, animal)
        os.makedirs(animal_output_path, exist_ok=True)
        
        # Get all image files
        image_files = glob.glob(os.path.join(animal_input_path, '*.png'))
        
        print(f"Processing {len(image_files)} images in {animal} folder...")
        
        for image_path in image_files:
            try:
                with Image.open(image_path) as img:
                    # Convert to grayscale for easier processing
                    grayscale = img.convert('L')
                    
                    # Get image dimensions
                    width, height = grayscale.size
                    
                    # Find content boundaries by detecting non-white pixels
                    # Adjust threshold as needed (255 is white)
                    threshold = 250
                    
                    # Find left boundary
                    left = 0
                    for x in range(width):
                        column_has_content = any(grayscale.getpixel((x, y)) < threshold 
                                               for y in range(height))
                        if column_has_content:
                            left = max(0, x - margin)
                            break
                    
                    # Find right boundary
                    right = width
                    for x in range(width-1, -1, -1):
                        column_has_content = any(grayscale.getpixel((x, y)) < threshold 
                                               for y in range(height))
                        if column_has_content:
                            right = min(width, x + margin + 1)
                            break
                    
                    # Find top boundary
                    top = 0
                    for y in range(height):
                        row_has_content = any(grayscale.getpixel((x, y)) < threshold 
                                            for x in range(width))
                        if row_has_content:
                            top = max(0, y - margin)
                            break
                    
                    # Find bottom boundary
                    bottom = height
                    for y in range(height-1, -1, -1):
                        row_has_content = any(grayscale.getpixel((x, y)) < threshold 
                                            for x in range(width))
                        if row_has_content:
                            bottom = min(height, y + margin + 1)
                            break
                    
                    # Crop the image
                    crop_box = (left, top, right, bottom)
                    cropped_img = img.crop(crop_box)
                    
                    # Save cropped image
                    filename = os.path.basename(image_path)
                    output_path = os.path.join(animal_output_path, filename)
                    cropped_img.save(output_path)
                    
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
        
        print(f"Completed processing {animal} folder")

# Usage examples:
if __name__ == "__main__":
    # Define your paths
    input_folder = "mel_spectrogram_images"  # Update this path
    output_folder = "clean_mel_images"   # Update this path
    
    # Method 1: Manual cropping with specific coordinates
    # Adjust these coordinates based on your image dimensions
    # (left, upper, right, lower)
    manual_crop_box = (80, 50, 450, 300)  # You'll need to adjust these values
    
    crop_spectrogram_images(input_folder, output_folder, crop_box=manual_crop_box)
    
    # Method 2: Automatic cropping (try this first)
    # auto_crop_spectrogram_images(input_folder, output_folder, margin=5)