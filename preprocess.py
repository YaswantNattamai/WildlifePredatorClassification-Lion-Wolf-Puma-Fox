import tensorflow as tf
import numpy as np
import os
import cv2
from tqdm import tqdm
import shutil

def create_resnet_preprocessed_dataset(source_dir, target_dir, target_size=(224, 224)):
    """
    Create separate preprocessed images for ResNet model
    
    Args:
        source_dir: Root directory with original animal folders
        target_dir: Root directory where preprocessed images will be saved
        target_size: Target size for ResNet (224x224)
    """
    
    animal_classes = ['Lion', 'Wolf', 'Fox', 'Puma']
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    print("Starting image preprocessing for ResNet...")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print(f"Target size: {target_size}\n")
    
    # Statistics
    total_processed = 0
    class_stats = {}
    
    for animal in animal_classes:
        source_animal_dir = os.path.join(source_dir, animal)
        target_animal_dir = os.path.join(target_dir, animal)
        
        # Skip if source directory doesn't exist
        if not os.path.exists(source_animal_dir):
            print(f"⚠️  Warning: {source_animal_dir} not found, skipping...")
            continue
            
        # Create target animal directory
        os.makedirs(target_animal_dir, exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(source_animal_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"🦁 Processing {animal}: {len(image_files)} images...")
        
        processed_count = 0
        for img_file in tqdm(image_files, desc=animal):
            try:
                # Source and target paths
                source_path = os.path.join(source_animal_dir, img_file)
                target_path = os.path.join(target_animal_dir, f"resnet_preprocessed_{img_file}")
                
                # Preprocess and save
                preprocess_and_save_image(source_path, target_path, target_size)
                processed_count += 1
                total_processed += 1
                
            except Exception as e:
                print(f"❌ Error processing {img_file}: {e}")
        
        class_stats[animal] = processed_count
        print(f"✅ {animal}: {processed_count}/{len(image_files)} images processed\n")
    
    # Print summary
    print("\n" + "="*50)
    print("📊 PREPROCESSING SUMMARY")
    print("="*50)
    for animal, count in class_stats.items():
        print(f"  {animal}: {count} images")
    print(f"\n🎯 Total: {total_processed} images preprocessed")
    print(f"📁 Output directory: {target_dir}")
    
    return total_processed, class_stats

def preprocess_and_save_image(source_path, target_path, target_size=(224, 224)):
    """
    Preprocess single image for ResNet and save to disk
    """
    # 1. Load image
    image = cv2.imread(source_path)
    if image is None:
        raise ValueError(f"Could not load image: {source_path}")
    
    # 2. Convert BGR to RGB (OpenCV loads as BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 3. Resize to target size
    image = cv2.resize(image, target_size)
    
    # 4. Convert to float32
    image = image.astype(np.float32)
    
    # 5. Apply ResNet-specific preprocessing:
    #    - Zero-center by mean pixel (ImageNet statistics)
    #    - Reverse channel order (RGB to BGR)
    mean = [103.939, 116.779, 123.68]  # ImageNet BGR means
    
    # Split into RGB channels
    red = image[:, :, 0]
    green = image[:, :, 1] 
    blue = image[:, :, 2]
    
    # Subtract means and convert to BGR order
    bgr_image = np.zeros_like(image)
    bgr_image[:, :, 0] = blue - mean[0]    # B channel
    bgr_image[:, :, 1] = green - mean[1]   # G channel
    bgr_image[:, :, 2] = red - mean[2]     # R channel
    
    # 6. Save as numpy array or image
    save_as_numpy(bgr_image, target_path)
    # Alternatively, you can save as image:
    # save_as_image(bgr_image, target_path)

def save_as_numpy(preprocessed_image, target_path):
    """
    Save preprocessed image as numpy file (.npy)
    """
    numpy_path = target_path.replace('.png', '.npy').replace('.jpg', '.npy').replace('.jpeg', '.npy')
    np.save(numpy_path, preprocessed_image)

def save_as_image(preprocessed_image, target_path):
    """
    Save preprocessed image as visualizable image (denormalized)
    Note: This is for inspection, not for model input
    """
    # Denormalize for visualization
    image_vis = denormalize_resnet(preprocessed_image)
    image_vis = np.clip(image_vis, 0, 255).astype(np.uint8)
    cv2.imwrite(target_path, cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))

def denormalize_resnet(preprocessed_image):
    """
    Denormalize ResNet preprocessed image for visualization
    """
    mean = [103.939, 116.779, 123.68]
    
    # Convert back from BGR to RGB
    blue = preprocessed_image[:, :, 0] + mean[0]
    green = preprocessed_image[:, :, 1] + mean[1]
    red = preprocessed_image[:, :, 2] + mean[2]
    
    rgb_image = np.stack([red, green, blue], axis=-1)
    return rgb_image

def create_dataset_from_preprocessed(preprocessed_dir, batch_size=32, validation_split=0.2):
    """
    Create tf.data dataset from preprocessed images
    """
    def load_preprocessed_numpy(image_path, label):
        # Load the preprocessed numpy file
        image = np.load(image_path.numpy().decode('utf-8'))
        return image, label
    
    # Collect all numpy files and labels
    numpy_paths = []
    labels = []
    animal_classes = ['Lion', 'Wolf', 'Fox', 'Puma']
    
    for label, animal in enumerate(animal_classes):
        animal_dir = os.path.join(preprocessed_dir, animal)
        if os.path.exists(animal_dir):
            npy_files = [os.path.join(animal_dir, f) for f in os.listdir(animal_dir) 
                        if f.endswith('.npy')]
            numpy_paths.extend(npy_files)
            labels.extend([label] * len(npy_files))
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((numpy_paths, labels))
    dataset = dataset.map(
        lambda x, y: tf.py_function(load_preprocessed_numpy, [x, y], [tf.float32, tf.int32]),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Split into train and validation
    dataset_size = len(numpy_paths)
    train_size = int(dataset_size * (1 - validation_split))
    
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    
    # Batch and optimize
    train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset, animal_classes

# Usage example
if __name__ == "__main__":
    # Set your directories
    SOURCE_DIR = "clean_mel_images"
    TARGET_DIR = "resnet_preprocessed_dataset"
    
    # Create preprocessed dataset
    total_images, stats = create_resnet_preprocessed_dataset(SOURCE_DIR, TARGET_DIR)
    
    print(f"\n🎉 Preprocessing complete!")
    print(f"📁 Your preprocessed images are saved in: {TARGET_DIR}")
    print(f"📊 You can now use these with your ResNet model!")
    
    # Optional: Create dataset from preprocessed files
    # train_ds, val_ds, classes = create_dataset_from_preprocessed(TARGET_DIR)