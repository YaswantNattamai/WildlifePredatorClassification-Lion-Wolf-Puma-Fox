import numpy as np
import os
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pickle

def simple_preprocess_for_cnn_balanced(
    data_folder,
    output_path="preprocessed_data_balanced",
    target_size=(224, 224),
    balance_mode="undersample"  # choose "undersample" or "oversample"
):
    """
    Preprocesses Mel-spectrogram images and balances the training data.
    
    balance_mode:
        - 'undersample' → reduce all classes to the smallest class count
        - 'oversample'  → increase all classes to the largest class count
    """
    
    images = []
    labels = []
    class_names = ['Fox', 'Lion', 'Puma', 'Wolf']
    
    os.makedirs(output_path, exist_ok=True)
    
    print("=== Loading and preprocessing images ===")
    for class_name in class_names:
        class_path = os.path.join(data_folder, class_name)
        if not os.path.exists(class_path):
            print(f"⚠️ Warning: Directory {class_path} not found, skipping.")
            continue
            
        image_files = glob.glob(os.path.join(class_path, '*.png'))
        print(f"Processing {len(image_files)} images for {class_name}")
        
        for image_path in image_files:
            try:
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image = image.resize(target_size, Image.Resampling.LANCZOS)
                image_array = np.array(image).astype('float32') / 255.0
                images.append(image_array)
                labels.append(class_name)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
    
    images = np.array(images)
    labels = np.array(labels)
    
    print("\nSplitting into train, validation, and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Before balancing → Train samples: {len(y_train)}")
    
    # ===== BALANCING TRAINING DATA =====
    print(f"\nBalancing training data using {balance_mode}...")
    X_train_balanced, y_train_balanced = [], []
    unique_classes, counts = np.unique(y_train, return_counts=True)
    print("Class distribution before balancing:", dict(zip(unique_classes, counts)))
    
    # Separate each class
    class_data = {cls: X_train[y_train == cls] for cls in unique_classes}
    
    if balance_mode == "undersample":
        min_count = min(len(v) for v in class_data.values())
        for cls in unique_classes:
            X_res = resample(class_data[cls], replace=False, n_samples=min_count, random_state=42)
            y_res = np.array([cls] * min_count)
            X_train_balanced.append(X_res)
            y_train_balanced.append(y_res)
    elif balance_mode == "oversample":
        max_count = max(len(v) for v in class_data.values())
        for cls in unique_classes:
            X_res = resample(class_data[cls], replace=True, n_samples=max_count, random_state=42)
            y_res = np.array([cls] * max_count)
            X_train_balanced.append(X_res)
            y_train_balanced.append(y_res)
    else:
        raise ValueError("balance_mode must be either 'undersample' or 'oversample'")
    
    X_train_balanced = np.concatenate(X_train_balanced)
    y_train_balanced = np.concatenate(y_train_balanced)
    
    print("Class distribution after balancing:",
          dict(zip(*np.unique(y_train_balanced, return_counts=True))))
    
    # ===== Save preprocessed data =====
    np.save(os.path.join(output_path, 'X_train.npy'), X_train_balanced)
    np.save(os.path.join(output_path, 'X_val.npy'), X_val)
    np.save(os.path.join(output_path, 'X_test.npy'), X_test)
    
    with open(os.path.join(output_path, 'y_train.pkl'), 'wb') as f:
        pickle.dump(y_train_balanced, f)
    with open(os.path.join(output_path, 'y_val.pkl'), 'wb') as f:
        pickle.dump(y_val, f)
    with open(os.path.join(output_path, 'y_test.pkl'), 'wb') as f:
        pickle.dump(y_test, f)
    with open(os.path.join(output_path, 'class_names.pkl'), 'wb') as f:
        pickle.dump(class_names, f)
    
    print("\n Preprocessing and balancing completed!")
    print(f"Balanced Training set: {X_train_balanced.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Data saved to {output_path}")

# Run simple preprocessing
if __name__ == "__main__":
    simple_preprocess_for_cnn_balanced("clean_mel_images", balance_mode="oversample")
