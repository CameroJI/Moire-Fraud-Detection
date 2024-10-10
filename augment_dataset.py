import os
import argparse
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def augment_image(image_path, output_dir, augmentor, augment_count=1):
    img = Image.open(image_path)
    
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, 0)

    augmented_images = 0

    for i, aug_img in enumerate(augmentor.flow(img_array, batch_size=1)):
        aug_img = aug_img[0].astype(np.uint8)
        aug_image_path = os.path.join(output_dir, f"aug_{i+1}_{os.path.basename(image_path)}")
        Image.fromarray(aug_img).save(aug_image_path)
        augmented_images += 1
        if i + 1 >= augment_count:
            break
    
    return augmented_images

def process_image(image_path, output_image_path, augmentor, augment_count=1):
    img = Image.open(image_path)
    
    img.save(output_image_path)
    
    return augment_image(image_path, os.path.dirname(output_image_path), augmentor, augment_count=augment_count)

def process_folder(input_dir, output_dir, augment_count=1, max_workers=4):
    for category in ['Ataque', 'Reales']:
        os.makedirs(os.path.join(output_dir, category), exist_ok=True)
    
    augmentor = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],
        channel_shift_range=50,
    )

    for category in ['Ataque', 'Reales']:
        category_path = os.path.join(input_dir, category)
        output_category_path = os.path.join(output_dir, category)

        original_images = 0
        augmented_images_total = 0

        image_files = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(image_files), desc=f"Processing {category}") as pbar:
            futures = []

            for filename in image_files:
                image_path = os.path.join(category_path, filename)
                output_image_path = os.path.join(output_category_path, filename)
                futures.append(executor.submit(process_image, image_path, output_image_path, augmentor, augment_count))

            for future in as_completed(futures):
                try:
                    augmented_images = future.result()
                    augmented_images_total += augmented_images
                    original_images += 1
                except Exception as e:
                    print(f"Error processing image: {e}")
                pbar.update(1)

        print(f"Category '{category}' - Original images: {original_images}, Augmented images: {augmented_images_total}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate augmented images and save originals.")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the input directory with 'Ataque' and 'Reales' subfolders.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory where the images will be saved.")
    parser.add_argument('--augment_count', type=int, default=1, help="Number of augmentations to generate per image.")
    parser.add_argument('--max_workers', type=int, default=4, help="Maximum number of threads to use for parallel processing.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    process_folder(args.input_dir, args.output_dir, augment_count=args.augment_count, max_workers=args.max_workers)