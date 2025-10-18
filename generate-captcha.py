#!/usr/bin/env python3

import os
import numpy
import random
import cv2
import argparse
import shutil
from captcha.image import ImageCaptcha


def create_data_yaml(output_dir, symbols, config_name="data.yaml"):
    """Generate YOLO format data.yaml configuration file"""
    file_path = os.path.join(output_dir, config_name)
    
    with open(file_path, "w") as f:
        # Write paths
        abs_path = os.path.abspath(output_dir)
        f.write(f"path: {abs_path}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"test:\n\n")
        
        # Write class names
        f.write("names:\n")
        for idx, char in enumerate(symbols):
            f.write(f"  {idx}: '{char}'\n")
    
    print(f"✓ Generated data.yaml at: {file_path}")


def split_train_val(output_dir, train_ratio=0.8):
    """Split images into train and val sets"""
    print(f"\nSplitting dataset (train ratio: {train_ratio})...")
    
    # Create train/val directory structure
    train_img_dir = os.path.join(output_dir, "images", "train")
    val_img_dir = os.path.join(output_dir, "images", "val")
    train_label_dir = os.path.join(output_dir, "labels", "train")
    val_label_dir = os.path.join(output_dir, "labels", "val")
    
    for d in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Get all subdirectories (font folders)
    subdirs = [d for d in os.listdir(output_dir) 
               if os.path.isdir(os.path.join(output_dir, d)) 
               and d not in ['images', 'labels']]
    
    total_train = 0
    total_val = 0
    
    for subdir in subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        images = [f for f in os.listdir(subdir_path) if f.endswith('.png')]
        
        # Shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy to train
        for img in train_images:
            src = os.path.join(subdir_path, img)
            dst = os.path.join(train_img_dir, img)
            shutil.copy(src, dst)
            
            # Copy corresponding label
            label_name = img.replace('.png', '.txt')
            label_src = os.path.join(subdir_path, label_name)
            if os.path.exists(label_src):
                shutil.copy(label_src, os.path.join(train_label_dir, label_name))
        
        # Copy to val
        for img in val_images:
            src = os.path.join(subdir_path, img)
            dst = os.path.join(val_img_dir, img)
            shutil.copy(src, dst)
            
            # Copy corresponding label
            label_name = img.replace('.png', '.txt')
            label_src = os.path.join(subdir_path, label_name)
            if os.path.exists(label_src):
                shutil.copy(label_src, os.path.join(val_label_dir, label_name))
        
        total_train += len(train_images)
        total_val += len(val_images)
        print(f"  {subdir}: {len(train_images)} train, {len(val_images)} val")
    
    print(f"\n✓ Split complete: {total_train} train, {total_val} val images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--font_dir', help='path of fonts directory OR individual font files', type=str,required=True)
    parser.add_argument('--mixed', help='Use mixed fonts (multiple fonts in one captcha)', action='store_true')
    parser.add_argument('--width', help='Width of captcha image', type=int, default=192)
    parser.add_argument('--height', help='Height of captcha image', type=int, default=96)
    parser.add_argument('--min_length', help='Minimum length of captchas', type=int, default=4)
    parser.add_argument('--max_length', help='Maximum length of captchas', type=int, default=6)
    parser.add_argument('--count', help='How many captchas to generate per font', type=int, required=True)
    parser.add_argument('--output_dir', help='Where to store the generated captchas', type=str, required=True)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str, required=True)
    args = parser.parse_args()

    # Read symbol sets
    with open(args.symbols, 'r') as f:
        captcha_symbols = f.readline().strip()
    
    print(f"Generating captchas with symbol set: {captcha_symbols}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Read all fonts
    font_names = os.listdir(args.font_dir)
    font_names = sorted(font_names)
    font_paths = [os.path.join(args.font_dir, f) for f in font_names]
    
    print(f"Found {len(font_names)} fonts")

    # If mixed mode, generate with all fonts combined
    if args.mixed:
        print(f"Generating {args.count} captchas with MIXED fonts")
        
        out_dir = os.path.join(args.output_dir, "mixed")
        os.makedirs(out_dir, exist_ok=True)
        
        # Create generator with ALL fonts
        generator = ImageCaptcha(width=args.width, height=args.height, fonts=font_paths)
        
        generator.character_warp_dx = (0.1, 0.5)
        generator.character_warp_dy = (0.2, 0.5)
        generator.character_rotate = (-45, 45)
        
        for i in range(args.count):
            length = random.randint(args.min_length, args.max_length)
            random_str = ''.join([random.choice(captcha_symbols) for _ in range(length)])
            image_path = os.path.join(out_dir, f'{random_str}_{i}.png')
            
            image = generator.generate_image(random_str)
            image_array = numpy.array(image)
            cv2.imwrite(image_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        
        print(f"  ✓ Saved {args.count} mixed-font images to {out_dir}")
    
    else:
        # Generate captchas separately for each font
        for font_name, font_path in zip(font_names, font_paths):
            font_base_name = font_name[:font_name.rfind(".")] if "." in font_name else font_name
            
            # Create subfolder for this font
            out_dir = os.path.join(args.output_dir, font_base_name)
            os.makedirs(out_dir, exist_ok=True)
            
            # Create captcha generator for this font
            generator = ImageCaptcha(width=args.width, height=args.height, fonts=[font_path])
            
            # Optional: customize character warping and rotation
            generator.character_warp_dx = (0.1, 0.5)
            generator.character_warp_dy = (0.2, 0.5)
            generator.character_rotate = (-45, 45)
            
            print(f"Generating {args.count} captchas for font: {font_base_name}")
            
            # Generate captchas with varying lengths
            for i in range(args.count):
                # Random length between min and max
                length = random.randint(args.min_length, args.max_length)
                
                # Generate random string
                random_str = ''.join([random.choice(captcha_symbols) for _ in range(length)])
                
                # Create filename: includes the text for easy verification
                image_path = os.path.join(out_dir, f'{random_str}_{i}.png')
                
                # Generate and save image
                image = generator.generate_image(random_str)
                image_array = numpy.array(image)
                cv2.imwrite(image_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
            
            print(f"  ✓ Saved {args.count} images to {out_dir}")
    
    print(f"\nDone! Generated captchas saved to: {args.output_dir}")


if __name__ == '__main__':
    main()