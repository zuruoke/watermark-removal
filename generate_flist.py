import os
import random

MASK_DIR = './utils/istock/landscape/mask.png'
OUTPUT_DIR = './output'

def generate_flist(input_image_dir, output_txt, mask_dir):
    
    # Create output_txt
    if not os.path.exists(output_txt):
        os.makedirs(output_txt)
    output_txt = os.path.join(output_txt, 'flist.txt')
    
    # Loop through input_image_dir
    with open(output_txt, 'w') as f:
        for image in os.listdir(input_image_dir):
            if image.endswith('.jpg'):
                image_path = os.path.join(input_image_dir, image)
                
                f.write(image_path + ' ' + MASK_DIR + ' ' + os.path.join(OUTPUT_DIR, image) + '\n')

input_image_dir = './images'
generate_flist(input_image_dir, './', './utils/istock/landscape')