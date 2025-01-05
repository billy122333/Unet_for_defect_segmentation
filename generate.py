import argparse
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from keras.models import load_model



def resize_and_crop(image, target_width, target_height):
    h, w = image.shape[:2]
    
    # Calculate target aspect ratio and current aspect ratio
    target_aspect = target_width / target_height
    aspect = w / h
    
    # Resize image based on the aspect ratio differences
    if aspect > target_aspect:
        # Current image is wider than target, resize based on height
        new_h = target_height
        new_w = int(aspect * new_h)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # Crop the width
        start_x = (new_w - target_width) // 2
        cropped = resized[:, start_x:start_x + target_width]
    else:
        # Current image is taller than target, resize based on width
        new_w = target_width
        new_h = int(new_w / aspect)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # Crop the height
        start_y = (new_h - target_height) // 2
        cropped = resized[start_y:start_y + target_height, :]

    return cropped

def image_preprocessing(image_list, img_type, mask = False):
    processed_images = []
    filenames = []
    display_count = 1
    for filename in image_list:
        if filename.endswith(img_type):  # Adjust the extension if needed
            img = cv2.imread(filename)
            if img is not None:
                processed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if mask:
                    processed_image[processed_image != 0] = 255

                processed_images.append(processed_image)
                filename = filename.split('/')[-1]
                print(filename)
                filenames.append(filename)

    return processed_images, filenames

     
"""Usage:
# test dent dataset
python generate.py --model_name best_dent.keras --data_folder ./data/dent/testing --result_folder ./result/inference/dent
# test scratch dataset
python generate.py --model_name best_scratch.keras --data_folder ./data/scratch/testing --result_folder ./result/inference/scratch


"""
if __name__ == '__main__':   
    
    args = argparse.ArgumentParser()
    args.add_argument('--model_name', type=str, default='best_scratch.keras')
    args.add_argument('--data_folder', type=str, default='./data/scratch/testing')
    args.add_argument('--result_folder', type=str, default='./result/inference')
    args = args.parse_args()

    # args
    model_name = args.model_name
    test_directory = args.data_folder
    result_folder = args.result_folder
    
    # folder to save the result
    os.makedirs(f'{result_folder}/test_image', exist_ok=True)
    
    if not os.path.exists(f'./models/{model_name}'):
        print(f"Model {model_name} not found")
        exit()
    if not os.path.exists(test_directory):
        print(f"Test directory {test_directory} not found")
        exit()

    #Load previously saved model
    model = load_model(f'./models/{model_name}', compile=False)
            

    test_image_names = glob.glob(f"{test_directory}/*.png")
    test_images, image_names = image_preprocessing(test_image_names, 'png')
    tst_image_dataset = np.array(test_images)
    tst_image_dataset = np.expand_dims(tst_image_dataset, axis=3)
    tst_image_dataset = tst_image_dataset/255.0

    for img, img_name in zip(tst_image_dataset, image_names):
        
        test_img_input=np.expand_dims(img, 0)
        prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)

        plt.figure(figsize=(16, 8))
        plt.subplot(231)
        plt.title('Testing Image')
        plt.imshow(img[:,:,0], cmap='gray')
        plt.subplot(232)
        plt.title('Prediction on test image')
        plt.imshow(prediction, cmap='gray')
        plt.savefig(f'{result_folder}/{img_name}.png')
        plt.close()
