import argparse
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Conv2D, MaxPool2D, Activation, Concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda, Add, Multiply
from keras.models import load_model
from keras.models import Model
from keras.optimizers import Adam, AdamW

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import normalize


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
    display_count = 1
    for filename in image_list:
        if filename.endswith(img_type):  # Adjust the extension if needed
            img = cv2.imread(filename)
            if img is not None:
                processed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if mask:
                    processed_image[processed_image != 0] = 255

                processed_images.append(processed_image)

                if display_count <= 3:
                    plt.figure(figsize=(16, 16))  # Adjust the figure size as needed
                    plt.subplot(2,5,display_count)
                    plt.imshow(processed_image, cmap='gray')  # Use cmap='gray' to display grayscale images
                    plt.title(f'Preprocessed Image {display_count+1}')
                    display_count += 1
    plt.show()
    return processed_images

def binarize_mask(mask):
    _, binary_mask = cv2.threshold(mask, thresh=0.5, maxval=1, type=cv2.THRESH_BINARY)
    binary_mask = binary_mask.reshape(mask.shape)  # 保持原始形状
    return binary_mask

def adjust_brightness(image, brightness_factor):
    return np.clip(image * brightness_factor, 0, 1)

def combine_generators(image_gen, mask_gen, brightness_factor_range=(0.8, 1.2)):
    while True:
        x = next(image_gen)
        y = next(mask_gen)
        # random brightness adjustment
        brightness_factor = np.random.uniform(brightness_factor_range[0], brightness_factor_range[1])

        # adjust brightness for each image in the batch
        x = np.array([adjust_brightness(img, brightness_factor) for img in x])
        yield x, y
     
     
"""Usage:
# train dent dataset
python Unet.py --data_folder ./data/dent --model_name best_dent.keras
# train scratch dataset
python Unet.py --data_folder ./data/scratch --model_name best_scratch.keras 

# test dent dataset
python Unet.py --model_name best_dent.keras --TRAIN False
# test scratch dataset
python Unet.py --model_name best_scratch.keras --TRAIN False

"""
if __name__ == '__main__':   
    
    args = argparse.ArgumentParser()
    args.add_argument('--data_folder', type=str, default='./data/scratch')
    args.add_argument('--epochs', type=int, default=800)
    args.add_argument('--batch_size', type=int, default=4)
    args.add_argument('--model_name', type=str, default='best_scratch.keras')
    args.add_argument('--DEBUG', type=bool, default=False)
    args.add_argument('--TRAIN', type=bool, default=True)
    args = args.parse_args()
    
    # data_path
    data_folder = args.data_folder
    image_directory = f'{data_folder}/img'
    mask_directory = f'{data_folder}/mask'
    test_directory = f'{data_folder}/testing'
    model_name = args.model_name

    # config
    epochs = args.epochs
    batch_size = args.batch_size
    DEBUG = args.DEBUG
    TRAIN = args.TRAIN

    # folder to save the result
    os.makedirs('result/training_process', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('result/stastics', exist_ok=True)
    os.makedirs('result/result/X_test', exist_ok=True)
    os.makedirs('result/result/test_image', exist_ok=True)

    # Sort the image names to correspond the mask names
    defect_image_names = glob.glob(f"{image_directory}/*.png")
    defect_image_names.sort()

    # get preprocess image list
    defect_images = image_preprocessing(defect_image_names, 'png')
    defect_image_dataset = np.array(defect_images)
    defect_image_dataset = np.expand_dims(defect_image_dataset, axis=3)

    if DEBUG:
        print("defect_image: ")
        print(defect_image_names)
        print(defect_image_dataset.shape)
        print("=====================================")
        
    # Sort the mask names to correspond the image names
    mask_image_names = glob.glob(f"{mask_directory}/*.png")
    mask_image_names.sort()

    # get preprocess mask list
    masked_images = image_preprocessing(mask_image_names, 'png', mask = True)
    masked_images_dataset = np.array(masked_images)
    masked_images_dataset = np.expand_dims(masked_images_dataset, axis=3)
    if DEBUG:
        print("mask_image: ")
        print(mask_image_names)
        print(masked_images_dataset.shape)
        print("=====================================")
        
    # check the shape of the image and mask
    print("Image data shape is: ", defect_image_dataset.shape)
    print("Mask data shape is: ", masked_images_dataset.shape)
    print("Max pixel value in image is: ", defect_image_dataset.max())
    print("Labels in the mask are : ", np.unique(masked_images_dataset))


    #Normalize images to 0 and 1 range
    image_dataset = defect_image_dataset /255.  
    #Do not normalize masks, just rescale to 0 to 1.
    mask_dataset = masked_images_dataset /255.  

    print("Image dataset range:", image_dataset.min(), image_dataset.max())
    print("Labels in the mask are : ", np.unique(mask_dataset))

    X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.2, random_state = 30, shuffle=True)

    if DEBUG:
        print("X_test.shape:", X_test.shape)

    # check if the image and mask are correctly matched 
    image_number = random.randint(0, len(X_train)-1)
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(X_train[image_number,:,:,0], cmap='gray')
    plt.subplot(122)
    plt.imshow(y_train[image_number,:,:,0], cmap='gray')
    plt.savefig('result/training_process/init_image_mask.png')
    plt.close()

    # arguments for data augmentation
    data_gen_args = dict(rotation_range=90.,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        shear_range=0.5,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        vertical_flip=True,
                        fill_mode='reflect')

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args, preprocessing_function=binarize_mask)

    # data generator
    seed = 1
    image_datagen.fit(X_train, augment=True, seed=seed)
    mask_datagen.fit(y_train, augment=True, seed=seed)

    image_generator = image_datagen.flow(X_train, batch_size=4, seed=seed)
    mask_generator = mask_datagen.flow(y_train, batch_size=4, seed=seed)
    train_generator = combine_generators(image_generator, mask_generator)

    if DEBUG:
        print("Labels in the mask are : ", np.unique(y_train))

    # model definition
    def attention_gate(f_g, f_l, inter_channels):
        # Gating signal processing
        g = Conv2D(inter_channels, (1, 1), padding='same', activation='relu')(f_g)
        # Skip connection processing
        x = Conv2D(inter_channels, (1, 1), padding='same', activation='relu')(f_l)
        # Combining the gating and skip connection
        combined = Add()([g, x])
        combined = Activation('relu')(combined)
        # Output attention coefficients
        psi = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(combined)
        # Applying attention coefficients
        return Multiply()([f_l, psi])

    def conv_block(input, num_filters):
        x = Conv2D(num_filters, 3, padding="same")(input)
        x = BatchNormalization()(x)   #Not in the original network. 
        x = Activation("relu")(x)

        x = Conv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x)  #Not in the original network
        x = Activation("relu")(x)

        return x

    def encoder_block(input, num_filters):
        x = conv_block(input, num_filters)
        p = MaxPool2D((2, 2))(x)
        return x, p   

    def decoder_block(input, skip_features, num_filters):
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
        
        # Apply attention gate on skip connection features
        attention_features = attention_gate(x, skip_features, num_filters)
        x = Concatenate()([x, attention_features])
        x = conv_block(x, num_filters)
        return x

    def build_attention_unet(input_shape, n_classes):
        inputs = Input(input_shape)

        # Encoder
        s1, p1 = encoder_block(inputs, 16)
        s2, p2 = encoder_block(p1, 32)
        s3, p3 = encoder_block(p2, 64)
        s4, p4 = encoder_block(p3, 128)
        # s5, p5 = encoder_block(p4, 256)

        # Bridge
        b1 = conv_block(p4, 256)

        # Decoder with 
        # d5 = decoder_block(b1, s5, 256)
        d1 = decoder_block(b1, s4, 128)
        d2 = decoder_block(d1, s3, 64)
        d3 = decoder_block(d2, s2, 32)
        d4 = decoder_block(d3, s1, 16)

        if n_classes == 1:  # Binary segmentation
            activation = 'sigmoid'
        else:
            activation = 'softmax'

        outputs = Conv2D(n_classes, 1, padding="same", activation=activation)(d4)
        model = Model(inputs, outputs, name="Attention_U-Net")
        return model

    # loss function1
    def weighted_binary_crossentropy(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        weights = tf.where(tf.equal(y_true, 1), 10., 1.)
        bce = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        weighted_bce = tf.reduce_mean(bce * weights)
        return weighted_bce

    # loss function2
    # current code is using dice loss
    def dice_loss(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        intersection = tf.reduce_sum(y_true * y_pred)
        dice_coeff = (2 * intersection + epsilon) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + epsilon)
        return 1 - dice_coeff

    # input shape
    IMG_HEIGHT = image_dataset.shape[1]
    IMG_WIDTH  = image_dataset.shape[2]
    IMG_CHANNELS = image_dataset.shape[3]

    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    model = build_attention_unet(input_shape, n_classes=1)
    model.compile(optimizer=Adam(learning_rate = 1e-4), loss=dice_loss, metrics=['accuracy'])
    model.summary()

    class VisualizationCallback(Callback):
        def __init__(self, X_train, y_train, num_samples=2):
            self.X_train = X_train
            self.y_train = y_train
            self.num_samples = num_samples

        def on_epoch_end(self, epoch, logs=None):
            predictions = self.model.predict(self.X_train[:self.num_samples])
            y_pred_thresholded = (predictions > 0.5).astype(np.uint8)

            for i in range(self.num_samples):
                if epoch%50==0:
                    plt.figure(figsize=(12, 4))

                    # 显示原始图像
                    plt.subplot(1, 3, 1)
                    plt.title('Original Image')
                    plt.imshow(self.X_train[i].squeeze(), cmap='gray')

                    # 显示真实掩码
                    plt.subplot(1, 3, 2)
                    plt.title('Ground Truth')
                    plt.imshow(self.y_train[i].squeeze(), cmap='gray')

                    # 显示预测的二值化图像
                    plt.subplot(1, 3, 3)
                    plt.title('Predicted Binary Image')
                    plt.imshow(y_pred_thresholded[i].squeeze(), cmap='gray')

                    plt.savefig(f'result/training_process/epoch_{epoch}_sample_{i}.png')
                    plt.close()

    class IoUCallback(Callback):
        def __init__(self, X_test, y_test, threshold=0.5):
            super().__init__()
            self.X_test = X_test
            self.y_test = y_test
            self.threshold = threshold
        
        def calculate_iou(self, pred_mask, true_mask):
            intersection = np.logical_and(pred_mask, true_mask).sum()
            union = np.logical_or(pred_mask, true_mask).sum()
            iou = intersection / union if union != 0 else 0
            return iou
        
        def on_epoch_end(self, epoch, logs=None):
            pred_masks = self.model.predict(self.X_test)
            pred_masks = (pred_masks > self.threshold).astype(np.uint8)
            
            ious = []
            for pred_mask, true_mask in zip(pred_masks, self.y_test):
                iou = self.calculate_iou(pred_mask, true_mask)
                ious.append(iou)
            
            mean_iou = np.mean(ious)
            print(f"Epoch {epoch+1}: Mean IoU on validation data = {mean_iou:.4f}")
            logs['val_iou'] = mean_iou  # Optionally log it for history

    # my callbacks
    visualization_callback = VisualizationCallback(X_train, y_train)
    iou_callback = IoUCallback(X_test, y_test)

    # callbacks
    earlyStopping = EarlyStopping(monitor='val_loss', patience=150, verbose=0, mode='min')
    mcp_save = ModelCheckpoint(f'./models/{model_name}', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, verbose=1, epsilon=1e-4, mode='min')

    if TRAIN:
        history = model.fit(train_generator,
                            steps_per_epoch=len(X_train) // batch_size,
                            epochs=epochs,
                            validation_data=(X_test, y_test),
                            callbacks=[earlyStopping, mcp_save, reduce_lr_loss, visualization_callback, iou_callback])


        # get the history of the training process
        epochs = range(1, len(history.history['val_iou']) + 1)
        val_iou = history.history['val_iou']

        # plot the validation IoU over epochs
        plt.figure()
        plt.plot(epochs, val_iou, 'b', label='Validation IoU')
        plt.title('Validation IoU over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('IoU')
        plt.legend()
        plt.grid(True)
        plt.savefig('result/stastics/validation_iou.png')
        plt.close()


        #plot the training and validation accuracy and loss at each epoch
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.figure()
        plt.plot(epochs, loss, 'y', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('result/stastics/training_validation_loss.png')
        plt.close()

        # plot the training and validation accuracy at each epoch
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        plt.figure()
        plt.plot(epochs, acc, 'y', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('result/stastics/training_validation_accuracy.png')
        plt.close()

    #Load previously saved model
    model = load_model(f'./models/{model_name}', compile=False)
            
    #IOU
    y_pred=model.predict(X_test)
    y_pred_thresholded = y_pred > 0.5
    print(y_pred.max())
    print(y_pred.min())

    n_classes = 2
    IOU_keras = MeanIoU(num_classes=n_classes)  
    IOU_keras.update_state(y_pred_thresholded, y_test)
    print("Mean IoU =", IOU_keras.result().numpy())

    threshold = 0.5
    test_img_list = [random.randint(0, len(X_test)-1) for _ in range(10)]
    for test_img_number in test_img_list:
        img = X_test[test_img_number]
        ground_truth=y_test[test_img_number]
        test_img_input=np.expand_dims(img, 0)
        print(test_img_input.shape)
        prediction = (model.predict(test_img_input)[0,:,:,0] > threshold).astype(np.uint8)
        print(prediction.shape)

        plt.figure(figsize=(16, 8))
        plt.subplot(231)
        plt.title('Testing Image')
        plt.imshow(img[:,:,0], cmap='gray')
        plt.subplot(232)
        plt.title('Testing Label')
        plt.imshow(ground_truth[:,:,0], cmap='gray')
        plt.subplot(233)
        plt.title('Prediction on test image')
        plt.imshow(prediction, cmap='gray')
        plt.savefig(f'result/result/X_test/test_img_{test_img_number}.png')
        plt.close()

    test_image_names = glob.glob(f"{test_directory}/*.png")
    test_image = image_preprocessing(test_image_names, 'png')
    tst_image_dataset = np.array(test_image)
    tst_image_dataset = np.expand_dims(tst_image_dataset, axis=3)
    tst_image_dataset = tst_image_dataset/255.0
    test_img_list = [random.randint(0, len(tst_image_dataset)-1) for _ in range(20)]

    for test_img_number in test_img_list:
        
        img = tst_image_dataset[test_img_number]
        test_img_input=np.expand_dims(img, 0)
        prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)

        plt.figure(figsize=(16, 8))
        plt.subplot(231)
        plt.title('Testing Image')
        plt.imshow(img[:,:,0], cmap='gray')
        plt.subplot(232)
        plt.title('Prediction on test image')
        plt.imshow(prediction, cmap='gray')
        plt.savefig(f'result/result/test_image/test_img_{test_img_number}.png')
        plt.close()
