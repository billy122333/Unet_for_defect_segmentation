# %% [markdown] {"id":"view-in-github"}
# <a href="https://colab.research.google.com/github/bnsreenu/python_for_image_processing_APEER/blob/master/tutorial118_binary_semantic_segmentation_using_unet.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] {"id":"BAywFlF3xZRP"}
# https://youtu.be/oBIkr7CAE6g

# %% [markdown] {"id":"DWtdrQ4CIadK"}
# Binary semantic segmentation using U-Net
# Dataset: https://www.epfl.ch/labs/cvlab/data/data-em/

# %% [code] {"id":"mGaY-5G4Ia6G","execution":{"iopub.status.busy":"2024-09-15T07:59:04.523147Z","iopub.execute_input":"2024-09-15T07:59:04.523511Z","iopub.status.idle":"2024-09-15T07:59:04.528966Z","shell.execute_reply.started":"2024-09-15T07:59:04.523470Z","shell.execute_reply":"2024-09-15T07:59:04.528087Z"}}
from tensorflow.keras.utils import normalize
import tensorflow as tf
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
import glob

debug =True

# 設置只使用第二張顯卡（GPU 1）
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 檢查是否正確使用指定的 GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


dent_directory = '/home/frieren/r12921062/Unet/data/training/dent_images'
dent_mask_directory = '/home/frieren/r12921062/Unet/data/training/dent_masks'

scratch_directory = '/home/frieren/r12921062/Unet/data/training/scratch_images'
scratch_mask_directory = '/home/frieren/r12921062/Unet/data/training/scratch_masks'

test_directory = '/home/frieren/r12921062/Unet/data/testing/new_dent'
test_directory2 = '/home/frieren/r12921062/Unet/data/testing/new_dent'

SIZE = 64
num_images = 40
img_width = 64
img_height = 64


# Load images and masks in order so they match

# %% [code] {"execution":{"iopub.status.busy":"2024-09-14T07:31:24.505375Z","iopub.execute_input":"2024-09-14T07:31:24.505700Z","iopub.status.idle":"2024-09-14T07:31:24.514085Z","shell.execute_reply.started":"2024-09-14T07:31:24.505670Z","shell.execute_reply":"2024-09-14T07:31:24.513249Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2024-09-14T07:31:24.515156Z","iopub.execute_input":"2024-09-14T07:31:24.515480Z","iopub.status.idle":"2024-09-14T07:31:24.525219Z","shell.execute_reply.started":"2024-09-14T07:31:24.515450Z","shell.execute_reply":"2024-09-14T07:31:24.524381Z"}}
def image_preprocessing(image_list, img_type, mask = False):
    processed_images = []
    display_count = 1
    for filename in image_list:
        if filename.endswith(img_type):  # Adjust the extension if needed
            img = cv2.imread(filename)
            if img is not None:
#                 processed_image = resize_and_crop(img, img_width, img_height)
                # Assuming you want to keep the images in grayscale
                processed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if mask:
                    processed_image[processed_image != 0] = 255
#                     _, processed_image = cv2.threshold(processed_image, thresh=1, maxval=255, type=cv2.THRESH_BINARY)
#                 processed_image = np.tile(np.expand_dims(processed_image,axis=-1),(1,1,3))

                processed_images.append(processed_image)

                if display_count <= 3:
                    plt.figure(figsize=(16, 16))  # Adjust the figure size as needed
                    plt.subplot(2,5,display_count)
                    plt.imshow(processed_image, cmap='gray')  # Use cmap='gray' to display grayscale images
                    plt.title(f'Preprocessed Image {display_count+1}')
                    display_count += 1
    plt.show()
    return processed_images

# ## dent image process

dent_image_names = glob.glob(f"{dent_directory}/*.png")
dent_image_names.sort()

dent_images = image_preprocessing(dent_image_names, 'png')

dent_image_dataset = np.array(dent_images)
dent_image_dataset = np.expand_dims(dent_image_dataset, axis=3)
print(dent_image_dataset.shape)

# scratch process

scratch_image_names = glob.glob(f"{scratch_directory}/*.png")
scratch_image_names.sort()

scratch_image = image_preprocessing(scratch_image_names, 'png')

scratch_image_dataset = np.array(scratch_image)
scratch_image_dataset = np.expand_dims(scratch_image_dataset, axis=3)
print(scratch_image_dataset.shape)

# %% [markdown]
# ## mask processing

dent_mask_image_names = glob.glob(f"{dent_mask_directory}/*.png")
dent_mask_image_names.sort()
dent_masked_images = image_preprocessing(dent_mask_image_names, 'png', mask = True)



dent_masked_images_dataset = np.array(dent_masked_images)
dent_masked_images_dataset = np.expand_dims(dent_masked_images_dataset, axis=3)

if debug :
    # list lenght
    print("dent_mask_name:", dent_mask_image_names.__len__())
    print("dent_mask:", dent_masked_images_dataset.shape)


# 对于 dent 图像来说，scratch 的掩码应该是全零矩阵
dent_padding_mask = np.zeros_like(dent_masked_images_dataset)  # 生成与 dent 掩码同样尺寸的全零矩阵

# 合并 dent 和 scratch 掩码 (合併在channel維度)
dent_masked_images_dataset = np.concatenate([dent_masked_images_dataset, dent_padding_mask], axis=3)

# %% [code] {"execution":{"iopub.status.busy":"2024-09-14T07:35:54.444143Z","iopub.execute_input":"2024-09-14T07:35:54.444449Z","iopub.status.idle":"2024-09-14T07:37:25.413730Z","shell.execute_reply.started":"2024-09-14T07:35:54.444424Z","shell.execute_reply":"2024-09-14T07:37:25.412760Z"}}
scratch_mask_image_names = glob.glob(f"{scratch_mask_directory}/*.png")
scratch_mask_image_names.sort()
scratch_masked_images = image_preprocessing(scratch_mask_image_names, 'png', mask = True)

# %% [code] {"execution":{"iopub.status.busy":"2024-09-14T07:37:25.414982Z","iopub.execute_input":"2024-09-14T07:37:25.415311Z","iopub.status.idle":"2024-09-14T07:37:25.442216Z","shell.execute_reply.started":"2024-09-14T07:37:25.415279Z","shell.execute_reply":"2024-09-14T07:37:25.441217Z"}}
scratch_masked_images_dataset = np.array(scratch_masked_images)
scratch_masked_images_dataset = np.expand_dims(scratch_masked_images_dataset, axis=3)

print(scratch_masked_images_dataset.shape)

# %% [code] {"execution":{"iopub.status.busy":"2024-09-14T07:37:25.443490Z","iopub.execute_input":"2024-09-14T07:37:25.443794Z","iopub.status.idle":"2024-09-14T07:37:25.511850Z","shell.execute_reply.started":"2024-09-14T07:37:25.443768Z","shell.execute_reply":"2024-09-14T07:37:25.511035Z"}}
# 对于 dent 图像来说，scratch 的掩码应该是全零矩阵
scratch_padding_mask = np.zeros_like(scratch_masked_images_dataset)  # 生成与 dent 掩码同样尺寸的全零矩阵

# 合并 dent 和 scratch 掩码  (合併在channel維度)
scratch_masked_images_dataset = np.concatenate([scratch_padding_mask, scratch_masked_images_dataset], axis=3)

# %% [markdown] {"id":"HGqh1PaAS19H"}
# ## Read masks the same way. 

# %% [code] {"id":"WPL2cpgwOMU4","outputId":"7195dec7-1f46-4ca8-e655-b3942c82370a","execution":{"iopub.status.busy":"2024-09-14T07:37:25.512943Z","iopub.execute_input":"2024-09-14T07:37:25.513240Z","iopub.status.idle":"2024-09-14T07:37:27.242416Z","shell.execute_reply.started":"2024-09-14T07:37:25.513215Z","shell.execute_reply":"2024-09-14T07:37:27.241448Z"}}
print("Image data shape is: ", dent_image_dataset.shape)
print("Mask data shape is: ", dent_masked_images_dataset.shape)
print("Max pixel value in image is: ", dent_image_dataset.max())
print("Labels in the mask are : ", np.unique(dent_masked_images_dataset))

# %% [code] {"execution":{"iopub.status.busy":"2024-09-14T07:37:27.243692Z","iopub.execute_input":"2024-09-14T07:37:27.244345Z","iopub.status.idle":"2024-09-14T07:37:27.339311Z","shell.execute_reply.started":"2024-09-14T07:37:27.244298Z","shell.execute_reply":"2024-09-14T07:37:27.338317Z"}}
# 合併在Batch維度 (B, h, w, c)
all_images = np.concatenate([dent_image_dataset, scratch_image_dataset], axis=0)
all_masks = np.concatenate([dent_masked_images_dataset, scratch_masked_images_dataset], axis=0)

# %% [code] {"execution":{"iopub.status.busy":"2024-09-14T07:37:27.340457Z","iopub.execute_input":"2024-09-14T07:37:27.340757Z","iopub.status.idle":"2024-09-14T07:37:27.345797Z","shell.execute_reply.started":"2024-09-14T07:37:27.340732Z","shell.execute_reply":"2024-09-14T07:37:27.344869Z"}}
print("Image data shape is: ", all_images.shape)
print("Mask data shape is: ", all_masks.shape)


# %% [markdown]
# ## Dataset Split

# %% [code] {"id":"az5lOo48Msva","execution":{"iopub.status.busy":"2024-09-14T07:37:27.347177Z","iopub.execute_input":"2024-09-14T07:37:27.347416Z","iopub.status.idle":"2024-09-14T07:37:27.524581Z","shell.execute_reply.started":"2024-09-14T07:37:27.347396Z","shell.execute_reply":"2024-09-14T07:37:27.523609Z"}}
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(all_images, all_masks, test_size = 0.2, random_state = 42)
print(X_test.shape)

# %% [code] {"id":"tycUwzOoMtXZ","outputId":"d9f05c72-e722-48be-f0d8-3877a83ea127","execution":{"iopub.status.busy":"2024-09-14T07:37:27.525746Z","iopub.execute_input":"2024-09-14T07:37:27.526029Z","iopub.status.idle":"2024-09-14T07:37:27.979924Z","shell.execute_reply.started":"2024-09-14T07:37:27.526004Z","shell.execute_reply":"2024-09-14T07:37:27.978960Z"}}
#Sanity check, view few mages
import random

image_number = random.randint(0, len(X_train)-1)
plt.figure(figsize=(12, 6))
plt.subplot(1,3,1)
plt.imshow(X_train[image_number,:,:,0], cmap='gray')
plt.subplot(1,3,2)
plt.imshow(y_train[image_number,:,:,0], cmap='gray')
plt.subplot(1,3,3)
plt.imshow(y_train[image_number,:,:,1], cmap='gray')
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-09-14T07:37:27.981162Z","iopub.execute_input":"2024-09-14T07:37:27.981772Z","iopub.status.idle":"2024-09-14T07:37:56.148200Z","shell.execute_reply.started":"2024-09-14T07:37:27.981745Z","shell.execute_reply":"2024-09-14T07:37:56.147117Z"}}
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf 

# preprocessing for img
def custom_image_preprocessing(image):
    # 對圖像進行亮度調整，這裡不會影響掩碼
    image = tf.image.random_brightness(image, max_delta=0.4)  # 亮度增強
    image = image / 255.0
    return image

# preprocessing for mask
def binarize_mask(mask):
    _, binary_mask = cv2.threshold(mask, thresh=0.5, maxval=1, type=cv2.THRESH_BINARY)
    binary_mask = binary_mask.reshape(mask.shape)  # 保持原始形状
    return binary_mask


data_gen_args = dict(rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
#                      shear_range=0.5,
#                      zoom_range=0.2,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect')


image_datagen = ImageDataGenerator(**data_gen_args, preprocessing_function=custom_image_preprocessing)
mask_datagen = ImageDataGenerator(**data_gen_args, preprocessing_function=binarize_mask)

# 使用相同的种子和批量大小来生成图像和掩码
seed = 1
image_datagen.fit(X_train, augment=True, seed=seed)
mask_datagen.fit(y_train, augment=True, seed=seed)

image_generator = image_datagen.flow(X_train, batch_size=32, seed=seed)
mask_generator = mask_datagen.flow(y_train, batch_size=32, seed=seed)
def combine_generators(image_gen, mask_gen):
    while True:
        x = next(image_gen)
        y = next(mask_gen)
        yield x, y
print("Labels in the mask are : ", np.unique(y_train))
train_generator = combine_generators(image_generator, mask_generator)


# %% [code] {"execution":{"iopub.status.busy":"2024-09-14T07:37:56.149444Z","iopub.execute_input":"2024-09-14T07:37:56.149810Z","iopub.status.idle":"2024-09-14T07:37:56.155359Z","shell.execute_reply.started":"2024-09-14T07:37:56.149777Z","shell.execute_reply":"2024-09-14T07:37:56.154269Z"}}
# # import matplotlib.pyplot as plt

# # Get a batch of augmented images and masks
# img_batch, mask_batch = next(zip(image_generator, mask_generator))

# # Define a function to display images and masks
# def display_images(images, masks):
#     for i in range(min(5, len(images))):  # Display first 5 images and masks
#         plt.figure(figsize=(12, 6))

#         plt.subplot(1, 2, 1)
#         plt.imshow(images[i], cmap='gray')
#         plt.title('Augmented Image')

#         plt.subplot(1, 2, 2)
#         plt.imshow(masks[i].squeeze(), cmap='gray')  # Assuming masks are single-channel
#         plt.title('Augmented Mask')

#         plt.show()

# # Display the augmented images and corresponding masks
# display_images(img_batch, mask_batch)

# print(np.unique(img_batch))


# %% [code] {"execution":{"iopub.status.busy":"2024-09-14T07:37:56.156581Z","iopub.execute_input":"2024-09-14T07:37:56.157145Z","iopub.status.idle":"2024-09-14T07:37:56.170176Z","shell.execute_reply.started":"2024-09-14T07:37:56.157112Z","shell.execute_reply":"2024-09-14T07:37:56.169326Z"}}
# for i in range(10):
#     img, msk = next(train_generator)
#     print("Labels in the mask are : ", np.unique(msk))
#     # 可视化第一张图像和掩码
#     plt.figure(figsize=(12, 4))

#     # 显示原始图像
#     plt.subplot(1, 2, 1)
#     plt.title('Original Image')
#     plt.imshow(img[0].squeeze(), cmap='gray')

#     # 显示真实掩码
#     plt.subplot(1, 2, 2)
#     plt.title('Mask')
#     plt.imshow(msk[0].squeeze(), cmap='gray')

#     plt.show()


# %% [code] {"id":"do_6g3dqMGEw","execution":{"iopub.status.busy":"2024-09-14T07:37:56.171579Z","iopub.execute_input":"2024-09-14T07:37:56.171899Z","iopub.status.idle":"2024-09-14T07:37:56.187112Z","shell.execute_reply.started":"2024-09-14T07:37:56.171870Z","shell.execute_reply":"2024-09-14T07:37:56.186286Z"}}
# Building Unet by dividing encoder and decoder into blocks

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda, Add, Multiply
from keras.optimizers import Adam
from keras.layers import Activation, MaxPool2D, Concatenate


def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)   #Not in the original network. 
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)  #Not in the original network
    x = Activation("relu")(x)

    return x

#Encoder block: Conv block followed by maxpooling


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p   

#Decoder block
#skip features gets input from encoder for concatenation

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

#Build Unet using the blocks
def build_unet(input_shape, n_classes):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024) #Bridge

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    if n_classes == 1:  #Binary
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    outputs = Conv2D(n_classes, 1, padding="same", activation=activation)(d4)  #Change the activation based on n_classes
    print(activation)

    model = Model(inputs, outputs, name="U-Net")
    return model

# %% [markdown]
# ### Trying Attention Unet

# %% [code] {"execution":{"iopub.status.busy":"2024-09-14T07:37:56.188769Z","iopub.execute_input":"2024-09-14T07:37:56.189190Z","iopub.status.idle":"2024-09-14T07:37:56.201461Z","shell.execute_reply.started":"2024-09-14T07:37:56.189159Z","shell.execute_reply":"2024-09-14T07:37:56.200583Z"}}
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

from tensorflow.keras.initializers import HeNormal

# %% [code] {"execution":{"iopub.status.busy":"2024-09-14T07:37:56.208006Z","iopub.execute_input":"2024-09-14T07:37:56.208289Z","iopub.status.idle":"2024-09-14T07:37:56.219563Z","shell.execute_reply.started":"2024-09-14T07:37:56.208267Z","shell.execute_reply":"2024-09-14T07:37:56.218221Z"}}
initializer = HeNormal()
def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same", kernel_initializer=initializer)(input)
    x = BatchNormalization()(x)   #Not in the original network. 
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same", kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)  #Not in the original network
    x = Activation("relu")(x)

    return x


def encoder_block(input, num_filters, dropout_rate=0.2):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    # NOTSURE
    p = Dropout(dropout_rate)(p)
    return x, p   

def decoder_block(input, skip_features, num_filters, dropout_rate=0.2):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    # Apply attention gate on skip connection features
    attention_features = attention_gate(x, skip_features, num_filters)
    x = Concatenate()([x, attention_features])
    #NOTSURE
    x = Dropout(dropout_rate)(x)  # Add dropout after concatenation
    x = conv_block(x, num_filters)
    return x

# %% [code] {"execution":{"iopub.status.busy":"2024-09-14T07:37:56.220740Z","iopub.execute_input":"2024-09-14T07:37:56.221063Z","iopub.status.idle":"2024-09-14T07:37:56.233296Z","shell.execute_reply.started":"2024-09-14T07:37:56.221030Z","shell.execute_reply":"2024-09-14T07:37:56.232559Z"}}
def build_attention_unet(input_shape, n_classes):
    inputs = Input(input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, 16)
    s2, p2 = encoder_block(p1, 32)
    s3, p3 = encoder_block(p2, 64)
    s4, p4 = encoder_block(p3, 128)

    # Bridge
    b1 = conv_block(p4, 256)

    # Decoder with attention
    d1 = decoder_block(b1, s4, 128)
    d2 = decoder_block(d1, s3, 64)
    d3 = decoder_block(d2, s2, 32)
    d4 = decoder_block(d3, s1, 16)

    if n_classes == 1:  # Binary segmentation
        activation = 'sigmoid'
    else:
#         activation = 'softmax' 
        activation = 'sigmoid'

    outputs = Conv2D(n_classes, 1, padding="same", activation=activation)(d4)
    model = Model(inputs, outputs, name="Attention_U-Net")
    return model

# %% [code] {"id":"KhNN7zEqMxaA","execution":{"iopub.status.busy":"2024-09-14T07:37:56.234245Z","iopub.execute_input":"2024-09-14T07:37:56.234681Z","iopub.status.idle":"2024-09-14T07:37:56.249485Z","shell.execute_reply.started":"2024-09-14T07:37:56.234652Z","shell.execute_reply":"2024-09-14T07:37:56.248507Z"}}
IMG_HEIGHT = dent_image_dataset.shape[1]
IMG_WIDTH  = dent_image_dataset.shape[2]
IMG_CHANNELS = dent_image_dataset.shape[3]

input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# TODO : Added dice loss
# def weighted_binary_crossentropy(y_true, y_pred):

#     epsilon = 1e-7
#     y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
#     y_true = tf.cast(y_true, tf.float32)

#     # 計算每個通道的 BCE 損失
#     bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)


#     # 計算每個通道的 Dice Loss
#     intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2], keepdims=True)  # 只對 height 和 width 求和
#     union = tf.reduce_sum(y_true + y_pred, axis=[1, 2], keepdims=True)         # 只對 height 和 width 求和
#     dice_loss = 1 - (2. * intersection + 1e-6) / (union + 1e-6)

#     # 結合損失
#     total_loss = bce + dice_loss

#     # 計算平均損失
#     loss = tf.reduce_mean(total_loss)

#     return loss

def weighted_binary_crossentropy(y_true, y_pred):
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    y_true = tf.cast(y_true, tf.float32)

    # 計算每個通道的 BCE 損失
    bce = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))

    # 計算每個通道的總和，判斷是否為全黑通道
    channel_sums = tf.reduce_sum(y_true, axis=[1, 2], keepdims=True)  # shape: (batch_size, 1, 1, n_classes)
    
    # 創建一個權重掩碼，對全黑通道賦予較低權重
    mask = tf.where(channel_sums > 0, 1.0, 0.4)  # 全黑通道權重為0.1，其他通道為1.0

    # 應用權重掩碼
    weighted_bce = bce * mask

    # 計算每個通道的總損失
    total_loss = tf.reduce_sum(weighted_bce) / tf.reduce_sum(mask)

    return total_loss

# def focal_loss(gamma=2., alpha=0.25):
#     def focal_loss_fixed(y_true, y_pred):
#         epsilon = 1e-7
#         y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
#         y_true = tf.cast(y_true, tf.float32)
        
#         cross_entropy = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
#         weight = alpha * y_true * (1 - y_pred) ** gamma + (1 - alpha) * (1 - y_true) * y_pred ** gamma
#         loss = weight * cross_entropy
#         return tf.reduce_mean(loss)
#     return focal_loss_fixed


# model = build_unet(input_shape, n_classes=1)
model = build_attention_unet(input_shape, n_classes=2)
model.compile(optimizer=Adam(learning_rate = 1e-4), loss=weighted_binary_crossentropy, metrics=['accuracy'])
model.summary()

# %% [code] {"execution":{"iopub.status.busy":"2024-09-14T08:01:12.216290Z","iopub.execute_input":"2024-09-14T08:01:12.217060Z","iopub.status.idle":"2024-09-14T08:01:12.230175Z","shell.execute_reply.started":"2024-09-14T08:01:12.217030Z","shell.execute_reply":"2024-09-14T08:01:12.229249Z"}}
import matplotlib.pyplot as plt
from keras.callbacks import Callback

class VisualizationCallback(Callback):
    def __init__(self, X_train, y_train, num_samples=2):
        self.X_train = X_train
        self.y_train = y_train
        self.num_samples = num_samples
        self.outputDir = '/home/frieren/r12921062/Unet/output'

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.X_train[:self.num_samples])
        y_pred_thresholded = (predictions > 0.5).astype(np.uint8)  # 二值化预测结果

        for i in range(self.num_samples):
            plt.figure(figsize=(12, 6))  # 调整图像大小

            # 显示原始图像
            plt.subplot(2, 3, 1)
            plt.title('Original Image')
            plt.imshow(self.X_train[i].squeeze(), cmap='gray')

            # 显示真实的 dent 掩码
            plt.subplot(2, 3, 2)
            plt.title('Ground Truth - Dent')
            plt.imshow(self.y_train[i, :, :, 0].squeeze(), cmap='gray')  # 第一个通道是 dent 掩码

            # 显示真实的 scratch 掩码
            plt.subplot(2, 3, 3)
            plt.title('Ground Truth - Scratch')
            plt.imshow(self.y_train[i, :, :, 1].squeeze(), cmap='gray')  # 第二个通道是 scratch 掩码

            # 显示预测的 dent 掩码
            plt.subplot(2, 3, 4)
            plt.title('Predicted Binary - Dent')
            plt.imshow(y_pred_thresholded[i, :, :, 0].squeeze(), cmap='gray')  # 第一个通道的预测结果

            # 显示预测的 scratch 掩码
            plt.subplot(2, 3, 5)
            plt.title('Predicted Binary - Scratch')
            plt.imshow(y_pred_thresholded[i, :, :, 1].squeeze(), cmap='gray')  # 第二个通道的预测结果

            # 显示预测结果的合并图像 (选用颜色叠加或灰度叠加)
            combined_mask = np.zeros_like(y_pred_thresholded[i, :, :, 0], dtype=np.float64)
            combined_mask += y_pred_thresholded[i, :, :, 0] * 0.5  # dent 用灰度 0.5 表示
            combined_mask += y_pred_thresholded[i, :, :, 1] * 1.0  # scratch 用灰度 1.0 表示

            plt.subplot(2, 3, 6)
            plt.title('Combined Prediction')
            plt.imshow(combined_mask, cmap='gray')

            plt.savefig(os.path.join(self.outputDir, f'epoch_{epoch}_sample_{i}.png'))
            plt.close()  

# %% [code] {"id":"SbxgEXNDM4DY","outputId":"557a1d80-6e90-4597-c6c4-aafc6052ebe0","execution":{"iopub.status.busy":"2024-09-14T08:01:14.300359Z","iopub.execute_input":"2024-09-14T08:01:14.300983Z"}}
visualization_callback = VisualizationCallback(X_train, y_train)
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
earlyStopping = EarlyStopping(monitor='val_loss', patience=300, verbose=0, mode='min')
mcp_save = ModelCheckpoint('/home/frieren/r12921062/Unet/models/AttentionUnet', save_best_only=True, monitor='val_loss', mode='min', save_format='tf')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, verbose=1, min_delta=1e-4, mode='min')

# history = model.fit(X_train, y_train, 
#                     batch_size = 4, 
#                     verbose=1, 
#                     epochs=20, 
#                     validation_data=(X_test, y_test), 
#                     shuffle=False,
#                     callbacks=[earlyStopping, mcp_save, reduce_lr_loss, visualization_callback])

img, msk = next(train_generator)
# check the shape of the image and mask
print("Image shape: ", img.shape)
print("Mask shape: ", msk.shape)

# check validation data
print("Validation data shape: ", X_test.shape)
print("Validation mask shape: ", y_test.shape)

history = model.fit(train_generator,
                    steps_per_epoch=len(X_train) // 4,
                    epochs=400,
                    workers=1,
                    use_multiprocessing=False,  # 禁用多進程
                    validation_data=(X_test, y_test),
                    callbacks=[earlyStopping, mcp_save, reduce_lr_loss, visualization_callback])



#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# %% [code] {"id":"rBmyuvr9U4MM","execution":{"iopub.status.busy":"2024-09-14T07:47:25.976568Z","iopub.status.idle":"2024-09-14T07:47:25.976865Z","shell.execute_reply.started":"2024-09-14T07:47:25.976715Z","shell.execute_reply":"2024-09-14T07:47:25.976728Z"}}
#Load previously saved model
from keras.models import load_model
model = load_model("/home/frieren/r12921062/Unet/models/AttentionUnet.keras", compile=False)
        

# %% [code] {"id":"XflE1c9IM_16","execution":{"iopub.status.busy":"2024-09-14T07:47:25.978008Z","iopub.status.idle":"2024-09-14T07:47:25.978357Z","shell.execute_reply.started":"2024-09-14T07:47:25.978193Z","shell.execute_reply":"2024-09-14T07:47:25.978207Z"}}
#IOU
y_pred=model.predict(X_test)
y_pred_thresholded = y_pred > 0.5
print(y_pred.max())
print(y_pred.min())

# %% [code] {"id":"NxwRS20eV05K","execution":{"iopub.status.busy":"2024-09-14T07:47:25.979825Z","iopub.status.idle":"2024-09-14T07:47:25.980304Z","shell.execute_reply.started":"2024-09-14T07:47:25.980035Z","shell.execute_reply":"2024-09-14T07:47:25.980053Z"}}
from tensorflow.keras.metrics import MeanIoU

# %% [code] {"id":"Sz9lZ-oLVumx","outputId":"46413096-2681-4fb5-d493-c2ed60363286","execution":{"iopub.status.busy":"2024-09-14T07:47:25.981656Z","iopub.status.idle":"2024-09-14T07:47:25.982158Z","shell.execute_reply.started":"2024-09-14T07:47:25.981869Z","shell.execute_reply":"2024-09-14T07:47:25.981918Z"}}
n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_pred_thresholded, y_test)
print("Mean IoU =", IOU_keras.result().numpy())

# %% [code] {"id":"rIV3cDBWWeXH","outputId":"3bc0bb1c-ef76-4f8b-be26-389b0f48c580","execution":{"iopub.status.busy":"2024-09-14T07:47:25.984126Z","iopub.status.idle":"2024-09-14T07:47:25.984673Z","shell.execute_reply.started":"2024-09-14T07:47:25.984389Z","shell.execute_reply":"2024-09-14T07:47:25.984410Z"}}
threshold = 0.5
test_img_list = [random.randint(0, len(X_test)-1) for _ in range(20)]
for test_img_number in test_img_list:
    test_img = X_test[test_img_number]
    ground_truth=y_test[test_img_number]
    test_img_input=np.expand_dims(test_img, 0)
    print(test_img_input.shape)
    prediction = (model.predict(test_img_input)[0,:,:,0] > threshold).astype(np.uint8)
    print(prediction.shape)

    plt.figure(figsize=(16, 8))
    plt.subplot(2,4,1)
    plt.title('Testing Image')
    plt.imshow(test_img[:,:,0], cmap='gray')
    plt.subplot(2,4,2)
    plt.title('Testing Label dent')
    plt.imshow(ground_truth[:,:,0], cmap='gray')
    plt.subplot(2,4,3)
    plt.title('Testing Label scratch')
    plt.imshow(ground_truth[:,:,1], cmap='gray')
    plt.subplot(2,4,4)
    plt.title('Prediction on test image')
    plt.imshow(prediction, cmap='gray')

    plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-09-14T07:47:25.985935Z","iopub.status.idle":"2024-09-14T07:47:25.986619Z","shell.execute_reply.started":"2024-09-14T07:47:25.986378Z","shell.execute_reply":"2024-09-14T07:47:25.986400Z"}}
test_image_names = glob.glob(f"{test_directory}/*.png")
test_image = image_preprocessing(test_image_names, 'png')
test_image_dataset = np.array(test_image)
test_image_dataset = np.expand_dims(test_image_dataset, axis=3)
test_image_dataset = test_image_dataset/255.0
test_image_pairs = list(zip(test_image_names, test_image_dataset))
test_img_list = random.sample(test_image_pairs, 50)


# %% [code] {"execution":{"iopub.status.busy":"2024-09-14T07:47:25.987817Z","iopub.status.idle":"2024-09-14T07:47:25.988176Z","shell.execute_reply.started":"2024-09-14T07:47:25.987985Z","shell.execute_reply":"2024-09-14T07:47:25.987999Z"}}
for test_image_name, test_img in test_img_list:
    test_img = test_img
    test_img_input=np.expand_dims(test_img, 0)
    prediction = (model.predict(test_img_input)[0,:,:,0] > 0.9).astype(np.uint8)
    print(prediction.shape)
    
    test_image_name = test_image_name.split("/")[-1]

    plt.figure(figsize=(16, 8))
    plt.subplot(2,3,1)
    plt.title(test_image_name)
    plt.imshow(test_img[:,:,0], cmap='gray')
    plt.subplot(2,3,2)
    plt.title(test_image_name)
    plt.imshow(test_img[:,:,1], cmap='gray')
    plt.subplot(2,3,3)
    plt.title('Prediction on test image')
    plt.imshow(prediction, cmap='gray')

    plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-09-14T07:47:25.989362Z","iopub.status.idle":"2024-09-14T07:47:25.989666Z","shell.execute_reply.started":"2024-09-14T07:47:25.989515Z","shell.execute_reply":"2024-09-14T07:47:25.989528Z"}}
test_image_names = glob.glob(f"{test_directory2}/*.png")
test_image = image_preprocessing(test_image_names, 'png')
test_image_dataset = np.array(test_image)
test_image_dataset = np.expand_dims(test_image_dataset, axis=3)
test_image_dataset = test_image_dataset/255.0
test_image_paris = list(zip(test_image_names, test_image_dataset))
test_img_list = random.sample(test_image_pairs, 50)


# %% [code] {"execution":{"iopub.status.busy":"2024-09-14T07:47:25.991255Z","iopub.status.idle":"2024-09-14T07:47:25.991591Z","shell.execute_reply.started":"2024-09-14T07:47:25.991426Z","shell.execute_reply":"2024-09-14T07:47:25.991441Z"}}
threshold = 0.45
for test_img_name, test_img in test_img_list:
    test_img_input=np.expand_dims(test_img, 0)
    print(test_img_input.min())
    prediction = (model.predict(test_img_input)[0,:,:,0] > 0.9).astype(np.uint8)
    print(prediction.shape)
    test_img_name = test_img_name.split("/")[-1]

    plt.figure(figsize=(16, 8))
    plt.subplot(2,3,1)
    plt.title(test_img_name)
    plt.imshow(test_img[:,:,0], cmap='gray')
    plt.subplot(2,3,2)
    plt.title(test_img_name)
    plt.imshow(test_img[:,:,1], cmap='gray')
    plt.subplot(2,3,3)
    plt.title('Prediction on test image')
    plt.imshow(prediction, cmap='gray')

    plt.show()

# %% [code]
