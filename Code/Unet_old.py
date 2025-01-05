
import cv2
import glob
import os

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Dropout, Activation, MaxPool2D, Concatenate, Add, Multiply
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import normalize
import tensorflow as tf

debug =False

# 檢查是否正確使用指定的 GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

dent_directory = './data/training_real/img_slt'
dent_mask_directory = './data/training_real/mask_slt'

test_directory = './data/testing/new_dent'
test_directory2 = './data/testing/new_dent'

# config
SIZE = 64
num_images = 40
img_width = 64
img_height = 64


# Load images and masks in order so they match
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
#                 processed_image = resize_and_crop(img, img_width, img_height)
                # Assuming you want to keep the images in grayscale
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


dent_image_names = glob.glob(f"{dent_directory}/*.png")
dent_image_names.sort()

dent_images = image_preprocessing(dent_image_names, 'png')
dent_image_dataset = np.array(dent_images)
dent_image_dataset = np.expand_dims(dent_image_dataset, axis=3)

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
    print("Image data shape is: ", dent_image_dataset.shape)
    print("Mask data shape is: ", dent_masked_images_dataset.shape)
    print("Max pixel value in image is: ", dent_image_dataset.max())
    print("Labels in the mask are : ", np.unique(dent_masked_images_dataset))



X_train, X_test, y_train, y_test = train_test_split(dent_image_dataset, dent_masked_images_dataset, test_size = 0.2, random_state = 42)
print(X_test.shape)

#Sanity check, view few mages
import random

if os.path.exists('./output') is False:
    os.makedirs('./output')

image_number = random.randint(0, len(X_train)-1)
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.imshow(X_train[image_number,:,:,0], cmap='gray')
plt.subplot(1,2,2)
plt.imshow(y_train[image_number,:,:,0], cmap='gray')
plt.savefig(f'./output/sample_image.png')

# preprocessing for img
def custom_image_preprocessing(image):
    # 對圖像進行亮度調整，這裡不會影響掩碼
    # image = tf.image.random_brightness(image, max_delta=0.2)  # 亮度增強
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
                     shear_range=0.5,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect')


image_datagen = ImageDataGenerator(**data_gen_args, preprocessing_function=custom_image_preprocessing)
mask_datagen = ImageDataGenerator(**data_gen_args, preprocessing_function=binarize_mask)

# 使用相同的种子和批量大小来生成图像和掩码
seed = 1
image_datagen.fit(X_train, augment=True, seed=seed)
mask_datagen.fit(y_train, augment=True, seed=seed)

image_generator = image_datagen.flow(X_train, batch_size=4, seed=seed)
mask_generator = mask_datagen.flow(y_train, batch_size=4, seed=seed)
def combine_generators(image_gen, mask_gen):
    while True:
        x = next(image_gen)
        y = next(mask_gen)
        yield x, y
print("Labels in the mask are : ", np.unique(y_train))
train_generator = combine_generators(image_generator, mask_generator)

for i in range(4):
    image, mask = next(train_generator)
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.title('Image')
    plt.imshow(image[0, :, :, 0], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('Mask')
    plt.imshow(mask[0, :, :, 0], cmap='gray')
    plt.savefig(f'./output/augmented_image_{i}.png')

    print("Image pixel range:", image.min(), image.max())
    print("Mask pixel range:", mask.min(), mask.max())





def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)   #Not in the original network. 
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)  #Not in the original network
    x = Activation("relu")(x)

    return x

# #Encoder block: Conv block followed by maxpooling


# def encoder_block(input, num_filters):
#     x = conv_block(input, num_filters)
#     p = MaxPool2D((2, 2))(x)
#     return x, p   

# #Decoder block
# #skip features gets input from encoder for concatenation

# def decoder_block(input, skip_features, num_filters):
#     x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
#     x = Concatenate()([x, skip_features])
#     x = conv_block(x, num_filters)
#     return x

# #Build Unet using the blocks
# def build_unet(input_shape, n_classes):
#     inputs = Input(input_shape)

#     s1, p1 = encoder_block(inputs, 64)
#     s2, p2 = encoder_block(p1, 128)
#     s3, p3 = encoder_block(p2, 256)
#     s4, p4 = encoder_block(p3, 512)

#     b1 = conv_block(p4, 1024) #Bridge

#     d1 = decoder_block(b1, s4, 512)
#     d2 = decoder_block(d1, s3, 256)
#     d3 = decoder_block(d2, s2, 128)
#     d4 = decoder_block(d3, s1, 64)

#     if n_classes == 1:  #Binary
#         activation = 'sigmoid'
#     else:
#         activation = 'softmax'

#     outputs = Conv2D(n_classes, 1, padding="same", activation=activation)(d4)  #Change the activation based on n_classes
#     print(activation)

#     model = Model(inputs, outputs, name="U-Net")
#     return model

# Attention U-Net

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

def encoder_block(input, num_filters, dropout_rate=0.3):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    # NOTSURE
    p = Dropout(dropout_rate)(p)
    return x, p   

def decoder_block(input, skip_features, num_filters, dropout_rate=0.3):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    # Apply attention gate on skip connection features
    attention_features = attention_gate(x, skip_features, num_filters)
    x = Concatenate()([x, attention_features])
    #NOTSURE
    x = Dropout(dropout_rate)(x)  # Add dropout after concatenation
    x = conv_block(x, num_filters)
    return x

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
        activation = 'softmax' 
        # activation = 'sigmoid'

    outputs = Conv2D(n_classes, 1, padding="same", activation=activation)(d4)
    model = Model(inputs, outputs, name="Attention_U-Net")
    return model

IMG_HEIGHT = dent_image_dataset.shape[1]
IMG_WIDTH  = dent_image_dataset.shape[2]
IMG_CHANNELS = dent_image_dataset.shape[3]
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)


# 改进的加权二元交叉熵损失
def weighted_binary_crossentropy(y_true, y_pred):
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    y_true = tf.cast(y_true, tf.float32)
    weights = tf.where(tf.equal(y_true, 1), 10., 1.)
    bce = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    weighted_bce = tf.reduce_mean(bce * weights)
    return weighted_bce

# Dice 损失
def dice_loss(y_true, y_pred):
    epsilon = 1e-7
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    # 计算交集和并集
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

    # 计算 Dice 系数
    dice = (2. * intersection + epsilon) / (union + epsilon)
    
    # 返回 1 减去 Dice 系数作为损失
    return 1 - dice

# 结合 BCE 和 Dice 的综合损失函数
def combined_dice_bce_loss(y_true, y_pred, bce_weight=0.5, dice_weight=0.5):
    bce = weighted_binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)

    # 综合损失，加权平均
    combined_loss = bce_weight * bce + dice_weight * dice
    return combined_loss


# model = build_unet(input_shape, n_classes=1)
model = build_attention_unet(input_shape, n_classes=1)
model.compile(optimizer=Adam(learning_rate = 5e-5, clipnorm=1.0), loss=weighted_binary_crossentropy, metrics=['accuracy'])
model.summary()


class VisualizationCallback(Callback):
    def __init__(self, X_train, y_train, num_samples=1):
        self.X_train = X_train
        self.y_train = y_train
        self.num_samples = num_samples
        self.outputDir = './output'

    def on_epoch_end(self, epoch, logs=None):
        rand = random.randint(0, len(self.X_train) - 1)
        predictions = self.model.predict(self.X_train[rand:(rand+self.num_samples)])
        y_pred_thresholded = (predictions > 0.5).astype(np.uint8)  # 二值化预测结果

        for i in range(self.num_samples):
            plt.figure(figsize=(12, 6))  # 调整图像大小

            nums = rand+i
            # 显示原始图像
            plt.subplot(1, 3, 1)
            plt.title('Original Image')
            plt.imshow(self.X_train[nums].squeeze(), cmap='gray')

            # 显示真实掩码
            plt.subplot(1, 3, 2)
            plt.title('Ground Truth')
            plt.imshow(self.y_train[nums].squeeze(), cmap='gray')

            # 显示预测的二值化图像
            plt.subplot(1, 3, 3)
            plt.title('Predicted Binary Image')
            plt.imshow(y_pred_thresholded[i].squeeze(), cmap='gray')
        
            plt.savefig(os.path.join(self.outputDir, f'epoch_{epoch}_sample_{i}.png'))
            plt.close()  

visualization_callback = VisualizationCallback(X_train, y_train)

earlyStopping = EarlyStopping(monitor='val_loss', patience=1000, verbose=0, mode='min')
mcp_save = ModelCheckpoint('./models/AttentionUnet.tf', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, verbose=1, min_delta=1e-4, mode='min')


X_test = X_test/255.0
y_test = y_test/255.0


history = model.fit(train_generator,
                    steps_per_epoch=len(X_train) // 4,
                    epochs=2500,
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

model = load_model("./models/AttentionUnet.tf", compile=False)
        

#IOU
y_pred=model.predict(X_test)
y_pred_thresholded = y_pred > 0.5
print(y_pred.max())
print(y_pred.min())
from tensorflow.keras.metrics import MeanIoU

# %% [code] {"id":"Sz9lZ-oLVumx","outputId":"46413096-2681-4fb5-d493-c2ed60363286","execution":{"iopub.status.busy":"2024-09-14T07:47:25.981656Z","iopub.status.idle":"2024-09-14T07:47:25.982158Z","shell.execute_reply.started":"2024-09-14T07:47:25.981869Z","shell.execute_reply":"2024-09-14T07:47:25.981918Z"}}
n_classes = 1
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

    plt.savefig(f'./output/test_image_{test_img_number}.png')


# %% [code] {"execution":{"iopub.status.busy":"2024-09-14T07:47:25.985935Z","iopub.status.idle":"2024-09-14T07:47:25.986619Z","shell.execute_reply.started":"2024-09-14T07:47:25.986378Z","shell.execute_reply":"2024-09-14T07:47:25.986400Z"}}
test_image_names = glob.glob(f"{test_directory}/*.png")
test_image = image_preprocessing(test_image_names, 'png')
test_image_dataset = np.array(test_image)
test_image_dataset = np.expand_dims(test_image_dataset, axis=3)
test_image_dataset = test_image_dataset/255.0
test_image_pairs = list(zip(test_image_names, test_image_dataset))
test_img_list = random.sample(test_image_pairs, 50)


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

    plt.savefig(f'./output/test_image2_{test_image_name}.png')

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

    plt.savefig(f'./output/test_image3_{test_img_name}.png')

