import tensorflow as tf

print(tf.__version__)
# print(tf.test.is_gpu_available())
# import tensorflow as tf
print("GPUs:", tf.config.list_physical_devices('GPU'))
