import tensorflow as tf

print("=" * 50)
print("TensorFlow Test")
print("=" * 50)

print("TensorFlow Version :", tf.__version__)

print("Num GPUs Available :", len(tf.config.list_physical_devices("GPU")))

print("TensorFlow Imported Successfully!")

print("\nDone.")