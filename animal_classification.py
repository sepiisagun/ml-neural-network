# -*- coding: utf-8 -*-
import tensorflow as tf

img_height, img_width = 128, 128
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    "Animals (copy)/train",
    image_size = (img_height, img_width),
    batch_size = batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "Animals (copy)/validation",
    image_size = (img_height, img_width),   
    batch_size = batch_size
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    "Animals (copy)/test",
    image_size = (img_height, img_width),
    batch_size = batch_size
)

model = tf.keras.Sequential(
    [
     tf.keras.layers.Rescaling(1./255),
     tf.keras.layers.Conv2D(64, 3, activation="relu"),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Conv2D(64, 3, activation="relu"),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Conv2D(64, 3, activation="relu"),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(32, activation="relu"),
     tf.keras.layers.Dense(12)
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics=['accuracy']
)

model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 10
)

model.evaluate(test_ds)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("animalModel.tflite", 'wb') as f:
  f.write(tflite_model)