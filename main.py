import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")

import random
import tensorflow as tf
import pickle
from datetime import datetime
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Convolution2D, BatchNormalization, Flatten , MaxPooling2D, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

def get_architecture_dataset(directory, image_height, image_width, batch_size):
     seed = random.getrandbits(32)

     train = tf.keras.utils.image_dataset_from_directory(
    directory, labels='inferred', label_mode='int',
    class_names=None, color_mode='rgb', batch_size=batch_size, image_size=(image_height,
    image_width), shuffle=True, seed=seed, validation_split=0.2, subset='training')
 
     test = tf.keras.utils.image_dataset_from_directory(
    directory, labels='inferred', label_mode='int',
    class_names=None, color_mode='rgb', batch_size=batch_size, image_size=(image_height,
    image_width), shuffle=True, seed=seed, validation_split=0.2, subset='validation')

     train = train.map(lambda x, y: (tf.divide(x, 255), y))
     test = test.map(lambda x, y: (tf.divide(x, 255), y))

     return train, test

def densenet(image_height, image_width, channels):
    dmodel = DenseNet169(
        include_top = False,
        weights = "imagenet",
        input_tensor = None,
        pooling = None
    )

    dmodel.trainable = True

    for layer in dmodel.layers:
        if 'conv5' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False

    input = tf.keras.Input(shape = (image_height, image_width, channels))
    preprocess = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (224, 224)), name='lamb')(input)
    initializer = tf.keras.initializers.he_normal(seed = 32)

    layer = dmodel(inputs = preprocess)
    layer = tf.keras.layers.Flatten()(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.Dense(units = 256,
                        activation = 'relu',
                        kernel_initializer = initializer
                        )(layer)
    layer = tf.keras.layers.Dropout(0.4)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.Dense(units = 128,
                        activation = 'relu',
                        kernel_initializer = initializer
                        )(layer)
    layer = tf.keras.layers.Dropout(0.4)(layer)
    layer = tf.keras.layers.Dense(units = 25,
                        activation = 'softmax',
                        kernel_initializer = initializer
                        )(layer)
    model = tf.keras.models.Model(inputs = input, outputs = layer)
    return model


if __name__ == "__main__":
    # Parameters
    directory = "./archive/architectural-styles-dataset"
    image_height = 700
    image_width = 700
    channels = 3
    batch_size = 32
    epochs = 50
    learning_rate = 2e-4

    train, test = get_architecture_dataset(directory, image_height, image_width, batch_size)

    model = densenet(image_height, image_width, channels)
    model.compile(optimizer = Adam(learning_rate = learning_rate), loss = "sparse_categorical_crossentropy", metrics=['accuracy'])
    model.fit(train, batch_size = batch_size, epochs = epochs, verbose = 2)

    print("\nnow testing...\n")
    model.evaluate(test, batch_size = batch_size, verbose = 2)

    print("\npickling model\n")
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    outfile = open(f"modelout_{now}", "wb")
    pickle.dump(model, outfile)
    outfile.close()

    tf.keras.utils.plot_model(model, show_shapes = True)

