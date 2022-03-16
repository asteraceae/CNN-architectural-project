#for windows
#import os
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")

import random
import pickle
from datetime import datetime
from tensorflow import divide
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Lambda, BatchNormalization, Flatten, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.image import resize
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.applications.densenet import DenseNet169
from tensorflow.keras.utils import image_dataset_from_directory, plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import he_normal

def get_architecture_dataset(directory, image_height, image_width, batch_size): 
    seed = random.getrandbits(32)
    '''
    train = image_dataset_from_directory(
    directory, labels='inferred', label_mode='int',
    class_names=None, color_mode='rgb', batch_size=batch_size, image_size=(image_height,
    image_width), shuffle=True, seed=seed, validation_split=0.2, subset='training')

    test = image_dataset_from_directory(
    directory, labels='inferred', label_mode='int',
    class_names=None, color_mode='rgb', batch_size=batch_size, image_size=(image_height,
    image_width), shuffle=True, seed=seed, validation_split=0.2, subset='validation')
    
    train = train.map(lambda x, y: (divide(x, 255), y))
    test = test.map(lambda x, y: (divide(x, 255), y))
'''

    aug = image.ImageDataGenerator(
        rotation_range = 5, # rotation
        width_shift_range = 0.2, # horizontal shift
        height_shift_range = 0.2, # vertical shift
        horizontal_flip = True,
        validation_split=0.2,
        
      ) 

    train = aug.flow_from_directory(
        directory = directory,
        color_mode = "rgb",
        batch_size = batch_size,
        target_size = (image_height, image_width),
        shuffle = True,
        seed = seed,
        subset = 'training'
    )

    test = aug.flow_from_directory(
        directory = directory,
        color_mode = "rgb",
        batch_size = batch_size,
        target_size = (image_height, image_width),
        shuffle = False,
        seed = seed,
        subset = 'validation'
    )

    return train, test

def densenet(image_height, image_width, channels):
    dmodel = DenseNet169(
        include_top = False,
        weights = "imagenet",
        input_tensor = None,
        input_shape = (224, 224, 3),
        pooling = None
    )

    dmodel.trainable = True

    for layer in dmodel.layers:
        if 'conv5' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False

    input = Input(shape = (image_height, image_width, channels))
    preprocess = Lambda(lambda x: resize(x, (224, 224)), name='lamb')(input)
    initializer = he_normal(seed = 32)

    layer = dmodel(inputs = preprocess)
    layer = Flatten()(layer)
    layer = BatchNormalization()(layer)
    layer = Dense(units = 256,
                        activation = 'relu',
                        kernel_initializer = initializer
                        )(layer)
    layer = Dropout(0.4)(layer)
    layer = BatchNormalization()(layer)
    layer = Dense(units = 128,
                        activation = 'relu',
                        kernel_initializer = initializer
                        )(layer)
    layer = Dropout(0.4)(layer)
    layer = Dense(units = 25,
                        activation = 'softmax',
                        kernel_initializer = initializer
                        )(layer)
    model = Model(inputs = input, outputs = layer)
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
    model.compile(optimizer = Adam(learning_rate = learning_rate), loss = "categorical_crossentropy", metrics=['accuracy'])
    
    model.fit(train, batch_size = batch_size, epochs = epochs, verbose = 1)

    print("\nnow testing...\n")
    model.evaluate(test, batch_size = batch_size, verbose = 1)

    print("\npickling model\n")
    now = datetime.now().strftime('modelout_%Y-%m-%d%_H%M%S.pickle')
    outfile = open(now, "wb")
    pickle.dump(model, outfile)
    outfile.close()

    plot_model(model, show_shapes = True)

