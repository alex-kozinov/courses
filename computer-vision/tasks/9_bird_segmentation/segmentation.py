from os.path import join
from os import listdir

from matplotlib.pyplot import imread
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.backend import flatten, sum
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 210

# target metric
def get_iou(gt, pred):
    gt = flatten(gt)
    pred = flatten(pred)
    intersection = sum(gt * pred)
    return (intersection + 0.1) / (sum(gt) + sum(pred) + 0.1 - intersection)

# loss function
def get_iou_loss(gt, pred):
    return -get_iou(gt, pred)

def preprocess_image(sample):
    if len(sample.shape) == 2:
        sample = np.dstack((sample, sample, sample))
    sample = resize(sample, IMG_SIZE)
    sample = preprocess_input(sample)
    return sample

def get_data_gen(inputs, labels):
    while True:
        input = inputs.next()
        label = labels.next()
        yield input, label

def get_model(img_size):
    inputs = tf.keras.Input(shape=img_size + (3,))
    model = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_tensor=inputs,
    )

    ### [Second half of the network: upsampling inputs] ###
    x = model.output
    previous_block_activation = x

    for filters in [256, 128, 64, 32, 16]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)

    # Define the model
    model = tf.keras.Model(inputs, outputs)
    # print(model.summary())
    return model

def train_model(train_data_path):
    input_dir = train_data_path + "/images/"
    target_dir = train_data_path + "/gt/"

    train_data_gen = ImageDataGenerator(
        rescale=1.0 / 255, horizontal_flip=True, preprocessing_function=preprocess_image
    )
    train_data_masks = ImageDataGenerator(rescale=1.0 / 255, horizontal_flip=True)
    train_gen = train_data_gen.flow_from_directory(
        input_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=None,
        seed=17,
    )
    train_gen_masks = train_data_masks.flow_from_directory(
        target_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=None,
        color_mode="grayscale",
        seed=17,
    )

    input_img_paths = sorted(
        [
            join(join(input_dir, dir_name), fname)
            for dir_name in listdir(input_dir)
            for fname in listdir(join(input_dir, dir_name))
            if fname.endswith(".jpg")
        ]
    )

    model = get_model(IMG_SIZE)
    model.compile(optimizer=Adam(1e-5), loss=get_iou_loss, metrics=[get_iou])
    checkpointer = [
        ModelCheckpoint(filepath="segmentation_model.hdf5", save_weights_only=False,)
    ]
    len_train_gen = len(input_img_paths)
    steps_per_epoch = len_train_gen // BATCH_SIZE

    hist = model.fit(
        get_data_gen(train_gen, train_gen_masks),
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        callbacks=checkpointer,
        shuffle=False,
    )
    model1 = load_model("segmentation_model.hdf5", compile=False)
    model1.save("segmentation_model.hdf5")
    return model


def predict(model, img_path):
    img = imread(img_path)
    needed_shape = img.shape[:2]
    img = resize(img, (224, 224, 3))
    pred = model.predict(img[None, ...])
    needed_pred = resize(pred[0, :, :, 0], needed_shape)
    return needed_pred