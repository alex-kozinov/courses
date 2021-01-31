import os

import albumentations as A

import numpy as np
import pandas as pd

from skimage.io import imread

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        TensorBoard)
import tensorflow.keras.layers as L
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.optimizers import Adam


AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_model():
    model = Sequential()

    model.add(
        L.Conv2D(
            3,
            32,
            padding='same',
            input_shape=(100, 100, 3)
        )
    )
    for filters in [None, 64, 128, 256, 512]:
        if filters:
            model.add(L.Conv2D(filters, 3, padding='same'))
            
        model.add(L.BatchNormalization())
        model.add(L.ReLU())
        model.add(L.MaxPool2D())

    model.add(L.Flatten())
    model.add(L.Dropout(0.3))
    model.add(L.Dense(1000))
    model.add(L.BatchNormalization())
    model.add(L.ReLU())
    model.add(L.Dense(1000))
    model.add(L.BatchNormalization())
    model.add(L.ReLU())
    model.add(L.Dense(1000))
    model.add(L.BatchNormalization())
    model.add(L.ReLU())
    model.add(L.Dense(28))
    model.compile(
        optimizer=Adam(learning_rate=0.0004),
        loss='mse',
    )
    return model

def get_transforms():
    train_transform = A.Compose(
        [
            A.Resize(width=100, height=100, always_apply=True),
            A.Rotate(limit=30, p=0.5),
            A.Normalize(always_apply=True)
        ],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=True)
    )

    val_transform = A.Compose(
        [
            A.Resize(width=100, height=100, always_apply=True),
            A.Normalize(always_apply=True)
        ],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=True)
    )
    return train_transform, val_transform


def parse_sample(
    sample
):
    num_str = sample['filename']
    coordinates = sample.values[1:]
    points = np.array(
        [
            (coordinates[2 * i], coordinates[2 * i + 1])
            for i in range(14)
        ]
    )
    return num_str, points


def dataset_gen(
    ds,
):
    def gen():
        for i in range(ds.shape[0]):
            sample = ds.iloc[i]
            img_path, points = parse_sample(sample)
            yield img_path, points
    return gen


@tf.autograph.experimental.do_not_convert
def load():
    def f(
        img_path,
        kp
    ):
        img = imread(img_path.decode('utf-8'))
        if len(img.shape) == 2:
            img = np.dstack([img, ]*3)
        return img.astype(np.float32), kp.astype(np.float32)
    return f


@tf.autograph.experimental.do_not_convert
def preprocess(
    transform,
    reserve_transform,
    num_trials=10
):
    def f(
        img: np.array,
        kp: np.array
    ):
        kp = np.clip(kp, 0, None)
        for t in range(num_trials):
            transformed = transform(image=img, keypoints=kp)
            if len(transformed['keypoints']) == 14:
                break
        else:
            transformed = reserve_transform(image=img, keypoints=kp)
        return transformed['image'].astype(np.float32), np.array(transformed['keypoints']).reshape(-1).astype(np.float32)

    return f


def create_dataset(
    ds,
    transform,
    reserve_transform,
    num_trials=20,
    batch_size=5,
    shuffle_buffer_size=200,
    prefetch_size=AUTOTUNE,
    num_calls=AUTOTUNE,
):
    gen = dataset_gen(ds)
    dataset = tf.data.Dataset.from_generator(
        gen,
        (tf.string, tf.float32),
        output_shapes=(tf.TensorShape([]), tf.TensorShape([14, 2]))
    )

    def loader(x, y):
        return tf.numpy_function(
            func=load(),
            inp=[x, y],
            Tout=(tf.float32, tf.float32)
        )
    dataset = dataset.map(
        loader,
        num_parallel_calls=num_calls
    )


    def preprocessor(x, y):
        return tf.numpy_function(
            func=preprocess(transform,
                            reserve_transform,
                            num_trials=num_trials),
            inp=[x, y],
            Tout=(tf.float32, tf.float32)
        )
    dataset = dataset.map(
        preprocessor,
        num_parallel_calls=num_calls
    )

    def set_shapes(img, kp, img_shape=(100, 100, 3)):
        img.set_shape(img_shape)
        kp.set_shape([28])
        return img, kp
    dataset = dataset.map(set_shapes, num_parallel_calls=num_calls)

    dataset = dataset.repeat()
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_size)
    return dataset


def process_data(
    train_csv_path,
    imgs_dir,
    test_size=0.2,
    shuffle=True,
    num_trials=20,
    batch_size=5,
    shuffle_buffer_size=200,
    prefetch_size=AUTOTUNE,
    num_calls=AUTOTUNE
):
    ds = pd.read_csv(train_csv_path)
    ds['filename'] = imgs_dir + ds['filename']

    ds_train, ds_val = train_test_split(
        ds,
        test_size=test_size,
        shuffle=shuffle
    )
    
    train_transform, val_transform = get_transforms()
    
    train_dataset = create_dataset(
        ds_train,
        train_transform,
        val_transform,
        num_trials=num_trials,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        prefetch_size=prefetch_size,
        num_calls=num_calls
    )
    val_dataset = create_dataset(
        ds_val,
        val_transform,
        val_transform,
        num_trials=num_trials,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        prefetch_size=prefetch_size,
        num_calls=num_calls
    )
    return train_dataset, ds_train.shape[0], val_dataset, ds_val.shape[0]



def get_callbacks(checkpoint_dir):
    reduce_lr = ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=7,
        min_delta=0.2,
        min_lr=1e-7,
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='loss',
        patience=20,
        min_delta=0.2,
        verbose=1
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + "facepoints_model.hdf5",
        monitor='val_loss',
        verbose=1,
        mode='min',
        save_best_only=True
    )
    return [reduce_lr, early_stopping, model_checkpoint_callback]


def train_detector(
        train_gt,
        train_img_dir,
        fast_train=False,
        checkpoint_dir="",
        batch_size=256,
):
    if fast_train:
        return

    model = get_model()
    
    ds_train, len_train, ds_val, len_val = process_data(
        train_gt,
        train_img_dir,
        batch_size=batch_size,
        test_size=0.001,
        shuffle_buffer_size=10,
        num_trials=2,
    )

    steps_per_epoch = np.ceil(len_train / batch_size).astype(int)
    validation_steps = np.ceil(len_val / batch_size).astype(int)

    model.fit(
        ds_train,
        epochs=1000,
        validation_data=ds_val,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=get_callbacks(checkpoint_dir)
    )
    return model


def read_img_shapes(filenames, img_dir):
    img_shapes = []
    for fname in filenames:
        img_shapes.append(imread(os.path.join(img_dir, fname)).shape[:2])
    return np.asarray(img_shapes)
    
    
def test_dataset_gen(
    ds,
    dir_path
):
    def gen():
        for filename in os.listdir(dir_path):
            img_path = os.path.join(dir_path, filename)
            points = np.ones([14, 2]) * 20
            yield img_path, points
    return gen

def create_test_dataset(
    test_img_dir,
    transform,
    reserve_transform,
    batch_size=5,
):
    dataset = tf.data.Dataset.from_generator(
        test_dataset_gen,
        (tf.string, tf.float32),
        output_shapes=(tf.TensorShape([]), tf.TensorShape([14, 2]))
    )

    def loader(x, y):
        return tf.numpy_function(
            func=load(),
            inp=[x, y],
            Tout=(tf.float32, tf.float32)
        )
    dataset = dataset.map(
        loader,
        num_parallel_calls=num_calls
    )

    def preprocessor(x, y):
        return tf.numpy_function(
            func=preprocess(transform,
                            reserve_transform,
                            num_trials=num_trials),
            inp=[x, y],
            Tout=(tf.float32, tf.float32)
        )
    dataset = dataset.map(
        preprocessor,
        num_parallel_calls=1
    )

    def set_shapes(img, kp, img_shape=(100, 100, 3)):
        img.set_shape(img_shape)
        kp.set_shape([28])
        return img, kp
    
    dataset = dataset.map(set_shapes, num_parallel_calls=1)
    dataset = dataset.batch(test_batch_size)
    dataset = dataset.prefetch(1)
    return dataset


def detect(model, test_img_dir):
    filenames = os.listdir(test_img_dir)
    test_bs = 4
    
    _, val_transform = get_transforms()
    
    test_dataset = create_test_dataset(
        test_img_dir,
        val_transform,
        val_transform,
        batch_size=test_bs,
    )
    pred = model.predict(
        test_dataset,
        steps=np.ceil(len(filenames) / test_bs)
    )
    shapes = read_img_shapes(filenames, test_img_dir)
    pred[:, ::2] *= shapes[:, 1:] / 100
    pred[:, 1::2] *= shapes[:, :1] / 100
    return {f: p for f, p in zip(filenames, pred)}
