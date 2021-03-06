set -o xtrace

setup_root() {
    apt-get install -qq -y \
        python3-pip \
        python3-tk \
        ffmpeg \
        libsm6 \
        libxext6
    python3 -m pip install --upgrade pip six setuptools
    pip3 install -qq \
        pytest \
        scikit-image \
        scikit-learn \
        matplotlib \
        h5py==2.10.0 \
        tensorflow \
        tensorflow-addons \
        pandas \
        tqdm \
        moviepy \
        albumentations
}

setup_checker() {
    python3 -c 'import matplotlib.pyplot'
    python3 -c 'from tensorflow.keras.applications import ResNet50, Xception, EfficientNetB4, MobileNetV2;\
                model = MobileNetV2(weights="imagenet", include_top=False);\
                model = ResNet50(weights="imagenet", include_top=False);\
                model = Xception(weights="imagenet", include_top=False);\
                model = EfficientNetB4(weights="imagenet", include_top=False);'
}

"$@"