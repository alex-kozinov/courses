set -o xtrace

setup_root() {
    apt-get install -qq -y \
        libsm6 \
        libgtk2.0-dev \
        libgl1-mesa-dev
    > /dev/null

    apt-get install -qq -y \
        python3-pip \
        python3-tk \
    > /dev/null

    python3 -m pip install --upgrade pip
    pip3 install -qq \
        numpy \
        scikit-image \
        opencv-python \
        opencv-contrib-python
}

setup_checker() {
    :
}

"$@"