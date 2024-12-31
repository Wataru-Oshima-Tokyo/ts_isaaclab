#!/bin/bash

set -e

# Check for UTF-8 locale
locale

# Update and install locales
sudo apt update -y
sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Verify settings
locale

sudo add-apt-repository -y universe



# Install various tools and dependencies
sudo apt update -q -y && sudo apt install -y \
  software-properties-common \
  curl \
  build-essential \
  cmake \
  iputils-ping \
  iproute2 \
  libasio-dev \
  libpcap-dev \
  python3-pip \
  snapd \
  git \
  liblcm-dev \
  tmux \
  vim \
  nano \
  wget \
  espeak \
  libusb-1.0-0-dev \
  zsh \
  linuxptp \
  kmod \
  rapidjson-dev \
  libqt5serialport5-dev \
  software-properties-common \
  gnome-terminal \
  xterm \
  dbus-x11 \
  libcanberra-gtk-module \
  libcanberra-gtk3-module \
  usbutils \
  socat \
  libhidapi-dev \
  libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev \
  libgstreamer-plugins-bad1.0-dev \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav \
  gstreamer1.0-tools \
  gstreamer1.0-x \
  gstreamer1.0-alsa \
  gstreamer1.0-gl \
  gstreamer1.0-gtk3 \
  gstreamer1.0-qt5 \
  gstreamer1.0-pulseaudio

echo "source $HOME/miniconda3/etc/profile.d/conda.sh" >> $HOME/.bashrc
echo "export PYTHONNOUSERSITE=True" >> $HOME/.bashrc

source $HOME/.bashrc