# Installation
To install goggle detection on a Raspberry Pi, you will need to install its hardware and software dependencies.

## Hardware
Simply follow [this guide](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html) to install the AI hat+ 2 that this software requires.

## Raspberry Pi configuration
You will need a Raspberry Pi 5 with a fresh install of Raspberry Pi OS and an internet connection. For ease of use, you may also wish to install Raspberry Pi Connect to the Raspberry Pi to facilitate remote desktop connections.

## Software
You will first need to install the Hailo Software Suite onto the Raspberry Pi.
1. Follow [this guide](https://www.raspberrypi.com/documentation/computers/ai.html) from the Raspberry Pi website to install the Hailo Runtime.
2. In the root directory of this project, create a virtual environment with the virtual environment tool (Conda, venv, etc) of your choice, ensuring that the system site packages are included in the virtual environment. For instance:
```
python3 -m venv .venv --system-site-packages
```
will install a virtual environment under the folder .venv with all system site packages included.
3. You will then need to enter the virtual environment.
```
source ./.venv/bin/activate
```
4. In the environment, you'll need to follow [this guide](https://github.com/hailo-ai/hailo-apps/blob/main/doc/user_guide/installation.md) to install the hailo-apps dependency into the project. Follow the *installing via pip* instructions.
5. Finally, you can install all other dependencies in the requirements.txt file in this directory.
```
pip install -r requirements.txt
```