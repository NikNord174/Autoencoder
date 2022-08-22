# Autoencoder
Neural network to compress and decompress images.

# Getting started
The project was made using Python 3.10 and tensor library PyTorch. Other necessary packages are noticed in requirements.txt. Be aware, that in requirements.txt torch==1.13.0.dev20220703 is noticed. This version allows accelerating the learning process on Mac with Apple Silicon. However, it can cause some issues with other OS, so you may want to change it to stable version 1.12.

First, clone the repository from Github and switch to the new directory:

```PYTHON
git clone git@github.com/USERNAME/{{ project_name }}.git
cd {{ project_name }}
```

Then create a new environment using

```PYTHON
python3 -m venv venv
```

and run environment

```PYTHON
source venv/bin/activate
```

Further activate the environment and instal all requirements using

```PYTHON
pip3 install -r requirements.txt
```

You may vary parameters including number of epochs in constants.py.
To start project type in terminal

```PYTHON
python3 main.py
```
