
# Deploy Detectron2 Model via a REST API wtih Flask

#### Task

Build a simple REST API. An API user should be able to post an image to your API and receive the image with drawn mask and the percentage between masks and unmasked pixels.

The user need not do model training.

#### Approach

I used the Flask framework in this task.

- Use the model weights that have been trained on the custom pandas dataset.
- Register the custom dataset
- Run Flask and post an image to the API
- Retrieve image with drawn masks

## 1. Setup 

This installation instruction is for Ubuntu 18.04. 

### 1.1 Change directory to detectron2-flask-api folder

```
cd detectron2-flask-api
```

### 1.2 Create and activate virtual environment

```
python3 -m venv dev
source dev/bin/activate
```

### 1.3 Install PyTorch without CUDA

```
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### 1.4 Install Detectron2 repo

Clone the Detectron2 repo and install

```
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo
```

Register custom dataset (pandas dataset)

```
python dataset.py
```

### 1.5 Install OpenCV

```
sudo apt update
sudo apt install python3-opencv
```

Verify installation of OpenCV

```
python -c "import cv2; print(cv2.__version__)"
```

My cv2 version is 3.2.0.

### 1.6 Install Flask

```
pip install Flask
```

### 1.7 Download Detectron2 model weights

As the weights file size is too large to be hosted on GitHub, download it from [this link](https://drive.google.com/drive/folders/1ATHGXHwJ-9R3QmsE3nauwK85zCYkpWh-?usp=sharing) and save it to a folder called `output`.

These weights were the output from retraining the Detectron2 model on my pandas dataset.

## 2. Run API

### 2.1 Run Flask server

In one terminal, run a Flask development server.

```
FLASK_ENV=development FLASK_APP=app.py flask run
```

### 2.2 Post image to API

In another terminal, post image to API.

```
curl -X POST -F file=@data/<IMAGE_NAME>.jpg http://localhost:5000/predict  --output data/result.png
```

For example,

```
curl -X POST -F file=@data/t08.jpg http://localhost:5000/predict  --output data/result.png
```

The resulting image with bounding box and percentage between masks and unmasked pixels will be saved to the data folder.

Image posted:

![t08](https://github.com/agrilive/detectron2-instance-segmentation/blob/master/flask-api/data/t08.jpg?raw=true))

Resulting image:

![result](https://github.com/agrilive/detectron2-instance-segmentation/blob/master/flask-api/data/result.png?raw=true))


