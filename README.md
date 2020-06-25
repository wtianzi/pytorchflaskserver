# pytorchflaskserver
Use PyTorch server and flask to serve a model inference calculation.

Given a trained model modellf_epoch_99.pth, a linux REST server is set to do inference, while using python requests or c# WebClient to send the input images.

# PyTorch server:
Refer to the instruction of [TorchServe](https://github.com/pytorch/serve) and [Flask](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html) to setup the environment. Or you can use the requirements.txt in the /linuxserver to install the packages.

To load a trained model in the app.py, you need the model structure and the model parameters. 
- The model structure is defined in a .py file, which has the pipeline of the model.
- The model parameters are the trained matrix that can be saved as .pth, .pt, .onnx files.

To run a flask for remote connection: 'flask run --host 0.0.0.0"

# Python client:
'''python
import requests
test_url = 'http://serveripaddress:5000/predict'

# single file post
resp = requests.post(test_url,files={"file": open('0.bmp','rb')})

## multiple files post
# resp = requests.post(test_url,files={"file1": open('kitten.jpg','rb'),"file2": open('kitten1.jpg','rb')})

## byte string post
#content_type = 'image/jpeg'
#headers = {'content-type': content_type}
#img = cv2.imread('kitten.jpg')
#_, img_encoded = cv2.imencode('.jpg', img)
#response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
#print(json.loads(response.text))

print(resp.json())
''''


* Notes: 
- my NVIDIA driver is old (NVIDIA 10010), so I use 'pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html' to install the torch and torchvision.
- anaconda and pip virtualenv can both work, but the opencv sometimes can not be recognized, just need to 'import opencv as cv2' in conda environment, or 'import cv2' in pip virtulenvironment.
- If you want to load in CPU, just change the name of app_cpu.py to app.py. 

# About ML.net and .ONNX format
ML.NET package enables the training and inference in c#. But currently, the ML.NET doesn't support 3D convTrans, which we need for this model, so we can't use the ML.NET.

If you have a 1D or 2D model, we can use the setup below:
