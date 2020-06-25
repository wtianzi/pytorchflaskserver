import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from pytorch_model import *
from collections import OrderedDict
'''
works for files transfer
'''
app = Flask(__name__)
'''
# GUP LOADING
model_path='modelf_epoch_99.pth'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9"
device = torch.device("cuda")
model = FineAP(3).to(device)
model = nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7,8,9])
model.load_state_dict(torch.load(model_path)) 
model.eval()

'''
# CPU LOADING
model_path='modelf_epoch_99.pth'
model = FineAP(3)
device = torch.device('cpu')
state_dict = torch.load(model_path,map_location=device)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()

def get_images(forlder='preimages/'):
    # load images from '/preimages/'
    frames=[]
    inp = []
    n_frames=16
    
    for i in range(10,30):
        filename="{0}/{1}.bmp".format(forlder,i)
        img=cv2.imread(filename,0)
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #print("load"+filename)
        
    frx = list(range(0,len(frames)-1,n_frames))
    for i in range(len(frx)):
        try:
            inp.append(np.array(frames[frx[i]:frx[i+1]])) 
        except IndexError:
            pass
    cv2.destroyAllWindows()
    return np.array(inp)

# this is where we load model, a predefined model for testing
def get_prediction():
    preds=[]
    
    test_array = get_images('preimages/')
    test_array = np.rollaxis(test_array,-1,1)
    test_array = torch.tensor(test_array,dtype=torch.float32,requires_grad=False)
    with torch.no_grad():
        for arr in test_array:
            p = model(arr[None]).squeeze()
            preds.append(p.cpu().numpy())
    
    frameindex=0
    reslist = []
    for arrframes in preds:
        for frame in arrframes:
            nx = np.argmax((np.max(frame,axis=1)))
            ny = np.argmax((np.max(frame,axis=0)))
            x = int(round(ny*20))
            y = int(round(nx*11.25))
            
            reslist.append([x,y])
    
    return reslist

def get_eyegaze_prediction(inputarr):
    outputs = model.forward(inputarr)

    return outputs

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        #batch_size=1
        #x = torch.randn(batch_size, 3,16, 64, 64, requires_grad=True)
        
        #class_id, class_name = get_prediction(image_bytes=img_bytes)
        #res=get_eyegaze_prediction(inputarr = x)
        res = get_prediction()
        print(res)
        return jsonify({'class_id': 1, 'class_name': 2,"rect":0})

@app.route('/')
def hello():
    return 'Hello World:)'

if __name__ == '__main__':
    app.run()
