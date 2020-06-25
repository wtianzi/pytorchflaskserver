import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from pytorch_model import *
from collections import OrderedDict
from imagepreprocess import *
import datetime;
from datetime import datetime
'''
works for files transfer
'''
app = Flask(__name__)

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
model_path = 'modelf_epoch_99.pth'
model = FineAP(3)
device = torch.device('cpu')
state_dict = torch.load(model_path, map_location=device)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()
'''

preloadimgs=get_images(forlder='preimages/',imgcount=16)
#print(preloadimgs.shape)
# this is where we load model, a predefined model for testing
def get_prediction(test_array):
    # test_array=np.array(frames[0:16])    
    # test_array = get_images('preimages/')  
    test_array = np.array(test_array)
    test_array = np.rollaxis(test_array, -1, 0)
    #print(test_array.shape)
    test_array = torch.tensor(test_array, dtype=torch.float32, requires_grad=False)
    
    t1=datetime.now()
    '''
    p = model(test_array[None]).squeeze()
    preds=p.cpu().detach().numpy()
    print(preds.shape)
    '''
    
    with torch.no_grad():        
        p = model(test_array[None]).squeeze()
        preds=p.cpu().numpy()
        print(preds.shape)
    
    
    t2=datetime.now()
    print("start:",t1.time(),"end:",t2.time(),"model gap: ", (t2-t1).total_seconds())
    
    frame=preds[-1]
    nx = np.argmax((np.max(frame, axis=1)))
    ny = np.argmax((np.max(frame, axis=0)))
    x = int(round(ny * 20))
    y = int(round(nx * 11.25))
    reslist=[x, y]

    return reslist

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        t1=datetime.now()
        global preloadimgs
        file = request.files['file']
        img_bytes = file.read()
        #UpdateImages(img_bytes)        
        t2=datetime.now()
        preloadimgs = updateimagearr(arrimages=preloadimgs,image_bytes=img_bytes)
        
        res = get_prediction(preloadimgs)
        t3=datetime.now()
        print("get:",t1.time(),"mid:",t2.time()," send: ",t3.time(),"gap: ", (t3-t1).total_seconds())
        return jsonify({"pos": res})


@app.route('/')
def hello():
    return 'Hello World:)'


if __name__ == '__main__':
    imagearr = get_images()
    app.run()
