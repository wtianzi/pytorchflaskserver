import io
import json
import numpy as np
import cv2
from PIL import Image

def get_images(forlder='preimages/',imgcount=16):
    # load images from '/preimages/'
    frames = []
    inp = []
    n_frames = 16

    for i in range(10, 10+imgcount):
        filename = "{0}/{1}.bmp".format(forlder, i)
        img = cv2.imread(filename, 0)
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # print("load"+filename)

    #frx = list([0,16,32,48])
    #inp=[frames[0:16],frames[1:17],...,]
    cv2.destroyAllWindows()
    return frames
    #return np.array(frames)

def updateimagearr(arrimages,image_bytes):
    pil_image = Image.open(io.BytesIO(image_bytes))
    opencv_Image = np.array(pil_image)
    
    #opencv_Image = cv2.cvtColor(numpy.array(pil_image),cv2.CV_RGB2BGR)
    #opencv_image=cv2.cvtColor(opencv_Image, cv2.cv.CV_BGR2RGB)

    '''
    test1=np.array(arrimages)
    test2=np.array(opencv_Image)
    
    print(test1.shape)
    print(test2.shape)
    '''
    arrimages = arrimages[1:]
    arrimages.append(opencv_Image.tolist())
    
    return arrimages