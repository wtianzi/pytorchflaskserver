import os
import numpy as np
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def get_frames(vidpath = None,overlapping = False,train=True,n_frames=16):
    
    #print(os.listdir(vidpath))
    if train:
        inp = []
        vid_frames = []
        vidpath = 'train/videos'
    
        
        for vid in os.listdir(vidpath):
        #print(vid,'read')
            cap = cv2.VideoCapture(vidpath+'/'+vid)
            opframes=[]
            frames = []
            i=0
       
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                opframes.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #frame = cv2.resize(frame1,(64,64))
                frames.append(cv2.cvtColor(cv2.resize(frame,(64,64)), cv2.COLOR_BGR2RGB))
            
    
            if overlapping:
                for i in range(len(frames)-n_frames):
                    inp.append(np.array(frames[i:i+n_frames]))


        #Omitting last frame because gaze coordinates of the last frame are not available 
            else:
                frx = list(range(0,len(frames)-1,n_frames))
       
                for i in range(len(frx)):
                    try:
                        inp.append( np.array( frames[frx[i]:frx[i+1]] )) 
                    except IndexError:
                        pass
        
        
            vid_frames.append(opframes)
        cap.release()
        cv2.destroyAllWindows()
        #print(len(inp),inp[0].shape,len(frames),frames[0].shape)
        return np.array(inp),vid_frames
    else:
        vidpath = vidpath
        #print(vidpath)
        inp = []
        
        cap = cv2.VideoCapture(vidpath)
        opframes=[]
        frames = []
        i=0
       
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            opframes.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #frame = cv2.resize(frame1,(64,64))
            frames.append(cv2.cvtColor(cv2.resize(frame,(64,64)), cv2.COLOR_BGR2RGB))
        
        frx = list(range(0,len(frames)-1,n_frames))
       
        for i in range(len(frx)):
            try:
                inp.append( np.array( frames[frx[i]:frx[i+1]] )) 
            except IndexError:
                pass
        
        cap.release()
        cv2.destroyAllWindows()
        #print(len(inp),inp[0].shape,len(frames),frames[0].shape)
        return np.array(inp),opframes
        
        


def gaussian_k(x0,y0,sigma, width, height):
        """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
        """
        x = np.arange(0, width, 1, float) ## (width,)
        y = np.arange(0, height, 1, float)[:, np.newaxis] ## (height,1)
        return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))

def generate_hm(height, width ,coords,s=1.5):
        
    
    inpgrid = []
    hm = np.zeros((height, width), dtype = np.float32)
    
    if not np.array_equal(coords, [-1,-1]):
         
        hm[:,:] = gaussian_k(coords[0],coords[1],
                                        s,height, width)
    else:
        hm[:,:] = np.zeros((height,width))
        
        
    
        
        
    return hm

        



def get_grids(labels_path = None,overlapping = False,train=True,n_frames=16):
    
    inpgrid = []
    gridarr=[]
    heatmap = []
    if train:
        labels_path = 'train/labels'
        vidpath = 'train/videos'
    else:
        labels_path = 'test/test_labels'
        vidpath = 'test/test_videos'
    gridlist = os.listdir(labels_path)
    vidlist = os.listdir(vidpath)
    newgridlist=[]
    for vid in vidlist:
        
        
        file = vid[:-4] + '.txt'
        #print(file,'read')
        gridarr=[]
        heatarr = []
        with open(labels_path+'/'+file, "r") as f:
            
        
            content = f.readlines()
            for i in range(len(content)):
            
                grid = np.zeros((4,4))
                x,y = content[i].strip().split(',')[0:2]
                coords = ([round(int(x)/20),round(int(y)/11.25)])
                grid[int(int(y)//180),int(int(x)//320)]=1
                gridarr.append(grid)
                heatarr.append(generate_hm(64, 64 ,coords,s=1.5))
            
        
       
        
        if overlapping:
            for i in range(len(gridarr)-n_frames+1):
                inpgrid.append(np.array(gridarr[i:i+n_frames]))
           
    
        else:
            gdx = list(range(0,len(gridarr),n_frames))
       
            for i in range(len(gridarr)):
                try:
                    inpgrid.append( np.array( gridarr[gdx[i]:gdx[i+1]] ))
                    heatmap.append(np.array(heatarr[gdx[i]:gdx[i+1]]))
                    
                except IndexError:
                    pass
        
        
            
    #print(np.expand_dims(np.array(heatmap).shape))
    return np.expand_dims(np.array(inpgrid),axis=-1),np.expand_dims(np.array(heatmap),axis=-1)


if __name__ == '__main__':
    
   
    #inp_array,frames_array = get_frames(vidpath='test/test_videos/Camera3_taskA_trial4_1573164947973.avi',train=False)
    inp_grid,hm = get_grids(train = False)
    print('heatmap = ',hm.shape,inp_grid.shape)
    plt.imshow(hm[0][0].squeeze())
    plt.savefig('fig.png')
    np.save('test_KLhm',hm)
    #print(type(inp_array))
    #print(len(frames_array),len(frames_array[0]),len(frames_array[1]))
    #print(len(inp_grid))
    #print((inp_grid).shape)
    #print(frames_array[0].shape,frames_array[1].shape)
    