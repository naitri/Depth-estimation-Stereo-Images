import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm



def get_blocks(img,window_size):
    height,width = img.shape
    window_set=[]
    window_loc = []
    for y in range(window_size, height - window_size):
        for x in range(window_size, width - window_size):
            img_window = img[y:y + window_size, x:x + window_size]
            window_set.append(img_window)
            window_loc.append((y,x))
    return window_set,window_loc

def get_boundary_condition(x,y,search_range,width,height):
    

    range_min = max(0, x-search_range)
    range_max = min(width,x+search_range )

    return range_max,range_min

def get_sdd(imgl_window,imgr_window):
    max = 1e10
    if imgl_window.shape == imgr_window.shape:
        diff = np.abs(imgl_window - imgr_window)

        SSD = np.sum(np.square(imgl_window - imgr_window))
        return SSD
    else:
        return max

def get_disparity(img1,img2,search_range,type):

    if type==1:
        print("method 1")
        imgl = img1.copy()
        imgr = img2.copy()
        disparityImg = np.zeros(img2.shape)
        window_size = 21
        
        
        #get a patch in left image
        height,width = img1.shape
        for y in tqdm(range(window_size,height-window_size)):
            for x in range(window_size,width-window_size):

                
                #get window boundary conditions
                # range_max,range_min = get_boundary_condition(x,y,50,width,height)
                range_min = max(0, x-100)
                range_max = min(width,x+100)
                imgl_window = imgl[y:y+window_size, x:x+window_size]
               
                first = True
                min_val = None
                index = None
                #search for corresponding patch in right image
                for i in range(range_min,range_max):

                    imgr_window = imgr[y:y+window_size, i:i+window_size]
              

                    SSD = get_sdd(imgl_window,imgr_window)
                    if first:
                        min_val = SSD
                        index = i
                        first = False
                    else:
                        if SSD < min_val:
                            min_val = SSD
                            index = i
                


                d = np.abs(index - x)

                disparityImg[y,x] = d
        
        return disparityImg

    if type == 2:
        print("method 2")
        # img1 = cv2.resize(img1, (int(img1.shape[1]*0.5), int(img1.shape[0] *0.5)))
        # img2 = cv2.resize(img2, (int(img2.shape[1]*0.5), int(img2.shape[0] *0.5)))
        imgl = img1.copy()
        imgr = img2.copy()
        disparityImg = np.zeros(img2.shape)
        window_size = 15
        #get a patch in left image
        height,width = img1.shape
        
        imgl = imgl.astype(np.int32)
        imgr = imgr.astype(np.int32)

        for y in tqdm(range(0,height-window_size)):

            imgl_window_set = []
            imgr_window_set = []

            for x in range(0,width-window_size):
                left_window = imgl[y:y+window_size, x:x+window_size]
                right_window = imgr[y:y+window_size, x:x+window_size]
                imgl_window_set.append(left_window.flatten())
                imgr_window_set.append(right_window.flatten())

            imgl_window_set = np.array(imgl_window_set)
            imgr_window_set = np.array(imgr_window_set)
            print(len(imgl_window_set))
            for i in range(len(imgl_window_set)):

                SSD = np.sum(np.square(imgr_window_set- imgl_window_set[i]),axis=1)
                index = np.argmin(SSD)
                disparity = np.abs(i - index)
                # disparity = ((disparity+1) / 2) * 255
                disparityImg[y,i] = np.uint8(disparity)



        return disparityImg

    else:   
        return np.zeros(img2.shape)




def get_depth(disparityImg,b,f):
    depth = (b * f) / (disparityImg + 1e-10)
    depth[depth > 10000] = 10000

    depth_map = np.uint8(depth * 255 / np.max(depth))
    return depth_map
