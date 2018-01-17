import os
import numpy as np
from PIL import Image, ImageFilter
import cv2


parent_path = '/home/web/nn'
all_directories = os.listdir(parent_path + '/')

training_inputs = []
training_outputs = []

test_inputs = []
test_outputs = []

count = 0

def imageprepare(argv):    
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (32,32), (255)) #creates white canvas of 28x28 pixels
    
    if width > height: #check which dimension is bigger
        #Width is bigger. Width becomes 20 pixels.
        nheight = int(round((30/width*height),0)) #resize height according to ratio width
        if (nheight == 0): #rare case but minimum is 1 pixel
            nheight = 1  
        # resize and sharpen
        img = im.resize((30,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((32 - nheight)/2),0)) #caculate horizontal pozition
        newImage.paste(img, (4, wtop)) #paste resized image on white canvas
    else:
        #Height is bigger. Heigth becomes 20 pixels. 
        nwidth = int(round((30/height*width),0)) #resize width according to ratio height
        if (nwidth == 0): #rare case but minimum is 1 pixel
            nwidth = 1
         # resize and sharpen
        img = im.resize((nwidth,30), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((32 - nwidth)/2),0)) #caculate vertical pozition
        newImage.paste(img, (wleft, 4)) #paste resized image on white canvas    
    #newImage.save("sample.png")
    tv = list(newImage.getdata()) #get pixel values    
    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255-x)/255 for x in tv] 
    return tva
    #print(tva)

def load_data():
    global count

    for i in [0,1,2,3,4,5,6,7,8,9]:
            #'a','b','c','d','e','f','g','h','i','j','k','l',
            #'m','n','o','p','q','r','s','t','u','v','w_','x','y','z',
            #'A','B','C','D','E','F','G','H','I','J','K','L',
            #'M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','white']:
        
        ## directory 0,1,2,3,4,5,6,7,8,9
        for z in all_directories:
            output_vector = np.zeros((63,1))
            path = parent_path + '/' + z

            if z.startswith(str(i)) and os.path.isdir(path):
                print(path)

                total_files = len(os.listdir(path))
                if total_files <= 2000:
                    training_files = int(90/100 * total_files)
                    test_files = total_files - training_files
                else:
                    training_files = int(90/100 * 2000)
                    test_files = 2000 - training_files

                lim = 0
                for file_name in os.listdir(path):
                    #input_image = Image.open(path + '/' + file_name).convert('L')
                    #input_image.thumbnail((64,64), Image.ANTIALIAS)
                    #tv = list(input_image.getdata()) #get pixel values
                    #tva = [(255-x)/255 for x in tv]
                    tva = imageprepare(path + '/' + file_name)
                    input_image = np.array(tva)
                    input_image = input_image.reshape(32,32,1)
                    if lim == 0:
                        cv2.imshow('noway', input_image)
                        cv2.waitKey(1)
                    if lim < training_files:
                        training_inputs.append(input_image)
                        output_vector[count] = 1
                        training_outputs.append(output_vector)
                    elif lim < training_files + test_files:
                        test_inputs.append(input_image)
                        output_vector[count] = 1
                        test_outputs.append(output_vector)
                    ## remove
                    else:
                        break
                    lim += 1
        count += 1

    return training_inputs, training_outputs, test_inputs, test_outputs

if __name__ == "__main__":
    training_inputs, training_outputs, test_inputs, test_outputs = load_data()
    np.save('training_inputs_0_9.npy', training_inputs)
    np.save('training_outputs_0_9.npy', training_outputs)
    np.save('test_inputs_0_9.npy', test_inputs)
    np.save('test_outputs_0_9.npy', test_outputs)
