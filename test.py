import tensorflow as tf
import numpy as np
from PIL import Image, ImageFilter
import cv2
from load_data import imageprepare
from pre import preprocess

def imageprepare(argv): 
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (32, 32), (255)) #creates white canvas of 28x28 pixels
    
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

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, shape = (None, n_H0, n_W0, n_C0), name='X')
    Y = tf.placeholder(tf.float32, shape = (None, n_y), name = 'Y')
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    return X,Y,keep_prob

def forward_propagation(X, keep_prob):
    tf.set_random_seed(1)

    # CONV 1
    W1 = tf.get_variable("W1", [4,4,1,64], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    b1 = tf.get_variable("b1", [64,1], initializer = tf.zeros_initializer())

    Z1 = tf.nn.conv2d(input = X, filter = W1, strides = [1,1,1,1], padding = "SAME")
    A1 = tf.nn.relu(Z1)
    # MAX POOL 1
    P1 = tf.nn.max_pool(A1, ksize = [1,4,4,1], strides = [1,4,4,1], padding = "SAME")

    # CONV 2
    W2 = tf.get_variable("W2", [2,2,64,128], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    b2 = tf.get_variable("b2", [128,1], initializer = tf.zeros_initializer())

    Z2 = tf.nn.conv2d(input = P1, filter = W2, strides = [1,1,1,1], padding = "SAME")
    A2 = tf.nn.relu(Z2)
    # MAX POOL 2
    P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = "SAME")
    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)
    # FULL CONNECT 1
    W3 = tf.get_variable('W3', [512, P2.shape[1:2].num_elements()], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    b3 = tf.get_variable('b3', [512,1], initializer = tf.zeros_initializer())
    Z3 = tf.add(tf.matmul(W3,tf.matrix_transpose(P2)), b3)
    A3 = tf.nn.relu(Z3)
    # FULL CONNECT 2
    W4 = tf.get_variable('W4', [256, 512], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    b4 = tf.get_variable('b4', [256,1], initializer = tf.zeros_initializer())
    A4_drop = tf.nn.dropout(A3, keep_prob)
    Z4 = tf.add(tf.matmul(W4,A4_drop), b4)
    A4 = tf.nn.relu(Z4)
    # FULL CONNECT 3
    W5 = tf.get_variable('W5', [63,256], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    b5 = tf.get_variable('b5', [63,1], initializer = tf.zeros_initializer())
    A5_drop = tf.nn.dropout(A4, keep_prob)
    Z5 = tf.add(tf.matmul(W5,A5_drop), b5)

    Z5 = tf.matrix_transpose(Z5)

    return Z5

def draw(result):
    a = result[0]
    print('------')
    if a<=9:
        print(a)
    elif a >= 10 and a <= 35:
        print(chr(a+87))
    else:
        print(chr(a+29))

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)


    tf.set_random_seed(1)
    seed = 3

    m, n_H0, n_W0, n_C0 = 1, 32, 32, 1
    X, Y, keep_prob = create_placeholders(n_H0, n_W0, n_C0, 63)


    Z3 = forward_propagation(X, keep_prob)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    new_saver = tf.train.import_meta_graph('my-model.ckpt.meta')

    while True:
        ret,frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = preprocess(frame)

        cv2.imshow('draw', frame)
        cv2.imwrite('draw.jpg', frame)

        #tva = imageprepare(Image.fromarray(frame))
        tva = imageprepare('draw.jpg')
        image = np.array(tva)
        np.save('frame.npy', image)

        cv2.imshow('love', image.reshape(n_H0,n_W0,1))

        #print(image.shape)
        image = image.reshape(1,n_H0,n_W0,1)
        prediction = tf.argmax(Z3, 1)

        with tf.Session() as sess:
            sess.run(init)
            new_saver.restore(sess, tf.train.latest_checkpoint('./'))
            result = sess.run(prediction, feed_dict = {X:image, keep_prob:0.9 })
            draw(result)

        cv2.waitKey(1)
