from PIL import Image , ImageDraw
from tqdm import tqdm
import numpy as np
import glob
import sys
import cv2
import psutil
from keras.models import model_from_json

print("Getting model")
json_file = open('model\car_detection.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model\model.h5")
print("Loaded model from disk")

responce = int(input("Single Image: 1\nList of Images: 2\nVideo: 3\nResponce: "))
print()
input_dim = 228

if responce == 1: #Single image
    
    path = r"testing\test.jpg"
    
    
    image = Image.open(path)
    image = image.resize((input_dim, input_dim))
    image = np.asarray(image) / 255.0
    
    image = np.expand_dims(image,axis=0)
    
    boxes = model.predict(image)
    image = image.squeeze(axis=0)
    #Grab the box locations and size them up to be the correct size.
    b = boxes[ 0 , 0 : 4 ] * input_dim
    img = image * input_dim
    #Rebuild the image.
    source_img = Image.fromarray( img.astype( np.uint8 ) , 'RGB' )
    #Draw the bounding box
    draw = ImageDraw.Draw( source_img )
    draw.rectangle( b , outline="blue" )
    #Save the image and add 1 to the image file name.
    print("Saving image")
    source_img.save( 'result\test_image_result.png' , 'png' )
    print("Image saved")
    
    
    
    
    
    
elif responce == 2:
    images = []
    boxes = []
    
    image_paths = glob.glob(r"testing\*.jpg")
    
    loader = tqdm(total=len(image_paths), position=0, leave=True, desc="Loading images...")
    for path in image_paths:
        image = Image.open(path)
        image = image.resize((input_dim, input_dim))
        image = np.asarray(image) / 255.0
        images.append(image)
        loader.update(1)
    
    loader.close()
    
    #Progress bar for predicting images and saving them to system
    images = np.asarray(images)
    
    loop = tqdm(total = len(image_paths), position=0, leave=True, desc="Predicting test images...")
    
    boxes = model.predict(images)
    
    #For each bounding box
    for i in range(len(image_paths)):
        #Grab the box locations and size them up to be the correct size.
        b = boxes[ i  , : ] * input_dim
        img = images[i] * input_dim
        
        #Rebuild the image.
        source_img = Image.fromarray( img.astype( np.uint8 ) , 'RGB' )
        #Draw the bounding box
        draw = ImageDraw.Draw( source_img )
        draw.rectangle( b , outline="blue" )
        #Save the image and add 1 to the image file name.
        source_img.save( 'result\image_{}.png'.format( i + 1 ) , 'png' )
        loop.update(1)
    
    loop.close()
    print("\nFinished")
    print("Complete")
    
    
    
elif responce == 3:
    #Create containers
    boxes = []
    images = []
    
    #Get video for capture
    cap = cv2.VideoCapture(r'testing\test.mp4')
    
    #If the video file cannot find data
    if cap.isOpened() == False:
        print("no video")
    
    #While there is data.
    while cap.isOpened():
        
        #Check if this frame exists and grab the frame regardless
        end, frame = cap.read()
        #If this is the end of the video exit the loop.
        if end == False:
            break
        
        #If memory usage goes above 95 stop the program to avoid a system crash.
        if psutil.virtual_memory()[2] > 95:
            print("Virtual memory at dangerously high levels. Closing application")
            sys.exit()
        
        #Perform preprocessing on the frame
        frame = cv2.resize(frame, (input_dim, input_dim))
        frame = np.asarray(frame) / 255
        #Store it in an array.
        images.append(frame)
    #Releases resources used by capture to reduce strain on memory.
    cap.release()
    
    #Make images into a numpy array
    images = np.asarray(images)
    
    #Initialise progress bar
    loop = tqdm(total = len(images), position=0, leave=True, desc="Predicting video...")
    
    #Predict the images
    boxes = model.predict(images)
    
    #Create video codec for MP4 videos that works on windows 10.
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    #Create the video maker.
    out = cv2.VideoWriter('result\result.mp4', fourcc, 30, (input_dim, input_dim), 1)
    #For each bounding box
    for i in range(len(images)):
        #Grab the box locations and size them up to be the correct size.
        b = boxes[ i  , : ] * input_dim
        img = images[i] * input_dim
        
        #Rebuild the image.
        source_img = Image.fromarray( img.astype( np.uint8 ) , 'RGB' )
        #Draw the bounding box
        draw = ImageDraw.Draw( source_img )
        draw.rectangle( b , outline="blue" )
        img = np.asarray(source_img)
        #add the frame to the video.
        out.write(img)
        loop.update(1)
    out.release()
    loop.close()
    
else:
    sys.exit()
