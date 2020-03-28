from PIL import Image , ImageDraw
from tqdm import tqdm
import numpy as np
import glob
from keras.models import model_from_json

print("Getting model")
json_file = open('car_detection.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

responce = int(input("Single Image: 1\nList of Images: 2\n"))
print()
input_dim = 228

if responce == 1: #Single image
    
    path = r"C:\Users\Local User\Pictures\cars_train\car tester.jpg"
    
    
    image = Image.open(path)
    image = image.resize((input_dim, input_dim))
    image = np.asarray(image) / 255.0
    
    image = np.expand_dims(image,axis=0)
    
    boxes = model.predict(image)
    image = image.squeeze(axis=0)
    #Grab the box locations and size them up to be the correct size.
    b = boxes[ 0 , 0 , 0 , 0 : 4 ] * input_dim
    img = image * input_dim
    #Rebuild the image.
    source_img = Image.fromarray( img.astype( np.uint8 ) , 'RGB' )
    #Draw the bounding box
    draw = ImageDraw.Draw( source_img )
    draw.rectangle( b , outline="blue" )
    #Save the image and add 1 to the image file name.
    print("Saving image")
    source_img.save( 'yolov3\image_results.png' , 'png' )
    print("Image saved")
    
    
    
    
    
    
else:
    images = []
    boxes = []
    
    image_paths = glob.glob(r"C:\Users\Local User\Pictures\cars_train\test images\*.jpg")
    
    loader = tqdm(total=len(image_paths), position=0, leave=True, desc="Loading images...")
    for path in image_paths:
        image = Image.open(path)
        image = image.resize((input_dim, input_dim))
        image = np.asarray(image) / 255.0
        image = np.expand_dims(image,axis=0)
        images.append(image)
        loader.update(1)
    
    loader.close()
    
    #Progress bar for predicting images and saving them to system
    
    images = np.array(images)
    print(images.shape)
    image = images.squeeze(axis=1)
    
    loop = tqdm(total = len(image_paths), position=0, leave=True, desc="Predicting test images...")
    
    #For each bounding box
    for i in range(len(image_paths)):
        boxes = model.predict(images[i])
        #Grab the box locations and size them up to be the correct size.
        b = boxes[ 0  , 0 , 0 , 0 : 4 ] * input_dim
        img = images[i] * input_dim
        
        img = img.squeeze(axis=0)
        #Rebuild the image.
        source_img = Image.fromarray( img.astype( np.uint8 ) , 'RGB' )
        #Draw the bounding box
        draw = ImageDraw.Draw( source_img )
        draw.rectangle( b , outline="blue" )
        #Save the image and add 1 to the image file name.
        source_img.save( 'yolov3\image_{}.png'.format( i + 1 ) , 'png' )
        loop.update(1)
    
    loop.close()
    print("\nFinished")
    print("Complete")
    
  
