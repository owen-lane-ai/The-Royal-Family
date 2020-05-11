from PIL import Image , ImageDraw, ImageOps
import xmltodict
from tqdm import tqdm
import os
import numpy as np
import glob
import sys
import psutil
import keras
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time

#-----------------------------------------------------------------------------

def day_time():
    dateString = f"Date: {time.gmtime()[2]}/{time.gmtime()[1]}/{time.gmtime()[0]}\nTime: {time.gmtime()[3]}:{time.gmtime()[4]}"
    return dateString

#-----------------------------------------------------------------------------

def calculate_iou( target_boxes , pred_boxes ):
    xA = tf.maximum( target_boxes[ ... , 0], pred_boxes[ ... , 0] )
    yA = tf.maximum( target_boxes[ ... , 1], pred_boxes[ ... , 1] )
    xB = tf.minimum( target_boxes[ ... , 2], pred_boxes[ ... , 2] )
    yB = tf.minimum( target_boxes[ ... , 3], pred_boxes[ ... , 3] )
    interArea = tf.maximum( 0.0 , xB - xA ) * tf.maximum( 0.0 , yB - yA )
    boxAArea = (target_boxes[ ... , 2] - target_boxes[ ... , 0]) * (target_boxes[ ... , 3] - target_boxes[ ... , 1])
    boxBArea = (pred_boxes[ ... , 2] - pred_boxes[ ... , 0]) * (pred_boxes[ ... , 3] - pred_boxes[ ... , 1])
    iou = interArea / ( boxAArea + boxBArea - interArea )
    return iou

#-----------------------------------------------------------------------------

def custom_loss( y_true , y_pred ):
    mse = tf.losses.mean_squared_error( y_true , y_pred ) 
    iou = calculate_iou( y_true , y_pred ) 
    return mse +  (1-iou)

#-----------------------------------------------------------------------------

def iou_metric( y_true , y_pred ):
    return calculate_iou( y_true , y_pred ) 

#-----------------------------------------------------------------------------

####################################START#####################################

#The size the image aswell as bounding boxes to be transformed to.
input_dim = 228

#Create keras model
model_layers = [       
    keras.layers.Conv2D( 256 , input_shape=( input_dim , input_dim , 3 ) , kernel_size=( 3 , 3 ) , strides=2 , activation='relu' ),
    keras.layers.Conv2D( 256 , kernel_size=( 3 , 3 ) , strides=2 , activation='relu' ),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),
    
    keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=2 , activation='relu' ),
    keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),
    keras.layers.Flatten(),

    keras.layers.Dense( 128 , activation='sigmoid' ),
    keras.layers.Dense( 128 , activation='sigmoid' ),
    keras.layers.Dropout(0.5),
    keras.layers.Dense( 64 , activation='sigmoid' ),
    keras.layers.Dense( 32 , activation='sigmoid' ),
    keras.layers.Dense( 4 ,  activation='sigmoid' ),
]

model = keras.Sequential( model_layers )

model.compile(
	optimizer=keras.optimizers.Adam(),
	loss=custom_loss,
    metrics=[iou_metric]
)

model.summary()

#Create an array for the images aswell as a list of pathways to them
images = []
image_paths = glob.glob( r'training\*.jpg')

#Create a array of bounding boxes aswell as the pathway to them
bboxes = []
annotations_paths = glob.glob(r'training\anno\*.xml')

#Progress bar for grabbing images and annotations
loop = tqdm(total = len(image_paths), position=0, leave=True)
loop.set_description("Loading images and annotations")

for IMGorXML in zip(image_paths, annotations_paths):
    
    #IMAGES---------------------------------------------------------------
    
    image = Image.open( IMGorXML[0] )
    #If virtual memory is above 80% used stop the program to avoid a crash.
    if psutil.virtual_memory()[2] > 85:
        print("\n\nMemory usage too high")
        sys.exit()
        
    #Grab thewidth and height of the image for bounding box manipulation
    width, height = image.size
    widthinput, heightinput = input_dim / width, input_dim / height
    #Resize the image
    image = image.resize( ( input_dim , input_dim ))
    #Make all pixel colours between 0 and 1 for processing
    image = np.asarray( image ) / 255.0
    images.append( image )
    
    
    #ANNOTATIONS----------------------------------------------------------
    
    xml = xmltodict.parse(open(IMGorXML[1], 'rb'))
    bndbox = xml['annotation']['object']['bndbox']
    #Grab all xml bounding boxes and shrink them appropriatly to fit the image where they originally where
    bndbox = np.array([ int(bndbox[ 'xmin' ]) * widthinput , int(bndbox[ 'ymin' ]) * heightinput , int(bndbox[ 'xmax' ]) * widthinput , int(bndbox[ 'ymax' ]) * heightinput ])
    #Shrink to be between 0 and 1 for processing
    bndbox = bndbox / input_dim
    bboxes.append(bndbox)
    
    loop.update(1)

loop.close()

#Create numpy arrays of each
bboxes = np.array(bboxes)
images = np.array(images)

#Present the number of images aswell as bounding boxes.
print(f"Number of images: {len(images)}")
print(f"Number of bounding boxes: {len(bboxes)}")

images, x_test, bboxes, y_test = train_test_split( images, bboxes, test_size=0.2 )
input_shape = ( input_dim , input_dim , 3 )

print("Applying keras")

epoc = 40

history = model.fit( 
    #The train data
    images ,
    bboxes , 
    #checking against test data
    validation_data=( x_test , y_test ),
    #How many steps through
    epochs= epoc,
    #How many images to be checked at any one time. Affecting this will affect both train times aswell as over fitting
    batch_size=3
)
#Save the model
print("Saving model")
model_json = model.to_json()
with open("result/car_detection.json", "w") as json_file:
    json_file.write(model_json)
model.save( 'result/model.h5')
print("Model saved")

print("Writing model to file...")

with open("Car Detection Model review.txt", "a") as file:
    file.write("------------------------------------------------------------------------------------------------------------\n")
    model.summary(print_fn=lambda x: file.write(x + '\n'))
    file.write(f"Running on {epoc} epochs it achieved an accuracy of {round(history.history['iou_metric'][-1] * 100, 1)}%\n")
    file.write("This model is NOT using data augmentation\n")
    length = len(images)
    file.write(f"using {length} images")
    file.write(f"{day_time()}\n")

#Predict boxes on test set.
boxes = model.predict( x_test )

#Progress bar for predicting images and saving them to system
loop = tqdm(total = boxes.shape[0], position=0, leave=True)
loop.set_description("Predicting test images...")

#For each bounding box
for i in range( boxes.shape[0] ):
    #Grab the box locations and size them up to be the correct size.
    b = boxes[ i , : ] * input_dim
    img = x_test[i] * input_dim
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
