from PIL import Image , ImageDraw
import xmltodict
import os
import numpy as np
import glob
import keras
from keras.layers import *
import tensorflow as tf
from sklearn.model_selection import train_test_split

#The size the image aswell as bounding boxes to be transformed to.
input_dim = 228

#Create an array for the images aswell as a list of pathways to them
images = []
image_paths = glob.glob( r'C:\Users\Local User\Pictures\cars_train\cars_train\*.jpg')

#Create a array of bounding boxes aswell as the pathway to them
bboxes = []
annotations_paths = glob.glob(r'C:\Users\Local User\Pictures\cars_train\cars_train\anno\*.xml')

print("Loading images and Applying annotations")

# Apply manipulations to both and store them in their arrays
for IMGorXML in zip(image_paths, annotations_paths):
    image = Image.open( IMGorXML[0] )
    
    #Grab thewidth and height of the image for bounding box manipulation
    width, height = image.size
    widthinput, heightinput = input_dim / width, input_dim / height
    
    #Resize the image
    image = image.resize( ( input_dim , input_dim ))
    #Make all pixel colours between 0 and 1 for processing
    image = np.asarray( image ) / 255.0
    images.append( image )
    
    
    xml = xmltodict.parse(open(IMGorXML[1], 'rb'))
    bndbox = xml['annotation']['object']['bndbox']
    #Grab all xml bounding boxes and shrink them appropriatly to fit the image where they originally where
    bndbox = np.array([ int(bndbox[ 'xmin' ]) * widthinput , int(bndbox[ 'ymin' ]) * heightinput , int(bndbox[ 'xmax' ]) * widthinput , int(bndbox[ 'ymax' ]) * heightinput ])
    #Shrink to be between 0 and 1 for processing
    bndbox = bndbox / input_dim
    bboxes.append(bndbox)

#Create numpy arrays of each
Y = np.array(bboxes)
X = np.array(images)

#Present the number of images aswell as bounding boxes.
print(f"Number of images: {len(X)}")
print(f"Number of bounding boxes: {len(Y)}")

x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size=0.2 )
input_shape = ( input_dim , input_dim , 3 )
dropout_rate = 0.6
alpha = 0.2
    
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

def custom_loss( y_true , y_pred ):
    mse = tf.losses.mean_squared_error( y_true , y_pred ) 
    iou = calculate_iou( y_true , y_pred ) 
    return mse + ( 1 - iou )

def iou_metric( y_true , y_pred ):
    return calculate_iou( y_true , y_pred ) 

print("Applying keras")

#Create keras model
model_layers = [       
    keras.layers.Conv2D( 256 , input_shape=( input_dim , input_dim , 3 ) , kernel_size=( 3 , 3 ) , strides=2 , activation='relu' ),
    keras.layers.Conv2D( 256 , kernel_size=( 3 , 3 ) , strides=2 , activation='relu' ),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D( 256 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 256 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Dropout(0.5),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D( 256 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 256 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D( 256 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 256 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D( 64 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 64 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D( 64 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 64 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),
    
    keras.layers.Conv2D( 16 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 16 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D( 4 , kernel_size=( 2 , 2 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 4 , kernel_size=( 2 , 2 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 4 , kernel_size=( 2 , 2 ) , strides=1 , activation='sigmoid' ),
]

model = keras.Sequential( model_layers )

model.compile(
	optimizer=keras.optimizers.Adam(),
	loss=custom_loss,
    metrics=[ iou_metric ]
)

model.summary()

model.fit( 
        #The train data
    x_train ,
    y_train , 
    #checking against test data
    validation_data=( x_test , y_test ),
    #How many steps through
    epochs=10 ,
    #How many images to be checked at any one time. Affecting this will affect both train times aswell as over fitting
    batch_size=3 
)
#Save the model
model.save( 'model.h5')

#Predict boxes on test set.
boxes = model.predict( x_test )

#For each bounding box
for i in range( boxes.shape[0] ):
    #Grab the box locations and size them up to be the correct size.
    b = boxes[ i , 0 , 0 , 0 : 4 ] * input_dim
    img = x_test[i] * input_dim
    #Rebuild the image.
    source_img = Image.fromarray( img.astype( np.uint8 ) , 'RGB' )
    #Draw the bounding box
    draw = ImageDraw.Draw( source_img )
    draw.rectangle( b , outline="blue" )
    #Save the image and add 1 to the image file name.
    source_img.save( 'yolov3\image_{}.png'.format( i + 1 ) , 'png' )
    
    
#https://towardsdatascience.com/getting-started-with-bounding-box-regression-in-tensorflow-743e22d0ccb3
