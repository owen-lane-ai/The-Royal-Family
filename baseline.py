from PIL import Image , ImageDraw
import xmltodict
import os
import numpy as np
import glob
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split

#variable declaration
input_dim = 228

#create an array and get the path to the images folder
images = []
image_paths = glob.glob( r'C:\Users\Local User\Pictures\cars_train\cars_train\*.jpg')
for imagefile in image_paths:
    image = Image.open( imagefile ).resize( ( input_dim , input_dim ))
    image = np.asarray( image ) / 255.0
    images.append( image )

#create an array and get the path to the boxes folder
bboxes = []
annotations_paths = glob.glob(r'C:\Users\Local User\Pictures\cars_train\cars_train\anno\*.xml')

for xmlfile in annotations_paths:
    x = xmltodict.parse(open(xmlfile, 'rb'))
    bndbox = x['annotation']['object']['bndbox']
    bndbox = np.array([ int(bndbox[ 'xmin' ]) , int(bndbox[ 'ymin' ]) , int(bndbox[ 'xmax' ]) , int(bndbox[ 'ymax' ]) ])
    bboxes.append( bndbox / input_dim )

Y = np.array(bboxes)
X = np.array(images)

print(len(X))
print(len(Y))

#split the data into test and training sets
x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size=0.1 )

input_shape = ( input_dim , input_dim , 3 )

#these are unused variables and will be used for tweaking the model to try and improve performance
dropout_rate = 0.5
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

#define the model layers
model_layers = [
    keras.layers.Conv2D( 256 , input_shape=( input_dim , input_dim , 3 ) , kernel_size=( 3 , 3 ) , strides=2 , activation='relu' ),
    keras.layers.Conv2D( 256 , kernel_size=( 3 , 3 ) , strides=2 , activation='relu' ),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D( 256 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 256 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D( 256 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 256 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D( 256 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 256 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D( 64 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 64 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D( 64 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 64 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D( 16 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 16 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),

    keras.layers.Conv2D( 4 , kernel_size=( 2 , 2 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 4 , kernel_size=( 2 , 2 ) , strides=1 , activation='relu' ),
    keras.layers.Conv2D( 4 , kernel_size=( 2 , 2 ) , strides=1 , activation='sigmoid' ),
]
#create a sequential model
model = keras.Sequential( model_layers )

#compile the model using Adam as the optimizer
model.compile(
	optimizer=keras.optimizers.Adam( lr=0.0001 ),
	loss=custom_loss,
    metrics=[ iou_metric ]
)
#printint the model summary
model.summary()

#fit the model to the dataset
model.fit(
    x_train ,
    y_train ,
    validation_data=( x_test , y_test ),
    epochs=20 ,
    batch_size=3
)

#save the model weights
model.save( 'model.h5')

#preditct the boxes based on the trained model
boxes = model.predict( x_test )
for i in range( boxes.shape[0] ):
    b = boxes[ i , 0 , 0 , 0 : 4 ] * input_dim
    img = x_test[i] * 255
    source_img = Image.fromarray( img.astype( np.uint8 ) , 'RGB' )
    draw = ImageDraw.Draw( source_img )
    draw.rectangle( b , outline="black" )
    source_img.save( 'yolov3\image_{}.png'.format( i + 1 ) , 'png' )