Bounding box car detection with Keras
========

This project will detect cars with bounding boxes, all done with keras and tensorflow libraries.

To get it up and running:

1. clone the repo
2. open baseline implementation. 
3. Create a new conda environment from the environment(.yml file) file within the repo.
4. Run the model and see first results.

The baseline implementation is the first file to get the code working. Not the most recent and most
accurate model. Therefore, for better results. Please use the other .py files. 

Features
--------

- Car detection with bounding boxes
- Using keras to create and train cnns
- A gui has been created

Installation
------------
Create conda environment from our .yml file:

    conda env create -f <filename>.yml

The first line of the .yml file sets the new environment's name.

Activate the new environment:

    conda activate <name>

Check the environment has been created successfully:

    conda env list

The new environment should appear in this list.

**Make sure to select this environment in your editor.**

Execution
---------
> Training

Should you wish to create your own model using either train or train aug. First open the command console. Next navigate to the folder that the .py files are stored.

If you wish to create a new model from scratch, run either train.py or trainAug.py by doing...

    python train.py

    python trainAug.py

Make sure before you do this that you have the images you wish to test on inside the folder called "training" with their annotations in a folder called "anno"

To create the annotations for these images please use this tool <https://pypi.org/project/labelImg/>

Once this is all done and each image is ordered correctly. (The easiest way to do this is to call each image a number (Image_[numbers]))

> Testing

To test the program, please run the following.

    python test.py

The program will give you 3 options.
A single image
A file of images
A video.

A single image.

Please make sure the image is in the testing folder with the name test.jpg. the program will get the image and output its results to the result folder.

A file of images

This option will use the entirity of the testing file. and output the result to the result folder.

A video

this option will require a video named "test.mp4" in the testing folder. Again the result will be sent to the results folder

Support
-------

If you are having issues, please check the environment has been set up and all dependencies have been covered. Otherwise, email us. 

Data
-------

For the data set please go to this link: <https://drive.google.com/open?id=18VxKi1Y1H2gFBkG7Qu84o0F9d56iWJql>

For the pretrained model go to this link: <https://drive.google.com/open?id=1AqQSAsd7Vbdh1YZsKEEgVH-RKcB4zpgU>
