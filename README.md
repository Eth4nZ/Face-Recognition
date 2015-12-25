# Face-Recognition

_This repository is our Digital Image Proccessing Course Exercise._

We implemented a basic face recognition program based on the [sample of OpenCV](http://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html#face-recognition-with-opencv).

##Requirements
* OpenCV 2.4.9
* Webcam
* python2.7+
* g++

##Preparation
###Modify facerec_video.cpp
you need to modify few lines in *facerec_video.cpp*.

**line 60 & 61** modify the path to your haarcascades*.xml.  
the available Haar-Cascades for face detection are located in the data folder of your OpenCV installation.

**line 64** modify the deviceID of your Webcam (0 for me).

###Prepare images
using crop_face.py to crop every image.(in the same folder of your raw images).
<pre><code>python crop_face.py
</code></pre>
you need to give the approximately (x,y)-position of two eyes to crop every face.

###Prepare .csv file
to run our facerec_video, we need a .csv file to store paths of our database.
if our images in hierarchie like this
> /basepath/<subject>/<image.png>
├── 0  
│   ├── 0.png_20_20_70_70.png  
│   ├── 1.png_20_20_70_70.png  
│   ├── 2.png_20_20_70_70.png  
│   ├── 3.png_20_20_70_70.png  
│   ├── 4.png_20_20_70_70.png  
│   └── 5.png_20_20_70_70.png  
├── 1  
│   ├── 0.png_20_20_70_70.png  
│   ├── 1.png_20_20_70_70.png  
│   └── 2.png_20_20_70_70.png  
├── 2  
│   ├── 0.png_20_20_70_70.png  
│   └── 1.png_20_20_70_70.png  
├── 3  
│   ├── 0.png_20_20_70_70.png  
│   ├── 1.png_20_20_70_70.png  
│   └── 2.png_20_20_70_70.png  

you can simply call create_csv.py with the path to the folder and redirect output to yourcsv.csv
<pre><code>python create_csv.py yourpath > yourcsv.csv
</code></pre>

##Usage
<pre><code>opencv facerec_video.cpp
./facerec_video yourcsv.csv
</code></pre>

if you want to take attendance in _50_ frames
<pre><code>./facerec_video yourcsv.csv 50
</code></pre>

