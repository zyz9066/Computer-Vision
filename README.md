# CS563-Image-Analysis
Courseworks and Project for image analysis.

## Assignment 1 Binary Image Processing

The objective of this project is to gain experience with connected components analysis, morphological filters and the use of features computed within it for recognition of objects. A secondary objective is to gain experience with image formats. 

### Requirements

The student is to implement and/or modify a connected components program so that it will report on all objects found in a gray-level input image. The figure below shows an image of two overlapping playing cards. The program should be able to detect all 11 objects. Moreover, the features reported should indicate which objects are similar to others.

### Library
Pandas: use dataframe to store each object's features

OpenCV: process image

## Assignment 2 Image Enhancement and Edge Detection

The first objective of the assignment is to implement image enhancement techniques. After enhancing the image, your solution must provide a selection of edge detectors to extract the contours of the image.

In this assignment, I have implemented a simple GUI for user adjusting different techniques to get the best result they needed.

### Libraries
OpenCV, numpy, pillow, matplotlib, tkinter


## Final Project: Face Tracking using Meanshift
Project name: Face tracking using meanshift

### Quick demo
Please visit [here](https://drive.google.com/file/d/13NmlB9p51czy72iKu2ggzDUmaqRnlIAm/view?usp=sharing) for CPU version and [here](https://drive.google.com/file/d/1i6L8PB3JzvICeJUm6hnmtzqDKfvhrMGl/view?usp=sharing) for GPU version.

Obviously GPU implementation achieved a speedup compared to traditional CPU-based tracking methods.  

### Dependency
OpenCV, numpy, matplotlib, time, PyCUDA, CUDA Toolkit 10.1

GPU: GeForce GTX 1050

All dependency on local machine has to be set up before running it.

Input: computer camera

Output: runtime for calculating back projection in each step

"face_tracking_by_meanshift_cpu.py" incorporates CPU version of the histogram calculation and back projection calculation.
It includes basic face detection and tracking task as well as printing runtime.

"face_tracking_by_meanshift_gpu.py" incorporates GPU version of the histogram calculation and back projection calculation.
It includes basic face detection and tracking task as well as printing runtime and speedup.

To run the code, simply run "python face_tracking_by_meanshift_cpu.py" or "python face_tracking_by_meanshift_gpu.py".

The computer must be equiped with camera before running source code. It is recommended that user might select a good initial tracking window before start tracking.
