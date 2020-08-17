# Face Tracking using Meanshift
Project name: Face tracking using meanshift

## Quick demo
Please visit [here](https://drive.google.com/file/d/13NmlB9p51czy72iKu2ggzDUmaqRnlIAm/view?usp=sharing) for CPU version and [here](https://drive.google.com/file/d/1i6L8PB3JzvICeJUm6hnmtzqDKfvhrMGl/view?usp=sharing) for GPU version.

Obviously GPU implementation achieved a speedup compared to traditional CPU-based tracking methods.  

## Dependency
OpenCV, numpy, matplotlib, time, PyCUDA, CUDA Toolkit 10.1

GPU: GeForce GTX 1050

All dependency on local machine has to be set up before running it.

Input: computer camera

Output: runtime for calculating back projection in each step

`face_tracking_by_meanshift_cpu.py` incorporates CPU version of the histogram calculation and back projection calculation.
It includes basic face detection and tracking task as well as printing runtime.

`face_tracking_by_meanshift_gpu.py` incorporates GPU version of the histogram calculation and back projection calculation.
It includes basic face detection and tracking task as well as printing runtime and speedup.

To run the code, simply run `python face_tracking_by_meanshift_cpu.py` or `python face_tracking_by_meanshift_gpu.py`.

The computer must be equiped with camera before running source code. It is recommended that user might select a good initial tracking window before start tracking.
