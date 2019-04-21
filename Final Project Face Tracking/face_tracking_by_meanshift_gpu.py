###################################################################
# face_tracking_by_meanshift_gpu.py
# GPU version for final project
# CS563 Image Analysis, Bishop's University, Winter 2019
# Auther: Tianye Zhao
# Test: Zhaoxuan Qin
# Modified: 4/20/2019
###################################################################

#########################################
# Libraries
#########################################
import numpy as np
import cv2
import time
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import pycuda.autoinit
#import matplotlib.pyplot as plt

#########################################
# Functions
#########################################

# calculate the centroid
def centroid(data):
    total = np.sum(data)
    indices = np.ogrid[[slice(0, i) for i in data.shape]]

    # note the output array is reversed to give (x, y) order
    return np.array([np.sum(indices[axis] * data) / total for axis in range(data.ndim)])[::-1]

# meanshift algorithm
def meanShift(_probImage, window, epsilon=1., maxCount=10):

    mat = _probImage
    cn = mat[0][0].size
    height, width = mat.shape
    cur_rect = window

    assert(cn == 1)
    
    if cur_rect[2] <= 0 or cur_rect[3] <= 0:
        raise AssertionError('Input window has non-positive sizes')

    eps = max(epsilon, 0.)
    eps = round(eps**2)

    niters = max(maxCount, 1)

    for i in range(0, niters):
        
        cur_rect = (max(cur_rect[0],0), max(cur_rect[1],0), min(cur_rect[2],width), min(cur_rect[3],height))
        lst = list(cur_rect)
        if lst[0] == lst[1] == lst[2] == lst[3] == 0:
            lst[0] = size[1]/2
            lst[1] = size[0]/2
            
        lst[2] = max(lst[2], 1)
        lst[3] = max(lst[3], 1)

        region = mat[lst[1]:lst[1]+lst[3],lst[0]:lst[0]+lst[2]]

        # calculate center of mass
        cx, cy = centroid(region)

        dx = round(cx - cur_rect[2]*0.5)
        dy = round(cy - cur_rect[3]*0.5)

        nx = min(max(lst[0]+dx, 0), width-lst[2])
        ny = min(max(lst[1]+dy, 0), height-lst[3])

        dx = nx - lst[0]
        dy = ny - lst[1]
        lst[0] = int(nx)
        lst[1] = int(ny)
        cur_rect = tuple(lst)

        # check for coverage centers mass & window
        if dx**2+dy**2 < eps:
            break

    return cur_rect

##########################################
# Kernel code
# histogram and back projection
##########################################
kernel = """
# define TILE_SIZE %(TILE_SIZE)s
# define MATRIX_LENGTH %(MATRIX_LENGTH)s
# define MATRIX_WIDTH %(MATRIX_WIDTH)s

// Calculate histogram, use shared memory, thread syncronizing and atomic operation
__global__ void calc_hist(unsigned char *img, unsigned int *bins){
    const unsigned int P = %(P)s;
    unsigned int k;
    volatile __shared__ unsigned int bins_loc[256];
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    for (k = 0; k < 256; ++k)
        bins_loc[k] = 0.0;
    for (k = 0; k < P; ++k)
        ++bins_loc[img[i*P + k]];
    // Set the barrier for all the threads
    __syncthreads();
    for (k = 0; k < 256; ++k)
        // Use atomic addition
        atomicAdd(&bins[k], bins_loc[k]);
}


// Calculate back projection, use tiling and check boundary condition
__global__ void back_proj(unsigned int *frame, unsigned int *bin, unsigned int *backproj){
    int frame_hue;
    int pvalue;
    int x = blockIdx.y * TILE_SIZE + threadIdx.x;
    int y = blockIdx.x * TILE_SIZE + threadIdx.y;
    if (x < MATRIX_LENGTH && y < MATRIX_WIDTH)
	frame_hue = frame[y*(MATRIX_LENGTH)+x];
    __syncthreads();

    if (x < MATRIX_LENGTH && y < MATRIX_WIDTH)
        pvalue = bin[frame_hue];
    __syncthreads();

    if (x < MATRIX_LENGTH && y < MATRIX_WIDTH)
        backproj[y*(MATRIX_LENGTH)+x] = pvalue;
    __syncthreads();

}
""" 

#########################################
# Main
#########################################

# grab from default camera using Direct Show as backend
cap = cv2.VideoCapture(cv2.CAP_DSHOW)

green = (0,255,0)
blue = (255,0,0)
line_width = 2

tracking = False

# setup the termination criteria, either 10 iteration or move by at least 1 pt
epsilon = 1.
maxCount = 10

# load Haar cascade for face detection
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
time_gpu = []

print('Press "y" to start face tracking and "n" to stop...')
print('Press "q" to quit...')
while(True):
    ret, frame = cap.read()
    
    if tracking:        
        # check if camera frame returned successfully
        if ret:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_gpu = gpuarray.to_gpu(hsv[:, :, 0].astype(np.uint32))

            # initialize GPU memory for back projection calculation
            bin_apu = gpuarray.to_gpu(roi_hist.astype(np.uint32))
            backproj_gpu = gpuarray.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint32)
            backproj_cpu = np.zeros_like(frame[:, :, 0], dtype=np.uint32)
            
            start = time.time_ns()
            calcBackProject(hsv_gpu, bin_gpu, backproj_gpu, block=(16, 16, 1), grid=(100, 200))
            end = time.time_ns()
            sec = (end - start) / 10**6
            print('Time for Back Projection by GPU is {} ms.'.format(sec))
            time_gpu.append(sec)

            backproj_cpu = backproj_gpu.get()
            
            # apply meanshift to get the new location
            track_window = meanShift(np.float32(backproj_cpu), track_window, epsilon, maxCount)

            # draw on frame
            x,y,w,h = track_window
            frame = cv2.rectangle(frame, (x,y), (x+w, y+h), blue, line_width)
                
        else:
            break
        
    else:
        if ret:
            # convert to grayscale image
            gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).copy()

            kernel = kernel % {'P': 96, 'TILE_SIZE': 32, 'MATRIX_LENGTH': frame.shape[0], 'MATRIX_WIDTH': frame.shape[1]}

            # compile kernel code
            mod = SourceModule(kernel)
            calcHist = mod.get_function("calc_hist")
            calcBackProject = mod.get_function("back_proj")

            # detect face locations on current frame
            faces = cascade.detectMultiScale(gray_img)

            if len(faces) > 0:
                # initial location of tracking window
                x, y, w, h = faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), green, line_width)
                
                if 'x' in locals():
                    # setup location of window
                    track_window = faces[0]

                    # setup the ROI for tracking
                    roi = frame[y:y+h, x:x+w]
                    # transform RGB to HSV
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                    # extract Hue from hsv_roi
                    hue, s, v = cv2.split(hsv_roi)
                            
                    # initialize GPU memory for histogram calculation
                    hue_gpu = gpuarray.to_gpu(hue)
                    bin_gpu = gpuarray.zeros((180, 1), np.uint32)

                    # calculate histogram using GPU
                    start = time.time_ns()
                    calcHist(hue_gpu, bin_gpu, block=(1, 1, 1), grid=(int(h * w / 96), 1, 1))
                    end = time.time_ns()
                    sec = (end - start) / 10**6
                    #print('Time for Histogram by GPU is {} ns.'.format(sec))

                    roi_hist = bin_gpu.get().astype(np.float32)
                    # normalization before back projection
                    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
        else:
            break

    cv2.imshow('Camera', frame)
    ch = cv2.waitKey(60) & 0xFF
    if ch == ord('q') or ch == ord('Q'):
        break
    elif ch == ord('y') or ch == ord('Y'):
        tracking = True
    elif ch == ord('n') or ch == ord('N'):
        tracking = False
    else:
        #cv2.imwrite('frame_gpu.jpg', frame)
        pass

cap.release()
cv2.destroyAllWindows()

print('Program runs over...')

'''
np.savetxt('time_gpu.txt', time_gpu)

plt.ioff()
plt.gcf()
plt.plot(time_gpu, label='GPU')
plt.legend(loc='upper left')
plt.title('Time of Back Projection by GPU')
plt.xlabel('frame index')
plt.ylabel('time / ms')
plt.gca().set_ylim(top=10)
plt.savefig('runtime_gpu.png')
'''
