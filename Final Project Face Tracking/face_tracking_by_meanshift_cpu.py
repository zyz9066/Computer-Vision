###################################################################
# face_tracking_by_meanshift_cpu.py
# CPU version for final project
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
#import matplotlib.pyplot as plt

np.seterr(divide='ignore',invalid='ignore')


#########################################
# Functions
#########################################

# calculate the centroid
def centroid(data):
    total = np.sum(data)
    indices = np.ogrid[[slice(0, i) for i in data.shape]]

    # note the output array is reversed to give (x, y) order
    return np.array([np.sum(indices[axis] * data) / total for axis in range(data.ndim)])[::-1]

# compute image histogram, discard low light values to avoid false values
def calcHist(hsv_roi, histSize=180):
    
    # extract the hue from hsv_roi
    hue, s, v = cv2.split(hsv_roi)
            
    bins = np.zeros(histSize, np.float32)
    for p in hue.flat:
        bins[p] += 1
    return bins.reshape(histSize, 1)

# compute back projection
def calcBackProject(hsv, roi_hist):
    # find ratio as palette
    R = roi_hist / calcHist(hsv)
    
    h, s, v = cv2.split(hsv)
    # create a image with every pixel as its corresponding probability of being target
    B = R[h.ravel()]
    # apply condition
    B = np.minimum(B, 1)
    return B.reshape(hsv.shape[:2])
    

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
time_cpu = []

print('Press "y" to start face tracking and "n" to stop...')
print('Press "q" to quit...')
while(True):
    ret, frame = cap.read()
    
    if tracking:        

        if ret:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            start = time.time_ns()
            dst = calcBackProject(hsv, roi_hist)
            end = time.time_ns()
            sec = (end - start) / 10**6
            print('Time for Back Projection by CPU is {} ms.'.format(sec))
            time_cpu.append(sec)
            
            # apply meanshift to get the new location
            track_window = meanShift(dst, track_window, epsilon, maxCount)

            # draw on frame
            x,y,w,h = track_window
            cv2.rectangle(frame, (x,y), (x+w, y+h), blue, line_width)
                
        else:
            break
        
    else:
        if ret:
            # convert to grayscale image
            gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).copy()

            # detect face locations on current camera frame
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
                    
                    # calculate histogram using CPU
                    start = time.time_ns()
                    roi_hist = calcHist(hsv_roi)
                    end = time.time_ns()
                    sec = (end - start) / 10**6
                    #print('Time for Histogram by CPU is {}.'.format(sec))

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
        #cv2.imwrite('frame_cpu.jpg', frame)
        pass

cap.release()
cv2.destroyAllWindows()

print('Program runs over...')
'''
np.savetxt('time_cpu.txt', time_cpu)

plt.ioff()
plt.gcf()
plt.plot(time_cpu, label='CPU')
plt.legend(loc='upper left')
plt.title('Time of Back Projection by CPU')
plt.xlabel('frame index')
plt.ylabel('time / ms')
plt.savefig('runtime_cpu.png')
'''
