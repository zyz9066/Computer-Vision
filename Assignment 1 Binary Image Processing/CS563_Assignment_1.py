'''
    File name: CS563_Assignment_1.py
    Author: Tianye Zhao
    Test: Zhaoxuan Qin
    Date created: 1/27/2019
    Date last modified: 2/7/2019
    Python Version: 3.7
    IDE: Python IDLE
    ================================
    There may be slight difference between report and program,
    since this program is edited continuously till final submission.
    
    What is commentted out at first was developed at first stage.
    It implements and tests connnected components algorithm.
    Since this algorithm is embedded in opencv, it isn't used
    during assignment implementation.
'''

import cv2 as cv
import math
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
# from matplotlib import pyplot as plt

# Global variables
connectivity = 8
black = 0
white = 255
contourIdx = 0
radius = 1

# Greeting
print('CS563 Assignment 1')
print('Author: Tianye Zhao')
print('================================================================')
print('This program is designed for Binary Image Analysis. '+
      'Techniques such as connected components analysis,'+
      'morphological filters and feature extraction are implemented.')
print('================================================================')

# Ask user choose image
root = tk.Tk()
root.withdraw()
img_path = filedialog.askopenfilename(initialdir='',title='Choose an Image to Analyze',
                                     filetypes=(('PGM','*.pgm'),('PBM','*.pbm'),('PPM','*.ppm')))

# Open image
flg = 0 # Return grayscale image
img = cv.imread(img_path, flg)

if not img.data: # Check loading error
    print('Problem loading image!')
    exit()
    
height, width = img.shape[0:2]


# Resize large image
if width > 1024:
    img = cv.resize(img, (1024, int(1024/width*height)))
    height, width = img.shape[0:2]
#cv.imshow('Original',img)

# Threshold image using Otsu's binarization and binary invert for further manipulation
retVal, binary = cv.threshold(img,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

# Preprocessing on image
if height > 600 and retVal < 160: # Large image with low threshold value
    # Apply adaptive threshold to get more foreground information with low illumination
    th = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,5,2)
    # Detect edges for segmenting objects
    edges = cv.Canny(img,int(0.33*retVal),retVal)
    # Combine all foreground information
    adding = (binary+th+edges).astype('uint8')

    contours, hierarchy = cv.findContours(adding, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    detect = np.ones((height, width), 'uint8')
    detect *= (2**8-1)
    
    # Preserve holes by checking parent and child of contours
    for idx, relation in enumerate(hierarchy[0]):
        cntr = contours[idx]
        area = cv.contourArea(cntr)
        # Exclude small noise dots
        if area > 25:
            perimeter = cv.arcLength(cntr, True)
            circularity = perimeter**2/area
            # Exclude lines
            if circularity < 100:
                if relation[3] == -1:
                    cv.drawContours(detect, [cntr], contourIdx, black, -1)
                elif relation[2] == -1:
                    cv.drawContours(detect, [cntr], contourIdx, white, -1)
                else:
                    cv.drawContours(detect, [cntr], contourIdx, black, -1)
                    
    binary = cv.bitwise_not(detect) # Binary invert
    
elif height > 300: # Apply opening on medium image to close edges
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    binary = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)
    
elif retVal > 180: # Remove lines on small image with high threshold value
    # Detect horizontal line
    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT,(18,1))
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT,(1,19))
    # Detect vertical line
    horizontal = cv.morphologyEx(binary,cv.MORPH_OPEN,horizontalStructure)
    vertical = cv.morphologyEx(binary,cv.MORPH_OPEN,verticalStructure)
    # Exclude lines
    binary = binary^horizontal^vertical
    
# Connect components to get a few objects features
output = cv.connectedComponentsWithStats(binary, connectivity, cv.CV_16U)

# Above function returns whole image as first result, ignore it when passing value
num_labels = output[0]-1
stats = output[2][1:]
centroids = output[3][1:]

# Create report template
report = pd.DataFrame(columns=['Object', 'Area', 'Perimeter', 'Circularity1', 'Circularity2',
                               'Centroid', 'mu_rr', 'mu_rc', 'mu_cc',
                               'Bounding Box', 'Rotated Bounding Box', 'Similar Obj', 'Contours'])

# Create blank template to draw features
objects = np.ones((height, width), 'uint8')
objects *= (2**8-1)

count = 0
for i in range(num_labels):
    
    w = stats[i, cv.CC_STAT_WIDTH]
    h = stats[i, cv.CC_STAT_HEIGHT]
    area = stats[i, cv.CC_STAT_AREA]

    # Exclude dots, large background and irregular shapes
    if area < 20 or area > 9000 or area/(w*h) < 0.2:
        continue
    # Exclude dots in medium image
    if height < 600 and height > 300 and area < 100:
        continue
    
    count += 1
    x = stats[i, cv.CC_STAT_LEFT]
    y = stats[i, cv.CC_STAT_TOP]
    
    '''
    # Draw bounding box
    cv.rectangle(objects, (x, y), (x+w, y+h), black)
    '''
    
    cx = round(centroids[i,0], 1)
    cy = round(centroids[i,1], 1)
    cv.circle(objects, (int(cx), int(cy)), radius, black)
    
    row = {'Object': count, 'Area': area, 'Bounding Box': [(x,y),(x+w,y+h)], 'Centroid': (cx,cy)}
    report = report.append(row, ignore_index=True)

# Get more features in contours
contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

for cntr in contours:
    
    M = cv.moments(cntr)
    # Skip dots whose area is 0
    if M['m00'] == 0:
        continue
    
    # Centroid
    cx = M['m10']/M['m00']
    cy = M['m01']/M['m00']

    perimeter = round(cv.arcLength(cntr, True),1)

    # Loop through dataframe to find exact object by checking centroid distance
    for idx, row in report.iterrows():
        dist = math.sqrt((row['Centroid'][0]-cx)**2+(row['Centroid'][1]-cy)**2)
        if dist < 3:
            # Check if permeter of current object is null or shorter than current perimeter of contour
            # because for objects with holes, it may return hole's perimeter
            if pd.isnull(row['Perimeter']) or row['Perimeter'] < perimeter:
                
                # Second moments
                mu_rr = round(M['mu20']/M['m00'],2)
                mu_rc = round(M['mu11']/M['m00'],2)
                mu_cc = round(M['mu02']/M['m00'],2)
                
                # Axis of least inertia
                slope = math.tan(math.radians((math.degrees(math.atan(2*mu_rc/(mu_rr-mu_cc))))/2))
                bx,by,bw,bh = cv.boundingRect(cntr)
                d = 10*math.sqrt(bw**2+bh**2)
                dx = math.sqrt(d/(1+slope**2))
                dy = math.fabs(dx*slope)

                # Circularity2
                mean_sum = 0
                for point in cntr:
                    mean_sum += math.sqrt((point[0,0]-cx)**2+(point[0,1]-cy)**2)
                mean = mean_sum/len(cntr)
                var_sum = 0
                for point in cntr:
                    var_sum += (math.sqrt((point[0, 0]-cx)**2 + (point[0, 1]-cy)**2) -mean)**2
                variance = var_sum/len(cntr)

                # Rotated Bounding Box
                minRect = cv.minAreaRect(cntr)
                box = cv.boxPoints(minRect)
                box = np.intp(box)

                # Append features to report
                report.at[idx, 'Perimeter'] = perimeter
                report.at[idx, 'Circularity1'] = round(perimeter**2/report.at[idx, 'Area'],1)
                report.at[idx, 'Circularity2'] = round(mean/math.sqrt(variance),2)
                report.at[idx, 'mu_rr'] = mu_rr
                report.at[idx, 'mu_rc'] = mu_rc
                report.at[idx, 'mu_cc'] = mu_cc
                report.at[idx, 'Rotated Bounding Box'] = box
                report.at[idx, 'Contours'] = cntr

                # Draw contours and bounding box
                cv.drawContours(objects, [cntr], contourIdx, black)
                cv.drawContours(objects, [box], contourIdx, black)

                # Draw axis of least inertia for large image only
                if height > 300:
                    cv.line(objects, (int(cx-dx),int(cy-dy)), (int(cx+dx),int(cy+dy)), black)

# Find similar object
for i in range(count):
    lst = []
    for j in range(count):
        if i == j:
            continue
        else:
            ret = cv.matchShapes(report.at[i,'Contours'],report.at[j,'Contours'],cv.CONTOURS_MATCH_I2,0)
            if ret < 0.3:
                lst.append(j+1)
    report.at[i,'Similar Obj'] = lst

# Display objects detection results with features and print report
cv.imshow('Objects', objects)
print(report.drop('Contours',axis=1).to_string(index=False))

'''
# Display images using matplot
titles = ['Original','Binary','Objects']
images = [img,binary,objects]
for i in range(len(images)):
    plt.subplot(1,3,i+1), plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

cv.imwrite('binary.png',binary)
cv.imwrite('objects.png',objects)
report.drop('Contours',axis=1).to_csv('report.csv', sep='\t',encoding='utf-8')
'''

# Press any key to close
cv.waitKey(0)
cv.destroyAllWindows()
