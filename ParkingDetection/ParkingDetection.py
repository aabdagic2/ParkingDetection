import cv2 as cv2
from matplotlib import image, pyplot as plt
from numpy.core.multiarray import array
from imutils import contours
import numpy as np
img=cv2.imread("C:\\Users\\amina\\Videos\\parkingbg.jpeg")
im=cv2.imread("C:\\Users\\amina\\Videos\\p1.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
cv2.imshow('grayy',edges)
rho = 1  
theta = np.pi / 180  
threshold = 15  
min_line_length = 50  
max_line_gap = 20  
line_image = np.copy(img) * 0  

lines = cv2.HoughLinesP(edges, rho, theta, threshold,0,
                    min_line_length, max_line_gap)

for line in lines:
    for x1,y1,x2,y2 in line:
      cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

cv2.imshow('',line_image)
# Nacrtaj  linije na slici pozadine
lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

cv2.imshow('result',lines_edges)
counter=0

gray = cv2.cvtColor(lines_edges, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

  
# Invertuj sliku
invert = 255 - thresh
offset, old_cY, first = 10, 0, True
visualize = cv2.cvtColor(invert, cv2.COLOR_GRAY2BGR)
# Pronađi konture i sortiraj ih
cnts = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
(cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")
i=0
print(cnts)
x=[]
y=[]
for c in cnts:
    # pronadji centar
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
 
    if(i!=0):
         cv2.circle(im, (cX, cY), 15, (36, 255, 12), -1) 
         x.append(cX)
         y.append(cY)
    counter=counter+1
    cv2.imshow('visualize', im)
    cv2.waitKey(200) 
    i=i+1 

numberOfParkingSpaces=counter-1
print("Total number of parking spaces: ")
print(numberOfParkingSpaces)

#Segmentacija slike parkiranih automobila

image1 = cv2.imread("C:\\Users\\amina\\Videos\\p1.jpg")
 

# konvertuje sliku u grayscale verziju
img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
 
# primjena thresholda da bi dobili inverznu binarnu sliku
# svi pikseli preko 120 će biti postavljeni na 255
ret, thresh2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
 

cv2.imshow('Binary Threshold Inverted',thresh2)
kernel = np.ones((5,5), np.uint8)
img_dilation = cv2.erode(thresh2, kernel, iterations=1)
img_dilation = cv2.dilate(img_dilation, kernel, iterations=3)

cv2.imshow('res',img_dilation)

j=0
countNotAvailable=0
while j!=counter-1:
    rectX = x[j] 
    rectY = y[j] 
    crop_img = img_dilation[rectY:(rectY+15), rectX:(rectX+15)]
    j=j+1
    if cv2.countNonZero(crop_img) == 0:
     continue
    else:
     cv2.circle(im, (x[j-1], y[j-1]), 15, (0,0,255), -1) 
     countNotAvailable=countNotAvailable+1
   
print('Spaces available:')
print(numberOfParkingSpaces-countNotAvailable)
print('Spaces not available:')
print(countNotAvailable)
cv2.imshow('vis', im)
cv2.waitKey(0)
cv2.destroyAllWindows()