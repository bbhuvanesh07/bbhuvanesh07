import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
def plotImg(img):
if len(img.shape) == 2:
plt.imshow(img, cmap='gray')
plt.show()
else:
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
source="Sample_Inputs/8.png"
img = cv2.imread(source)
img22 = cv2.imread(source)
cv2.imshow("Input image",img)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
binary_img = cv2.adaptiveThreshold(gray_img, 255,
cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

cv2.THRESH_BINARY_INV, 133, 15)



#binary_img = cv2.adaptiveThreshold(binary_img, 255,
cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
# cv2.THRESH_BINARY_INV, 101, 45)
#adaptiveThreshold(src, dst, maxValue, adaptiveMethod, thresholdType,
blockSize, C)
plotImg(binary_img)
_, _, boxes, _ = cv2.connectedComponentsWithStats(binary_img)
# first box is the background
boxes = boxes[1:]
filtered_boxes = []
rot=0
for x,y,w,h,pixels in boxes:
if pixels < 10000 and h < 200 and w < 200 and h > 10 and w > 10:
filtered_boxes.append((x,y,w,h))
rot+=1
for x,y,w,h in filtered_boxes:
cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255),2)
found=0
img1=cv2.imread(source)
# Convert it to HSV
folder='Bacterial blight'
for filename in os.listdir(folder):


img2 = cv2.imread(os.path.join(folder,filename))
if img2 is not None:
img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
# Calculate the histogram and normalize it
hist_img1 = cv2.calcHist([img1_hsv], [0,1], None, [180,256], [0,180,0,256])
cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1,
norm_type=cv2.NORM_MINMAX);
hist_img2 = cv2.calcHist([img2_hsv], [0,1], None, [180,256], [0,180,0,256])
cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1,
norm_type=cv2.NORM_MINMAX);
# find the metric value
metric_val = cv2.compareHist(hist_img1, hist_img2,
cv2.HISTCMP_BHATTACHARYYA)
if metric_val == 0.0:
print("Affected by Bacterial blight")
cv2.imshow("Bacterial blight",img)
plotImg(img)
found=found+1
#print("\n")
folder='Bacterial leafstreak'
for filename in os.listdir(folder):



img2 = cv2.imread(os.path.join(folder,filename))
if img2 is not None:
img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
# Calculate the histogram and normalize it
hist_img1 = cv2.calcHist([img1_hsv], [0,1], None, [180,256], [0,180,0,256])
cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1,
norm_type=cv2.NORM_MINMAX);
hist_img2 = cv2.calcHist([img2_hsv], [0,1], None, [180,256], [0,180,0,256])
cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1,
norm_type=cv2.NORM_MINMAX);
# find the metric value
metric_val = cv2.compareHist(hist_img1, hist_img2,
cv2.HISTCMP_BHATTACHARYYA)
if metric_val == 0.0:
print("Affected by Bacterial leaf streak")
cv2.imshow("Bacterial leaf streak",img)
plotImg(img)
found=found+1
#print("\n")
folder='Blast'
for filename in os.listdir(folder):



img2 = cv2.imread(os.path.join(folder,filename))
if img2 is not None:
img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

# Calculate the histogram and normalize it
hist_img1 = cv2.calcHist([img1_hsv], [0,1], None, [180,256], [0,180,0,256])
cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1,
norm_type=cv2.NORM_MINMAX);
hist_img2 = cv2.calcHist([img2_hsv], [0,1], None, [180,256], [0,180,0,256])
cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1,
norm_type=cv2.NORM_MINMAX);
# find the metric value
metric_val = cv2.compareHist(hist_img1, hist_img2,
cv2.HISTCMP_BHATTACHARYYA)
if metric_val == 0.0:
print("Affected by Blast")
cv2.imshow("Blast",img)
plotImg(img)
found=found+1
#print("\n")
folder='Brown spot'

for filename in os.listdir(folder):
img2 = cv2.imread(os.path.join(folder,filename))
if img2 is not None:
img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

# Calculate the histogram and normalize it
hist_img1 = cv2.calcHist([img1_hsv], [0,1], None, [180,256], [0,180,0,256])
cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1,
norm_type=cv2.NORM_MINMAX);
hist_img2 = cv2.calcHist([img2_hsv], [0,1], None, [180,256], [0,180,0,256])
cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1,
norm_type=cv2.NORM_MINMAX);
# find the metric value
metric_val = cv2.compareHist(hist_img1, hist_img2,
cv2.HISTCMP_BHATTACHARYYA)
if metric_val == 0.0:
print("Affected by Brown spot")
cv2.imshow("Brown spot",img)
plotImg(img)
found=found+1
#print("\n")

folder='False smut'
for filename in os.listdir(folder):
img2 = cv2.imread(os.path.join(folder,filename))
if img2 is not None:
img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

# Calculate the histogram and normalize it
hist_img1 = cv2.calcHist([img1_hsv], [0,1], None, [180,256], [0,180,0,256])
cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1,
norm_type=cv2.NORM_MINMAX);
hist_img2 = cv2.calcHist([img2_hsv], [0,1], None, [180,256], [0,180,0,256])
cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1,
norm_type=cv2.NORM_MINMAX);
# find the metric value
metric_val = cv2.compareHist(hist_img1, hist_img2,
cv2.HISTCMP_BHATTACHARYYA)
if metric_val == 0.0:
print("Affected by False smut")
cv2.imshow("False smut",img22)
#plotImg(img)
found=found+1

#print("\n")
folder='Sheath blight'
for filename in os.listdir(folder):
img2 = cv2.imread(os.path.join(folder,filename))
if img2 is not None:
img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

# Calculate the histogram and normalize it
hist_img1 = cv2.calcHist([img1_hsv], [0,1], None, [180,256], [0,180,0,256])
cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1,
norm_type=cv2.NORM_MINMAX);
hist_img2 = cv2.calcHist([img2_hsv], [0,1], None, [180,256], [0,180,0,256])
cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1,
norm_type=cv2.NORM_MINMAX);
# find the metric value
metric_val = cv2.compareHist(hist_img1, hist_img2,
cv2.HISTCMP_BHATTACHARYYA)
if metric_val == 0.0:
print("Affected by Sheath blight")
cv2.imshow("Sheath blight",img)
plotImg(img)

found=found+1
#print("\n")
if found==0:
if rot<=10:
cv2.imshow("Bacterial blight",img)
print("Affected by Backterial Blight")
if rot>10 and rot<=20:
cv2.imshow("Blast",img)
print("Affected by Blast")
if rot>20:
cv2.imshow("Bacterial leaf streak",img)
print("Affected by Bacterial leafstreak")
