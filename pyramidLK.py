import numpy as np
import cv2
import copy
from scipy.interpolate import RectBivariateSpline
import glob
from lktracker_bolt import LucasKanade


# function to calculate 1st order derivative of gaussian
def gaussian(sigma,x,y):
    a= 1/(np.sqrt(2*np.pi)*sigma)
    b= np.exp(-(x**2+y**2)/(2*(sigma**2)))
    c = a*b
    return a*b


## getting kernel from  gaussian for [-1,0,1]
def gaussian_kernel():
    G=np.zeros((5,5))
    for i in range(-2,3):
        for j in range(-2,3):
            G[i+1,j+1]=gaussian(1.5,i,j)
    return G

def Lucas_Kanade_Reduce(I1):
    w, h = I1.shape
    newWidth = int(w / 2)
    newHei = int(h / 2)
    G = gaussian_kernel()
    newImage = np.ones((newWidth, newHei))
    for i in range(2, I1.shape[0] - 2, 2):  # making image of half size by skiping alternate pixels
        for j in range(2, I1.shape[1] - 2, 2):
            newImage[int(i / 2), int(j / 2)] = np.sum(I1[i - 2:i + 3, j - 2:j + 3] * G)  # convolving with gaussian mask

    return newImage

def LK_Reduce_Iterative(Img,Level):
    if Level==0:#level 0 means current level i.e. no change
        return Img
    i=0
    #newImage=cv2.imread(Img,0)
    while(i<Level):
        newImage=Lucas_Kanade_Reduce(Img)
        i=i+1

    return newImage


path = "./data/Bolt2/img"
#path = "./data/DragonBaby/img"
ext = ".jpg"
images = glob.glob(path+'/*' + ext)
images.sort()

rectangle = [269, 75, 303, 139]
#rectangle = [160, 83, 216, 148]

l = rectangle[2] - rectangle[0]
b = rectangle[3] - rectangle[1]

nlevels = 2
rectangle0 = copy.deepcopy(rectangle)
rectangle_l = copy.deepcopy(rectangle)
capture_in = cv2.imread(images[0])
capture_gray_in = cv2.cvtColor(capture_in, cv2.COLOR_BGR2GRAY)
h, w = capture_gray_in.shape
out = cv2.VideoWriter('Bolt-pyramid.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (w, h))

for index in range(len(images)-1):
    capture = cv2.imread(images[index])
    capture_gray = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)
    #capture_gray = cv2.equalizeHist(capture_gray)

    cv2.rectangle(capture,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[0])+l,int(rectangle[1])+b),(255, 0, 0),3)
    cv2.imshow('Tracking Human', capture)
    out.write(capture)

    capture_next = cv2.imread(images[index+1])
    capture_gray_next = cv2.cvtColor(capture_next, cv2.COLOR_BGR2GRAY)
    #capture_gray_next = cv2.equalizeHist(capture_gray_next)

    in_temp_x = capture_gray_in / 255.
    in_temp = capture_gray / 255.
    in_temp_a = capture_gray_next / 255.
    #stop = LucasKanade(in_temp, in_temp_a, rectangle0)
    stop = [0, 0]
    for level in range(nlevels,0,-1):
        img1 = LK_Reduce_Iterative(in_temp_x,level)
        img2 = LK_Reduce_Iterative(in_temp_a,level)
        stop += LucasKanade(img1, img2, rectangle0)
        #stop = g + stop
    rectangle_l[0] = stop[0] + rectangle0[0]
    rectangle_l[1] = stop[1] + rectangle0[1]
    rectangle_l[2] = stop[0] + rectangle0[2]
    rectangle_l[3] = stop[1] + rectangle0[3]


    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break
    out.release()
