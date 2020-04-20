import numpy as np
import cv2
import copy
from scipy.interpolate import RectBivariateSpline
import glob
from InverseCompositionAffine import InverseCompositionAffine
from scipy.ndimage import shift, affine_transform


def LucasKanade(in_temp, in_temp_a, rectangle, s=np.zeros(2)):

    x1, y1, x2, y2 = rectangle[0], rectangle[1], rectangle[2], rectangle[3]
    temp_y, temp_x = np.gradient(in_temp_a)
    ds = 1
    thresh = 0.001
    u_size = 50 #87
    v_size = 50 #36

    while np.square(ds).sum() > thresh:

        s_x, s_y = s[0], s[1]
        w_x1, w_y1, w_x2, w_y2 = x1 + s_x, y1 + s_y, x2 + s_x, y2 + s_y

        u = np.linspace(x1, x2, u_size)
        v = np.linspace(y1, y2, v_size)
        u0, v0 = np.meshgrid(u, v)

        w_u = np.linspace(w_x1, w_x2, u_size)
        w_v = np.linspace(w_y1, w_y2, v_size)
        w_u0, w_v0 = np.meshgrid(w_u, w_v)

        x = np.arange(0, in_temp.shape[0], 1)
        y = np.arange(0, in_temp.shape[1], 1)

        spline = RectBivariateSpline(x, y, in_temp)
        S = spline.ev(v0, u0)

        spline_a = RectBivariateSpline(x, y, in_temp_a)
        img_warp = spline_a.ev(w_v0, w_u0)


        error = S - img_warp
        img_error = error.reshape(-1, 1)


        spline_x = RectBivariateSpline(x, y, temp_x)
        warpTemp_x = spline_x.ev(w_v0, w_u0)

        spline_y = RectBivariateSpline(x, y, temp_y)
        warpTemp_y = spline_y.ev(w_v0, w_u0)

        temp = np.vstack((warpTemp_x.ravel(), warpTemp_y.ravel())).T


        #jac_matrix = np.array([[1, 0], [0, 1]])
        jac_matrix = np.array([[1, 0], [0, 1]])


        hess_matrix = temp @ jac_matrix

        H = hess_matrix.T @ hess_matrix



        ds = np.linalg.inv(H) @ (hess_matrix.T) @ img_error


        s[0] += ds[0, 0]
        s[1] += ds[1, 0]

    stop = s
    return stop


if __name__=="__main__":
    #path = "./data/Car4/img"
    #path = "./data/Bolt2/img"
    path = "./data/DragonBaby/img"
    ext = ".jpg"
    images = glob.glob(path+'/*' + ext)
    images.sort()
    #rectangle = [70, 51, 107, 87]
    #rectangle = [70, 51, 177, 138]
    #rectangle = [269, 75, 303, 139]
    rectangle = [160, 83, 216, 148]

    l = rectangle[2] - rectangle[0]
    b = rectangle[3] - rectangle[1]

    rectangle0 = copy.deepcopy(rectangle)
    capture_in = cv2.imread(images[0])
    capture_gray_in = cv2.cvtColor(capture_in, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #capture_gray_in = clahe.apply(capture_gray_in)
    #capture_gray_in = cv2.equalizeHist(capture_gray_in)
    h,w = capture_gray_in.shape
    #out = cv2.VideoWriter('Car.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (w,h))
    #out = cv2.VideoWriter('Baby.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (w,h))

    for index in range(len(images)-1):
        capture = cv2.imread(images[index])
        capture_gray = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)
        #capture_gray = cv2.equalizeHist(capture_gray)
        #capture_gray = clahe.apply(capture_gray)

        cv2.rectangle(capture,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[0])+l,int(rectangle[1])+b),(255, 0, 0),3)
        cv2.imshow('Tracking Human', capture)
        #hout.write(capture)

        capture_next = cv2.imread(images[index+1])
        capture_gray_next = cv2.cvtColor(capture_next, cv2.COLOR_BGR2GRAY)
        #capture_gray_next = cv2.equalizeHist(capture_gray_next)
        #capture_gray_next = clahe.apply(capture_gray_next)

        in_temp_x = capture_gray_in / 255.
        in_temp = capture_gray / 255.
        in_temp_a = capture_gray_next / 255.
        #stop = LucasKanade(in_temp, in_temp_a, rectangle0)
        #stop = LucasKanade(in_temp_x, in_temp_a, rectangle0)
        M = InverseCompositionAffine(in_temp_x, in_temp_a)
        #M = InverseCompositionAffine(in_temp_x[rectangle[0]:rectangle[2], rectangle[1]:rectangle[3]], in_temp_a,rectangle)
        #warp = cv2.rectangle(in_temp_a.copy(),(int(rectangle[0]),int(rectangle[1])),(int(rectangle[0])+l,int(rectangle[1])+b),(255, 0, 0),3)
        #warp_img = affine_transform(warp, np.flip(M)[..., [1, 2, 0]])
        #cv2.imshow('warp',warp_img)
        rectangle = (M @ np.vstack((np.reshape(rectangle0, (2,2)).T,np.ones(2)))).T.flatten()

        #rectangle[0] = stop[0] + rectangle0[0]
        #rectangle[1] = stop[1] + rectangle0[1]
        #rectangle[2] = stop[0] + rectangle0[2]
        #rectangle[3] = stop[1] + rectangle0[3]


        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break
    #out.release()
