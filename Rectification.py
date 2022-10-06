import numpy as np
import cv2
import glob

def rectify(img1,img2,F,pts1,pts2):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]

    _,H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=(w1, h1))
    img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
    img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))

    H1_inv = np.linalg.inv(H1)
    H2_inv = np.linalg.inv(H2)
    F_rectified = np.dot(H2_inv.T, np.dot(F,H1_inv))

    return img1_rectified,img2_rectified,F_rectified

def draw_lines(lines_set,image,points):
    for line,pt in zip(lines_set,points):
        x0,y0 = map(int, [0, -line[2]/line[1] ])
        x1,y1 = map(int, [image.shape[1]-1, -line[2]/line[1] ])
        img = cv2.line(image, (x0,y0), (x1,y1), (0,255,0) ,2)
        img = cv2.circle(image, (int(pt[0]),int(pt[1])), 5, (0,255,0), 2)
    return img

def epipolar_lines(pts1_,pts2_,F,image1,image2):
    lines1_,lines2_ = [], []
    for i in range(len(pts1_)):
        p1 = np.array([pts1_[i,0], pts1_[i,1], 1])
        p2 = np.array([pts2_[i,0], pts2_[i,1], 1])
    
        lines1_.append(np.dot(F.T, p2))
        lines2_.append(np.dot(F,p1))
    img1 = draw_lines(lines1_,image1,pts1_)
    img2 = draw_lines(lines2_,image2,pts2_)

   

    out = np.hstack((img1, img2))
    cv2.imwrite("epipolar_lines.png",out)
    return lines1_,lines2_
    