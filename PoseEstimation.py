
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt




def point_triangulation(k1,k2,pt1,pt2,R1,C1,R2,C2):
    points_3d = []

    I = np.identity(3)
    C1 = C1.reshape(3,1)
    C2 = C2.reshape(3,1)

    #calculating projection matrix P = K[R|T]
    P1 = np.dot(k1,np.dot(R1,np.hstack((I,-C1))))
    P2 = np.dot(k2,np.dot(R2,np.hstack((I,-C2))))
  
    #homogeneous coordinates for images
    xy = np.hstack((pt1,np.ones((len(pt1),1))))
    xy_cap = np.hstack((pt2,np.ones((len(pt1),1))))

    
    p1,p2,p3 = P1
    p1_cap, p2_cap,p3_cap = P2

    #constructing contraints matrix
    for i in range(len(xy)):
        A = []
        x = xy[i][0]
        y = xy[i][1]
        x_cap = xy_cap[i][0]
        y_cap = xy_cap[i][1] 
        
        A.append((y*p3) - p2)
        A.append((x*p3) - p1)
        
        A.append((y_cap*p3_cap)- p2_cap)
        A.append((x_cap*p3_cap) - p1_cap)

        A = np.array(A).reshape(4,4)

        _, _, v = np.linalg.svd(A)
        x_ = v[-1,:]
        x_ = x_/x_[-1]
        x_ = x_[:3]
        points_3d.append(x_)


    return points_3d

def linear_triangulation(R_Set,T_Set,pt1,pt2,k1,k2):
    R1_ = np.identity(3)
    T1_ = np.zeros((3,1))
    points_3d_set = []
    for i in range(len(R_Set)):
        points3d = point_triangulation(k1,k2,pt1,pt2,R1_,T1_,R_Set[i],T_Set[i])
        points_3d_set.append(points3d)

    return points_3d_set

def get_RTset(E):

    U, S, V = np.linalg.svd(E,full_matrices=True)
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])

    R1 = np.dot(U,np.dot(W,V))
    R2 = np.dot(U,np.dot(W,V))
    R3 = np.dot(U,np.dot(W.T,V))
    R4 = np.dot(U,np.dot(W.T,V))

    T1 = U[:,2]
    T2 = -U[:,2]
    T3 = U[:,2]
    T4 = -U[:,2]

    R = [R1,R2,R3,R4]
    T = [T1,T2,T3,T4]

    for i in range(len(R)):
        if (np.linalg.det(R[i]) < 0):
            R[i] = -R[i]
            T[i] = -T[i]

    return R, T

def compute_cheriality(pt,r3,t):
    count_depth = 0
    for xy in pt:
        if np.dot(r3,(xy-t)) > 0 and t[2] > 0:
            count_depth +=1
    return count_depth

def extract_pose(E,pt1,pt2,k1,k2):
    #get four rotation and translation matrices
    R_set, T_set = get_RTset(E)

    #get 3D points using triangulation
    pts_3d = linear_triangulation(R_set,T_set,pt1,pt2,k1,k2)
    threshold = 0
    #Four sets are available for each possibility
    for i in range(len(R_set)):
        R = R_set[i]
        T = T_set[i]
        r3 = R[2]
        pt3d = pts_3d[i]

        #calculating which R satisfies the condition
        num_depth_positive = compute_cheriality(pt3d,r3,T)
        if num_depth_positive > threshold:
            index = i 
            threshold = num_depth_positive

    R_best = R_set[index]
    T_best = T_set[index]

    return R_best,T_best
