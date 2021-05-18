import numpy as np
import cv2

def distEclud(vecA, vecB):
    return np.linalg.norm(vecA-vecB)

def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = np.mat(minJ + rangeJ * np.random.rand(k,1))
    return centroids
# dataSet have k center point
# disMeas the distance, we use eclud
# createCent,the initial center point
def kMeans(dataSet, k, length,width,distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0] #num of sample
    new_arr=(np.zeros((length,width,3)))
    clusterAssment = np.mat(np.zeros((m,2))) #m*2 matrix
    centroids = createCent(dataSet, k) #initial k centers
    clusterChanged = True
    round=0
    while clusterChanged:  # while center still changes
        clusterChanged = False
        for i in range(m):
            minDist = np.inf;
            minIndex = -1
            for j in range(k):  # find nearest mass center
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI;
                    minIndex = j
            if clusterAssment[i, 0] != minIndex: clusterChanged = True
            # the first column is the center while the second column is the distance
            clusterAssment[i, :] = minIndex, minDist ** 2

        # change the mass center
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
        round=round+1
    #return the new image
    for i in range(length):
        for j in range(width):
#            print(clusterAssment[i*width+j,0],np.int(clusterAssment[i*width+j,0]))
            new_arr[i,j,:]=centroids[np.int(clusterAssment[i*width+j,0]),:]


    return centroids, clusterAssment,new_arr


img = cv2.imread('star.ppm')
length=img.shape[0]
width=img.shape[1]
Z = np.mat(img.reshape((-1,3)))
Z = np.float32(Z)
K = 3
center,ret,img2=kMeans (Z,K,length,width)

cv2.imwrite("after.png", img2)

#cv2.imshow('img2',img2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
