# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import cv2
import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift

parser = argparse.ArgumentParser(description=' robot visio  n knn homework')
parser.add_argument('--case',default='3D', type=str, choices=['3D','5D'],
                        help='case 3D or 5D KNN clusrtring')
parser.add_argument('--dir',default='./TestImages', type=str,
                        help='Test image directory')
parser.add_argument('--name',default='2apples.jpg', type=str,
                    help = 'image name')
parser.add_argument('--k',default=2, type=int,
                    help = 'number of segmentation')
parser.add_argument('--dim',default=3, type=int, choices = [3,5],
                    help = '3(R,G,B) or 5(R,G,B,x,y) dimension of image')
parser.add_argument('--resize',default=[100,100], type=list,
                    help = 'resize image shape of image')
parser.add_argument('--plot',default=True, type=bool,
                    help = '3d scatter plot')
parser.add_argument('--grab',default=True, type=bool,
                    help = 'use grabcut')



def scatter_plot(points):
    # Creating figure
    # fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    ax.view_init(60, 60)
    ax.set_xlabel('B')
    ax.set_ylabel('G')
    ax.set_zlabel('R')


    # colors = np.array(
    #     ["red", "green", "blue", "yellow", "pink", "black", "orange", "purple", "beige", "brown", "gray", "cyan",
    #      "magenta"])
    # if len(points) > len(colors):

    for i in range(len(points)):
        colors = cm.rainbow(i / len(points))

        for j in range(len(points[i])):
            ax.scatter3D(points[i][j][0], points[i][j][1], points[i][j][2], color=colors)

    plt.title("3D scatter plot")
    plt.show()

def get_image(*args):
    """
    :param args:
    :return:
    resize도 여기서 함
    """
    os.chdir(args[0].dir)
    img = cv2.imread(args[0].name, cv2.IMREAD_COLOR)
    img = cv2.resize(img, args[0].resize)
    return img

def calc_distance(a,b):
    a=a[:len(b)]
    assert len(a) == len(b), 'dim must be same'

    dis =[(x - y)**2 for x, y in zip(a, b)]
    return math.sqrt(sum(dis))

def clusturing(Ctd, img):
    """
    :param Ctd:
    :param img:
        diss : distance , list
            각 각의 k 번째 Centroid 마다의 거리를 저장.
    :return: points
    """
    seg_dim = len(Ctd)

    # for row, rows  in enumerate(img):
    #     for col, pixel in enumerate(rows):
    #         for k in Ctd:
    #             print("")
    points = []
    for i in range(seg_dim):
            points.append([])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            diss = []
            for k in range(seg_dim):
                dis = calc_distance([img[i][j][0], img[i][j][1], img[i][j][2], i, j], Ctd[k])
                diss.append(dis)
            idx = np.argmin(np.array(diss))
            points[idx].append([img[i][j][0],img[i][j][1], img[i][j][2],i,j])
    return points
    # np.argmin(np.array([dis[0][0] for i in zip(diss)]))

def update_Ctds(points,seg_dim, Ctds):
    new_Ctds = []
    for i in range(seg_dim):
        if not points[i]: # empty list check , True : empty
            new_Ctds.append(np.array(Ctds[i]))
        else:
            new_Ctds.append(np.sum(np.array(points[i]), 0) / len(points[i]))

    if seg_dim ==3:
        new_Ctds = new_Ctds[:3]

    for i in range(seg_dim):
        if np.isnan(new_Ctds[i]).any():
            assert 'empty points component'

    return new_Ctds

def move_centroid(Ctds, img):
    points = clusturing(Ctds, img)
    Ctds = update_Ctds(points=points,seg_dim = len(Ctds), Ctds=Ctds)

    return Ctds

def show_seg_img(points,img):
    seg_img = np.zeros(img.shape,dtype=np.uint8)
    for i in range(len(points)):
        color = np.random.randint(0,255,3)
        for j in range(len(points[i])):
            seg_img[points[i][j][3],points[i][j][4]] = color.tolist()

    print("plot")
    cv2.imshow('img',img)
    cv2.imshow('seg_img',seg_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def grabcut(img):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (50, 50, 450, 290)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    plt.imshow(img), plt.colorbar(), plt.show()

def k_means(img, *args):
    """
        Ctd : Centroid , list
        Ctds : Centroids , list
        diss : distances , list
    :return:
    """
    #initialize
    Ctds = []
    pre_Ctds = []
    for i in range(args[0].k):
        pre_Ctd = []
        Ctd = []

        for i in range(args[0].dim):
            if i < 3:
                pre_Ctd.append(math.inf)
                Ctd.append(random.randint(0,255))
            else:
                pre_Ctd.append(math.inf)
                Ctd.append(random.randint(0,img.shape[i-3]))

        Ctds.append(Ctd)
        pre_Ctds.append(pre_Ctd)

    #Iteration
    while(True):
        pre_Ctds = Ctds
        Ctds = move_centroid(Ctds,img)

        diss = [calc_distance(Ctds[i],pre_Ctds[i]) for i in range(len(Ctds))]

        if max(diss) < 1:
            break

    points = clusturing(Ctds, img)
    show_seg_img(points,img)
    if args[0].plot:
        print('scatter plot')
        scatter_plot(points)


def main():

    args = parser.parse_args()
    img = get_image(args)
    k_means(img, args)
    if args.grab:
        grabcut(img)
if __name__ == '__main__':
    main()




