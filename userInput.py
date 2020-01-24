from skeleton import thickenSkeletonAdd, thickenSkeletonMult
import Frep as f

#FILE_NAME = "Bird.stl"
FILE_NAME = "hand_low.stl"

PRIMITIVE = False

BUFFER = 10
RESOLUTION = 150
GTHRESH = 0.85
STHRESH = -1

PLOTTING = True
SAVE_PLOTS = False

OBJ_MESH = True
SKELE_MESH = True
REFLESH_MESH = True

COMPOUND_MODIFICATIONS = False

COMPLEMENTAL_SKELETON = True
COMP_SKELE_MESH = True
COMP_REFLESH_MESH = True

def skeletonMods(u):
    i,j,k = u.shape
    impact = -1
    ray = j-1
    while impact<0:
        if u[i//2-20,ray,100]<0:
            impact = 1
            u[i//2-20,ray,100]-=10
        else:
            ray -=1
    return u

def complementalSkeletonMods(u):
    i,j,k = u.shape
    impact = -1
    ray = j-1
    while impact<0:
        if u[i//2-20,ray,100]<0:
            impact = 1
            u[i//2-20,ray,100]-=10
        else:
            ray -=1
    return u