from skeleton import thickenSkeletonAdd, thickenSkeletonMult
import Frep as f

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

#FILE_NAME = "Bird.stl"
#FILE_NAME = "E.stl"
#FILE_NAME = "3DBenchy_up.stl"
#FILE_NAME = "3DBenchy.stl"
#FILE_NAME = "bust_low.stl"
#FILE_NAME = "wavySurface.stl"
FILE_NAME = "hand_low.stl"

PRIMITIVE = True   #Set to true if you would prefer a simple geometric object, then uncomment the one you want
PRIMITIVE_TYPE = "Heart"
#PRIMITIVE_TYPE = "Cube"
#PRIMITIVE_TYPE = "Sphere"
#PRIMITIVE_TYPE = "Cylinder"
#PRIMITIVE_TYPE = "Silo"

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