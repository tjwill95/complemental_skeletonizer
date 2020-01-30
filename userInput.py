from skeleton import thickenSkeletonAdd, thickenSkeletonMult
import Frep as f

BUFFER = 10     #Number of empty layers around the object
RESOLUTION = 150    #Maximum number of voxels in the X/Y directions
#Gradient function sensitivity (Higher value = more gradient voxels)
GTHRESH = 0.85
#SDF Skeletal Sensitivity (Negative, higher magnitude = more skeletal voxels)
STHRESH = -1

PLOTTING = True     #Print plots as it goes through the skeletizing process
SAVE_PLOTS = False  #Save the afore-printed plots to the Output folder

#Return the PLY of the original object after voxelizing and smoothing
OBJ_MESH = True     
SKELE_MESH = True   #Return the PLY of the skeleton
REFLESH_MESH = True #Return the PLY of the object recreated from the skeleton

#Recreate the object with changes to the skeleton and complemental skeleton
COMPOUND_MODIFICATIONS = False  

COMPLEMENTAL_SKELETON = True    #Find the Complemental Skeleton of the object
COMP_SKELE_MESH = True          #Return the PLY of the Complemental Skeleton
#Return the PLY of the object recreated from the complemental skeleton
COMP_REFLESH_MESH = True

#Put the name of the desired file below, or uncomment one of the example files
#This file must be in the Input folder, set to be in the same directory as the
#python files.

#FILE_NAME = "Bird.stl"
#FILE_NAME = "E.stl"
#FILE_NAME = "3DBenchy.stl"
#FILE_NAME = "bust_low.stl"
#FILE_NAME = "wavySurface.stl"
#FILE_NAME = "hand_low.stl"

#If you would prefer a simple geometric object, uncomment the one you want and
#make sure that all FILE_NAME options are commented out.
PRIMITIVE_TYPE = "Heart"
#PRIMITIVE_TYPE = "Cube"
#PRIMITIVE_TYPE = "Sphere"
#PRIMITIVE_TYPE = "Cylinder"
#PRIMITIVE_TYPE = "Silo"

def skeletonMods(u):
    #This section allows you to make modifications to the skeleton of the 
    #object. Reminder that internal voxels are negative, external voxels are 
    #positive, and the magnitude of the voxel defines the size of the maximal
    #sphere at that point
    i,j,k = u.shape
    """
    ray = j-1
    impact = -1
    while impact<0:
        if u[i//2-20,ray,100]<0:
            impact = 1
            u[i//2-20,ray,100]-=10
        else:
            ray -=1
    """
    return u

def complementalSkeletonMods(u):
    #This section allows you to make modifications to the complemental skeleton
    #of the object.  Reminder that internal voxels are negative, external 
    #voxels are positive, and the magnitude of the voxel defines the size of 
    #the maximal sphere at that point
    i,j,k = u.shape
    """
    impact = -1
    ray = j-1
    while impact<0:
        if u[i//2-20,ray,100]<0:
            impact = 1
            u[i//2-20,ray,100]-=10
        else:
            ray -=1
    """
    return u