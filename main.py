import os #Just used to set up file directory
import userInput as u
import numpy as np
import Frep as f
from SDF3D import SDF3D
from skeleton import skeleton, reflesh
from visualizeSlice import multiPlot
from meshExport import generateMesh
from voxelize import voxelize

def main():
    scale = [1,1,1]
    try: os.mkdir(os.path.join(os.path.dirname(__file__),'Output')) #Creates an output folder if there isn't one yet
    except: pass
    try:    FILE_NAME = u.FILE_NAME #Checks to see if a file name has been set
    except: FILE_NAME = ""
    try:    PRIMITIVE_TYPE = u.PRIMITIVE_TYPE #Checks to see if a primitive type has been set
    except: PRIMITIVE_TYPE = ""
    if FILE_NAME != "":
        #This section retrieves the input file and voxelizes the input STL
        shortName = FILE_NAME[:-4]
        filepath = os.path.join(os.path.dirname(__file__), 'Input',FILE_NAME)
        res = u.RESOLUTION-u.BUFFER*2
        origShape, objectBox = voxelize(filepath, res, u.BUFFER)
        gridResX, gridResY, gridResZ = origShape.shape
        scale[0] = objectBox[0]/(gridResX-u.BUFFER*2)
        scale[1] = max(objectBox[1:])/(gridResY-u.BUFFER*2)
        scale[2] = scale[1]
    elif PRIMITIVE_TYPE != "":
        #This section generates the desired primitive
        shortName = PRIMITIVE_TYPE
        if PRIMITIVE_TYPE == "Heart":
            x0 = np.linspace(-1.5,1.5,u.RESOLUTION)
            y0, z0 = x0, x0
            origShape = f.heart(x0,y0,z0,0,0,0)
        elif PRIMITIVE_TYPE == "Egg":
            x0 = np.linspace(-5,5,u.RESOLUTION)
            y0, z0 = x0, x0
            origShape = f.egg(x0,y0,z0,0,0,0)
            #eggknowledgement to Molly Carton for this feature.
        else:
            x0 = np.linspace(-50,50,u.RESOLUTION)
            y0, z0 = x0, x0
            if PRIMITIVE_TYPE == "Cube":
                origShape = f.rect(x0,y0,z0,80,80,80)
            elif PRIMITIVE_TYPE == "Silo":
                origShape = f.union(f.sphere(x0,y0,z0,40),f.cylinderY(x0,y0,z0,-40,0,40))
            elif PRIMITIVE_TYPE == "Cylinder":
                origShape = f.cylinderX(x0,y0,z0,-40,40,40)
            elif PRIMITIVE_TYPE == "Sphere":
                origShape = f.sphere(x0,y0,z0,40)
            else:
                print("Given primitive type has not yet been implemented.")
    else:
        print("Provide either a file name or a desired primitive.")
        return
    
    #This section cleans up the input object
    rawObject = SDF3D(f.condense(origShape,2*u.BUFFER))
    i,j,k = rawObject.shape
    origShape = f.smooth(rawObject,iteration=1)
    print("Object Smoothed")
    if u.OBJ_MESH: generateMesh(origShape,scale,modelName=shortName+"Object")
    if u.PLOTTING: multiPlot(origShape,shortName+' Object',u.SAVE_PLOTS)
    
    #This section finds the skeleton of the input object
    oSkele = skeleton(origShape,u.GTHRESH,u.STHRESH)
    oSkele = u.skeletonMods(oSkele)
    print("Skeleton Computed")
    if u.PLOTTING: multiPlot(oSkele,shortName+' Skeleton',u.SAVE_PLOTS)
    if u.SKELE_MESH: generateMesh(oSkele,scale,modelName=shortName+"Skeleton")
    
    #If needed, this section refleshes the skeleton
    if u.COMPOUND_MODIFICATIONS or u.REFLESH_MESH:
        refleshed = f.smooth(reflesh(oSkele,iteration = u.RESOLUTION//8))
        print("Skeleton Refleshed")
        if u.REFLESH_MESH: generateMesh(refleshed,scale,modelName=shortName+"Refleshed")
        if u.COMPOUND_MODIFICATIONS: rawObject = refleshed
    
    #If needed, this section finds the complemental skeleton
    if u.COMPLEMENTAL_SKELETON:
        flesh = SDF3D(f.subtract(oSkele,rawObject))
        flesh = f.smooth(flesh,iteration=1)
        cSkele = skeleton(flesh,u.GTHRESH,u.STHRESH)
        cSkele = u.complementalSkeletonMods(cSkele)
        print("Complemental Skeleton Computed")
        if u.PLOTTING: multiPlot(cSkele,shortName+' Complemental Skeleton',u.SAVE_PLOTS)
        if u.COMP_SKELE_MESH: generateMesh(cSkele,scale,modelName=shortName+"Complemental Skeleton")
        
        #If needed, this section refleshes the complemental skeleton
        if u.COMP_REFLESH_MESH:
            cRefleshed = reflesh(cSkele,iteration = u.RESOLUTION//8)
            recreatedObject = f.smooth(f.union(f.thicken(SDF3D(oSkele),2.0),cRefleshed))
            print("Complemental Skeleton Refleshed")
            if u.PLOTTING: multiPlot(recreatedObject,shortName+' Remade',u.SAVE_PLOTS)
            generateMesh(recreatedObject,scale,modelName=shortName+"Remade")
    
if __name__ == '__main__':
    main()