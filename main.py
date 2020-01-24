import os #Just used to set up file directory
import userInput as u
import numpy as np
import Frep as f
from SDF3D import SDF3D
from skeleton import skeleton, reflesh
from visualizeSlice import multiPlot, slicePlot
from meshExport import generateMesh
from voxelize import voxelize

def main():
    scale = [1,1,1]
    try: os.mkdir(os.path.join(os.path.dirname(__file__),'Output')) #Creates an output folder if there isn't one yet
    except: pass
    try:    FILE_NAME = u.FILE_NAME
    except: FILE_NAME = ""
    if FILE_NAME != "":
        shortName = FILE_NAME[:-4]
        filepath = os.path.join(os.path.dirname(__file__), 'Input',FILE_NAME)
        res = u.RESOLUTION-u.BUFFER*2
        origShape, objectBox = voxelize(filepath, res, u.BUFFER)
        gridResX, gridResY, gridResZ = origShape.shape
        scale[0] = objectBox[0]/(gridResX-u.BUFFER*2)
        scale[1] = max(objectBox[1:])/(gridResY-u.BUFFER*2)
        scale[2] = scale[1]
    elif u.PRIMITIVE == True:
        shortName = u.PRIMITIVE_TYPE
        x0 = np.linspace(-50,50,u.RESOLUTION)
        y0, z0 = x0, x0
        #origShape = f.cylinderX(x0,y0,z0,-40,40,20)
        origShape = f.rect(x0,y0,z0,80,40,60)
    else:
        print("Provide either a file name or a desired primative.")
        return
    
    rawObject = SDF3D(f.condense(origShape,2*u.BUFFER))
    i,j,k = rawObject.shape
    
    origShape = rawObject
    origShape = f.smooth(rawObject,iteration=1)
    print("Object Smoothed")
    if u.OBJ_MESH: generateMesh(origShape,scale,modelName=shortName+"Object")
    if u.PLOTTING: multiPlot(origShape,shortName+' Object',u.SAVE_PLOTS)
    
    oSkele = skeleton(origShape,u.GTHRESH,u.STHRESH)
    oSkele = u.skeletonMods(oSkele)
    print("Skeleton Computed")
    if u.PLOTTING: multiPlot(oSkele,shortName+' Skeleton',u.SAVE_PLOTS)
    if u.SKELE_MESH: generateMesh(oSkele,scale,modelName=shortName+"Skeleton")
    
    if u.COMPOUND_MODIFICATIONS or u.REFLESH_MESH:
        refleshed = f.smooth(reflesh(oSkele,iteration = u.RESOLUTION//8))
        print("Skeleton Refleshed")
        if u.REFLESH_MESH: generateMesh(refleshed,scale,modelName=shortName+"Refleshed")
        if u.COMPOUND_MODIFICATIONS: rawObject = refleshed
    
    if u.COMPLEMENTAL_SKELETON:
        flesh = SDF3D(f.subtract(oSkele,rawObject))
        flesh = f.smooth(flesh,iteration=1)
        #if u.PLOTTING: multiPlot(flesh,shortName+' Flesh',u.SAVE_PLOTS)
        cSkele = skeleton(flesh,u.GTHRESH,u.STHRESH)
        cSkele = u.complementalSkeletonMods(cSkele)
        print("Complemental Skeleton Computed")
        if u.PLOTTING: multiPlot(cSkele,shortName+' Complemental Skeleton',u.SAVE_PLOTS)
        if u.COMP_SKELE_MESH: generateMesh(cSkele,scale,modelName=shortName+"Complemental Skeleton")
        
        cRefleshed = reflesh(cSkele,iteration = u.RESOLUTION//8)
        recreatedObject = f.smooth(f.union(f.thicken(SDF3D(oSkele),2.0),cRefleshed))
        print("Complemental Skeleton Refleshed")
        if u.PLOTTING: multiPlot(recreatedObject,shortName+' Remade',u.SAVE_PLOTS)
        if u.COMP_REFLESH_MESH: generateMesh(recreatedObject,scale,modelName=shortName+"Remade")
    
if __name__ == '__main__':
    main()