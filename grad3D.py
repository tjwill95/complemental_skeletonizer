from numba import cuda
import numpy as np
TPB = 6

@cuda.jit(device = True)
def CFD(n,p,h):
    return ((p-n)/(2*h))**2

@cuda.jit
def grad3DKernel(d_u,d_v,maxVal):
    i,j,k = cuda.grid(3)
    dims = d_u.shape
    if i<dims[0] or j<dims[1] or k<dims[2]:
        h1 = 1 #step size
        h2 = 2**(1/2)
        h3 = 5**(1/2)

        dx = CFD(d_u[i-1,j,k],d_u[i+1,j,k],h1)
        dy = CFD(d_u[i,j-1,k],d_u[i,j+1,k],h1)
        dz = CFD(d_u[i,j,k-1],d_u[i,j,k+1],h1)
    
        dPxPy = CFD(d_u[i-1,j-1,k],d_u[i+1,j+1,k],h2)
        dNxPy = CFD(d_u[i-1,j+1,k],d_u[i+1,j-1,k],h2)
        dPyPz = CFD(d_u[i,j-1,k-1],d_u[i,j+1,k+1],h2)
        dNyPz = CFD(d_u[i,j-1,k+1],d_u[i,j+1,k-1],h2)
        dPzPx = CFD(d_u[i-1,j,k-1],d_u[i+1,j,k+1],h2)
        dNzPx = CFD(d_u[i-1,j,k+1],d_u[i+1,j,k-1],h2)
        d5, d6, d7, d8, d9, d10 = 1,1,1,1,1,1
        """
        dPPxPy = CFD(d_u[i-2,j-1,k],d_u[i+2,j+1,k],h3)
        dPPxNy = CFD(d_u[i-2,j+1,k],d_u[i+2,j-1,k],h3)
        dPPyPz = CFD(d_u[i,j-2,k-1],d_u[i,j+2,k+1],h3)
        dPPyNz = CFD(d_u[i,j-2,k+1],d_u[i,j+2,k-1],h3)
        dPPzPx = CFD(d_u[i-1,j,k-2],d_u[i+1,j,k+2],h3)
        dPPzNx = CFD(d_u[i+1,j,k-2],d_u[i-1,j,k+2],h3)
        
        dNNxPy = CFD(d_u[i+2,j-1,k],d_u[i-2,j+1,k],h3)
        dNNxNy = CFD(d_u[i+2,j+1,k],d_u[i-2,j-1,k],h3)
        dNNyPz = CFD(d_u[i,j+2,k-1],d_u[i,j-2,k+1],h3)
        dNNyNz = CFD(d_u[i,j+2,k+1],d_u[i,j-2,k-1],h3)
        dNNzPx = CFD(d_u[i-1,j,k+2],d_u[i+1,j,k-2],h3)
        dNNzNx = CFD(d_u[i+1,j,k+2],d_u[i-1,j,k-2],h3)
        d5 = (dPPxPy+dNNxPy+dz)
        d6 = (dPPxNy+dNNxNy+dz)
        d7 = (dPPyPz+dNNyPz+dx)
        d8 = (dPPyNz+dNNyNz+dx)
        d9 = (dPPzPx+dNNzPx+dy)
        d10= (dPPzNx+dNNzNx+dy)
        """
        d1 = (dx+dy+dz)
        
        d2 = (dPxPy+dNxPy+dz)
        d3 = (dx+dPyPz+dNyPz)
        d4 = (dPzPx+dy+dNzPx)
        
        d_v[i,j,k]=min(maxVal,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10)**(1/2)
    
def grad3D(u,maxVal=2):
    #u = SDF of our desired voxel model
    #maxVal = Upper cap on the gradient.  If any gradient values fall above 
    #this value, it is set to the maxVal.  This is for visualization purposes.
    #Outputs the magnitude of the gradient vector at each cell in the voxel model.
    dims = u.shape
    d_u = cuda.to_device(u)
    d_v = cuda.to_device(np.ones(dims))
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB,(dims[2]+TPB-1)//TPB]
    blockSize = [TPB, TPB, TPB]
    grad3DKernel[gridSize, blockSize](d_u,d_v,maxVal)
    return d_v.copy_to_host()    