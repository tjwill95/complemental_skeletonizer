from numba import cuda
from grad3D import grad3D
TPB = 6

@cuda.jit
def skeletonKernel(d_u,d_uGrad,gThresh,sThresh):
    i,j,k = cuda.grid(3)
    dims = d_u.shape
    if i>=dims[0] or j>=dims[1] or k>=dims[2]:
        return
    if d_u[i,j,k]>sThresh or d_uGrad[i,j,k]>gThresh:
            d_u[i,j,k] = 1

def skeleton(u,gThresh,sThresh):
    dims = u.shape
    d_u = cuda.to_device(u)
    d_uGrad = cuda.to_device(grad3D(u, maxVal = 1))
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB,(dims[2]+TPB-1)//TPB]
    blockSize = [TPB, TPB, TPB]
    skeletonKernel[gridSize, blockSize](d_u,d_uGrad,gThresh,sThresh)
    return d_u.copy_to_host()

@cuda.jit
def addKernel(d_u,t):
    i,j,k = cuda.grid(3)
    dims = d_u.shape
    if i>=dims[0] or j>=dims[1] or k>=dims[2]:
        return
    if d_u[i,j,k]<0:
            d_u[i,j,k] +=t

def thickenSkeletonAdd(u,t):
    dims = u.shape
    d_u = cuda.to_device(u)
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB,(dims[2]+TPB-1)//TPB]
    blockSize = [TPB, TPB, TPB]
    addKernel[gridSize, blockSize](d_u,t)
    return d_u.copy_to_host()

@cuda.jit
def multKernel(d_u,t):
    i,j,k = cuda.grid(3)
    dims = d_u.shape
    if i>=dims[0] or j>=dims[1] or k>=dims[2]:
        return
    if d_u[i,j,k]<0:
            d_u[i,j,k] *=t

def thickenSkeletonMult(u,t):
    dims = u.shape
    d_u = cuda.to_device(u)
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB,(dims[2]+TPB-1)//TPB]
    blockSize = [TPB, TPB, TPB]
    multKernel[gridSize, blockSize](d_u,t)
    return d_u.copy_to_host()

@cuda.jit
def refleshKernel(d_r,d_w,template):
    i,j,k = cuda.grid(3)
    dims = d_r.shape
    if i>=dims[0] or j>=dims[1] or k>=dims[2]:
        return
    for index in range(template**3):
        checkPos = (i+((index//(template**2))%template-template//2),
                    j+((index//template)%template-template//2),
                    k+(index%template-template//2))
        if checkPos[0]<dims[0] and checkPos[1]<dims[1] and checkPos[2]<dims[2] and min(checkPos)>0:
            updated = d_r[checkPos]+((i-checkPos[0])**2+(j-checkPos[1])**2+(k-checkPos[2])**2)**(1/2)
            d_w[i,j,k]=min(d_w[i,j,k],updated)

def reflesh(skeleton,iteration = 25,template = 5):
    dims = skeleton.shape
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB,(dims[2]+TPB-1)//TPB]
    blockSize = [TPB, TPB, TPB]
    d_r = cuda.to_device(skeleton)
    d_w = cuda.to_device(skeleton)
    for count in range(iteration):
        refleshKernel[gridSize, blockSize](d_r,d_w,template)
        d_r,d_w = d_w,d_r
    return d_r.copy_to_host()