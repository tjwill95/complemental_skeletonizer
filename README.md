# Complemental Skeleton Generator

The Complemental Skeleton Generator is a python script written to enable the user to generate and manipulate the skeleton of a 3D model.  It also can generate and manipulate the Complemental Skeleton of the model.  The Complemental Skeleton is a new geometric construct that lies between the surface of the object and the original skeleton of the model.

This code (and the algorithms within) are described by this paper: https://www.nasampe.org/store/ViewProduct.aspx?id=16280994
The paper is also freely available here: https://www.researchgate.net/publication/342143916_Voxelized_Skeletal_Modeling_Techniques_via_Complemental_Skeletons

The presentation (Provided by the SAMPE 2020 Virtual Series) accompanying the paper can be found for free here: https://www.youtube.com/watch?v=yGzciiRlPcc


## Setting Up Your Machine

This script uses CUDA, a parallel processing library.  It requires that you have an Nvidia graphics card installed and that your computer is set up with the CUDA Toolkit.  Instructions are available here: https://developer.nvidia.com/how-to-cuda-python

The following packages are used:
numba, numpy, math, os, time, matplotlib, PIL, struct, operator, skimage, mpl_toolkits

I recommend using the Anaconda Python 3.X package, which comes with all the relevant packages pre-installed.
https://www.anaconda.com/distribution/

Some alternatives to having your own GPU installed are listed on the above website.


## Running the Script

First, place the STL of the file you want to find the skeleton (or complemental skeleton) of into the 'Input' folder.

Next, open the userInput.py file in a text editor or scripting environment, such as Spyder (Which comes packaged with Anaconda) or Notepad.

Edit the options to best fit your needs.  It is recommended that most settings be left to their default values to begin with.  If you want to run it on a primative, uncomment one of the primative options, found on lines 41 to 46.  If you want to try one of the preloaded objects, uncomment one of the file name options, found on lines 32 to 37.  If you want to use one of your own objects, place the STL of the object into the input folder, then modify the FILE_NAME variable to direct to the input STLs, following the same naming convention as the one used for the demo objects.

Modifications can be made to the skeleton and the complemental skeleton through the two functions at the bottom of the userInput.py file.  Examples of modifications are currently commented out between the triple-quote marks.

The script can be run through main.py.

After running the script, you may want to perform some post-processing to pretty up your model.  I recommend MeshLab, a free mesh-editing software available here: http://www.meshlab.net/.  To clean the model, I recommend using Filters > Cleaning and Repairing > Remove Non-Manifold Faces.  To smooth out the resulting model, I recommend the HC Laplacian Smooth filter, found under Filters > Smoothing Fairing and Deformation > HC Laplacian Smooth.  The HC Laplacian filter can be used iteratively to achieve the desired surface finish.  Finally, to export your model from MeshLab, go to File > Export Mesh As, and save it as a file type compatible with your preferred model viewer.
