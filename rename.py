#-*- coding:utf-8 -*-
from PIL import Image
import os.path
import numpy as np
from scipy.misc import imsave
#test
path1 = "groundtruth/"
path2 = "baseline_frrn/"
savepath = "tmp/"
i = 0
filelist1 = os.listdir(path1)
filelist2 = os.listdir(path2)
filelist1 = sorted(filelist1)
filelist2.sort(key= lambda x:int(x[:-11]))

for file in filelist2:
    
    filepath = path2 + file
    img = Image.open(filepath)
    newname = filelist1[i]
    imsave(savepath+newname,img)
    i += 1


   