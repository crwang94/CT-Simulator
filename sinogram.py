import os
from PIL import Image


xRayImagePath = 'head'
files = os.listdir(xRayImagePath)
sorted(files)
files.sort()
for fi in files:
    print(fi)
