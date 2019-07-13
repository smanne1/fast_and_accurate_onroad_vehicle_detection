import cv2
import numpy as np
import copy
img=cv2.imread('img.jpg',1)
res = cv2.resize(img,None,fx=0.2, fy=0.2, interpolation = cv2.INTER_CUBIC)
##print (img)
res2=np.zeros(res.shape)
res = cv2.normalize(res.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
fblur=copy.deepcopy(res)
cv2.GaussianBlur(res, (5,5), 0.8, res2, 0.8,cv2.BORDER_CONSTANT)

fblur=cv2.GaussianBlur(fblur,(5,5),0)
print(fblur-res2)




cv2.imshow('',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
