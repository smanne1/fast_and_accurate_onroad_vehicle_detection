import numpy as np
import cv2
import glob
import copy
import time
start=time.time()
def tracker(Rep_ref,Count_ref,MaxLaneNum,ExpLaneNum,Enable,Line,TrackThreshold,CountUpperThresh):

    List = np.ones((MaxLaneNum, ExpLaneNum))*1.7976931348623157e+308
    for i in range(MaxLaneNum):
        for j in range(ExpLaneNum):
            if (Count_ref[0,i] > 0) & (Enable[0,j] == 1):
                List[i, j] = abs(Line[0, j].transpose() - Rep_ref[0,i]) + abs(Line[1, j].transpose() - Rep_ref[1, i]) * 200
##    print("Line",Line)
##    print("Rep:",Rep_ref)
##    print("List",List)
        
    #Find the best matches between the current lines and those in the repository.
    #Match_dis  = intmax('int16')*ones(1, MaxLaneNum, 'int16');
    Match_dis  = np.ones((1, MaxLaneNum))*1.7976931348623157e+308
    Match_list = np.uint8(np.zeros((2, MaxLaneNum)))
    for i in range(ExpLaneNum):
        if i > 0:
            #Reset the row and column where the minimum element lies on.
            List[rowInd, :] = np.ones((1, ExpLaneNum))*1.7976931348623157e+308
            List[:, colInd] = 1.7976931348623157e+308
        
        # In the 1st iteration, find minimum element (corresponds to
        # best matching targets) in the distance matrix. Then, use the
        # updated distance matrix where the minimun elements and their
        # corresponding rows and columns have been reset.
        Val = List.min()
        rowInd,colInd = np.unravel_index(List.argmin(), List.shape) 
        Match_dis[0,i]    = Val
        Match_list[:,i] = np.array([rowInd,colInd]).transpose()

    # Update reference target list.
    # If a line in the repository matches with an input line, replace
    # it with the input one and increase the count number by one;
    # otherwise, reduce the count number by one. The count number is
    # then saturated.
    Count_ref = Count_ref - 1
##    print("Match_dis:",Match_dis)
##    print("Match_list:",Match_list)
    
    for i in range(ExpLaneNum):
        if Match_dis[0,i] > TrackThreshold:
            # Insert in an unused space in the reference target list
            NewInd = Count_ref.argmin()
            Rep_ref[:, NewInd] = Line[:, Match_list[1, i]]
            Count_ref[0,NewInd] = Count_ref[0,NewInd] + 2
##            print("Count_ref1:",Count_ref)
        if Match_dis[0,i] < TrackThreshold:
            # Update the reference list
            Rep_ref[:, Match_list[0, i]] = Line[:, Match_list[1, i]]
            Count_ref[0,Match_list[0, i]] = Count_ref[0,Match_list[0, i]] + 2
##            print("Count_ref2:",Count_ref)
        
    Count_ref[Count_ref < 0] = 0
    Count_ref[Count_ref > CountUpperThresh] = CountUpperThresh
    return [ Rep_ref, Count_ref]


##def f(x):
##    f1slope=((0.1))/(3000*0.0001)
##    fslope=((-0.05))/(7000*0.0001)
##    if x > 0.3:
##        x=((fslope)*(x-0.3))+0.3
##    if x == 0:
##        x=x
##    if (x<0.3)&(x>0):
##        x=((f1slope)*x)+0.2
##        
##    return x










rhoL=0
rhoR=0
thetaL=0
thetaR=0
MaxLaneNum = 20  # Maximum number of lanes to store in the tracking repository.
ExpLaneNum = 2  # Maximum number of lanes to find in the current frame.
Rep_ref   = np.zeros((ExpLaneNum, MaxLaneNum)) # Stored lines
Count_ref = np.zeros((1, MaxLaneNum))        # Count of each stored line
TrackThreshold = 75  # Maximum allowable change of lane distance

##f = np.vectorize(f)
frameFound=5
frameLost=20
npi=np.pi/180
NumRows = 180
test_count=0
images = cv2.VideoCapture('rain.mp4')
D=0
while (images.isOpened()):
    
    ret, frame = images.read()
    cv2.imwrite("frame%d.jpg" % D, frame)
    D=D+1
##    print(time.time()-start)
    imlow1=frame[500:600,100:800,:]
    test_count=test_count+1
##    print(test_count)
    imlow=cv2.cvtColor(imlow1,cv2.COLOR_BGR2GRAY)
    imlow = cv2.normalize(imlow.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
##    laplacian = cv2.Laplacian(imlow,cv2.CV_64F)
##    laplacian1 = np.absolute(laplacian)
##    laplacian2 = np.uint8(laplacian1)
    kernel = np.array([[-1, 0, 1]])
    dst = cv2.filter2D(imlow, -1, kernel)

    dst[dst<0] = 0  # Where values are low
    dst[dst>1] = 1  # Where values are high
    dst=dst*255
    dst=np.uint8(dst)
    ret,th3 = cv2.threshold(dst,20,255,cv2.THRESH_BINARY)

    lines = cv2.HoughLines(th3,1,np.pi/180,1)
    lines=np.double(lines)
    condition11=1
    condition12=1
    
    if lines!=[]:
        for rho,theta in lines[:,0,:]:
            
            if (theta<80*npi)&(theta>0*npi)&(condition11==1):
                thetaL=theta
                rhoL=rho
                condition11=0
            if (theta>120*npi)&(theta<180*npi)&(condition12==1):
                thetaR=theta
                rhoR=rho
                condition12=0
            if (condition11==0)&(condition12==0):
                break
    Count1=2
    Enable = np.ones((1,Count1))
    Line = np.array([[rhoL,rhoR],[thetaL,thetaR]])

    Rep_ref, Count_ref = tracker(Rep_ref, Count_ref, MaxLaneNum, ExpLaneNum, Enable, Line, TrackThreshold,frameFound+frameLost);

    Count_ref1=copy.deepcopy(Count_ref)
    Count_refind=Count_ref1.argmax()
    rho1=Rep_ref[0,Count_refind]
    theta1=Rep_ref[1,Count_refind]
    Count_ref1[0, Count_refind]=0
    Count_refind=Count_ref1.argmax()
    rho2=Rep_ref[0,Count_refind]
    theta2=Rep_ref[1,Count_refind]
    
    a = np.cos(theta1)
    b = np.sin(theta1)
    x0 = a*rho1
    y0 = b*rho1
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(imlow1,(x1,y1),(x2,y2),(0,0,255),2)
##    print('left')


    a = np.cos(theta2)
    b = np.sin(theta2)
    x0 = a*rho2
    y0 = b*rho2
    x11 = int(x0 + 1000*(-b))
    y11 = int(y0 + 1000*(a))
    x22 = int(x0 - 1000*(-b))
    y22 = int(y0 - 1000*(a))
    cv2.line(imlow1,(x11,y11),(x22,y22),(0,0,255),2)
##    print(np.array([[x1,y1],[x2,y2]]))

    mask = np.zeros(imlow.shape, dtype=np.uint8)
    mask2 = np.zeros(imlow.shape, dtype=np.uint8)
    roi_corners = np.array([[(x1,y1),(x2,y2),(x11,y11),(x22,y22)]], dtype=np.int32)
##    channel_count = imlow.shape[2]

    if ((x1-x2)>0)&((y1-y2)<0):
        
        x1=x1+10
        x2=x2+10
    if ((x1-x2)<0)&((y1-y2)<0):
        
        x1=x1-10
        x2=x2-10
    if ((x11-x22)>0)&((y11-y22)<0):
        
        x11=x11+10
        x22=x22+10
    if ((x11-x22)<0)&((y11-y22)<0):
        
        x11=x11-10
        x22=x22-10

    if ((x1-x2)<0)&((y1-y2)>0):
        
        x1=x1+10
        x2=x2+10
    if ((x1-x2)>0)&((y1-y2)>0):
        
        x1=x1-10
        x2=x2-10
    if ((x11-x22)<0)&((y11-y22)>0):
        
        x11=x11+10
        x22=x22+10
    if ((x11-x22)>0)&((y11-y22)>0):
        
        x11=x11-10
        x22=x22-10
##    print(np.array([[x1,y1],[x2,y2],[x11,y11],[x22,y22]]))
    roi_corners2 = np.array([[(x1,y1),(x2,y2),(x11,y11),(x22,y22)]], dtype=np.int32)
    cv2.fillPoly(mask2, roi_corners2, 1)
    
    
    cv2.fillPoly(mask, roi_corners, 255)
    imlow=np.uint8(imlow*255)
    masked_image = cv2.bitwise_and(imlow, mask)
    masked_image = cv2.normalize(masked_image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    mask[mask>0]=1
    
    imlowf1=copy.deepcopy(masked_image)
    imlowf2=copy.deepcopy(masked_image)

    imlowf2[imlowf1<0.3]=0
    imlowf2[imlowf1>=0.3]=1
    imlowf1=imlowf1*imlowf2
    imlowf1=1-imlowf1
    imlowf1 = cv2.normalize(imlowf1.astype('float'), None, 0.25, 0.3214285714285714, cv2.NORM_MINMAX)
    imlowf1=imlowf1*imlowf2
    imlowf1=mask*imlowf1


    imlowf11=copy.deepcopy(masked_image)
    imlowf22=copy.deepcopy(masked_image)
    imlowf22[(imlowf11<0.3)]=1
    imlowf22[imlowf11>=0.3]=0
    imlowf11=imlowf11*imlowf22
    imlowf11[0,0]=1
    imlowf11 = cv2.normalize(imlowf11.astype('float'), None, 0.2, 0.5333333333333333, cv2.NORM_MINMAX)
    imlowf11=imlowf11*abs(imlowf2-1)
    imlowf11=mask*imlowf11
    imlowf1=imlowf1+imlowf11

    th3 = cv2.Canny(np.uint8(imlowf1*255),10,100,apertureSize = 3)
    th2 = cv2.Canny(np.uint8(masked_image*255),35,100,apertureSize = 3)
    th3=mask2*th3
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(50,5))
    closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
    dilation = cv2.dilate(closing,kernel,iterations = 1)
    
##    edges = cv2.filter2D(imlowf1, -1, kernel)
##    edges[edges<0] = 0  # Where values are low
##    edges[edges>1] = 1  # Where values are high
##    edges=edges*255
##    edges=np.uint8(edges)
##    ret,edges = cv2.threshold(edges,20,255,cv2.THRESH_BINARY)
    
    
    
    
##    imlowf=f(imlowf)
##    f1slope=((0.1))/(3000*0.0001)
##    fslope=((-0.05))/(7000*0.0001)
##    size1,size2=imlow.shape
##    for f1 in range(size1):
##        for f2 in range(size2):
##            if (imlowf[f1,f2]>0.3):
##                imlowf[f1,f2]=((fslope)*(imlowf[f1,f2]-0.3))+0.3
##
##            if (imlowf[f1,f2]<0.3)&(imlowf[f1,f2]>0):
##                imlowf[f1,f2]=((f1slope)*(imlowf[f1,f2]))+0.2
            
        
    

    
##    edges = cv2.Canny(imlow,50,150,apertureSize = 3)
##    lines = cv2.HoughLines(edges,1,np.pi/180,200)

##    for i in range(len(lines[:,0])):
##        theta=(180/np.pi)*lines[i,0,1]
##        if (theta<-78)|(theta>78):
##            lines[i,0,1]=0
##            lines[i,0,0]=0

    imlow1[:,:,1]=imlow1[:,:,1]+dilation*50
    
    cv2.imshow('frame',th3)
    cv2.imshow('frame2',th2)
    cv2.imshow('frame3',frame)
##    while cv2.waitKey(10)&0xff!=27:
##        pass
   
    if cv2.waitKey(10)&0xff == 27:
        break
cv2.destroyAllWindows()

















