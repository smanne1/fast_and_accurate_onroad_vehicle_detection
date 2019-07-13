clc
close all
clear all
%%
I = imread('circuit.tif');
%%
hedge = vision.EdgeDetector;
    hhoughtrans = vision.HoughTransform(pi/360,'ThetaRhoOutputPort', true);
    hfindmax = vision.LocalMaximaFinder(1,	'HoughMatrixInput', true);
    hhoughlines = vision.HoughLines('SineComputation','Trigonometric function');
    %%
    BW = step(hedge, I);
    [ht, theta, rho] = step(hhoughtrans, BW);
    idx = step(hfindmax, ht);
    linepts = step(hhoughlines, theta(idx(:,1)-1), rho(idx(:,2)-1), I);
    imshow(I); hold on;
   line(linepts(:,[1 3])-1, linepts(:,[2 4])-1,'color',[1 1 0]);
   
   
   
   