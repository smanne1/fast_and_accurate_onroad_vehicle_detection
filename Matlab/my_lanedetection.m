clc
close all
NumRows = 180;
MaxLaneNum = 20;
ExpLaneNum = 2;
Rep_ref   = zeros(ExpLaneNum, MaxLaneNum); 
Count_ref = zeros(1, MaxLaneNum);
TrackThreshold = 75;
Square=zeros(8,1);
slope=zeros(1,2);
frameFound = 5;
frameLost = 20;
RLshift=100;
offset = int32([0, NumRows, 0, NumRows]);
%%
% load('dataset_matfile.mat');
%%
convert = vision.ImageDataTypeConverter; 
opticalFlow = vision.OpticalFlow('ReferenceFrameDelay', 1);
opticalFlow.OutputValue ='Horizontal and vertical components in complex form';
hColorConv1 = vision.ColorSpaceConverter( ...
                    'Conversion', 'RGB to intensity');
hColorConv2 = vision.ColorSpaceConverter( ...
                    'Conversion', 'RGB to YCbCr');
hFilter2D = vision.ImageFilter( ...
                    'Coefficients', [-1 0 1], ...
                    'OutputSize', 'Same as first input', ...
                    'PaddingMethod', 'Replicate', ...
                    'Method', 'Correlation');
hAutothreshold = vision.Autothresholder;
hdilate = vision.MorphologicalDilate('Neighborhood', ones(500,5));
hedge = vision.EdgeDetector;
blob = vision.BlobAnalysis(...
       'CentroidOutputPort', false, 'AreaOutputPort', false, ...
       'BoundingBoxOutputPort', true, ...
       'MinimumBlobAreaSource', 'Property', 'MinimumBlobArea', .....
       250,'ExcludeBorderBlobs',false);
%%
hHough = vision.HoughTransform( ...
                    'ThetaRhoOutputPort', true, ...
                    'OutputDataType', 'single');
                
hLocalMaxFind1 = vision.LocalMaximaFinder( ...
                        'MaximumNumLocalMaxima', ExpLaneNum, ...
                        'NeighborhoodSize', [301 81], ...
                        'Threshold', 1, ...
                        'HoughMatrixInput', true, ...
                        'IndexDataType', 'uint16');
                    
hHoughLines1 = vision.HoughLines(.....
               'SineComputation', 'Trigonometric function');
%%
hdilate = vision.MorphologicalDilate('Neighborhood', ones(5,5));
videoPlayer = vision.VideoPlayer('Position', [0, 400, 700, 400],'Name','Road_detection');
 maskPlayer = vision.VideoPlayer('Position', [740, 400, 700, 400],'Name','Road_seperation');
 maskPlayer2 = vision.VideoPlayer('Position', [0, 0, 700, 400],'Name','Object detection');
  maskPlayer3 = vision.VideoPlayer('Position', [740, 0, 700, 400],'Name','Illumination variation filter');
 Frame = 0;
NumNormalDriving = 0;
Broken = false;

%%
for(j=1:180)
   RGB = squeeze(y(j,:,:,:))/255;
    Imlow  = RGB(NumRows+1:600,100:800, :);
   RGB1=Imlow;
%    red=RGB1(:,:,1);
%    green=RGB1(:,:,2);
%    blue=RGB1(:,:,3);
%    cutx=470;
%    cuty=140;
%    cutx1=670;
%    cuty1=180;
%    diffx=cutx1-cutx+1;
%    diffy=cuty1-cuty+1;
%    R1=red(cuty:cuty1,cutx:cutx1);
%    G1=green(cuty:cuty1,cutx:cutx1);
%    B1=blue(cuty:cuty1,cutx:cutx1);
%    R11=min(min(R1));
%    G11=min(min(G1));
%    B11=min(min(B1));
%    R12=(sum(sum(R1))/(diffx*diffy))+0.15;
%    G12=(sum(sum(G1))/(diffx*diffy))+0.15;
%    B12=(sum(sum(B1))/(diffx*diffy))+0.15;
%    Mfilter=(red<(R12))&(red>=(R11))&(green<(G12))....
%        &(green>=G11)&(blue<(B12))&(blue>(B11));

  
   Imlow2  = RGB(NumRows+1:end,100:800, :);
   Imlow = step(hColorConv1, Imlow);
   Imlow1=Imlow;
   [size1 size2]=size(Imlow);
%   Edge2=edge(Imlow,'canny');
%    Edge3=step(hdilate,Edge2);
% nshadow=3;
% k=0;
% for (i1=1:size1)
% for (j6=1:nshadow:(floor(size2/nshadow))*nshadow)
%     shadow=[Imlow(i1,j6) Imlow(i1,j6+1) Imlow(i1,j6+2)] ; 
%     shadowMax=max(shadow);
%     shadowMin=min(shadow);
%     shadowDiff=shadowMax-shadowMin;
%     if (shadowDiff>0.02)&&(shadowDiff<0.04)
%         for(j7=1:3)
%             if shadow(j7)==shadowMax
%                 Imlow(i1,j6+j7-1)=1;
%                 k=k+1;
%             end    
%         end  
%     end    
% end
% end
   I = step(hFilter2D, Imlow);
   I(I < 0) = 0;
   I(I > 1) = 1;
   Edge = step(hAutothreshold, I);


%%





%%
   [H, Theta, Rho] = step(hHough,Edge);
   H1 = H;
   H1(:, 1:12) = 0;
   H1(:, end-12:end) = 0;
   Idx1 = step(hLocalMaxFind1, H1);
   Count1 = size(Idx1,1);
   Line = [Rho(Idx1(:, 2)); Theta(Idx1(:, 1))]
   Enable = [ones(1,Count1) zeros(1, ExpLaneNum-Count1)]; 
   [Rep_ref, Count_ref] = videolanematching(Rep_ref, Count_ref, ...
                                MaxLaneNum, ExpLaneNum, Enable, Line, ...
                                TrackThreshold, frameFound+frameLost);
   Pts = step(hHoughLines1, Rep_ref(2,:), Rep_ref(1,:), Imlow);
   Count_ref1=Count_ref;
   col1=zeros(1,2);
   for (j1=1:2)
       [ val col1(j1)]=max(Count_ref1);
       Count_ref1(col1(j1))=0;
   end
   
  
   
       Frame = Frame + 1;
    if Frame >= 5
        
   Two=[Pts(col1(1),:); Pts(col1(2),:)];
   
   Two=double(Two);
    for (j2=1:2)
       slope(j2)=((Two(j2,4)-Two(j2,2)))/((Two(j2,3)-Two(j2,1)));
       C(j2)=Two(j2,2)-(slope(j2))*Two(j2,1);
   for (j3=1:size1)
       X(j2,j3)=abs(floor((j3-C(j2))/(slope(j2))));
   end
    end
    
   for (j4=1:size1)
       M1=min(X(:,j4));
       M2=max(X(:,j4));
       if M1<0
       M1=0;
       end
       if M2>1241
       M2=1241;
       end
   Imlow1(j4,:)=[zeros(1,M1) ones(1,length(M1+1:M2+1)) zeros(1,size2-M2-1)];
   end
    for(j5=1:3)
        Imlow2(:,:,j5)=Imlow2(:,:,j5).*Imlow1;
    end
   Square=[Two(1,1);Two(1,2)+NumRows;Two(1,3);....
       Two(1,4)+NumRows;Two(2,3);Two(2,4)+NumRows;Two(2,1);...
       Two(2,2)+NumRows];
   RGB = insertShape(RGB,'FilledPolygon',Square',...
                              'Color',[1 0 0],'Opacity',0.2);
    end
    Imlowf=Imlow.*Imlow1;
     alalal=Imlow.*Imlow1;
%  fslope=(-(0.3-mean(Imlowf(Imlowf>0)))/(7000*0.0001));
 f1slope=((0.1))/(3000*0.0001);
 fslope=((-0.05))/(7000*0.0001);
    
    for f1=1:size1
        for f2=1:size2
            if (Imlowf(f1,f2)>0.3)
   Imlowf(f1,f2)=double((fslope)*(Imlowf(f1,f2)-0.3))+0.3;
            end
        end
    end
    for f1=1:size1
        for f2=1:size2
            if (Imlowf(f1,f2)<0.3)&(Imlowf(f1,f2)>0)
   Imlowf(f1,f2)=double((f1slope)*(Imlowf(f1,f2)))+0.2;
            end
        end
    end
% Foptic=abs(step(opticalFlow,Imlowf));
%  I = step(hFilter2D, Foptic);
I1=step(hedge,Imlowf);
    I1exp=I1;
Imlowf1=Imlow.*Imlow1;
I3main=step(hedge,Imlowf1);

if Frame>5
for (j8=-4:4)
for (j6=1:size1)
    j7=2;
    if (X(j7,j6)+j8<=0) 
      X(j7,j6)=-j8+1;  
    end
        I1exp(j6,X(j7,j6)+j8)=0;
    end
end
for (j8=-4:4)
for (j6=1:size1)
    j7=1;
    if (X(j7,j6)+j8<=0)
      X(j7,j6)=-j8+1;  
    end
        I1exp(j6,X(j7,j6)+j8)=0;
    end
end
end
I1exp2 = bwpropfilt(I1exp, 'Area', [10 + eps(10), Inf]);
I1exp=I1exp2;
I1exp=~I1exp;
se = strel('disk',11);  
se1 = strel('disk',6); 
I1exp = imerode(I1exp,se);
% I1exp = imerode(I1exp,strel('rectangle', [35, 14]));
I1exp=~I1exp;


% newf=new&I3;
% I1=imfill(I1,'holes');
% new=imclose(I1, strel('disk', 20));
% 
% new=imclose(new, strel('rectangle', [80, 5]));
box=step(blob,I1exp);





Xco=box(:,1);
Yco=box(:,2);
Xco1=0;
Width=box(:,3);
Height=box(:,4);
boxarea=Width.*Height;
levsize=size(Xco)-size(Xco1);
sXco=size(Xco);
sXco1=size(Xco1);
if levsize(1)<0
        Xco((sXco+1:sXco+abs(levsize(1))),1)=0;
        Yco((sXco+1:sXco+abs(levsize(1))),1)=0;
        Width((sXco+1:sXco+abs(levsize(1))),1)=0;
        Height((sXco+1:sXco+abs(levsize(1))),1)=0;
        boxarea((sXco+1:sXco+abs(levsize(1))),1)=0;
        box((sXco+1:sXco+abs(levsize(1))),1:4)=0;
end
if levsize(1)>0
    Xco1((sXco1+1:sXco1+abs(levsize(1))),1)=0;
    Yco1((sXco1+1:sXco1+abs(levsize(1))),1)=0;
    Width1((sXco1+1:sXco1+abs(levsize(1))),1)=0;
    Height1((sXco1+1:sXco1+abs(levsize(1))),1)=0;
    boxarea1((sXco1+1:sXco1+abs(levsize(1))),1)=0;
end
for j10=1:size(Xco1)
for j9=1:size(Xco1)
    duffx=abs(Xco(j10,1)-Xco1(j9,1));
    duffy=abs(Yco(j10,1)-Yco1(j9,1));
    if (duffx<15)&&(duffy<15)
        if boxarea1(j9,1)>boxarea(j10,1)
            Width(j10,1)=Width1(j9,1);
            Height(j10,1)=Height1(j9,1);
        end
    end
end
end

box(:,3)=Width;
box(:,4)=Height;
storemax=0;
bcount=0;
for j11=1:size(Xco)
    if Yco(j11,1)>=50
        if boxarea(j11,1)>storemax
          if bcount==1
            box(store,1:4)=0;
          end
          bcount=1;
           store=j11;
           storemax=boxarea(j11,1);
           
        end
        if boxarea(j11,1)<storemax
           box(j11,1:4)=0;
           
        end
    end
end
% if (bcount==1)
% box(store,3)=236;
% box(store,4)=117;
% end
Xco1=box(:,1);
Yco1=box(:,2);

Width1=box(:,3);
Height1=box(:,4);
boxarea1=Width1.*Height1;

out=insertObjectAnnotation(Imlow, 'rectangle',box,1);


 step(maskPlayer,RGB);
 step(videoPlayer,I3main);
step(maskPlayer2,Imlowf);
step(maskPlayer3,I1exp2);  
    
end

                