clc
close all
clear all
convert = vision.ImageDataTypeConverter; 
opticalFlow = vision.OpticalFlow('ReferenceFrameDelay', 1);
 opticalFlow.OutputValue ='Horizontal and vertical components in complex form';
 blob = vision.BlobAnalysis(...
       'CentroidOutputPort', false, 'AreaOutputPort', false, ...
       'BoundingBoxOutputPort', true, ...
       'MinimumBlobAreaSource', 'Property', 'MinimumBlobArea', 250,'ExcludeBorderBlobs',true);
videoFReader = vision.VideoFileReader('cars.mp4');
 videoPlayer = vision.VideoPlayer('Position', [0, 400, 700, 400],'Name','Video');
 maskPlayer = vision.VideoPlayer('Position', [740, 400, 700, 400],'Name','Velocity vector mask');
 maskPlayer2 = vision.VideoPlayer('Position', [0, 0, 700, 400],'Name','Black colour filter mask');
 maskPlayer3 = vision.VideoPlayer('Position', [740, 0, 700, 400],'Name','velocity vector + Black colur filter mask');
while ~isDone(videoFReader)
  videoFrame = step(videoFReader);
red=videoFrame(:,:,1);
green=videoFrame(:,:,2);
blue=videoFrame(:,:,3);
filt =(red<0.7)&(red>0.35)&(green<(0.7))&(green>0.35)&(blue<(0.7))&(blue>0.35);
BW_out1 = imfill(filt, 'holes');

% Filter image based on image properties.
filt2 = bwpropfilt(BW_out1, 'Area', [108 + eps(108), Inf]);
filt2=~filt2;
filt3=filt2;
filt3(1:160,:)=0;
vFrame=rgb2gray(videoFrame);
of = step(opticalFlow, vFrame);
    of=abs(of);
    exp = tsmovavg(of,'e',12,1);
    %mesh(of-exp);
    of=of-2*exp;
    level = graythresh(of);
    BW = im2bw(of,level); 
    % Fill holes in regions.
BW_out = imfill(BW, 'holes');

% Filter image based on image properties.
BW2 = bwpropfilt(BW_out, 'Area', [98 + eps(98), Inf]);
   
    BW2=imclose(BW2, strel('rectangle', [80, 20]));
    BW2=imfill(BW2,'holes');
new=BW2&filt3; 
new=imfill(new,'holes');
new=imclose(new, strel('rectangle', [20, 20]));
bbox=step(blob,new);
new=step(convert,new);
filt4=step(convert,filt3);


% out=insertObjectAnnotation(new, 'rectangle',bbox,1); 
out2=insertObjectAnnotation(videoFrame, 'rectangle',bbox,1);   
% out3=insertObjectAnnotation(filt4, 'rectangle',bbox,1); 
% out4=insertObjectAnnotation(of, 'rectangle',bbox,1); 
% step(maskPlayer,out4);
% step(maskPlayer3,out);
% step(maskPlayer2,out3);
step(videoPlayer, out2);
end
release(videoPlayer);
release(videoFReader);
release(maskPlayer);
release(maskPlayer2);
release(maskPlayer3);