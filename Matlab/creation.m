videoSource = vision.VideoFileReader('cars.mp4','ImageColorSpace','Intensity','VideoOutputDataType','single');

detector = vision.ForegroundDetector('NumGaussians', 3, ...
            'NumTrainingFrames', 40, 'MinimumBackgroundRatio', 0.7);
   
 blob = vision.BlobAnalysis(...
       'CentroidOutputPort', false, 'AreaOutputPort', false, ...
       'BoundingBoxOutputPort', true, ...
       'MinimumBlobAreaSource', 'Property', 'MinimumBlobArea', 250,'ExcludeBorderBlobs',true);
 opticalFlow = vision.OpticalFlow('ReferenceFrameDelay', 1);
 opticalFlow.OutputValue ='Horizontal and vertical components in complex form';
 shapeInserter = vision.ShapeInserter('BorderColor','White');
 convert = vision.ImageDataTypeConverter; 
 shapeInserter = vision.ShapeInserter('BorderColor','White');
 videoPlayer = vision.VideoPlayer('Position', [20, 400, 700, 400]);
 maskPlayer = vision.VideoPlayer('Position', [740, 400, 700, 400]);
while ~isDone(videoSource)
     frame  = step(videoSource);
    fgMask = step(detector, frame);
    of = step(opticalFlow, frame);
    of=abs(of);
    level = graythresh(of);
    BW = im2bw(of,level); 
    % Fill holes in regions.
BW_out = imfill(BW, 'holes');

% Filter image based on image properties.
BW2 = bwpropfilt(BW_out, 'Area', [98 + eps(98), Inf]);
   
    BW2=imclose(BW2, strel('rectangle', [50, 20]));
    BW2=imfill(BW2,'holes');
   BW2(1:160,:)=0;
    bbox   = step(blob, BW2);
    BW3=step(convert,BW2); 
      %fgMask=step(convert,fgMask); 
    % out    = step(shapeInserter, frame, bbox);
    %    out2    = step(shapeInserter, BW2, bbox);
    out = insertObjectAnnotation(BW3, 'rectangle',bbox,1);
                
    out2 = insertObjectAnnotation(frame, 'rectangle',bbox,1); 
    step(videoPlayer, out);
    step(maskPlayer, out2);
end
release(videoPlayer);
release(videoSource);



