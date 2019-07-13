clc
close all
clear all
vdd = vision.VideoFileReader('cars.mp4','ImageColorSpace','Intensity','VideoOutputDataType','uint16');
convert = vision.ImageDataTypeConverter; 
opticalFlow = vision.OpticalFlow('ReferenceFrameDelay', 1);
opticalFlow.OutputValue ='Horizontal and vertical components in complex form';
shapeInserter = vision.ShapeInserter('Shape','lines','BorderColor','custom', 'CustomBorderColor', 200);
vddp = vision.VideoPlayer('Position', [0, 200, 700, 400],'Name','MSra1');


while ~isDone(vdd)
    frame = step(vdd);
    frame=imresize(frame,[360,640]);
    image = step(convert, frame);
   
    of = step(opticalFlow, image);
    of=abs(of);
    level = graythresh(of);
    BW = im2bw(of,level);
    lines = videooptflowlines(BW, 10);
    if ~isempty(lines)
      out =  step(shapeInserter, image, lines); 
      step(vddp, out);
    
    end
end



