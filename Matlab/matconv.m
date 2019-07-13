clc
close all
clear 
%untar('matconvnet-1.0-beta19.tar.gz') ;
%cd matconvnet-1.0-beta18
%run vl_compilenn ;

% Download a pre-trained CNN from the web (needed once).
%urlwrite(...
 % 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat', ...
 % 'imagenet-vgg-f.mat') ;

% Setup MatConvNet.
run vl_setupnn ;

% Load a model and upgrade it to MatConvNet current version.
net = load('imagenet-caffe-alex.mat') ;
net = vl_simplenn_tidy(net) ;

% Obtain and preprocess an image.
im = imread('peppers.png') ;
im_ = single(im) ; % note: 255 range

im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_ = im_ - net.meta.normalization.averageImage ;

% Run the CNN.
res = vl_simplenn(net, im_) ;

% Show the classification result.
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',net.meta.classes.description{best}, best, bestScore)) ;