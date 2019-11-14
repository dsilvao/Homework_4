% Homework 4  
%% Import Image
RGB = imread('3096_colorPlane.jpg');

imshow(RGB) 
%% RGB and X,Y information for feature set. 

wavelength = 2.^(0:5) * 3;
orientation = 0:45:135;
g = gabor(wavelength,orientation);
I = rgb2gray(im2single(RGB));
gabormag = imgaborfilt(I,g);
for i = 1:length(g)
    sigma = 0.5*g(i).Wavelength;
    gabormag(:,:,i) = imgaussfilt(gabormag(:,:,i),3*sigma); 
end
nrows = size(RGB,1);
ncols = size(RGB,2);
[X,Y] = meshgrid(1:ncols,1:nrows); 
featureSet = cat(3,I,gabormag,X,Y);
%% Kmeans clustering. K= 2
L2 = imsegkmeans(featureSet,2,'NormalizeInput',true);
C = labeloverlay(RGB,L2);
imshow(C)  
title('K = 2 Image') 
%% Kmeans clustering. K= 3
L2 = imsegkmeans(featureSet,3,'NormalizeInput',true);
D = labeloverlay(RGB,L2);
imshow(D)  
title('K = 3 Image') 
%% Kmeans clustering. K= 4
L2 = imsegkmeans(featureSet,4,'NormalizeInput',true);
F = labeloverlay(RGB,L2);
imshow(F)  
title('K = 4 Image') 
%% Kmeans clustering. K= 5
L2 = imsegkmeans(featureSet,5,'NormalizeInput',true);
G = labeloverlay(RGB,L2);
imshow(G)  
title('K = 5 Image')

