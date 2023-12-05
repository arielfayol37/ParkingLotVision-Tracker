 %codes
clc;
close all;
% clear all;
%image acquisition
f=imread('image.jpeg');
f=imresize(f,[800 NaN]); % image loading unit
figure (1)
imshow(f)
g=rgb2gray(f);
g=medfilt2(g,[5 5]);
figure (2)
imshow (g)
% morphological image processing 
conc=strel('disk',5);
gi=imdilate(g,conc);
conc1=strel('disk',5);
ge=imerode(gi,conc1); % morphological image processing
gdiff=imsubtract(gi,ge);
gdiff1=mat2gray(gdiff);
figure (4)
imshow (gdiff1)
gdiff2=conv2(gdiff1,[1 1;1 1]);
figure  (5)
imshow (gdiff2)
gdiff3=imadjust(gdiff2,[0.4 0.9],[0 1],1);
figure (6)
imshow (gdiff3)
B=logical(gdiff3);
[a1 b1]=size(B);
figure(7)
imshow(B)
er=imerode(B,strel('line',60,8));
figure(8)
imshow(er)
out1=imsubtract(B,er);
F=imfill(out1,'holes'); %filling the object
H=bwmorph(F,'thin',0.5);
H=imerode(H,strel('line',8,55));
figure(9)
imshow(H)
%Normalization% & Object Recognition 
I=bwareaopen(H,floor((a1/18)*(b1/18)));
I(1:floor(.9*a1),1:2)=1;
I(a1:-1:(a1-20),b1:1:(b1-2))=1;
figure(10)
imshow(I)
%Cars detection in image
figure (11)
imshowpair (f,I)
%Create bounding box on detected cars
Iprops=regionprops(I,'BoundingBox','Image');
hold on
text(8,785,strcat('\color{green}Cars Detected:',num2str(length(Iprops))))
hold on
for n=2:size(Iprops,1)
rectangle('Position',Iprops(n).BoundingBox,'EdgeColor','g','LineWidth',2);
end
hold off