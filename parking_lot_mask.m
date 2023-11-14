image = imread("DJI_0134.JPG");

%array of masks for each parking lot, with a classification for each? 

x = [994 4706 4737 973; 898 4858 4800 934]; %x and y coordinates of each polygon 
y= [1781 1712 1980 2101; 2876 2805 2441 2527];

[im_length, im_width] = size(im2gray(image));
final_mask = false(im_length, im_width); %create empty mask of correct size

for i = 1:height(x)
    mask = poly2mask(x(i, :), y(i, :), im_length, im_width); %create sub mask with specified coordinates
    final_mask = final_mask | mask; %add sub mask to final mask
end

%mask image on each RGB channel separately 
% Separate the RGB channels
redChannel = image(:, :, 1);
greenChannel = image(:, :, 2);
blueChannel = image(:, :, 3);

% Apply the binary mask to each channel separately
redChannel(final_mask == 0) = 0; % Set masked pixels to 0 in the red channel
greenChannel(final_mask == 0) = 0; % Set masked pixels to 0 in the green channel
blueChannel(final_mask == 0) = 0; % Set masked pixels to 0 in the blue channel

masked_RGB = cat(3, redChannel, greenChannel, blueChannel);


%method to find lines 
gray = im2gray(image); %convert to grayscale
se = strel("rectangle", [10 10]);
opened = imopen(gray, se);

diff_lines = gray - opened; %find only the lines 
sharpened = imsharpen(diff_lines);
binary_lines = im2bw(sharpened, 0.2);
binary_lines_dilated = imdilate(binary_lines, se);

%imshow(masked_RGB);
%sweeped = sweep_open(binary_lines_dilated, 150);

%method to find cars
se2 = strel("rectangle", [10 10]);

eroded = imerode(gray, se); %erode image to remove small details

diff_eroded = gray - eroded; %find difference 

dilated = imdilate(diff_eroded, se2); %dilate image

dilated_diff = dilated - eroded; 
BW2 = imbinarize(dilated_diff);


imshowpair(BW2, gray, "montage");

function s = sweep_open(I, length)
[r c] = size(I);
s=zeros(r,c);
    %sweep through full circle  
    for i=1:10:180
        %create structuring element for corresponding angle 
        se = strel('line', length, i);
        %open image with current angle and add to image 
        s = s | imopen(I,se);
        imshow(s);
        drawnow;
    end
end


function h = has_car(M)
    %input is a matrix (region of the image)
    % calculates the sum and returns whether it meets the treshold
    S = sum(M, "all");
    h = (S > 1000);
end



