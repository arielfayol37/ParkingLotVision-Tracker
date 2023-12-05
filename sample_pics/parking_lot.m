% Demo to find available parking spaces.  By Image Analyst, May 12, 2020.
% Requires 3 images: 'Parking Lot without Cars.jpg', 'Parking Lot with Cars.jpg', 'Parking Lot Mask.png'

% Clean up.
clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clearvars;
workspace;  % Make sure the workspace panel is showing.
format long g;
format compact;
fontSize = 16;
fprintf('Beginning to run %s.m ...\n', mfilename);

%-----------------------------------------------------------------------------------------------------------------------------------
% Read in reference image with no cars (empty parking lot).
folder = pwd;
baseFileName = 'Parking Lot without Cars.jpg';
fullFileName = fullfile(folder, baseFileName);
% Check if file exists.
if ~exist(fullFileName, 'file')
	% The file doesn't exist -- didn't find it there in that folder.
	% Check the entire search path (other folders) for the file by stripping off the folder.
	fullFileNameOnSearchPath = baseFileName; % No path this time.
	if ~exist(fullFileNameOnSearchPath, 'file')
		% Still didn't find it.  Alert user.
		errorMessage = sprintf('Error: %s does not exist in the search path folders.', fullFileName);
		uiwait(warndlg(errorMessage));
		return;
	end
end
rgbEmptyImage = imread(fullFileName);
[rows, columns, numberOfColorChannels] = size(rgbEmptyImage);
% Display the test image full size.
subplot(2, 3, 2);
imshow(rgbEmptyImage, []);
axis('on', 'image');
caption = sprintf('Reference Image : "%s"', baseFileName);
title(caption, 'FontSize', fontSize, 'Interpreter', 'None');
drawnow;
hp = impixelinfo(); % Set up status line to see values when you mouse over the image.

%-----------------------------------------------------------------------------------------------------------------------------------
% Read in test image (image with cars parked on the parking lot).
folder = pwd;
baseFileName = 'Parking Lot with Cars.jpg';
fullFileName = fullfile(folder, baseFileName);
% Check if file exists.
if ~exist(fullFileName, 'file')
	% The file doesn't exist -- didn't find it there in that folder.
	% Check the entire search path (other folders) for the file by stripping off the folder.
	fullFileNameOnSearchPath = baseFileName; % No path this time.
	if ~exist(fullFileNameOnSearchPath, 'file')
		% Still didn't find it.  Alert user.
		errorMessage = sprintf('Error: %s does not exist in the search path folders.', fullFileName);
		uiwait(warndlg(errorMessage));
		return;
	end
end
rgbTestImage = imread(fullFileName);
[rows, columns, numberOfColorChannels] = size(rgbTestImage);
% Display the original image full size.
subplot(2, 3, 1);
imshow(rgbTestImage, []);
axis('on', 'image');
caption = sprintf('Test Image : "%s"', baseFileName);
title(caption, 'FontSize', fontSize, 'Interpreter', 'None');
drawnow;
hp = impixelinfo(); % Set up status line to see values when you mouse over the image.

% Set up figure properties:
% Enlarge figure to full screen.
hFig1 = gcf;
hFig1.Units = 'Normalized';
hFig1.WindowState = 'maximized';
% Get rid of tool bar and pulldown menus that are along top of figure.
% set(gcf, 'Toolbar', 'none', 'Menu', 'none');
% Give a name to the title bar.
hFig1.Name = 'Processing Pipeline';

%-----------------------------------------------------------------------------------------------------------------------------------
% Read in mask image that defines where the spaces are.
folder = pwd;
baseFileName = 'Parking Lot Mask.png';
fullFileName = fullfile(folder, baseFileName);
% Check if file exists.
if ~exist(fullFileName, 'file')
	% The file doesn't exist -- didn't find it there in that folder.
	% Check the entire search path (other folders) for the file by stripping off the folder.
	fullFileNameOnSearchPath = baseFileName; % No path this time.
	if ~exist(fullFileNameOnSearchPath, 'file')
		% Still didn't find it.  Alert user.
		errorMessage = sprintf('Error: %s does not exist in the search path folders.', fullFileName);
		uiwait(warndlg(errorMessage));
		return;
	end
end
maskImage = imread(fullFileName);
[rows, columns, numberOfColorChannels] = size(maskImage);
% Create a binary mask from seeing where the min value is 255.
mask = min(maskImage, [], 3) == 255;
% Display the test image full size.
subplot(2, 3, 3);
imshow(mask, []);
axis('on', 'image');
caption = sprintf('Mask Image : "%s"', baseFileName);
title(caption, 'FontSize', fontSize, 'Interpreter', 'None');
drawnow;
hp = impixelinfo(); % Set up status line to see values when you mouse over the image.

%-----------------------------------------------------------------------------------------------------------------------------------
% Find the cars.
% First, get the absolute difference image.
diffImage = imabsdiff(rgbEmptyImage, rgbTestImage);
% Display the gray scale image.
subplot(2, 3, 4);
imshow(diffImage, []);
axis('on', 'image');
caption = sprintf('Difference Image');
title(caption, 'FontSize', fontSize, 'Interpreter', 'None');
drawnow;
hp = impixelinfo(); % Set up status line to see values when you mouse over the image.

% Convert to gray scale and mask it with the spaces mask.
diffImage = rgb2gray(diffImage);
diffImage(~mask) = 0;
% Get a histogram of the image so we can see where to threshold it at.
subplot(2, 3, 5);
histogram(diffImage(diffImage>0));
%drawnow;
% Display the gray scale image.
imshow(diffImage, []);
axis('on', 'image');
caption = sprintf('Gray Scale Difference Image');
title(caption, 'FontSize', fontSize, 'Interpreter', 'None');
drawnow;
hp = impixelinfo(); % Set up status line to see values when you mouse over the image.

% Threshold the image to find pixels that are substantially different from the background.
kThreshold = 40; % Determined by examining the histogram.
parkedCars = diffImage > kThreshold;
% Fill holes.
parkedCars = imfill(parkedCars, 'holes');
% Get convex hull.
parkedCars = bwconvhull(parkedCars, 'objects');
% Display the mask image.
subplot(2, 3, 6);
imshow(parkedCars, []);
impixelinfo;
axis('on', 'image');
caption = sprintf('Parked Cars Binary Image with Threshold = %.1f', kThreshold);
title(caption, 'FontSize', fontSize, 'Interpreter', 'None');
drawnow;

%-----------------------------------------------------------------------------------------------------------------------------------
% Measure the percentage of white pixels within each rectangular mask.
props = regionprops(mask, parkedCars, 'MeanIntensity', 'Centroid', 'BoundingBox');
centroids = vertcat(props.Centroid);
%-----------------------------------------------------------------------------------------------------------------------------------
% Optional sorting by row with kmeans.  Only appropriate for parking lot with rectangular grid aligned with image edges.
% Sort y centroids 
numRows = 4; % There are 4 rows in the parking lot to park in.
[indexesY, clusterCenterY] = kmeans(centroids(:, 2), numRows);
% Put lines across at the y centroids of the spaces in the mask rows.
for k = 1 : numRows
	yline(clusterCenterY(k), 'Color', 'g', 'LineWidth', 2);
end
% Put yellow bounding boxes for each space (whether taken or available).
for k = 1 : length(props)
	rectangle('Position', props(k).BoundingBox, 'EdgeColor', 'y');
end
% Now the clusters are not necessarily sorted with cluster 1 at the top, cluster 2 right below it, cluster 3 below that, and cluster 4 at the bottom.
% So we need to sort the clusters by the y value of the cluster center.
[~, sortOrder] = sort(clusterCenterY, 'ascend');
% Now sort the indexes and we will have them going from top to bottom.
originalIndexes = indexesY; % Save a copy so we can compare to make sure it did it correctly.
fprintf('\nOriginal Class Number    New Class #\n');
for k = 1 : length(indexesY)
	currentClass = indexesY(k);
	newClass = find(sortOrder == currentClass);
	fprintf('                  %d       %d\n', newClass, currentClass);
	indexesY(k) = newClass;
% 	text(centroids(k, 1), centroids(k, 2)-15, num2str(currentClass), 'Color', 'r');
% 	text(centroids(k, 1), centroids(k, 2)+15, num2str(newClass), 'Color', 'g');
end
% comparedLabels = [originalIndexes(:), indexesY(:)];
newLabels = 1 : length(props); % Look up table so we can map the original index to a new sorted index.
pointer = 0;
for k = 1 : numRows
	% For this class (row of parking), get the elements of props that are in this row (y value).
	thisClass = find(indexesY == k);
	% The x centroid values are already sorted from left to right so we're ok there - no need to sort.
	newLabels((pointer + 1) : (pointer + length(thisClass))) = thisClass;
	pointer = pointer + length(thisClass);
end
% Now that we have the new sort order, apply it.
props = props(newLabels);
% Re-extract these vectors with the new order.
percentageFilled = [props.MeanIntensity]
centroids = vertcat(props.Centroid);
%-----------------------------------------------------------------------------------------------------------------------------------


%-----------------------------------------------------------------------------------------------------------------------------------
% Place a red x on the image if the space is filled, and a green circle if the space is available to be parked on (it's empty).
% Go through each rectangle and say whether it's filled with a car or not.
% We'll say it's filled if 10% of the pixels are filled.
hFig2 = figure;
imshow(rgbTestImage);
hFig2.WindowState = 'maximized';
% Give a name to the title bar.
hFig2.Name = 'Labeling free and occupied spots';
hold on;
numberOfCars = 0;
for k = 1 : length(props)
	x = centroids(k, 1);
	y = centroids(k, 2);
	blobLabel = sprintf('%d', k);
	if percentageFilled(k) > 0.40
        numberOfCars = numberOfCars + 1;
		% It has a car in that rectangle.
		plot(x, y, 'rx', 'MarkerSize', 30, 'LineWidth', 4);
		% Put up the blob label.
		text(x, y+20, blobLabel, 'Color', 'r', 'FontSize', 15, 'FontWeight', 'bold');
	else
		% No car is parked there.  The space is available.
		plot(x, y, 'g.', 'MarkerSize', 40, 'LineWidth', 4);
		% Put up the blob label.
		text(x, y+20, blobLabel, 'Color', 'g', 'FontSize', 15, 'FontWeight', 'bold');
	end
	
end
title('Marked Spaces.  Green Spot = Available.  Red X = Taken.', 'FontSize', fontSize);
fprintf('Done running %s.m ...\n', mfilename);
fprintf('Number of parked cars: %d\n', numberOfCars);
