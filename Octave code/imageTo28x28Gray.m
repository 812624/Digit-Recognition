function vectorImage = imageTo28x28Gray(img, cropPercentage=0)
%IMAGETO20X20GRAY display reduced image and converts for digit classification
%
Image3DmatrixRGB = img;
% Convert to NTSC image (YIQ)
Image3DmatrixYIQ = rgb2ntsc(Image3DmatrixRGB );
% Convert to grays keeping only luminance (Y) and discard chrominance (IQ)
Image2DmatrixBW  = Image3DmatrixYIQ(:,:,1);
% Get the size of your image
oldSize = size(Image2DmatrixBW);

cropDelta = floor((oldSize - min(oldSize)) .* (cropPercentage/100));
finalSize = oldSize - cropDelta;
cropOrigin = floor(cropDelta / 2) + 1;
copySize = cropOrigin + finalSize - 1;
croppedImage = Image2DmatrixBW( ...
                    cropOrigin(1):copySize(1), cropOrigin(2):copySize(2));

scale = [28 28] ./ finalSize;

% Compute back the new image size (extra step to keep code general)
newSize = max(floor(scale .* finalSize),1);
% Compute a re-sampled set of indices:
rowIndex = min(round(((1:newSize(1))-0.5)./scale(1)+0.5), finalSize(1));
colIndex = min(round(((1:newSize(2))-0.5)./scale(2)+0.5), finalSize(2));
% Copy just the indexed values from old image to get new image
newImage = croppedImage(rowIndex,colIndex,:);

% Invert black and white
invertedImage = - newImage;
% Find min and max grays values in the image
maxValue = max(invertedImage(:));
minValue = min(invertedImage(:));
% Compute the value range of actual grays
delta = maxValue - minValue;
% Normalize grays between 0 and 1
normImage = (invertedImage - minValue) / delta;
% Add contrast. Multiplication factor is contrast control.
x1 = (normImage -0.5) * 5;
contrastedImage = 1.0 ./ (1.0 + exp(-x1));
% Show image as seen by the classifier
imshow(contrastedImage, [-1, 1] );
% Output the matrix as a unrolled vector
vectorImage = reshape(contrastedImage, 1, newSize(1)*newSize(2));

end
