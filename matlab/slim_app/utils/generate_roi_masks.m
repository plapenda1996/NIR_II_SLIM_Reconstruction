function roi_mask = generate_roi_masks(tform_list, image_size, roi_ratio, blur_range)
    % Set default values for optional parameters
    if nargin < 3
        roi_ratio = 0.9;
    end
    if nargin < 4
        blur_range = 7;
    end

    % Define the number of rows and columns based on image size
    numRows = image_size(1);
    numCols = image_size(2);

    % Calculate the center and radii of the ellipse
    centerX = numCols / 2;
    centerY = numRows / 2;
    radiusX = (numCols * roi_ratio) / 2;
    radiusY = (numRows * roi_ratio) / 2;

    % Create a meshgrid for the image
    [x, y] = meshgrid(1:numCols, 1:numRows);

    % Generate the binary mask for the ellipse
    roi_mask_center = ((x - centerX).^2 / radiusX^2 + (y - centerY).^2 / radiusY^2) <= 1;

    % Create soft ROI mask using Gaussian filter
    roi_mask_center = imgaussfilt(double(roi_mask_center), blur_range);

    % Determine the number of views and depths
    numViews = size(tform_list, 1);
    numDepths = size(tform_list, 2);

    % Initialize the ROI mask
    roi_mask = zeros([size(roi_mask_center), numViews]);

    % Loop through each view and depth to generate and display the ROI masks
    for i = 1:numViews
        % do integration for depths
        roi_mask_tmp = zeros([size(roi_mask_center), numDepths]);
        for j=1:numDepths
            roi_mask_tmp(:,:,j) = imwarp(roi_mask_center, tform_list(i, j), 'OutputView', imref2d(size(roi_mask_center)));
        end
        roi_mask_tmp = sum(roi_mask_tmp,3);
        roi_mask_tmp(roi_mask_tmp>1) = 1;
        roi_mask(:,:,i) = roi_mask_tmp;
    end

end