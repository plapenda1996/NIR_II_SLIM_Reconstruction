function interactive_stitch_viewer(recon_array)
    % This function creates an interactive viewer for displaying stitched 3D images stored in a cell array
    % Each cell in recon_array should contain a 3D matrix of the same size
    % The function allows adjustment of the depth slice and x/y spacing (overlap)
    % Overlapping regions will be cropped from each cell rather than overwritten
    
    % Check if input is empty
    if isempty(recon_array)
        error('Input cell array is empty.');
    end
    
    % Get dimensions of the cell array
    [num_rows, num_cols] = size(recon_array);
    
    % Get dimensions of the first 3D image
    [height, width, depth] = size(recon_array{1,1});
    
    % Check if all elements in the cell array have the same dimensions
    for r = 1:num_rows
        for c = 1:num_cols
            if ~isequal(size(recon_array{r,c}), [height, width, depth])
                error('All 3D images in the cell array must have the same dimensions.');
            end
        end
    end
    
    % Create figure for interactive display
    fig = figure('Name', '3D Image Stitching Viewer', 'Position', [100, 100, 1000, 700]);
    
    % Create main axes for displaying the stitched image
    ax = axes('Parent', fig, 'Position', [0.1, 0.25, 0.8, 0.7]);
    
    % Create depth selection slider
    depth_slider = uicontrol('Parent', fig, 'Style', 'slider', ...
        'Position', [100, 30, 300, 20], ...
        'Min', 1, 'Max', depth, 'Value', 1, ...
        'SliderStep', [1/(depth-1), 10/(depth-1)], ...
        'Callback', @update_display);
    
    % Create text label for depth slider
    uicontrol('Parent', fig, 'Style', 'text', ...
        'Position', [100, 55, 300, 20], ...
        'String', 'Depth (Z-axis)');
    
    % Create text to display current depth value
    depth_text = uicontrol('Parent', fig, 'Style', 'text', ...
        'Position', [410, 30, 50, 20], ...
        'String', '1');
    
    % Create X-direction spacing slider (from 50% overlap to 0% overlap)
    x_spacing_slider = uicontrol('Parent', fig, 'Style', 'slider', ...
        'Position', [500, 30, 300, 20], ...
        'Min', 0.5, 'Max', 1, 'Value', 0.75, ...
        'SliderStep', [0.05/(1-0.5), 0.1/(1-0.5)], ...
        'Callback', @update_display);
    
    % Create text label for X-direction spacing slider
    uicontrol('Parent', fig, 'Style', 'text', ...
        'Position', [500, 55, 300, 20], ...
        'String', 'X Spacing (0.5=50% overlap, 1=no overlap)');
    
    % Create text to display current X-direction spacing value
    x_spacing_text = uicontrol('Parent', fig, 'Style', 'text', ...
        'Position', [810, 30, 50, 20], ...
        'String', '0.75');
    
    % Create Y-direction spacing slider (from 50% overlap to 0% overlap)
    y_spacing_slider = uicontrol('Parent', fig, 'Style', 'slider', ...
        'Position', [500, 80, 300, 20], ...
        'Min', 0.5, 'Max', 1, 'Value', 0.75, ...
        'SliderStep', [0.05/(1-0.5), 0.1/(1-0.5)], ...
        'Callback', @update_display);
    
    % Create text label for Y-direction spacing slider
    uicontrol('Parent', fig, 'Style', 'text', ...
        'Position', [500, 105, 300, 20], ...
        'String', 'Y Spacing (0.5=50% overlap, 1=no overlap)');
    
    % Create text to display current Y-direction spacing value
    y_spacing_text = uicontrol('Parent', fig, 'Style', 'text', ...
        'Position', [810, 80, 50, 20], ...
        'String', '0.75');
    
    % Initial display
    update_display();
    
    function update_display(~, ~)
        % Get current slider values
        current_depth = round(get(depth_slider, 'Value'));
        current_x_spacing = get(x_spacing_slider, 'Value');
        current_y_spacing = get(y_spacing_slider, 'Value');
        
        % Update text displays
        set(depth_text, 'String', num2str(current_depth));
        set(x_spacing_text, 'String', num2str(current_x_spacing, '%.2f'));
        set(y_spacing_text, 'String', num2str(current_y_spacing, '%.2f'));
        
        % Calculate overlap ratios (from 0.5 to 0)
        x_overlap_ratio = 1 - current_x_spacing;  % Slider at 0.5 means 50% overlap, slider at 1 means 0% overlap
        y_overlap_ratio = 1 - current_y_spacing;  % Slider at 0.5 means 50% overlap, slider at 1 means 0% overlap
        
        % Calculate actual pixel overlap amounts
        x_overlap = round(width * x_overlap_ratio);
        y_overlap = round(height * y_overlap_ratio);
        
        % Calculate step sizes (distance between adjacent image starting positions)
        x_step = width - x_overlap;
        y_step = height - y_overlap;
        
        % Calculate amount to crop from each edge of overlapping regions
        x_crop = round(x_overlap / 2);
        y_crop = round(y_overlap / 2);
        
        % Calculate dimensions of the stitched image (now smaller due to cropping)
        effective_width = width - x_crop * 2;  % Effective width after cropping
        effective_height = height - y_crop * 2;  % Effective height after cropping
        
        % Only apply cropping if there's overlap
        if x_overlap > 0
            total_width = x_step * (num_cols - 1) + width - x_crop * 2;
        else
            total_width = x_step * (num_cols - 1) + width;
        end
        
        if y_overlap > 0
            total_height = y_step * (num_rows - 1) + height - y_crop * 2;
        else
            total_height = y_step * (num_rows - 1) + height;
        end
        
        % Create empty canvas for the stitched image
        stitched_image = zeros(ceil(total_height), ceil(total_width));
        
        % Place each subimage on the canvas after appropriate cropping
        for r = 1:num_rows
            for c = 1:num_cols
                % Extract the current depth slice from the current subimage
                subimage = recon_array{r, c}(:, :, current_depth);
                
                % Determine crop regions based on position
                crop_left = 0;
                crop_right = 0;
                crop_top = 0;
                crop_bottom = 0;
                
                % Apply cropping if there's overlap
                if x_overlap > 0
                    % Crop left side if not first column
                    if c > 1
                        crop_left = x_crop;
                    end
                    
                    % Crop right side if not last column
                    if c < num_cols
                        crop_right = x_crop;
                    end
                end
                
                if y_overlap > 0
                    % Crop top side if not first row
                    if r > 1
                        crop_top = y_crop;
                    end
                    
                    % Crop bottom side if not last row
                    if r < num_rows
                        crop_bottom = y_crop;
                    end
                end
                
                % Calculate cropped image dimensions
                cropped_width = width - crop_left - crop_right;
                cropped_height = height - crop_top - crop_bottom;
                
                % Extract the cropped region from the subimage
                cropped_subimage = subimage((1+crop_top):(height-crop_bottom), ...
                                           (1+crop_left):(width-crop_right));
                
                % Calculate position to place this cropped subimage
                if x_overlap > 0
                    if c > 1
                        x_start = 1 + (c-1) * x_step - x_crop;
                    else
                        x_start = 1 + (c-1) * x_step;
                    end
                else
                    x_start = 1 + (c-1) * x_step;
                end
                
                if y_overlap > 0
                    if r > 1
                        y_start = 1 + (r-1) * y_step - y_crop;
                    else
                        y_start = 1 + (r-1) * y_step;
                    end
                else
                    y_start = 1 + (r-1) * y_step;
                end
                
                % Ensure start positions are valid integers
                x_start = round(max(1, x_start));
                y_start = round(max(1, y_start));
                
                % Calculate end positions
                x_end = x_start + cropped_width - 1;
                y_end = y_start + cropped_height - 1;
                
                % Ensure we don't exceed canvas boundaries
                x_end = min(x_end, ceil(total_width));
                y_end = min(y_end, ceil(total_height));
                
                % Adjust image region to place based on boundary check
                img_width = x_end - x_start + 1;
                img_height = y_end - y_start + 1;
                
                % Place cropped subimage on the canvas
                stitched_image(y_start:y_end, x_start:x_end) = ...
                    cropped_subimage(1:img_height, 1:img_width);
            end
        end
        
        % Display the stitched image
        imagesc(stitched_image, 'Parent', ax);
        colormap(ax, bone);  % Use bone colormap as requested
        axis(ax, 'equal');
        axis(ax, 'off');
        title(ax, ['Depth: ', num2str(current_depth), ' / ', num2str(depth)]);
        drawnow;
    end
end