% NIR-II SLIM - zebrafish chamber segmentation (MATLAB U-Net) - produces Fig. 2b-h masks
% environment: MATLAB R2024a
% copied from D:\NIR_SLIM2\test_segmentation_XYZT_fish_251110.m on 2026-09-17
% Load and segment XYZT data with dual labels
% This script loads multiple z-layers and processes them together
%% Configuration
addpath('./utils/');
base_dataset_name = 'record_02092025_183907_rl';
z_range = [6, 13];  % Define the range of z-layers to load
data_folder = 'E:\250902_fish\6';

% Load all z-layers
fprintf('Loading z-layers from %d to %d...\n', z_range(1), z_range(2));

% First, load one layer to get dimensions
first_layer_name = sprintf('%s_%d', base_dataset_name, z_range(1));
first_im = load_tif_block(data_folder, [first_layer_name, '.tif']);
[height, width, num_frames] = size(first_im);
num_layers = z_range(2) - z_range(1) + 1;

% Initialize 4D array for all data
img_4d = zeros(height, width, num_layers, num_frames);

% Load all layers
for z_idx = z_range(1):z_range(2)
    layer_name = sprintf('%s_%d', base_dataset_name, z_idx);
    fprintf('Loading layer %d (%s)...\n', z_idx, layer_name);
    
    im_temp = load_tif_block(data_folder, [layer_name, '.tif']);
    
    % Store in 4D array
    layer_index = z_idx - z_range(1) + 1;
    img_4d(:, :, layer_index, :) = im_temp;
end

fprintf('All layers loaded! Data size: %dx%dx%dx%d (X,Y,Z,T)\n', ...
        height, width, num_layers, num_frames);

% Create dataset folder
dataset_path = fullfile(pwd, 'dataset_seg', [base_dataset_name, '_XYZT']);
if ~exist(dataset_path, 'dir')
    mkdir(dataset_path);
end

%% Manual cropping (optional)
% This will apply the same crop to all layers and frames
crop_size = [128, 128];  % Define crop size
config_file = fullfile(dataset_path, 'crop_config');

% Select middle layer and middle frame as reference for cropping
mid_layer = ceil(num_layers/2);
mid_frame = ceil(num_frames/2);
ref_img = img_4d(:, :, mid_layer, mid_frame);

% Manual crop and save crop parameters (first time)
fprintf('Select crop region on reference image (Layer %d, Frame %d)...\n', ...
        mid_layer + z_range(1) - 1, mid_frame);

% Show reference image for cropping
figure('Name', 'Select Crop Region');
imshow(ref_img, []);
title(sprintf('Reference: Layer %d, Frame %d - Select center point', ...
              mid_layer + z_range(1) - 1, mid_frame));

% Get center point for cropping
[cx, cy] = ginput(1);
close(gcf);

% Calculate crop boundaries
x_start = max(1, round(cx - crop_size(2)/2));
x_end = min(width, x_start + crop_size(2) - 1);
y_start = max(1, round(cy - crop_size(1)/2));
y_end = min(height, y_start + crop_size(1) - 1);

% Adjust if crop goes out of bounds
if x_end > width
    x_start = width - crop_size(2) + 1;
    x_end = width;
end
if y_end > height
    y_start = height - crop_size(1) + 1;
    y_end = height;
end

% Save crop parameters
crop_params = struct('x_start', x_start, 'x_end', x_end, ...
                    'y_start', y_start, 'y_end', y_end, ...
                    'center_x', cx, 'center_y', cy, ...
                    'crop_size', crop_size);
save(config_file, 'crop_params');

% Apply crop to all data
img_4d_cropped = img_4d(y_start:y_end, x_start:x_end, :, :);

fprintf('Data cropped to size: %dx%dx%dx%d\n', ...
        size(img_4d_cropped, 1), size(img_4d_cropped, 2), ...
        size(img_4d_cropped, 3), size(img_4d_cropped, 4));

%% Alternative: Load existing crop parameters
% If you've already saved crop parameters and want to reuse them:

% config_file = fullfile(dataset_path, 'crop_config');
% load(config_file, 'crop_params');

img_4d_cropped = img_4d(crop_params.y_start:crop_params.y_end, ...
                       crop_params.x_start:crop_params.x_end, :, :);

fprintf('Data cropped to size: %dx%dx%dx%d\n', ...
        size(img_4d_cropped, 1), size(img_4d_cropped, 2), ...
        size(img_4d_cropped, 3), size(img_4d_cropped, 4));

%% Run segmentation on XYZT data
fprintf('\nStarting dual-label segmentation tool for XYZT data...\n');
fprintf('Z-layer range: %d to %d (%d layers total)\n', ...
        z_range(1), z_range(2), num_layers);

segmentation_data = keyframeSegToolDualLabel_XYZT(img_4d_cropped);

%% Save segmentation data
seg_file = fullfile(dataset_path, 'segmentation_data_XYZT_dual.mat');
save(seg_file, 'segmentation_data', 'z_range', 'crop_params');

% Display summary
if isfield(segmentation_data, 'label1')
    fprintf('\n=== Segmentation Summary ===\n');
    fprintf('Total possible segments: %d (layers) x %d (frames) = %d\n', ...
            num_layers, size(img_4d_cropped, 4), ...
            num_layers * size(img_4d_cropped, 4));
    
    fprintf('\nLabel 1:\n');
    fprintf('  Total segments: %d\n', segmentation_data.label1.total_segmented);
    fprintf('  Segments per layer:\n');
    for z = 1:num_layers
        if segmentation_data.label1.num_segmented_per_layer(z) > 0
            fprintf('    Layer %d: %d segments\n', ...
                    z + z_range(1) - 1, ...
                    segmentation_data.label1.num_segmented_per_layer(z));
        end
    end
    
    fprintf('\nLabel 2:\n');
    fprintf('  Total segments: %d\n', segmentation_data.label2.total_segmented);
    fprintf('  Segments per layer:\n');
    for z = 1:num_layers
        if segmentation_data.label2.num_segmented_per_layer(z) > 0
            fprintf('    Layer %d: %d segments\n', ...
                    z + z_range(1) - 1, ...
                    segmentation_data.label2.num_segmented_per_layer(z));
        end
    end
    
    fprintf('\nData saved to: %s\n', seg_file);
end

%% LOAD SEGMENTATION DATA (if haven't)
% % Assuming you've already run the segmentation tool and have the data
dataset_path = fullfile(pwd, 'dataset_seg', [base_dataset_name, '_XYZT']);
load(fullfile(dataset_path, 'segmentation_data_XYZT_dual.mat'), ...
     'segmentation_data', 'z_range', 'crop_params');
% Display summary
if isfield(segmentation_data, 'label1')
    fprintf('\n=== Segmentation Summary ===\n');
    fprintf('Total possible segments: %d (layers) x %d (frames) = %d\n', ...
            num_layers, size(img_4d_cropped, 4), ...
            num_layers * size(img_4d_cropped, 4));
    
    fprintf('\nLabel 1:\n');
    fprintf('  Total segments: %d\n', segmentation_data.label1.total_segmented);
    fprintf('  Segments per layer:\n');
    for z = 1:num_layers
        if segmentation_data.label1.num_segmented_per_layer(z) > 0
            fprintf('    Layer %d: %d segments\n', ...
                    z + z_range(1) - 1, ...
                    segmentation_data.label1.num_segmented_per_layer(z));
        end
    end
    
    fprintf('\nLabel 2:\n');
    fprintf('  Total segments: %d\n', segmentation_data.label2.total_segmented);
    fprintf('  Segments per layer:\n');
    for z = 1:num_layers
        if segmentation_data.label2.num_segmented_per_layer(z) > 0
            fprintf('    Layer %d: %d segments\n', ...
                    z + z_range(1) - 1, ...
                    segmentation_data.label2.num_segmented_per_layer(z));
        end
    end
end

%% GENERATE DATASETS FOR BOTH LABELS
imageSize = [128, 128, 3];  % Target size for U-Net

fprintf('\n=== Generating dataset for Label 1 with depth encoding ===\n');
[dsTrain_label1, ~, ~] = buildDatasetXYZT(img_4d_cropped, segmentation_data, 1, ...
    fullfile(dataset_path, 'label1'), [base_dataset_name '_label1_XYZT'], imageSize, ...
    'EnableAugmentation', true, ...
    'AugmentationFactor', 4, ...
    'RotationRange', [-3.5, 3.5], ...
    'FlipProbability', 0, ...
    'ScaleRange', [0.8, 1.1], ...
    'TranslationRange', [-7, 7], ...
    'BrightnessRange', [-0.1, 0.1], ...
    'ContrastRange', [0.85, 1.15], ...
    'NoiseStd', 0);  % Enable depth channel

% Dataset for Label 2 with depth channel
fprintf('\n=== Generating dataset for Label 2 with depth encoding ===\n');
[dsTrain_label2, ~, ~] = buildDatasetXYZT(img_4d_cropped, segmentation_data, 2, ...
    fullfile(dataset_path, 'label2'), [base_dataset_name '_label2_XYZT'], imageSize, ...
    'EnableAugmentation', true, ...
    'AugmentationFactor', 4, ...
    'RotationRange', [-3.5, 3.5], ...
    'FlipProbability', 0, ...
    'ScaleRange', [0.8, 1.1], ...
    'TranslationRange', [-7, 7], ...
    'BrightnessRange', [-0.1, 0.1], ...
    'ContrastRange', [0.85, 1.15], ...
    'NoiseStd', 0);  % Enable depth channel

%% INITIALIZE AND TRAIN TWO U-NETS WITH 2-CHANNEL INPUT
lr = 2e-4;

% Initialize U-Net for Label 1 with 2 input channels
unetNetwork_label1 = unet(imageSize, 2, EncoderDepth=3);

% Initialize U-Net for Label 2 with 2 input channels
unetNetwork_label2 = unet(imageSize, 2, EncoderDepth=3);

% Define combined loss function
function loss = diceCrossEntropyLoss(Y, T)
    % Dice Loss part
    smooth = 1e-6;
    intersection = sum(Y .* T, [1,2]);
    dice_coeff = (2 * intersection + smooth) ./ (sum(Y, [1,2]) + sum(T, [1,2]) + smooth);
    dice_loss = 1 - mean(dice_coeff, 'all');
    
    % Cross Entropy part
    ce_loss = crossentropy(Y, T);
    
    % Combined loss (weighted)
    loss = 0.7 * dice_loss + 0.3 * ce_loss;
end

% Training options with adjusted parameters
options = trainingOptions("adam", ...
    InitialLearnRate=lr, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.5, ...
    LearnRateDropPeriod=100, ...
    MaxEpochs=75, ...  % Increased epochs
    MiniBatchSize=32, ...  % Slightly reduced batch size
    Metrics = ["accuracy","fscore"], ...
    Verbose=false, ...
    Plots="training-progress");

% Train U-Net for Label 1
fprintf('\n=== Training U-Net for Label 1 ===\n');
reset(dsTrain_label1);
net_label1 = trainnet(dsTrain_label1, unetNetwork_label1, @diceCrossEntropyLoss, options);
save(fullfile(dataset_path, 'label1', 'net_label1_XYZT.mat'), 'net_label1');

% Train U-Net for Label 2
fprintf('\n=== Training U-Net for Label 2 ===\n');
reset(dsTrain_label2);
net_label2 = trainnet(dsTrain_label2, unetNetwork_label2, @diceCrossEntropyLoss, options);
save(fullfile(dataset_path, 'label2', 'net_label2_XYZT.mat'), 'net_label2');

%% 5. LOAD EXISTING TRAINED NETWORKS (if already trained)
load(fullfile(dataset_path, 'label1', 'net_label1_XYZT.mat'));
load(fullfile(dataset_path, 'label2', 'net_label2_XYZT.mat'));

%% 6. INFERENCE WITH DEPTH ENCODING AND BETTER POST-PROCESSING
fprintf('\n=== Running inference on XYZT data ===\n');

% Get dimensions
[height, width, num_layers, num_frames] = size(img_4d_cropped);
% *** MODIFICATION START: Get network input size, which is now [H, W, 3] ***
networkInputSize = net_label1.Layers(1).InputSize(1:2); % e.g., [128, 128]

% Initialize 4D mask arrays for both labels
all_masks_label1 = false(height, width, num_layers, num_frames);
all_masks_label2 = false(height, width, num_layers, num_frames);

% Define classes for semantic segmentation
classes = categorical(["object", "background"]);

% Process each frame in each layer
total_frames = num_layers * num_frames;
frame_counter = 0;

fprintf('Processing %d layers x %d frames = %d total frames\n', ...
        num_layers, num_frames, total_frames);

for layer_idx = 1:num_layers
    fprintf('\nProcessing layer %d/%d...\n', layer_idx, num_layers);

    for frame_idx = 1:num_frames
        frame_counter = frame_counter + 1;
        
        if mod(frame_counter, 50) == 0 || frame_counter == total_frames
            fprintf('  Overall progress: %d/%d frames (%.1f%%)\n', ...
                    frame_counter, total_frames, 100*frame_counter/total_frames);
        end
        
        % Extract current frame
        currentFrame = img_4d_cropped(:, :, layer_idx, frame_idx);
        
        % Preprocess for 3-channel network input ***
        % Channel 1: Image data
        img_channel = imresize(rescale(double(currentFrame)), networkInputSize);
        % Channel 2: Depth data
        if num_layers > 1
            z_normalized = (layer_idx - 1) / (num_layers - 1);
        else
            z_normalized = 0;
        end
        z_channel = ones(networkInputSize) * z_normalized;
        % Combine into a 3-channel image for the network
        I = cat(3, img_channel, z_channel, zeros(networkInputSize));
        I = uint8(255 * I);

        % Segment with Label 1 network
        [C1, ~] = semanticseg(I, net_label1, Classes = classes);
        mask1 = C1 == "object";
        
        % Morphological processing for Label 1
        se_open = strel('disk', 3);
        mask1 = imopen(mask1, se_open);
        nexttile; imagesc(mask1);
        se_close = strel('disk', 3);
        mask1 = imclose(mask1, se_close);
        mask1 = bwareaopen(mask1, 3);
        % Keep only the largest connected component
        CC1 = bwconncomp(mask1);
        if CC1.NumObjects > 0
            numPixels = cellfun(@numel, CC1.PixelIdxList);
            [~, idx] = max(numPixels);
            mask1 = false(size(mask1));
            mask1(CC1.PixelIdxList{idx}) = true;
        end
        % Resize back to original size and store
        all_masks_label1(:, :, layer_idx, frame_idx) = imresize(mask1, [height, width], 'nearest');
        

        % Segment with Label 2 network
        [C2, ~] = semanticseg(I, net_label2, Classes = classes);
        mask2 = C2 == "object";
        % Morphological processing for Label 2
        mask2 = imopen(mask2, se_open);
        mask2 = imclose(mask2, se_close);
        mask2 = bwareaopen(mask2, 3);
        % Keep only the largest connected component ***
        CC2 = bwconncomp(mask2);
        if CC2.NumObjects > 0
            numPixels = cellfun(@numel, CC2.PixelIdxList);
            [~, idx] = max(numPixels);
            mask2 = false(size(mask2));
            mask2(CC2.PixelIdxList{idx}) = true;
        end
        % Resize back to original size and store
        all_masks_label2(:, :, layer_idx, frame_idx) = imresize(mask2, [height, width], 'nearest');
    end
end

fprintf('\nSegmentation complete for all layers and frames!\n');

% APPLY 3D TEMPORAL-SPATIAL SMOOTHING and save video
fprintf('\nApplying 3D temporal-spatial smoothing...\n');

% Apply smoothing per layer (temporal smoothing)
smoothed_masks_label1 = zeros(size(all_masks_label1));
smoothed_masks_label2 = zeros(size(all_masks_label2));

for layer_idx = 1:num_layers
    % Extract layer data
    layer_masks1 = squeeze(all_masks_label1(:, :, layer_idx, :));
    layer_masks2 = squeeze(all_masks_label2(:, :, layer_idx, :));
    
    smoothed_masks_label1(:, :, layer_idx, :) = pca_temporal_smoothing(layer_masks1, 4, 3);
    smoothed_masks_label2(:, :, layer_idx, :) = pca_temporal_smoothing(layer_masks2, 4, 3);
end

% SAVE RESULTS
fprintf('\nSaving results...\n');
save(fullfile(dataset_path, 'dual_masks_XYZT.mat'), ...
     'smoothed_masks_label1', 'smoothed_masks_label2', ...
     'all_masks_label1', 'all_masks_label2', '-v7.3');  % Use -v7.3 for large files

%% VISUALIZATION: use dashed lines for selected frames
target_resolution = [400, 400];
current_layer = 5;
frame_range = 200:4:400;
fps = 600;
figure
for i = 1:length(frame_range)
    frame_idx = frame_range(i);
    % Get current frame and masks for this layer
    img = img_4d_cropped(:, :, current_layer, frame_idx);
    mask1 = smoothed_masks_label1(:, :, current_layer, frame_idx);
    mask2 = smoothed_masks_label2(:, :, current_layer, frame_idx);
    % Resize images to target resolution
    img_resized = imresize(img, target_resolution, 'bicubic');
    mask1_resized = imresize(mask1, target_resolution, 'nearest'); % Use nearest to preserve mask edges
    mask2_resized = imresize(mask2, target_resolution, 'nearest');

    imshow(img_resized, []);  % Display resized image
    title(sprintf('Time: %.0f/%.0f ms', i/fps*1000, length(frame_range)/fps*1000), ...
        'FontSize', 14, 'FontWeight', 'bold');
    hold on

    % add an overlay of mask1_resized and mask2_resized as dashed red and
    % blue lines, respectively
    [B1, ~] = bwboundaries(mask1_resized, 'noholes');
    for k = 1:length(B1)
        boundary = B1{k};
        plot(boundary(:,2), boundary(:,1), 'r--', 'LineWidth', 2.5);
    end

    [B2, ~] = bwboundaries(mask2_resized, 'noholes');
    for k = 1:length(B2)
        boundary = B2{k};
        plot(boundary(:,2), boundary(:,1), 'b--', 'LineWidth', 2.5);
    end

    % add a scale bar with annotation in um
    % original (before resize), every pixel is 5 um
    original_pixel_size_um = 5;
    scale_factor = target_resolution(2) / size(img, 2);
    resized_pixel_size_um = original_pixel_size_um / scale_factor;

    % Create a 100 μm scale bar
    scalebar_length_um = 100;
    scalebar_length_pixels = scalebar_length_um / resized_pixel_size_um;

    % Position in bottom right corner
    scalebar_x = target_resolution(2) - scalebar_length_pixels - 20;
    scalebar_y = target_resolution(1) - 20;

    % Draw scale bar
    plot([scalebar_x, scalebar_x + scalebar_length_pixels], [scalebar_y, scalebar_y], ...
        'r-', 'LineWidth', 3);
    text(scalebar_x + scalebar_length_pixels/2, scalebar_y - 20, ...
        sprintf('%d μm', scalebar_length_um), ...
        'Color', 'red', 'FontSize', 10, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center');

    hold off
    drawnow
    pause(0.1)
end


%% VISUALIZATION: CREATE HIGH-RESOLUTION VIDEOS FOR ALL LAYERS AND 4D TIFF
fprintf('\nCreating high-resolution visualization for all layers...\n');

% === PARAMETERS ===
target_resolution = [400, 400]; % Target resolution (adjustable, e.g., [1024, 1024])
frame_rate = 60;
compression_quality = 85; % FFmpeg compression quality (0-100)

% === 1. CREATE HIGH-RESOLUTION VIDEOS FOR ALL LAYERS ===
fprintf('Creating videos for %d layers...\n', num_layers);

% Create figure with target resolution (create once, reuse for all layers)
figure('Position', [100, 100, target_resolution(1)+50, target_resolution(2)+100], 'Visible', 'off');
ax = axes('Position', [0.05, 0.1, 0.9, 0.85]); % Adjust axes position

% Create video for each layer
for layer_idx = 1:num_layers
    current_layer = layer_idx;
    layer_z_pos = current_layer + z_range(1) - 1; % Real Z position
    
    % File names
    output_avi = fullfile(dataset_path, sprintf('segmentation_layer%d_hires.avi', layer_z_pos));
    output_mp4 = fullfile(dataset_path, sprintf('segmentation_layer%d_compressed.mp4', layer_z_pos));
    
    fprintf('\nProcessing Layer %d/%d (Z=%d)...\n', layer_idx, num_layers, layer_z_pos);
    
    % Create high-resolution video writer for this layer
    v = VideoWriter(output_avi, 'Motion JPEG AVI'); % Use high quality encoding
    v.Quality = 95; % High quality
    v.FrameRate = frame_rate;
    open(v);
    
    fprintf('Processing %d frames at %dx%d resolution...\n', num_frames, target_resolution(1), target_resolution(2));
    
    for frame_idx = 1:num_frames
        % Get current frame and masks for this layer
        img = img_4d_cropped(:, :, current_layer, frame_idx);
        mask1 = smoothed_masks_label1(:, :, current_layer, frame_idx);
        mask2 = smoothed_masks_label2(:, :, current_layer, frame_idx);
        
        % Resize images to target resolution
        img_resized = imresize(img, target_resolution, 'bicubic');
        mask1_resized = imresize(mask1, target_resolution, 'nearest'); % Use nearest to preserve mask edges
        mask2_resized = imresize(mask2, target_resolution, 'nearest');
        
        % Create RGB overlay
        img_rgb = repmat(rescale(img_resized), [1, 1, 3]);
        
        % Overlay Label 1 (red channel)
        img_rgb(:, :, 1) = img_rgb(:, :, 1) + 0.3 * mask1_resized;
        
        % Overlay Label 2 (blue channel) 
        img_rgb(:, :, 3) = img_rgb(:, :, 3) + 0.3 * mask2_resized;
        
        % Ensure values are in [0, 1]
        img_rgb = min(1, img_rgb);
        
        % Display in axes
        imshow(img_rgb, 'Parent', ax);
        title(ax, sprintf('Layer %d, Frame %d/%d', layer_z_pos, frame_idx, num_frames), ...
              'FontSize', 14, 'FontWeight', 'bold');
        
        % Capture frame at exact target resolution
        frame = getframe(ax);
        
        % Ensure consistent frame size
        if size(frame.cdata, 1) ~= target_resolution(2) || size(frame.cdata, 2) ~= target_resolution(1)
            frame.cdata = imresize(frame.cdata, [target_resolution(2), target_resolution(1)]);
        end
        
        writeVideo(v, frame);
        
        % Show progress
        if mod(frame_idx, 20) == 0
            fprintf('  Layer %d: Processed frame %d/%d\n', layer_idx, frame_idx, num_frames);
        end
    end
    
    close(v);
    fprintf('Layer %d AVI saved to: %s\n', layer_idx, output_avi);
    
    % === COMPRESS THIS LAYER WITH FFMPEG TO MP4 ===
    fprintf('Compressing Layer %d with FFmpeg...\n', layer_idx);
    ffmpeg_cmd = sprintf(['ffmpeg -i "%s" -c:v libx264 -crf %d -preset medium ' ...
                         '-pix_fmt yuv420p -y "%s"'], ...
                         output_avi, 51-compression_quality*0.51, output_mp4);
    
    try
        [status, result] = system(ffmpeg_cmd);
        if status == 0
            fprintf('Layer %d MP4 saved: %s\n', layer_idx, output_mp4);
            % Delete AVI file to save space
            delete(output_avi);
        else
            fprintf('FFmpeg compression failed for layer %d. Error: %s\n', layer_idx, result);
        end
    catch ME
        fprintf('FFmpeg error for layer %d: %s\n', layer_idx, ME.message);
    end
end

close(gcf);
fprintf('\nAll layer videos completed!\n');

% === 2. CREATE RGB 4D TIFF FOR IMAGEJ (NO COMPRESSION) ===
fprintf('\nCreating RGB 4D TIFF for ImageJ...\n');
output_rgb_tiff = fullfile(dataset_path, 'segmentation_4D_RGB.tif');

fprintf('Preparing RGB 4D overlay data (%d layers x %d frames)...\n', num_layers, num_frames);

for z = 1:num_layers
    for t = 1:num_frames
        % Create RGB version
        img = img_4d_cropped(:, :, z, t);
        mask1 = smoothed_masks_label1(:, :, z, t);
        mask2 = smoothed_masks_label2(:, :, z, t);
        
        % RGB overlay
        img_rgb = repmat(rescale(img), [1, 1, 3]);
        img_rgb(:, :, 1) = img_rgb(:, :, 1) + 0.3 * mask1; % Red channel
        img_rgb(:, :, 3) = img_rgb(:, :, 3) + 0.3 * mask2; % Blue channel
        img_rgb = min(1, img_rgb);
        img_rgb_uint8 = uint8(img_rgb * 255);
        
        if z == 1 && t == 1
            imwrite(img_rgb_uint8, output_rgb_tiff, 'WriteMode', 'overwrite');
        else
            imwrite(img_rgb_uint8, output_rgb_tiff, 'WriteMode', 'append');
        end
        
        % Show progress
        if mod((z-1)*num_frames + t, 50) == 0
            fprintf('  Processed slice %d/%d\n', (z-1)*num_frames + t, num_layers*num_frames);
        end
    end
end

fprintf('RGB 4D TIFF saved to: %s\n', output_rgb_tiff);
fprintf('ImageJ import settings: %d x %d x %d x %d (width x height x slices x frames)\n', ...
        size(img_4d_cropped, 2), size(img_4d_cropped, 1), num_layers, num_frames);

fprintf('\n=== SUMMARY ===\n');
fprintf('Files created:\n');

% Count MP4 files
mp4_files = dir(fullfile(dataset_path, '*compressed.mp4'));
fprintf('- MP4 videos: %d files\n', length(mp4_files));
for i = 1:length(mp4_files)
    fprintf('  • %s\n', mp4_files(i).name);
end

fprintf('- RGB 4D TIFF: %s\n', output_rgb_tiff);

% Display file size information
all_files = [dir(fullfile(dataset_path, '*compressed.mp4')); dir(fullfile(dataset_path, '*4D_RGB.tif'))];
fprintf('\nFile sizes:\n');
total_size = 0;
for i = 1:length(all_files)
    file_size_mb = all_files(i).bytes/1024/1024;
    fprintf('- %s: %.1f MB\n', all_files(i).name, file_size_mb);
    total_size = total_size + file_size_mb;
end
fprintf('Total size: %.1f MB\n', total_size);

%% VENTRICULAR AREA ANALYSIS - TIME SERIES FOR ALL LAYERS
fprintf('\n=== VENTRICULAR AREA ANALYSIS ===\n');

% === Z-AXIS PARAMETERS (in micrometers) ===
z_step_um = 30; % 0.03 mm = 30 µm
z_coords_um = (0:(num_layers-1)) * z_step_um;

% === OTHER PARAMETERS ===
actual_frame_rate = 600;

% Physical parameters
ratio = target_resolution(1)/size(img,1);
pixel_to_um_ratio = (2.4e-3 * 1000) / ratio; % Convert mm ratio to µm ratio
area_unit = 'µm²'; % Updated area unit

% Color schemes for the two ventricles
color_ventricle1 = [1, 0, 0]; % Red for Label 1
color_ventricle2 = [0, 0, 1]; % Blue for Label 2

% === CALCULATE AREAS FOR ALL LAYERS AND FRAMES ===
fprintf('Calculating areas for %d layers and %d frames...\n', num_layers, num_frames);
% Initialize area arrays
area_label1 = zeros(num_layers, num_frames);
area_label2 = zeros(num_layers, num_frames);

% Calculate areas for each layer and frame
for layer_idx = 1:num_layers
    for frame_idx = 1:num_frames
        mask1 = smoothed_masks_label1(:, :, layer_idx, frame_idx);
        mask2 = smoothed_masks_label2(:, :, layer_idx, frame_idx);
        
        area_label1(layer_idx, frame_idx) = sum(mask1(:)); 
        area_label2(layer_idx, frame_idx) = sum(mask2(:));
    end
    
    if mod(layer_idx, 10) == 0
        fprintf('  Processed layer %d/%d\n', layer_idx, num_layers);
    end
end

% Convert pixel areas to physical units (µm²)
area_label1 = area_label1 * (pixel_to_um_ratio^2);
area_label2 = area_label2 * (pixel_to_um_ratio^2);

% === CREATE TIME AXIS ===
time_per_frame = 1 / actual_frame_rate;
time_axis = (0:num_frames-1) * time_per_frame;

% === VISUALIZATION ===
fprintf('\nCreating multi-layer area visualization...\n');
fig = figure('Position', [100, 50, 1400, 900], 'Name', 'Ventricular Area Analysis');

% === SUBPLOT 1: Ventricle 1 (Label 1) ===
subplot(2, 2, 1);
hold on;
alpha_values = linspace(0.3, 1, num_layers);
for layer_idx = 1:num_layers
    layer_z_pos_um = z_coords_um(layer_idx); 
    layer_color = color_ventricle1 * alpha_values(layer_idx);
    
    plot(time_axis, area_label1(layer_idx, :), ...
         'Color', layer_color, ...
         'LineWidth', 1.5, ...
         'DisplayName', sprintf('Layer Z=%.0f µm', layer_z_pos_um));
end
xlabel('Time (seconds)', 'FontSize', 12);
ylabel(sprintf('Area (%s)', area_unit), 'FontSize', 12);
title('A (Red Label) - Area Over Time', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
legend('Location', 'eastoutside', 'FontSize', 8);
xlim([0, max(time_axis)]);

% === SUBPLOT 2: Ventricle 2 (Label 2) ===
subplot(2, 2, 2);
hold on;
for layer_idx = 1:num_layers
    layer_z_pos_um = z_coords_um(layer_idx);
    layer_color = color_ventricle2 * alpha_values(layer_idx);
    
    plot(time_axis, area_label2(layer_idx, :), ...
         'Color', layer_color, ...
         'LineWidth', 1.5, ...
         'DisplayName', sprintf('Layer Z=%.0f µm', layer_z_pos_um));
end
xlabel('Time (seconds)', 'FontSize', 12);
ylabel(sprintf('Area (%s)', area_unit), 'FontSize', 12);
title('V (Blue Label) - Area Over Time', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
legend('Location', 'eastoutside', 'FontSize', 8);
xlim([0, max(time_axis)]);

% === SUBPLOT 3: Combined View ===
subplot(2, 2, [3, 4]);
hold on;
for layer_idx = 1:num_layers
    layer_z_pos_um = z_coords_um(layer_idx); 
    
    % Ventricle 1 (Red shades)
    layer_color1 = color_ventricle1 * alpha_values(layer_idx);
    plot(time_axis, area_label1(layer_idx, :), ...
              'Color', layer_color1, ...
              'LineWidth', 2, ...
              'LineStyle', '-', ...
              'DisplayName', sprintf('A-Layer Z=%.0f µm', layer_z_pos_um));
    
    % Ventricle 2 (Blue shades)  
    layer_color2 = color_ventricle2 * alpha_values(layer_idx);
    plot(time_axis, area_label2(layer_idx, :), ...
              'Color', layer_color2, ...
              'LineWidth', 2, ...
              'LineStyle', '--', ...
              'DisplayName', sprintf('V-Layer Z=%.0f µm', layer_z_pos_um));
end
xlabel('Time (seconds)', 'FontSize', 12);
ylabel(sprintf('Area (%s)', area_unit), 'FontSize', 12);
title('Both - Area Over Time (All Layers)', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
legend('Location', 'eastoutside', 'FontSize', 8, 'NumColumns', 2);
xlim([0, max(time_axis)]);

% Annotation
annotation('textbox', [0.02, 0.02, 0.3, 0.05], ...
           'String', sprintf('Frame Rate: %.1f Hz | Total Duration: %.2f s | Z-Step: %.0f µm', ...
                            actual_frame_rate, max(time_axis), z_step_um), ...
           'FontSize', 10, 'EdgeColor', 'none');

% ---
% === CREATE HEATMAP OF AREA CHANGES ===
figure('Position', [100, 100, 1200, 500], 'Name', 'Area Heatmaps');
z_tick_labels = arrayfun(@(x) sprintf('Z=%.0f µm', x), z_coords_um, 'UniformOutput', false);

% Heatmap for Ventricle 1
subplot(1, 2, 1);
imagesc(time_axis, 1:num_layers, area_label1);
colorbar;
xlabel('Time (seconds)', 'FontSize', 12);
ylabel(sprintf('Z Position (%s)', 'µm'), 'FontSize', 12); 
title('A - Area Heatmap', 'FontSize', 14, 'FontWeight', 'bold');
colormap(subplot(1, 2, 1), hot);
yticks(1:num_layers);
yticklabels(z_tick_labels); 

% Heatmap for Ventricle 2
subplot(1, 2, 2);
imagesc(time_axis, 1:num_layers, area_label2);
colorbar;
xlabel('Time (seconds)', 'FontSize', 12);
ylabel(sprintf('Z Position (%s)', 'µm'), 'FontSize', 12); 
title('V - Area Heatmap', 'FontSize', 14, 'FontWeight', 'bold');
colormap(subplot(1, 2, 2), hot);
yticks(1:num_layers);
yticklabels(z_tick_labels);

%% === STATISTICAL ANALYSIS ===
fprintf('\n=== AREA STATISTICS ===\n');

% Calculate statistics for each ventricle and layer
for ventricle_idx = 1:2
    if ventricle_idx == 1
        area_data = area_label1;
        ventricle_name = 'Ventricle 1 (Red)';
    else
        area_data = area_label2;
        ventricle_name = 'Ventricle 2 (Blue)';
    end
    
    fprintf('\n%s:\n', ventricle_name);
    fprintf('%-10s %-15s %-15s %-15s %-15s %-15s\n', ...
            'Layer', 'Mean Area', 'Std Dev', 'Min Area', 'Max Area', 'Range');
    fprintf('%-10s %-15s %-15s %-15s %-15s %-15s\n', ...
            '-----', '---------', '-------', '--------', '--------', '-----');
    
    for layer_idx = 1:num_layers
        layer_z_pos = layer_idx + z_range(1) - 1;
        layer_areas = area_data(layer_idx, :);
        
        mean_area = mean(layer_areas);
        std_area = std(layer_areas);
        min_area = min(layer_areas);
        max_area = max(layer_areas);
        range_area = max_area - min_area;
        
        fprintf('Z=%-8d %-15.2f %-15.2f %-15.2f %-15.2f %-15.2f\n', ...
                layer_z_pos, mean_area, std_area, min_area, max_area, range_area);
    end
end

% === PULSATION ANALYSIS ===
fprintf('\n=== PULSATION ANALYSIS ===\n');

% Calculate pulsation metrics
for ventricle_idx = 1:2
    if ventricle_idx == 1
        area_data = area_label1;
        ventricle_name = 'Ventricle 1';
    else
        area_data = area_label2;
        ventricle_name = 'Ventricle 2';
    end
    
    fprintf('\n%s Pulsation Metrics:\n', ventricle_name);
    
    for layer_idx = 1:num_layers
        layer_z_pos = layer_idx + z_range(1) - 1;
        layer_areas = area_data(layer_idx, :);
        
        % Calculate ejection fraction surrogate (simplified)
        max_area = max(layer_areas);
        min_area = min(layer_areas);
        if max_area > 0
            ejection_fraction = (max_area - min_area) / max_area * 100;
        else
            ejection_fraction = 0;
        end
        
        % Find peaks (systole) and valleys (diastole)
        [peaks, peak_locs] = findpeaks(layer_areas, 'MinPeakDistance', 5);
        [valleys, valley_locs] = findpeaks(-layer_areas, 'MinPeakDistance', 5);
        valleys = -valleys;
        
        if ~isempty(peak_locs) && length(peak_locs) > 1
            % Calculate heart rate from peak intervals
            peak_intervals = diff(peak_locs) * time_per_frame; % in seconds
            heart_rate = 60 / mean(peak_intervals); % beats per minute
            
            fprintf('  Layer Z=%d: EF≈%.1f%%, HR≈%.1f bpm, Peaks=%d\n', ...
                    layer_z_pos, ejection_fraction, heart_rate, length(peaks));
        else
            fprintf('  Layer Z=%d: EF≈%.1f%%, Insufficient peaks for HR\n', ...
                    layer_z_pos, ejection_fraction);
        end
    end
end




%% GUI VERSION: Interactive frame viewer with slider and save button
create_interactive_viewer(img_4d_cropped,smoothed_masks_label1,smoothed_masks_label2)
function create_interactive_viewer(img_4d_cropped,smoothed_masks_label1,smoothed_masks_label2)
    % Get frame range from data
    total_frames = size(img_4d_cropped, 4);
    
    % Parameters
    target_resolution = [400, 400];
    current_layer = 5;
    fps = 600;
    original_pixel_size_um = 5;
    
    % Create figure with controls
    fig = figure('Name', 'Interactive Frame Viewer', ...
                 'Position', [100, 100, 500, 550], ...
                 'NumberTitle', 'off');
    
    % Create axes for image display
    ax = axes('Parent', fig, ...
              'Position', [0.1, 0.25, 0.8, 0.7]);
    
    % Create slider
    slider = uicontrol('Style', 'slider', ...
                      'Parent', fig, ...
                      'Min', 1, 'Max', total_frames, ...
                      'Value', 1, ...
                      'SliderStep', [1/(total_frames-1), 10/(total_frames-1)], ...
                      'Position', [50, 80, 400, 20]);
    
    % Add listener for continuous value change (real-time response)
    addlistener(slider, 'ContinuousValueChange', @(src, evt) update_display());
    
    % Create frame label
    frame_label = uicontrol('Style', 'text', ...
                           'Parent', fig, ...
                           'Position', [50, 105, 400, 20], ...
                           'String', sprintf('Frame: 1 / %d', total_frames), ...
                           'FontSize', 10);
    
    % Create save button
    save_btn = uicontrol('Style', 'pushbutton', ...
                        'Parent', fig, ...
                        'String', 'Save Figure', ...
                        'Position', [175, 30, 150, 30], ...
                        'FontSize', 11, ...
                        'Callback', @save_figure);
    
    % Initial display
    update_display();
    
    % Nested function to update display
    function update_display()
        frame_idx = round(get(slider, 'Value'));
        
        % Get current frame and masks for this layer
        img = img_4d_cropped(:, :, current_layer, frame_idx);
        mask1 = smoothed_masks_label1(:, :, current_layer, frame_idx);
        mask2 = smoothed_masks_label2(:, :, current_layer, frame_idx);
        
        % Resize images to target resolution
        img_resized = imresize(img, target_resolution, 'bicubic');
        mask1_resized = imresize(mask1, target_resolution, 'nearest');
        mask2_resized = imresize(mask2, target_resolution, 'nearest');
        
        % Clear axes and display image
        cla(ax);
        imshow(img_resized, [], 'Parent', ax);
        time_ms = (frame_idx / fps) * 1000;
        title(ax, sprintf('Time: %.1f ms', time_ms), ...
              'FontSize', 14, 'FontWeight', 'bold');
        hold(ax, 'on');
        
        % Add mask1 overlay (red dashed)
        [B1, ~] = bwboundaries(mask1_resized, 'noholes');
        for k = 1:length(B1)
            boundary = B1{k};
            plot(ax, boundary(:,2), boundary(:,1), 'r--', 'LineWidth', 2.5);
        end
        
        % Add mask2 overlay (blue dashed)
        [B2, ~] = bwboundaries(mask2_resized, 'noholes');
        for k = 1:length(B2)
            boundary = B2{k};
            plot(ax, boundary(:,2), boundary(:,1), 'b--', 'LineWidth', 2.5);
        end
        
        % Add scale bar
        scale_factor = target_resolution(2) / size(img, 2);
        resized_pixel_size_um = original_pixel_size_um / scale_factor;
        
        scalebar_length_um = 100;
        scalebar_length_pixels = scalebar_length_um / resized_pixel_size_um;
        
        scalebar_x = target_resolution(2) - scalebar_length_pixels - 20;
        scalebar_y = target_resolution(1) - 20;
        
        plot(ax, [scalebar_x, scalebar_x + scalebar_length_pixels], ...
             [scalebar_y, scalebar_y], 'r-', 'LineWidth', 3);
        text(ax, scalebar_x + scalebar_length_pixels/2, scalebar_y - 30, ...
             sprintf('%d μm', scalebar_length_um), ...
             'Color', 'red', 'FontSize', 18, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'center');
        
        hold(ax, 'off');
        
        % Update frame label
        set(frame_label, 'String', sprintf('Frame: %d / %d', frame_idx, total_frames));
        
        drawnow;  % Force immediate update
    end
    
    % Nested function to save figure (without controls)
    function save_figure(~, ~)
        % Create a new figure for saving (without controls)
        temp_fig = figure('Visible', 'off', ...
                         'Position', [0, 0, target_resolution(2), target_resolution(1)]);
        temp_ax = axes('Parent', temp_fig, 'Position', [0, 0, 1, 1]);
        
        % Copy current axes content to temp figure
        frame_idx = round(get(slider, 'Value'));
        
        % Recreate the image in temp figure
        img = img_4d_cropped(:, :, current_layer, frame_idx);
        mask1 = smoothed_masks_label1(:, :, current_layer, frame_idx);
        mask2 = smoothed_masks_label2(:, :, current_layer, frame_idx);
        
        img_resized = imresize(img, target_resolution, 'bicubic');
        mask1_resized = imresize(mask1, target_resolution, 'nearest');
        mask2_resized = imresize(mask2, target_resolution, 'nearest');
        
        imshow(img_resized, [], 'Parent', temp_ax);
        hold(temp_ax, 'on');
        
        % Add mask overlays
        [B1, ~] = bwboundaries(mask1_resized, 'noholes');
        for k = 1:length(B1)
            boundary = B1{k};
            plot(temp_ax, boundary(:,2), boundary(:,1), 'r--', 'LineWidth', 2.5);
        end
        
        [B2, ~] = bwboundaries(mask2_resized, 'noholes');
        for k = 1:length(B2)
            boundary = B2{k};
            plot(temp_ax, boundary(:,2), boundary(:,1), 'b--', 'LineWidth', 2.5);
        end
        
        % Add scale bar
        scale_factor = target_resolution(2) / size(img, 2);
        resized_pixel_size_um = original_pixel_size_um / scale_factor;
        scalebar_length_um = 100;
        scalebar_length_pixels = scalebar_length_um / resized_pixel_size_um;
        scalebar_x = target_resolution(2) - scalebar_length_pixels - 20;
        scalebar_y = target_resolution(1) - 20;
        
        plot(temp_ax, [scalebar_x, scalebar_x + scalebar_length_pixels], ...
             [scalebar_y, scalebar_y], 'r-', 'LineWidth', 3);
        text(temp_ax, scalebar_x + scalebar_length_pixels/2, scalebar_y - 30, ...
             sprintf('%d μm', scalebar_length_um), ...
             'Color', 'red', 'FontSize', 20, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'center');
        
        hold(temp_ax, 'off');
        axis(temp_ax, 'off');
        
        % Save file
        [filename, pathname] = uiputfile({'*.png'; '*.jpg'; '*.tif'; '*.fig'}, ...
                                        'Save Figure', ...
                                        sprintf('frame_%04d.png', frame_idx));
        if filename ~= 0
            full_path = fullfile(pathname, filename);
            exportgraphics(temp_ax, full_path, 'Resolution', 300);
            msgbox(sprintf('Figure saved to:\n%s', full_path), 'Save Successful');
        end
        
        close(temp_fig);
    end
end