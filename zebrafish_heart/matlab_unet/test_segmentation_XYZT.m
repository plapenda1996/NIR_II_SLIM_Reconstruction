% NIR-II SLIM - zebrafish chamber segmentation (MATLAB U-Net) - produces Fig. 2b-h masks
% environment: MATLAB R2024a
% copied from D:\NIR_SLIM2\test_segmentation_XYZT.m on 2026-09-17
% Load and segment XYZT data with dual labels
% This script loads multiple z-layers and processes them together
%% Configuration
addpath('./utils/');
base_dataset_name = 'record_23092025_004521_rl';
z_range = [4, 15];  % Define the range of z-layers to load
data_folder = 'E:\Selected data\250922_mutant_fish\28';

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

%% Apply cropping (optional)
% This will apply the same crop to all layers and frames
crop_size = [100, 100];  % Define crop size
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
% load(config_file, 'crop_params');
% img_4d_cropped = img_4d(crop_params.y_start:crop_params.y_end, ...
%                        crop_params.x_start:crop_params.x_end, :, :);

%% Run dual-label segmentation on XYZT data
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
% dataset_path = fullfile(pwd, 'dataset_seg', [base_dataset_name, '_XYZT']);
% load(fullfile(dataset_path, 'segmentation_data_XYZT_dual.mat'), ...
%      'segmentation_data', 'z_range', 'crop_params');

%% GENERATE DATASETS FOR BOTH LABELS
imageSize = [128, 128, 1];  % Target size for U-Net

% Dataset for Label 1
fprintf('\n=== Generating dataset for Label 1 ===\n');
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
    'NoiseStd', 0);

% Dataset for Label 2
fprintf('\n=== Generating dataset for Label 2 ===\n');
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
    'NoiseStd', 0);

%% INITIALIZE AND TRAIN TWO U-NETS
lr = 3e-5;

% Initialize U-Net for Label 1
unetNetwork_label1 = unet(imageSize, 2, EncoderDepth=3);

% Initialize U-Net for Label 2
unetNetwork_label2 = unet(imageSize, 2, EncoderDepth=3);

% % Define combined loss function
% % function loss = diceCrossEntropyLoss(Y, T)
% %     Dice Loss part
% %     smooth = 1e-6;
% %     intersection = sum(Y .* T, [1,2]);
% %     dice_coeff = (2 * intersection + smooth) ./ (sum(Y, [1,2]) + sum(T, [1,2]) + smooth);
% %     dice_loss = 1 - mean(dice_coeff, 'all');
% %     
% %     Cross Entropy part
% %     ce_loss = crossentropy(Y, T);
% %     
% %     Combined loss (weighted)
% %     loss = 0.7 * dice_loss + 0.3 * ce_loss;
% % end

% Training options
options = trainingOptions("adam", ...
    InitialLearnRate=lr, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.5, ...
    LearnRateDropPeriod=100, ...
    MaxEpochs=80, ...
    MiniBatchSize=36, ...
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
% load(fullfile(dataset_path, 'label1', 'net_label1_XYZT.mat'));
% load(fullfile(dataset_path, 'label2', 'net_label2_XYZT.mat'));

%% 6. INFERENCE: TEST BOTH U-NETS ON ALL XYZT DATA
fprintf('\n=== Running inference on XYZT data ===\n');

% Get dimensions
[height, width, num_layers, num_frames] = size(img_4d_cropped);
inputSize = net_label1.Layers(1).InputSize;

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
        
        % Preprocess for network input
        I = imresize(double(currentFrame), inputSize(1:2));
        I = uint8(255 * rescale(I));
        
        % Segment with Label 1 network
        [C1, scores1] = semanticseg(I, net_label1, Classes = classes);
        mask1 = C1 == "object";
        
        % Morphological processing for Label 1
        se_open = strel('disk', 1);
        mask1 = imopen(mask1, se_open);
        se_close = strel('disk', 3);
        mask1 = imclose(mask1, se_close);
        mask1 = bwareaopen(mask1, 3);
        
        % Resize back to original size and store
        all_masks_label1(:, :, layer_idx, frame_idx) = imresize(mask1, [height, width], 'nearest');
        
        % Segment with Label 2 network
        [C2, scores2] = semanticseg(I, net_label2, Classes = classes);
        mask2 = C2 == "object";
        
        % Morphological processing for Label 2
        mask2 = imopen(mask2, se_open);
        mask2 = imclose(mask2, se_close);
        mask2 = bwareaopen(mask2, 3);
        
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
    
    % Apply temporal smoothing (if you have pca_temporal_smoothing function)
    if exist('pca_temporal_smoothing', 'file')
        smoothed_masks_label1(:, :, layer_idx, :) = pca_temporal_smoothing(layer_masks1, 4, 3);
        smoothed_masks_label2(:, :, layer_idx, :) = pca_temporal_smoothing(layer_masks2, 4, 3);
    else
        % Simple median filtering as alternative
        for t = 1:num_frames
            if t > 2 && t < num_frames-1
                % Median filter across 5 frames
                temp_window1 = all_masks_label1(:, :, layer_idx, t-2:t+2);
                temp_window2 = all_masks_label2(:, :, layer_idx, t-2:t+2);
                smoothed_masks_label1(:, :, layer_idx, t) = median(temp_window1, 4) > 0.5;
                smoothed_masks_label2(:, :, layer_idx, t) = median(temp_window2, 4) > 0.5;
            else
                smoothed_masks_label1(:, :, layer_idx, t) = all_masks_label1(:, :, layer_idx, t);
                smoothed_masks_label2(:, :, layer_idx, t) = all_masks_label2(:, :, layer_idx, t);
            end
        end
    end
end

% SAVE RESULTS
fprintf('\nSaving results...\n');
save(fullfile(dataset_path, 'dual_masks_XYZT.mat'), ...
     'smoothed_masks_label1', 'smoothed_masks_label2', ...
     'all_masks_label1', 'all_masks_label2', '-v7.3');  % Use -v7.3 for large files

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
