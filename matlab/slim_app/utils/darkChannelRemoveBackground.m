%% 修改
function output_image = darkChannelRemoveBackground(input_image, method, options)
% DARKCHANNELREMOVEBACKGROUND Removes background from microscopy images using various methods
%
% Syntax:
%   output_image = darkChannelRemoveBackground(input_image)
%   output_image = darkChannelRemoveBackground(input_image, method)
%   output_image = darkChannelRemoveBackground(input_image, method, options)
%
% Input:
%   input_image - 2D or 3D image array (required)
%   method      - Background removal method: 'Dark', 'Tophat', or 'Rolling ball' (default: 'Dark')
%   options     - Structure with fields for parameters (optional)
%
% Optional Structure Fields:
%   strel_size     - Structuring element size for Tophat and Rolling ball methods (default: 15)
%   BackgroundMode - Background removal strictness for Dark method: 0-medium, 1-strict (default: 1)
%   Padding        - Edge padding mode for Dark method: 0-zeros, 1-symmetric (default: 1)
%   Denoise        - Apply Gaussian denoising for Dark method: 0-no, 1-yes (default: 0)
%   Threshold      - Low frequency dehazing threshold for Dark method (default: 70)
%   DivideCoeff    - Frequency division coefficient for Dark method (default: 0.5)
%   NA             - Numerical aperture for Dark method (default: 0.06)
%   Wavelength     - Emission wavelength in nm for Dark method (default: 1100)
%   PixelSizeX     - Pixel size in X direction in nm for Dark method (default: 6000)
%   PixelSizeY     - Pixel size in Y direction in nm for Dark method (default: 6000)
%
% Output:
%   output_image - Background-removed image with same dimensions as input
%
% Example:
%   img_out = darkChannelRemoveBackground(img_in);
%   img_out = darkChannelRemoveBackground(img_in, 'Tophat');
%   opts = struct('strel_size', 20);
%   img_out = darkChannelRemoveBackground(img_in, 'Tophat', opts);
%   opts = struct('strel_size', 50);
%   img_out = darkChannelRemoveBackground(img_in, 'Rolling ball', opts);
%   % Example with different x/y pixel sizes
%   opts = struct('PixelSizeX', 5000, 'PixelSizeY', 7000);
%   img_out = darkChannelRemoveBackground(img_in, 'Dark', opts);
%
% Based on: "Single Image Haze Removal Using Dark Channel Prior" by Kaiming He, et al.

% Handle input arguments
if nargin < 2 || isempty(method)
    method = 'dark';
end

% Set default options
defaultOptions = struct(...
    'strel_size', 15, ...
    'BackgroundMode', 1, ...
    'Padding', 1, ...
    'Denoise', 0, ...
    'Threshold', 70, ...
    'DivideCoeff', 0.5, ...
    'NA', 0.06, ...
    'Wavelength', 1100, ...
    'PixelSizeX', 6000, ...
    'PixelSizeY', 6000);

% If options not provided, use defaults
if nargin < 3 || isempty(options)
    options = defaultOptions;
else
    % Merge with defaults for any missing fields
    optionFields = fieldnames(defaultOptions);
    for i = 1:length(optionFields)
        if ~isfield(options, optionFields{i})
            options.(optionFields{i}) = defaultOptions.(optionFields{i});
        end
    end
    
    % For backward compatibility: if PixelSize exists but not PixelSizeX/Y
    if isfield(options, 'PixelSize') && ~isfield(options, 'PixelSizeX')
        if length(options.PixelSize) == 1
            % Single value provided
            options.PixelSizeX = options.PixelSize;
            options.PixelSizeY = options.PixelSize;
        elseif length(options.PixelSize) >= 2
            % Array provided, use first two elements
            options.PixelSizeX = options.PixelSize(1);
            options.PixelSizeY = options.PixelSize(2);
        end
    end
end

% Get original dimensions
original_image = double(input_image);
if ndims(original_image) == 2
    original_image = reshape(original_image, size(original_image, 1), size(original_image, 2), 1); % Convert 2D to 3D
end
[Nx0, Ny0, Nz] = size(original_image);

% Initialize output
output_image = zeros(Nx0, Ny0, Nz);

% Process based on method
switch lower(method)
    case 'tophat'
        % Apply tophat filter to each slice
        se = strel('disk', options.strel_size);
        for z = 1:Nz
            im = original_image(:,:,z);
            im = rescale(imtophat(im, se));
            output_image(:,:,z) = im;
        end

    case 'rolling ball'
        % Improved Rolling Ball algorithm implementation
        radius = options.strel_size;

        for z = 1:Nz
            % Get current slice
            img = original_image(:,:,z);

            % Step 1: Padding the image to handle edge effects
            pad_size = radius + 5; % Extra padding to avoid edge artifacts
            if ismatrix(img)
                padded_img = padarray(img, [pad_size pad_size], 'symmetric');
            else
                padded_img = img;
            end

            % Step 2: Create properly sized structuring element for opening
            se = strel('disk', radius);

            % Step 3: Estimate background using morphological opening
            % Opening is erosion followed by dilation - effectively removes peaks
            background = imopen(padded_img, se);

            % Step 4: Crop background to original size
            if ismatrix(img)
                background = background(pad_size+1:pad_size+size(img,1), pad_size+1:pad_size+size(img,2));
            end

            % Step 5: Subtract background and normalize
            result = img - background;

            % Step 6: Adjust result to have positive values and scale to [0,1]
            result = result - min(result(:));
            if max(result(:)) > 0
                result = result / max(result(:));
            end

            output_image(:,:,z) = result;
        end
    case {'dark', 'default'}
        % h = waitbar(0,'Background removing...');
        % Use existing Dark Channel Prior method
        image0 = original_image;

        % Normalize input to [0, 255] range
        image0 = 255 * (image0 - min(image0(:))) ./ (max(image0(:)) - min(image0(:)));

        % Make square if necessary (for processing)
        [Nx, Ny, ~] = size(image0);
        if Ny > Nx
            image0(Nx+1:Ny, :, :) = 0; % Zero-pad rows
        elseif Ny < Nx
            image0(:, Ny+1:Nx, :) = 0; % Zero-pad columns
        end
        [Nx, Ny, Nz] = size(image0);

        % Extract parameters
        background = options.BackgroundMode;
        pad = options.Padding;
        denoise = options.Denoise;
        thres = options.Threshold;
        divide = options.DivideCoeff;
        result_stack = zeros(Nx, Ny, Nz);
        Lo_process_stack = zeros(Nx, Ny, Nz);
        Hi_stack = zeros(Nx, Ny, Nz);

        % Padding size
        pad_size = 15;

        % Pad edges
        image = zeros(Nx + 2*floor(Nx/pad_size) + 2, Ny + 2*floor(Ny/pad_size) + 2, Nz);
        for jz = 1:Nz
            if pad == 1
                image(:,:,jz) = padarray(image0(:,:,jz), [floor(Nx/pad_size)+1, floor(Ny/pad_size)+1], 'symmetric');
            else
                image(:,:,jz) = padarray(image0(:,:,jz), [floor(Nx/pad_size)+1, floor(Ny/pad_size)+1]);
            end
        end

        % Basic parameters
        params.Nx = size(image, 1);
        params.Ny = size(image, 2);
        params.NA = options.NA;
        params.emwavelength = options.Wavelength;
        params.pixelsizeX = options.PixelSizeX;
        params.pixelsizeY = options.PixelSizeY;
        params.factor = 2;

        % Background removal settings
        if background == 1
            maxtime = 2;  % Strict mode: two iterations
            deg_matrix = [6, 3, 1.2]; % Radius parameters for each iteration
            dep_matrix = [3, 3, 2];   % Attenuation depth parameters
            hl_matrix = [1, 1, 1];    % High frequency scaling parameters
        else
            maxtime = 1;  % Medium mode: one iteration
            deg_matrix = 6;
            dep_matrix = 3;
            hl_matrix = 1;
        end

        % Crop indices for removing padding
        start_i = floor(Nx/pad_size) + 2;
        start_j = floor(Ny/pad_size) + 2;

        % Dark channel sectioning process
        for time = 1:maxtime
            for jz = 1:Nz
                % Get current iteration parameters
                if length(deg_matrix) > 1
                    deg = deg_matrix(time);
                    dep = dep_matrix(time);
                    hl = hl_matrix(time);
                else
                    deg = deg_matrix;
                    dep = dep_matrix;
                    hl = hl_matrix;
                end

                % Separate high and low frequency components
                [Hi, Lo, lp, EL] = separateHiLo(squeeze(image(:,:,jz)), params, deg, divide);

                % Determine dehazing block size based on lp
                block_size = confirm_block(params, lp);

                % Apply fast dehazing to low frequency component
                Lo_process = dehaze_fast2(Lo, 0.95, block_size, EL, dep, thres);

                % Reconstruct image: dehazing low frequency + high frequency
                result = Lo_process/hl + Hi;

                % Crop to remove padding
                Lo_process = Lo_process(start_i:start_i+Nx-1, start_j:start_j+Ny-1);
                Hi = Hi(start_i:start_i+Nx-1, start_j:start_j+Ny-1);
                result = result(start_i:start_i+Nx-1, start_j:start_j+Ny-1);

                % Store results
                result_stack(:,:,jz) = result;
                Lo_process_stack(:,:,jz) = Lo_process;
                Hi_stack(:,:,jz) = Hi;
            end

            % Update image for next iteration
            if time < maxtime
                image0 = result_stack;
                for jz = 1:Nz
                    if pad == 1
                        image(:,:,jz) = padarray(image0(:,:,jz), [floor(Nx/pad_size)+1, floor(Ny/pad_size)+1], 'symmetric');
                    else
                        image(:,:,jz) = padarray(image0(:,:,jz), [floor(Nx/pad_size)+1, floor(Ny/pad_size)+1]);
                    end
                end
            end
        end

        % Optional denoising
        result_final = zeros(Nx, Ny, Nz);
        for jz = 1:Nz
            if pad == 1
                temp = padarray(result_stack(:,:,jz), [floor(Nx/pad_size)+1, floor(Ny/pad_size)+1], 'symmetric');
            else
                temp = padarray(result_stack(:,:,jz), [floor(Nx/pad_size)+1, floor(Ny/pad_size)+1]);
            end

            if denoise == 0
                temp1 = temp; % No denoising
            else
                % Gaussian filter for denoising
                W = fspecial('gaussian', [2,2], 1);
                temp1 = imfilter(temp, W, 'replicate');
            end

            % Crop padding
            result_final(:,:,jz) = temp1(start_i:start_i+Nx-1, start_j:start_j+Ny-1);
        end

        % Crop back to original size if necessary
        if Nx0 ~= Nx || Ny0 ~= Ny
            if Nx > Nx0
                result_final(Nx0+1:end,:,:) = [];
            end
            if Ny > Ny0
                result_final(:,Ny0+1:end,:) = [];
            end
        end

        output_image = rescale(result_final);
        % close(h)

    otherwise
        error('Unknown method specified. Valid options are "Dark", "Tophat", or "Rolling ball".');
end
end

%% 辅助函数
%%%%%%%%%

function [Hi,Lo,lp,EL] = separateHiLo(image,params,deg,divide)

% 基本参数
Nx = params.Nx;
Ny = params.Ny;
NA = params.NA;
emwavelength = params.emwavelength;
pixelsize_x = params.pixelsizeX;
pixelsize_y = params.pixelsizeY;

% 其他参数
res = 0.5 * emwavelength / NA / params.factor;     % resolution
k_m_x = Nx / (res / pixelsize_x);    % objective cut-off frequency in x-direction
k_m_y = Ny / (res / pixelsize_y);    % objective cut-off frequency in y-direction

% Use the smaller cutoff frequency for filter design
k_m = min(k_m_x, k_m_y);
kc = nearest(k_m * 0.2);             % cut-off frequency between hp and lp filter
sigmaLP = kc*2/2.355;                % Finding sigma value for low pass

% Create filters accounting for pixel aspect ratio
aspect_ratio = pixelsize_y / pixelsize_x;  % Calculate pixel aspect ratio
lp = lpgauss_elliptical(Nx, Ny, sigmaLP*2*divide, sigmaLP*2*divide*aspect_ratio);
hp = hpgauss_elliptical(Nx, Ny, sigmaLP*2*divide, sigmaLP*2*divide*aspect_ratio);
elp = lpgauss_elliptical(Nx, Ny, sigmaLP/deg, sigmaLP/deg*aspect_ratio);
ehp = hpgauss_elliptical(Nx, Ny, sigmaLP/deg, sigmaLP/deg*aspect_ratio);

% 得到高低频和极低频率
fft_image = fftshift(fft2(image));
Hi = real(ifft2(ifftshift(fft_image.*fftshift(hp))));
Lo = real(ifft2(ifftshift(fft_image.*fftshift(lp))));

EL = real(ifft2(fft2(image).*elp));
EH = real(ifft2(fft2(image).*ehp));
end

function [out] = hpgauss_elliptical(H, W, SIGMA_X, SIGMA_Y)
% Creates a 2D elliptical Gaussian high-pass filter for a Fourier space image
out = 1 - lpgauss_elliptical(H, W, SIGMA_X, SIGMA_Y);
end

function [out] = lpgauss_elliptical(H, W, SIGMA_X, SIGMA_Y)
% Creates a 2D elliptical Gaussian low-pass filter for a Fourier space image
% W is the number of columns, H is the number of rows
% SIGMA_X and SIGMA_Y are the standard deviations in X and Y directions
H = double(H);
W = double(W);
[x, y] = meshgrid(-floor(W/2):floor((W-1)/2), -floor(H/2):floor((H-1)/2));
temp = -(x.^2/(SIGMA_X^2) + y.^2/(SIGMA_Y^2));
out = ifftshift(exp(temp));
end

function [out] = hpgauss(H,W,SIGMA)
%   Creates a 2D Gaussian filter for a Fourier space image of height H and
%   width W. SIGMA is the standard deviation of the Gaussian.
out=1-lpgauss(H,W,SIGMA);
end

function [out] = lpgauss(H,W,SIGMA)
%   Creates a 2D Gaussian filter for a Fourier space image
%   W is the number of columns of the source image and H is the number of
%   rows. SIGMA is the standard deviation of the Gaussian.
H = double(H);
W = double(W);
kcx = (SIGMA);
kcy = ((H/W)*SIGMA);
temp0 = -floor(W/2);
[x,y] = meshgrid(-floor(W/2):floor((W-1)/2), -floor(H/2):floor((H-1)/2));
temp = -(x.^2/(kcx^2)+y.^2/(kcy^2));
out = ifftshift(exp(temp));
% out = ifftshift(exp(-(x.^2/(kcx^2)+y.^2/(kcy^2))));
end

function [ radiance ] = dehaze_fast2(  image, omega, win_size,EL,dep,thres )
% 对低频分量进行快速去雾处理
% Copyright (c) 2014 Stephen Tierney
[Nx,Ny] = size(image);
if ~exist('omega', 'var')
    omega = 0.95;
end

if ~exist('win_size', 'var')
    win_size = 15;
end

r = 15;
res = 0.001;

[m, n, ~] = size(image);

Mask = zeros(Nx,Ny);
Mask(image<thres) = 1;
dark_channel = get_dark_channel_gpu(image.*Mask, win_size);
min_atmosphere = get_atmosphere(image.*Mask, dark_channel);

dark_channel = get_dark_channel_gpu(image, win_size);
max_atmosphere = get_atmosphere(image, dark_channel);
EL = EL - min(min(EL));
rep_atmosphere_process = EL/max(max(EL))*(max_atmosphere-min_atmosphere)+min_atmosphere;
rep_atmosphere_process = dep*rep_atmosphere_process;
trans_est = get_transmission_estimate(rep_atmosphere_process,image, omega, win_size);
x = guided_filter(image, trans_est, r, res);
transmission = reshape(x, m, n);
radiance = get_radiance(rep_atmosphere_process,image, transmission);
end

function block_size = confirm_block(params,lp)
% 根据低通滤波器确定去雾块大小
PSF = PSF_Generator(params.emwavelength, params.pixelsizeX, params.pixelsizeY, params.NA, params.Nx, params.Ny, params.factor);
PSF_Lo = abs(ifft2(fftshift(fft2(PSF)).*fftshift(lp)));
PSF_Lo = PSF_Lo./max(max(PSF_Lo));
% figure;plot(PSF(floor(params.Nx/2)+1,:))
for count_x = floor(params.Nx/2):params.Nx
    if PSF_Lo(count_x,floor(params.Nx/2)) <0.01
        break;
    end
end
block_size = count_x-floor(params.Nx/2);
end

function PSF = PSF_Generator(lambada, pixelsize_x, pixelsize_y, NA, w, h, factor)
% Generate Point Spread Function with possibly different pixel sizes in x/y directions
% If h is not provided, assume a square PSF
if nargin < 6
    h = w;
end

[X,Y] = meshgrid(linspace(0,w-1,w), linspace(0,h-1,h));

% Account for different pixel sizes in X and Y
scale_x = 2*pi*NA/lambada*pixelsize_x*factor;
scale_y = 2*pi*NA/lambada*pixelsize_y*factor;

% Create an elliptical distance map
X_scaled = min(X,abs(X-w)) * (scale_x/scale_y);
Y_scaled = min(Y,abs(Y-h));
R = sqrt(X_scaled.^2 + Y_scaled.^2);
R = R * scale_y;  % Apply scale to the computed distances

% Generate the PSF
PSF = abs(2*besselj(1,R+eps,1)./(R+eps)).^2;
PSF = PSF/(sum(sum(PSF)));
PSF = fftshift(PSF);
end


function atmosphere = get_atmosphere(image, dark_channel)
% Copyright (c) 2014 Stephen Tierney

[m, n, ~] = size(image);
n_pixels = m * n;

n_search_pixels = floor(n_pixels * 0.01);

dark_vec = reshape(dark_channel, n_pixels, 1);

image_vec = reshape(image, n_pixels,1);

[~, indices] = sort(dark_vec, 'descend');

accumulator = 0;

for k = 1 : n_search_pixels
    accumulator = accumulator + image_vec(indices(k),:);
end

atmosphere = accumulator / n_search_pixels;

end

function trans_est = get_transmission_estimate(rep_atmosphere, image, omega, win_size)
% Copyright (c) 2014 Stephen Tierney
[m, n, ~] = size(image);

%rep_atmosphere = repmat(atmosphere, m, n);

trans_est = 1 - omega * get_dark_channel_gpu( image ./ rep_atmosphere, win_size);

end

function radiance = get_radiance(rep_atmosphere,image, transmission)
% Copyright (c) 2014 Stephen Tierney

[m, n, ~] = size(image);

max_transmission = max(transmission, 0.1);

radiance = ((image - rep_atmosphere) ./ max_transmission) + rep_atmosphere;

end

function q = guided_filter(guide, target, radius, eps)
% Copyright (c) 2014 Stephen Tierney

[h, w] = size(guide);

avg_denom = window_sum_filter(ones(h, w), radius);

mean_g = window_sum_filter(guide, radius) ./ avg_denom;
mean_t = window_sum_filter(target, radius) ./ avg_denom;

corr_gg = window_sum_filter(guide .* guide, radius) ./ avg_denom;
corr_gt = window_sum_filter(guide .* target, radius) ./ avg_denom;

var_g = corr_gg - mean_g .* mean_g;
cov_gt = corr_gt - mean_g .* mean_t;

a = cov_gt ./ (var_g + eps);
b = mean_t - a .* mean_g;

mean_a = window_sum_filter(a, radius) ./ avg_denom;
mean_b = window_sum_filter(b, radius) ./ avg_denom;

q = mean_a .* guide + mean_b;

end


function sum_img = window_sum_filter(image, r)

% sum_img(x, y) = = sum(sum(image(x-r:x+r, y-r:y+r)));

[h, w] = size(image);
sum_img = zeros(size(image));

% Y axis
im_cum = cumsum(image, 1);

sum_img(1:r+1, :) = im_cum(1+r:2*r+1, :);
sum_img(r+2:h-r, :) = im_cum(2*r+2:h, :) - im_cum(1:h-2*r-1, :);
sum_img(h-r+1:h, :) = repmat(im_cum(h, :), [r, 1]) - im_cum(h-2*r:h-r-1, :);

% X axis
im_cum = cumsum(sum_img, 2);

sum_img(:, 1:r+1) = im_cum(:, 1+r:2*r+1);
sum_img(:, r+2:w-r) = im_cum(:, 2*r+2:w) - im_cum(:, 1:w-2*r-1);
sum_img(:, w-r+1:w) = repmat(im_cum(:, w), [1, r]) - im_cum(:, w-2*r:w-r-1);

end

%% 原版

% function output_image = darkChannelRemoveBackground(input_image, method, options)
% % DARKCHANNELREMOVEBACKGROUND Removes background from microscopy images using various methods
% %
% % Syntax:
% %   output_image = darkChannelRemoveBackground(input_image)
% %   output_image = darkChannelRemoveBackground(input_image, method)
% %   output_image = darkChannelRemoveBackground(input_image, method, options)
% %
% % Input:
% %   input_image - 2D or 3D image array (required)
% %   method      - Background removal method: 'Dark', 'Tophat', or 'Rolling ball' (default: 'Dark')
% %   options     - Structure with fields for parameters (optional)
% %
% % Optional Structure Fields:
% %   strel_size     - Structuring element size for Tophat and Rolling ball methods (default: 15)
% %   BackgroundMode - Background removal strictness for Dark method: 0-medium, 1-strict (default: 1)
% %   Padding        - Edge padding mode for Dark method: 0-zeros, 1-symmetric (default: 1)
% %   Denoise        - Apply Gaussian denoising for Dark method: 0-no, 1-yes (default: 0)
% %   Threshold      - Low frequency dehazing threshold for Dark method (default: 70)
% %   DivideCoeff    - Frequency division coefficient for Dark method (default: 0.5)
% %   NA             - Numerical aperture for Dark method (default: 0.06)
% %   Wavelength     - Emission wavelength in nm for Dark method (default: 1100)
% %   PixelSize      - Pixel size in nm for Dark method (default: 6000)
% %
% % Output:
% %   output_image - Background-removed image with same dimensions as input
% %
% % Example:
% %   img_out = darkChannelRemoveBackground(img_in);
% %   img_out = darkChannelRemoveBackground(img_in, 'Tophat');
% %   opts = struct('strel_size', 20);
% %   img_out = darkChannelRemoveBackground(img_in, 'Tophat', opts);
% %   opts = struct('strel_size', 50);
% %   img_out = darkChannelRemoveBackground(img_in, 'Rolling ball', opts);
% %
% % Based on: "Single Image Haze Removal Using Dark Channel Prior" by Kaiming He, et al.
% 
% % Handle input arguments
% if nargin < 2 || isempty(method)
%     method = 'Dark';
% end
% 
% % Set default options
% defaultOptions = struct(...
%     'strel_size', 15, ...
%     'BackgroundMode', 1, ...
%     'Padding', 1, ...
%     'Denoise', 0, ...
%     'Threshold', 70, ...
%     'DivideCoeff', 0.5, ...
%     'NA', 0.06, ...
%     'Wavelength', 1100, ...
%     'PixelSize', 6000);
% 
% % If options not provided, use defaults
% if nargin < 3 || isempty(options)
%     options = defaultOptions;
% else
%     % Merge with defaults for any missing fields
%     optionFields = fieldnames(defaultOptions);
%     for i = 1:length(optionFields)
%         if ~isfield(options, optionFields{i})
%             options.(optionFields{i}) = defaultOptions.(optionFields{i});
%         end
%     end
% end
% 
% % Get original dimensions
% original_image = double(input_image);
% if ndims(original_image) == 2
%     original_image = reshape(original_image, size(original_image, 1), size(original_image, 2), 1); % Convert 2D to 3D
% end
% [Nx0, Ny0, Nz] = size(original_image);
% 
% % Initialize output
% output_image = zeros(Nx0, Ny0, Nz);
% 
% % Process based on method
% switch lower(method)
%     case 'tophat'
%         % Apply tophat filter to each slice
%         se = strel('disk', options.strel_size);
%         for z = 1:Nz
%             im = original_image(:,:,z);
%             im = rescale(imtophat(im, se));
%             output_image(:,:,z) = im;
%         end
% 
%     case 'rolling ball'
%         % Improved Rolling Ball algorithm implementation
%         radius = options.strel_size;
% 
%         for z = 1:Nz
%             % Get current slice
%             img = original_image(:,:,z);
% 
%             % Step 1: Padding the image to handle edge effects
%             pad_size = radius + 5; % Extra padding to avoid edge artifacts
%             if ismatrix(img)
%                 padded_img = padarray(img, [pad_size pad_size], 'symmetric');
%             else
%                 padded_img = img;
%             end
% 
%             % Step 2: Create properly sized structuring element for opening
%             se = strel('disk', radius);
% 
%             % Step 3: Estimate background using morphological opening
%             % Opening is erosion followed by dilation - effectively removes peaks
%             background = imopen(padded_img, se);
% 
%             % Step 4: Crop background to original size
%             if ismatrix(img)
%                 background = background(pad_size+1:pad_size+size(img,1), pad_size+1:pad_size+size(img,2));
%             end
% 
%             % Step 5: Subtract background and normalize
%             result = img - background;
% 
%             % Step 6: Adjust result to have positive values and scale to [0,1]
%             result = result - min(result(:));
%             if max(result(:)) > 0
%                 result = result / max(result(:));
%             end
% 
%             output_image(:,:,z) = result;
%         end
%     case {'dark', 'default'}
% 
%         % Use existing Dark Channel Prior method
%         image0 = original_image;
% 
%         % Normalize input to [0, 255] range
%         image0 = 255 * (image0 - min(image0(:))) ./ (max(image0(:)) - min(image0(:)));
% 
%         % Make square if necessary (for processing)
%         [Nx, Ny, ~] = size(image0);
%         if Ny > Nx
%             image0(Nx+1:Ny, :, :) = 0; % Zero-pad rows
%         elseif Ny < Nx
%             image0(:, Ny+1:Nx, :) = 0; % Zero-pad columns
%         end
%         [Nx, Ny, Nz] = size(image0);
% 
%         % Extract parameters
%         background = options.BackgroundMode;
%         pad = options.Padding;
%         denoise = options.Denoise;
%         thres = options.Threshold;
%         divide = options.DivideCoeff;
%         result_stack = zeros(Nx, Ny, Nz);
%         Lo_process_stack = zeros(Nx, Ny, Nz);
%         Hi_stack = zeros(Nx, Ny, Nz);
% 
%         % Padding size
%         pad_size = 15;
% 
%         % Pad edges
%         image = zeros(Nx + 2*floor(Nx/pad_size) + 2, Ny + 2*floor(Ny/pad_size) + 2, Nz);
%         for jz = 1:Nz
%             if pad == 1
%                 image(:,:,jz) = padarray(image0(:,:,jz), [floor(Nx/pad_size)+1, floor(Ny/pad_size)+1], 'symmetric');
%             else
%                 image(:,:,jz) = padarray(image0(:,:,jz), [floor(Nx/pad_size)+1, floor(Ny/pad_size)+1]);
%             end
%         end
% 
%         % Basic parameters
%         params.Nx = size(image, 1);
%         params.Ny = size(image, 2);
%         params.NA = options.NA;
%         params.emwavelength = options.Wavelength;
%         params.pixelsize = options.PixelSize;
%         params.factor = 2;
% 
%         % Background removal settings
%         if background == 1
%             maxtime = 2;  % Strict mode: two iterations
%             deg_matrix = [6, 3, 1.2]; % Radius parameters for each iteration
%             dep_matrix = [3, 3, 2];   % Attenuation depth parameters
%             hl_matrix = [1, 1, 1];    % High frequency scaling parameters
%         else
%             maxtime = 1;  % Medium mode: one iteration
%             deg_matrix = 6;
%             dep_matrix = 3;
%             hl_matrix = 1;
%         end
% 
%         % Crop indices for removing padding
%         start_i = floor(Nx/pad_size) + 2;
%         start_j = floor(Ny/pad_size) + 2;
% 
%         % Dark channel sectioning process
%         for time = 1:maxtime
%             for jz = 1:Nz
%                 % Get current iteration parameters
%                 if length(deg_matrix) > 1
%                     deg = deg_matrix(time);
%                     dep = dep_matrix(time);
%                     hl = hl_matrix(time);
%                 else
%                     deg = deg_matrix;
%                     dep = dep_matrix;
%                     hl = hl_matrix;
%                 end
% 
%                 % Separate high and low frequency components
%                 [Hi, Lo, lp, EL] = separateHiLo(squeeze(image(:,:,jz)), params, deg, divide);
% 
%                 % Determine dehazing block size based on lp
%                 block_size = confirm_block(params, lp);
% 
%                 % Apply fast dehazing to low frequency component
%                 Lo_process = dehaze_fast2(Lo, 0.95, block_size, EL, dep, thres);
% 
%                 % Reconstruct image: dehazing low frequency + high frequency
%                 result = Lo_process/hl + Hi;
% 
%                 % Crop to remove padding
%                 Lo_process = Lo_process(start_i:start_i+Nx-1, start_j:start_j+Ny-1);
%                 Hi = Hi(start_i:start_i+Nx-1, start_j:start_j+Ny-1);
%                 result = result(start_i:start_i+Nx-1, start_j:start_j+Ny-1);
% 
%                 % Store results
%                 result_stack(:,:,jz) = result;
%                 Lo_process_stack(:,:,jz) = Lo_process;
%                 Hi_stack(:,:,jz) = Hi;
%             end
% 
%             % Update image for next iteration
%             if time < maxtime
%                 image0 = result_stack;
%                 for jz = 1:Nz
%                     if pad == 1
%                         image(:,:,jz) = padarray(image0(:,:,jz), [floor(Nx/pad_size)+1, floor(Ny/pad_size)+1], 'symmetric');
%                     else
%                         image(:,:,jz) = padarray(image0(:,:,jz), [floor(Nx/pad_size)+1, floor(Ny/pad_size)+1]);
%                     end
%                 end
%             end
%         end
% 
%         % Optional denoising
%         result_final = zeros(Nx, Ny, Nz);
%         for jz = 1:Nz
%             if pad == 1
%                 temp = padarray(result_stack(:,:,jz), [floor(Nx/pad_size)+1, floor(Ny/pad_size)+1], 'symmetric');
%             else
%                 temp = padarray(result_stack(:,:,jz), [floor(Nx/pad_size)+1, floor(Ny/pad_size)+1]);
%             end
% 
%             if denoise == 0
%                 temp1 = temp; % No denoising
%             else
%                 % Gaussian filter for denoising
%                 W = fspecial('gaussian', [2,2], 1);
%                 temp1 = imfilter(temp, W, 'replicate');
%             end
% 
%             % Crop padding
%             result_final(:,:,jz) = temp1(start_i:start_i+Nx-1, start_j:start_j+Ny-1);
%         end
% 
%         % Crop back to original size if necessary
%         if Nx0 ~= Nx || Ny0 ~= Ny
%             if Nx > Nx0
%                 result_final(Nx0+1:end,:,:) = [];
%             end
%             if Ny > Ny0
%                 result_final(:,Ny0+1:end,:) = [];
%             end
%         end
% 
%         output_image = rescale(result_final);
% 
%     otherwise
%         error('Unknown method specified. Valid options are "Dark", "Tophat", or "Rolling ball".');
% end
% end
% 
% %% 辅助函数
% %%%%%%%%%
% 
% function [Hi,Lo,lp,EL] = separateHiLo(image,params,deg,divide)
% 
% % 基本参数
% Nx = params.Nx;
% Ny = params.Ny;
% NA = params.NA;
% emwavelength = params.emwavelength;
% pixel_size = params.pixelsize;
% 
% % 其他参数
% res = 0.5 * emwavelength / NA/ params.factor;     % resolution
% k_m = Ny / (res / pixel_size(1));    % objective cut-off frequency ???
% kc = nearest(k_m * 0.2);             % cut-off frequency between hp and lp filter
% sigmaLP = kc*2/2.355;                % Finding sigma value for low pass
% 
% % 滤波
% lp = lpgauss(Nx,Ny,sigmaLP*2*divide);
% hp = hpgauss(Nx,Ny,sigmaLP*2*divide);
% elp = lpgauss(Nx,Ny,sigmaLP/deg);
% ehp = hpgauss(Nx,Ny,sigmaLP/deg);
% 
% % 得到高低频和极低频率
% Hi = real(ifft2(fft2(image).*hp));
% Lo = real(ifft2(fft2(image).*lp));
% 
% fft_image = fftshift(fft2(image));
% Hi = real(ifft2(ifftshift(fft_image.*fftshift(hp))));
% Lo = real(ifft2(ifftshift(fft_image.*fftshift(lp))));
% 
% % elp = zeros(Nx,Ny);
% % elp(floor((Nx+1)/2)-1:floor((Nx+1)/2)+1,floor((Nx+1)/2)-1:floor((Nx+1)/2)+1) = 1;
% EL = real(ifft2(fft2(image).*elp));
% EH = real(ifft2(fft2(image).*ehp));
% end
% 
% function [ out ] = hpgauss(H,W,SIGMA)
% %   Creates a 2D Gaussian filter for a Fourier space image of height H and
% %   width W. SIGMA is the standard deviation of the Gaussian.
% out=1-lpgauss(H,W,SIGMA);
% end
% 
% function [ out ] = lpgauss(H,W,SIGMA)
% %   Creates a 2D Gaussian filter for a Fourier space image
% %   W is the number of columns of the source image and H is the number of
% %   rows. SIGMA is the standard deviation of the Gaussian.
% H = double(H);
% W = double(W);
% kcx = (SIGMA);
% kcy = ((H/W)*SIGMA);
% temp0 = -floor(W/2);
% [x,y] = meshgrid(-floor(W/2):floor((W-1)/2), -floor(H/2):floor((H-1)/2));
% temp = -(x.^2/(kcx^2)+y.^2/(kcy^2));
% out = ifftshift(exp(temp));
% % out = ifftshift(exp(-(x.^2/(kcx^2)+y.^2/(kcy^2))));
% end
% 
% function block_size = confirm_block(params,lp)
% % 根据低通滤波器确定去雾块大小
% PSF = PSF_Generator(params.emwavelength,params.pixelsize,params.NA,params.Nx,params.factor);
% PSF_Lo = abs(ifft2(fftshift(fft2(PSF)).*fftshift(lp)));
% PSF_Lo = PSF_Lo./max(max(PSF_Lo));
% % figure;plot(PSF(floor(params.Nx/2)+1,:))
% for count_x = floor(params.Nx/2):params.Nx
%     if PSF_Lo(count_x,floor(params.Nx/2)) <0.01
%         break;
%     end
% end
% block_size = count_x-floor(params.Nx/2);
% end
% 
% function [ radiance ] = dehaze_fast2(  image, omega, win_size,EL,dep,thres )
% % 对低频分量进行快速去雾处理
% % Copyright (c) 2014 Stephen Tierney
% [Nx,Ny] = size(image);
% if ~exist('omega', 'var')
%     omega = 0.95;
% end
% 
% if ~exist('win_size', 'var')
%     win_size = 15;
% end
% 
% r = 15;
% res = 0.001;
% 
% [m, n, ~] = size(image);
% 
% Mask = zeros(Nx,Ny);
% Mask(image<thres) = 1;
% dark_channel = get_dark_channel_gpu(image.*Mask, win_size);
% min_atmosphere = get_atmosphere(image.*Mask, dark_channel);
% 
% dark_channel = get_dark_channel_gpu(image, win_size);
% max_atmosphere = get_atmosphere(image, dark_channel);
% EL = EL - min(min(EL));
% rep_atmosphere_process = EL/max(max(EL))*(max_atmosphere-min_atmosphere)+min_atmosphere;
% rep_atmosphere_process = dep*rep_atmosphere_process;
% trans_est = get_transmission_estimate(rep_atmosphere_process,image, omega, win_size);
% x = guided_filter(image, trans_est, r, res);
% transmission = reshape(x, m, n);
% radiance = get_radiance(rep_atmosphere_process,image, transmission);
% end
% function PSF = PSF_Generator(lambada,pixelsize,NA,w,factor)
% 
% [X,Y]=meshgrid(linspace(0,w-1,w),linspace(0,w-1,w));
% scale=2*pi*NA/lambada*pixelsize;
% scale=scale*factor;
% 
% R=sqrt(min(X,abs(X-w)).^2+min(Y,abs(Y-w)).^2);
% PSF=abs(2*besselj(1,scale*R+eps,1)./(scale*R+eps)).^2;
% PSF = PSF/(sum(sum(PSF)));
% PSF=fftshift(PSF);
% 
% end
% 
% % function dark_channel = get_dark_channel_gpu(image, win_size)
% % 
% % [m, n, ~] = size(image);
% % 
% % pad_size = floor(win_size/2);
% % 
% % padded_image = padarray(image, [pad_size pad_size], Inf);
% % 
% % dark_channel = zeros(m, n); 
% % 
% % % for j = 1 : m
% % %     for i = 1 : n
% % %         patch = padded_image(j : j + (win_size-1), i : i + (win_size-1), :);
% % %         dark_channel(j,i) = min(patch(:));
% % %      end
% % % end
% % 
% % parfor k =  1:m*n       
% %         i = mod(k+m-1,m)+1;
% %         j = floor((k+m-1)/m);
% %         patch = padded_image(j : j + (win_size-1), i : i + (win_size-1), :);
% %         dark_channel_temp(k) = min(patch(:));
% % end
% % 
% % for k = 1:m*n
% %     dark_channel(floor((k+m-1)/m),mod(k+m-1,m)+1) = dark_channel_temp(k);
% % end
% % 
% % 
% % end
% 
% function dark_channel = get_dark_channel(image, win_size)
% % 修复索引问题的get_dark_channel函数
% 
% [m, n, ~] = size(image);
% 
% % 确保win_size不会过大
% win_size = min(win_size, min(m, n)/2); % 限制窗口大小不超过图像尺寸的一半
% win_size = max(3, win_size); % 确保win_size至少为3
% 
% pad_size = floor(win_size/2);
% padded_image = padarray(image, [pad_size pad_size], Inf);
% 
% dark_channel = zeros(m, n); 
% 
% % 预先分配数组防止动态大小变化
% dark_channel_temp = zeros(m*n, 1);
% 
% % 使用更安全的索引计算
% parfor k = 1:m*n       
%     row = ceil(k/n); % 行索引
%     col = mod(k-1, n) + 1; % 列索引
% 
%     % 从padded_image中获取相应的窗口
%     r_start = row;
%     r_end = row + win_size - 1;
%     c_start = col;
%     c_end = col + win_size - 1;
% 
%     % 确保索引不超出边界
%     r_end = min(r_end, r_start + 2*pad_size);
%     c_end = min(c_end, c_start + 2*pad_size);
% 
%     patch = padded_image(r_start:r_end, c_start:c_end, :);
%     dark_channel_temp(k) = min(patch(:));
% end
% % 填充dark_channel
% for k = 1:m*n
%     row = ceil(k/n);
%     col = mod(k-1, n) + 1;
%     dark_channel(row, col) = dark_channel_temp(k);
% end
% end
% 
% function atmosphere = get_atmosphere(image, dark_channel)
% % Copyright (c) 2014 Stephen Tierney
% 
% [m, n, ~] = size(image);
% n_pixels = m * n;
% 
% n_search_pixels = floor(n_pixels * 0.01);
% 
% dark_vec = reshape(dark_channel, n_pixels, 1);
% 
% image_vec = reshape(image, n_pixels,1);
% 
% [~, indices] = sort(dark_vec, 'descend');
% 
% accumulator = 0;
% 
% for k = 1 : n_search_pixels
%     accumulator = accumulator + image_vec(indices(k),:);
% end
% 
% atmosphere = accumulator / n_search_pixels;
% 
% end
% 
% function trans_est = get_transmission_estimate(rep_atmosphere, image, omega, win_size)
% % Copyright (c) 2014 Stephen Tierney
% [m, n, ~] = size(image);
% 
% %rep_atmosphere = repmat(atmosphere, m, n);
% 
% trans_est = 1 - omega * get_dark_channel_gpu( image ./ rep_atmosphere, win_size);
% 
% end
% 
% function radiance = get_radiance(rep_atmosphere,image, transmission)
% % Copyright (c) 2014 Stephen Tierney
% 
% [m, n, ~] = size(image);
% 
% max_transmission = max(transmission, 0.1);
% 
% radiance = ((image - rep_atmosphere) ./ max_transmission) + rep_atmosphere;
% 
% end
% 
% function q = guided_filter(guide, target, radius, eps)
% % Copyright (c) 2014 Stephen Tierney
% 
% [h, w] = size(guide);
% 
% avg_denom = window_sum_filter(ones(h, w), radius);
% 
% mean_g = window_sum_filter(guide, radius) ./ avg_denom;
% mean_t = window_sum_filter(target, radius) ./ avg_denom;
% 
% corr_gg = window_sum_filter(guide .* guide, radius) ./ avg_denom;
% corr_gt = window_sum_filter(guide .* target, radius) ./ avg_denom;
% 
% var_g = corr_gg - mean_g .* mean_g;
% cov_gt = corr_gt - mean_g .* mean_t;
% 
% a = cov_gt ./ (var_g + eps);
% b = mean_t - a .* mean_g;
% 
% mean_a = window_sum_filter(a, radius) ./ avg_denom;
% mean_b = window_sum_filter(b, radius) ./ avg_denom;
% 
% q = mean_a .* guide + mean_b;
% 
% end
% 
% 
% function sum_img = window_sum_filter(image, r)
% 
% % sum_img(x, y) = = sum(sum(image(x-r:x+r, y-r:y+r)));
% 
% [h, w] = size(image);
% sum_img = zeros(size(image));
% 
% % Y axis
% im_cum = cumsum(image, 1);
% 
% sum_img(1:r+1, :) = im_cum(1+r:2*r+1, :);
% sum_img(r+2:h-r, :) = im_cum(2*r+2:h, :) - im_cum(1:h-2*r-1, :);
% sum_img(h-r+1:h, :) = repmat(im_cum(h, :), [r, 1]) - im_cum(h-2*r:h-r-1, :);
% 
% % X axis
% im_cum = cumsum(sum_img, 2);
% 
% sum_img(:, 1:r+1) = im_cum(:, 1+r:2*r+1);
% sum_img(:, r+2:w-r) = im_cum(:, 2*r+2:w) - im_cum(:, 1:w-2*r-1);
% sum_img(:, w-r+1:w) = repmat(im_cum(:, w), [1, r]) - im_cum(:, w-2*r:w-r-1);
% 
% end
% 
