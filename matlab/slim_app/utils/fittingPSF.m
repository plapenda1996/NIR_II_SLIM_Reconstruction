function [psf_mean,psf_fit,optimal_params] = fittingPSF(gridX, gridY, img_fix, patch_size, dia_x,dia_y,iter)

psf_mean = zeros(patch_size);
all_patches = [];

for i = 1:size(gridX, 1)
    for j = 1:size(gridX, 2)
        % 假设 x 和 y 为 patch 的中心像素
        x = gridX(i, j);
        y = gridY(i, j);
        % 计算 patch 的起始位置
        startX = x - (patch_size(2) - 1) / 2;
        startY = y - (patch_size(1) - 1) / 2;
        % 裁剪出 patch
        patch = imcrop(img_fix, [startX, startY, patch_size(2) - 1, patch_size(1) - 1]);
        % deconv
        if nargin>4
            patch = deconvPSF(patch,dia_x,dia_y,iter);
        end
        % avg psf
        psf_mean = psf_mean + patch/size(gridX, 1)/size(gridX, 2);
        patch = patch - min(patch(:));
        patch = patch/sum(patch,"all");
        % 将所有patch的数据联合起来
        all_patches = [all_patches, patch(:)];
    end
end
psf_mean = psf_mean - min(psf_mean(:));
psf_mean = psf_mean/sum(psf_mean,'all');

% 初始参数估计 [amplitude, x0, y0, sigma_x, sigma_y, theta]
% 使用图像矩来估计中心位置、标准差和角度
[X, Y] = meshgrid(1:patch_size(2), 1:patch_size(1));
initial_params = estimateParams(psf_mean, X, Y);
% 拟合
options = optimset('Display', 'off');
optimal_params = lsqcurvefit(@gaussian2D, initial_params, [X(:), Y(:)], psf_mean(:));
% fun = @(x,xdata) gaussian2D(x,xdata)*size(gridX, 1) * size(gridX, 2);
% all_XY = repmat([X(:), Y(:)], size(gridX, 1) * size(gridX, 2), 1);
% optimal_params = lsqcurvefit(fun, initial_params, all_XY, all_patches(:));
% optimal_params = fminsearch(@(p) sum((gaussian2D(p, [X(:), Y(:)]) - psf_mean(:)).^2), initial_params, options);
% 生成拟合的PSF
psf_fit = reshape(gaussian2D(optimal_params, [X(:), Y(:)]), patch_size);
% % visualize
% figure
% c = [0,max(all_patches(:))];
% subplot(241)
% imagesc(reshape(all_patches(:,1),patch_size));clim(c)
% subplot(242)
% imagesc(reshape(all_patches(:,round(size(all_patches,2)/3)),patch_size));clim(c)
% subplot(243)
% imagesc(reshape(all_patches(:,round(size(all_patches,2)/2)),patch_size));clim(c)
% subplot(244)
% imagesc(reshape(all_patches(:,end),patch_size));clim(c)
% subplot(245)
% imagesc(psf_fit);clim(c);title('Fitted')
% subplot(246)
% imagesc(psf_mean);clim(c);title('Mean')
% pause(1)
% close

end

function F = gaussian2D(x, xdata)
% 高斯函数模型
% params = [x0, y0, sigma_x, sigma_y, theta]
x0 = x(1);
y0 = x(2);
sigma_x = x(3);
sigma_y = x(4);
theta = x(5);

x = xdata(:, 1);
y = xdata(:, 2);

a = (cos(theta)^2 / (2 * sigma_x^2)) + (sin(theta)^2 / (2 * sigma_y^2));
b = -(sin(2 * theta) / (4 * sigma_x^2)) + (sin(2 * theta) / (4 * sigma_y^2));
c = (sin(theta)^2 / (2 * sigma_x^2)) + (cos(theta)^2 / (2 * sigma_y^2));

F = exp(-(a * ((x - x0).^2) + 2 * b * (x - x0) .* (y - y0) + c * ((y - y0).^2)));
F = F/sum(F,"all");
end

function params = estimateParams(patch, X, Y)
% 使用图像矩估计初始参数
totalIntensity = sum(patch(:));
x0 = sum(X(:) .* patch(:)) / totalIntensity;
y0 = sum(Y(:) .* patch(:)) / totalIntensity;

x_var = sum((X(:) - x0).^2 .* patch(:)) / totalIntensity;
y_var = sum((Y(:) - y0).^2 .* patch(:)) / totalIntensity;
% xy_cov = sum((X(:) - x0) .* (Y(:) - y0) .* patch(:)) / totalIntensity;

sigma_x = sqrt(x_var);
sigma_y = sqrt(y_var);

% 通过计算图像的二阶中心矩来估计旋转角度theta
mu20 = sum((X(:) - x0).^2 .* patch(:)) / totalIntensity;
mu02 = sum((Y(:) - y0).^2 .* patch(:)) / totalIntensity;
mu11 = sum((X(:) - x0) .* (Y(:) - y0) .* patch(:)) / totalIntensity;

theta = 0.5 * atan2(2 * mu11, mu20 - mu02);
% 处理接近90度的情况
if abs(theta) > pi/4
    theta = theta - sign(theta) * pi/2;
end
params = [x0, y0, sigma_x, sigma_y, theta];
end