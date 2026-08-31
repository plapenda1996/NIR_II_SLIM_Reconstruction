function [tform,err] = optimizeCentroidTransform(gridX, gridY, img_fix, patch_size)
if nargin<4
    % 定义patch尺寸（奇数）
    patch_size = [21, 21]; % 根据实际情况设置
end
ptX = zeros(size(gridX));
ptY = zeros(size(gridX));
for i = 1:size(gridX,1)
    for j = 1:size(gridX,2)
        % 假设 x 和 y 为 patch 的中心像素
        x = gridX(i, j);
        y = gridY(i, j);
        % 计算 patch 的起始位置
        startX = x - (patch_size(2) - 1) / 2;
        startY = y - (patch_size(1) - 1) / 2;
        % 裁剪出 patch
        patch = imcrop(img_fix, [startX, startY, patch_size(2) - 1, patch_size(1) - 1]);
        % % 找最大值
        % [row, col] = find(patch == max(patch(:)));
        % 找质心
        centroid  = regionprops(imbinarize(patch), patch, 'Centroid').Centroid;
        row = centroid(2);
        col = centroid(1);
        % 转换为原图像中的索引
        ptY(i,j) = startY + row - 1;
        ptX(i,j) = startX + col - 1;
    end
end
movingpoints = [ptX(:),ptY(:)];
fixedpoints = [gridX(:),gridY(:)];
% 优化仿射变换
tform = fitgeotform2d(movingpoints,fixedpoints,"affine");
% tform = fitgeotform2d(movingpoints,fixedpoints,"projective");
% 拟合误差
% Match circle centers to the nearest grid coordinates using nearest neighbor search
fittedpoints = transformPointsForward(tform,movingpoints);
err = sqrt(sum((fixedpoints - fittedpoints).^2,2));
end