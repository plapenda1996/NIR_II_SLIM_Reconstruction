function [ptX, ptY] = rectCentroid(img_fix,rect)
    % rect = round(rect);
    % 裁剪出 patch
    patch = imcrop(img_fix, rect);
    patch = patch-min(patch(:));
    % 计算 patch 的起始位置
    startX = rect(1);
    startY = rect(2);
    % 找质心
    centroid = regionprops(imbinarize(patch), patch, 'Centroid').Centroid;
    % 转换为原图像中的索引
    ptY = startY + centroid(2) - 1;
    ptX = startX + centroid(1) - 1;
end