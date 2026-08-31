function patch_in_original = getRawPSF(patch, originalSize, gridX, gridY, tform)
%   tform      : 几何变换对象，必须支持 transformPointsInverse
%   rectWarped : [startX, startY, width, height]，表示在变换后图像上的矩形区域

%---------------------------
% 1. 为 patch 构造空间参考
%---------------------------
% 假设 patch 的大小如下
[patchHeight, patchWidth] = size(patch);  
Rpatch = imref2d(size(patch));

% 设置 patch 在“变换后坐标系”里的 X/Y 世界坐标范围
% 注意：X 对应 width 方向，Y 对应 height 方向
startX = gridX - (patchWidth - 1) / 2;
startY = gridY - (patchHeight - 1) / 2;
Rpatch.XWorldLimits = [startX, startX + patchWidth - 1]; 
Rpatch.YWorldLimits = [startY, startY + patchHeight - 1];

%---------------------------
% 2. 准备逆变换：inverse T
%---------------------------
tform_inv = invert(tform);  

%---------------------------
% 3. 把 patch 反变换回原图坐标系
%---------------------------
Roriginal = imref2d(originalSize);

% 用 imwarp + 逆变换把 “变换后坐标系”的 patch -> “原图坐标系”
patch_in_original = imwarp(patch, ...       % 要映射的图像
                           Rpatch, ...      % 告诉 imwarp：图像 patch 在变换后坐标系里的参考
                           tform_inv, ...   % 逆变换
                           'OutputView', Roriginal);

% 四个角点 (x,y)
cornersWarped = [startX, startY;...
                 startX+patchWidth-1, startY;...
                 startX, startY+patchHeight-1;...
                 startX+patchWidth-1 , startY+patchHeight-1];
% 逆变换 -> 原图坐标系
[xInOrig, yInOrig] = transformPointsInverse(tform, ...
                                            cornersWarped(:,1), ...
                                            cornersWarped(:,2));
% 在原图坐标中计算这四个点的 min/max
minX = min(xInOrig);
maxX = max(xInOrig);
minY = min(yInOrig);
maxY = max(yInOrig);

% 得到原图坐标下的 bounding box
rectInOriginal = [minX, minY, (maxX - minX), (maxY - minY)];
% 在贴回后的图像上，截取 bounding box
patch_in_original = imcrop(patch_in_original, rectInOriginal);

end