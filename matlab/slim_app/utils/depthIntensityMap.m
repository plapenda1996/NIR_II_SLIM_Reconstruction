function [RGB,cmap] = depthIntensityMap(depthNorm, I)
% depthIntensityMap  用 Hue 编码深度、Value 编码强度，返回 RGB 图像
%
%   RGB = depthIntensityMap(depthNorm, I)
%
% 输入：
%   depthNorm  — M×N 矩阵，元素范围 [0,1]，表示每个像素的深度归一化值
%   I          — M×N 矩阵，元素范围 [0,1]，表示每个像素的强度归一化值
%
% 输出：
%   RGB        — M×N×3 矩阵，hsv->rgb 转换后的彩色图像
%
% 使用示例：
%   % 计算 MIP 与深度
%   [MIP, idx] = max(vol, [], 3);
%   depthNorm = (idx-1) / (size(vol,3)-1);
%   I = mat2gray(MIP);
%   % 调用本函数
%   RGB = depthIntensityMap(depthNorm, I);
%   imshow(RGB);
%   title('MIP with Hue=Depth, Value=Intensity');

    % 检查输入尺寸一致
    if ~isequal(size(depthNorm), size(I))
        error('depthNorm 和 I 必须是相同尺寸的矩阵');
    end

    % 构造 HSV 三通道
    H = depthNorm;                % Hue 通道
    S = ones(size(H));            % Saturation 全 1 保持饱和
    V = I;                        % Value 通道严格对应强度
    HSV = cat(3, H, S, V);        % 合并成 M×N×3
    RGB = hsv2rgb(HSV);           % 转换到 RGB 空间
    
    hueVals = H(:);
    hmin = min(hueVals);
    hmax = max(hueVals);
    hues = linspace(hmin, hmax, 256)';    % 256×1
    cmap = hsv2rgb([ hues, ones(256,1), ones(256,1) ]);  % 256×3
end