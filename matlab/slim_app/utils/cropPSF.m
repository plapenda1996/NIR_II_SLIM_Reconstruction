
% 通用的任意维度cropPSF函数
function cropped = cropPSF(P, targetSize)
%CROPPSF 从填充的PSF中裁剪出指定大小的中心区域
% cropped = cropPSF(P, targetSize)
%
% 输入:
% P - 填充后的N维数组
% targetSize - 目标大小，必须是一个向量
%
% 输出:
% cropped - 裁剪后的数组

% 获取填充PSF的大小
padSize = size(P);
if isa(P, 'gpuArray')
    padSize = gather(padSize);
end

% 确保维度匹配
ndims_max = max(length(padSize), length(targetSize));
if length(padSize) < ndims_max
    padSize(end+1:ndims_max) = 1;
end
if length(targetSize) < ndims_max
    targetSize(end+1:ndims_max) = 1;
end

% 计算中心坐标
centerPad = floor(padSize / 2) + 1;
centerTarget = floor(targetSize / 2) + 1;
shift = centerPad - centerTarget;

% 创建裁剪索引
cropIdx = cell(1, ndims_max);
for d = 1:ndims_max
    start = shift(d);
    
    % 确保起始位置有效
    start = max(1, start);
    
    % 计算结束位置
    endPos = start + targetSize(d) - 1;
    
    % 确保结束位置不超过边界
    endPos = min(endPos, padSize(d));
    
    cropIdx{d} = start:endPos;
end

% 裁剪操作
cropped = P(cropIdx{:});

end