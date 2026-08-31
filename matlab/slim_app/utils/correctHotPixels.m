function [correctedImg, hotPixelMask, coldPixelMask] = correctHotPixels(img, varargin)
% CORRECTHOTPIXELS 使用选择性中值滤波校正灰度显微镜图像中的热像素和冷像素
%
% [correctedImg, hotPixelMask, coldPixelMask] = correctHotPixels(img) 
% 使用默认参数校正输入灰度图像中的热像素和冷像素
%
% [correctedImg, hotPixelMask, coldPixelMask] = correctHotPixels(img, 'ParameterName', ParameterValue, ...)
% 允许自定义像素校正过程的参数
%
% 输入参数:
% img - 输入灰度图像 (2D矩阵)
%
% 可选参数:
% 'zScore' - 检测阈值 (默认: 3.0)
%
% 'DetectCold' - 是否检测冷像素 (默认: true)
%               设为false时只检测热像素
%
% 'WinSize' - 滤波的局部邻域大小 (默认: [3 3])
%            可以是单个奇数 (如 3) 表示方形窗口，或
%            奇数值的向量 [行数 列数] (如 [3 3], [5 5])
%
% 'MaxSize' - 异常像素簇的最大大小 (默认: 1)
%            更大的连通组件不被视为异常像素
%
% 输出:
% correctedImg - 校正异常像素后的图像
% hotPixelMask - 显示检测到的热像素的二值掩码
% coldPixelMask - 显示检测到的冷像素的二值掩码
%
% ========== 第一部分：输入参数解析 ==========
% 创建输入参数解析器，用于处理函数的输入参数
p = inputParser;

% 添加必需参数：输入图像必须是数值类型
addRequired(p, 'img', @isnumeric);

% 添加可选参数：z分数阈值，默认为3.0
addParameter(p, 'zScore', 3.0, @(x) isnumeric(x) && x > 0);

% 添加可选参数：是否检测冷像素，默认为true
addParameter(p, 'DetectCold', true, @islogical);

% 添加可选参数：窗口大小，支持两种格式
addParameter(p, 'WinSize', [3 3], @(x) isnumeric(x) && ...
    ((isscalar(x) && mod(x,2)==1) || ... % 单个奇数
     (numel(x)==2 && all(mod(x,2)==1)))); % 两个奇数

% 添加可选参数：异常像素簇的最大大小，默认为1（只处理孤立像素）
addParameter(p, 'MaxSize', 1, @(x) isnumeric(x) && x > 0);

% 解析输入参数
parse(p, img, varargin{:});

% ========== 第二部分：参数提取和预处理 ==========
% 从解析器中提取参数值
zScore = p.Results.zScore;
hotThreshold = zScore;      % 热像素阈值（正数）
coldThreshold = -zScore;    % 冷像素阈值（负数）
detectCold = p.Results.DetectCold;          % 是否检测冷像素
windowSize = p.Results.WinSize;             % 窗口大小
maxSize = p.Results.MaxSize;                % 最大簇大小

% 如果窗口大小是标量，转换为[n n]格式（方形窗口）
if isscalar(windowSize)
    windowSize = [windowSize windowSize];
end

% 保存原始数据类型，并转换为double类型进行处理
originalClass = class(img);
imgDouble = double(img);

% ========== 第三部分：异常像素检测算法 ==========
% 计算局部统计量：

% 1. 计算局部均值：使用平均滤波器计算每个像素邻域的平均值
%    'replicate'边界处理方式：边界外的像素值通过复制边界像素值获得
localMean = imfilter(imgDouble, fspecial('average', windowSize), 'replicate');

% 2. 计算局部标准差：使用标准差滤波器计算每个像素邻域的标准差
%    ones(windowSize)创建一个全1的窗口用于标准差计算
localStd = stdfilt(imgDouble, ones(windowSize));

% 3. 计算z分数：衡量每个像素相对于其邻域的异常程度
%    z分数 = (像素值 - 局部均值) / 局部标准差
%    加eps防止除零错误
zScore = (imgDouble - localMean) ./ (localStd + eps);

% 4. 检测热像素：z分数超过正阈值的像素被标记为热像素
%    热像素特征：比邻域显著更亮
hotPixelMask = zScore > hotThreshold;

% 5. 检测冷像素：z分数低于负阈值的像素被标记为冷像素
%    冷像素特征：比邻域显著更暗
if detectCold
    coldPixelMask = zScore < coldThreshold;
else
    coldPixelMask = false(size(img)); % 如果不检测冷像素，创建全false掩码
end

% ========== 第四部分：异常像素簇处理 ==========
% 只保留小的连通区域（孤立的异常像素）：
% bwareaopen(mask, n) 移除面积小于n的连通组件
% 逻辑：保留面积≥1 且 面积≤maxSize 的连通组件

% 处理热像素簇
hotPixelMask = bwareaopen(hotPixelMask, 1) & ~bwareaopen(hotPixelMask, maxSize + 1);

% 处理冷像素簇
if detectCold
    coldPixelMask = bwareaopen(coldPixelMask, 1) & ~bwareaopen(coldPixelMask, maxSize + 1);
end

% ========== 第五部分：异常像素校正 ==========
% 创建校正后图像的副本
correctedImg = imgDouble;

% 对整个图像应用中值滤波，但只在异常像素位置使用滤波结果
medFiltered = medfilt2(imgDouble, windowSize);

% 选择性替换：只有被标记为异常像素的位置才用中值滤波的结果替换
% 合并热像素和冷像素掩码
allDefectMask = hotPixelMask | coldPixelMask;
correctedImg(allDefectMask) = medFiltered(allDefectMask);

% ========== 第六部分：输出处理 ==========
% 将结果转换回原始数据类型（uint8, uint16等）
correctedImg = cast(correctedImg, originalClass);

end