function mask = createCircularMask(centerX,centerY,roiRatio,imgSize,blurRadius)
    % createCircularMask - 创建一个带羽化边缘的圆形掩码
    %
    % 输入:
    %       - centerX: 圆形掩码中心的X坐标(像素)
    %       - centerY: 圆形掩码中心的Y坐标(像素)
    %       - roiRatio: 圆形掩码占全图的比例(0-1之间)
    %       - imgSize: 图像大小 [height, width]
    %       - blurRadius: 掩码边缘羽化半径(像素)
    %
    % 输出:
    %   mask - 二维掩码矩阵，大小与图像相同，值范围为0-1
    
    % 确保imgSize是有两个元素的向量 [height, width]
    if length(imgSize) ~= 2
        error('图像尺寸必须为 [height, width] 格式');
    end
    
    height = imgSize(1);
    width = imgSize(2);
    
    % 计算圆形掩码的半径
    % 使用较小的尺寸来确保圆完全在图像内
    minDimension = min(height, width);
    radius = minDimension * roiRatio / 2;
    
    % 创建二维网格
    [X, Y] = meshgrid(1:width, 1:height);
    
    % 计算每个像素到中心的距离
    distanceMap = sqrt((X - centerX).^2 + (Y - centerY).^2);
    
    % 创建基本的二进制掩码
    mask = zeros(height, width);
    
    if blurRadius <= 0
        % 无羽化边缘的掩码
        mask(distanceMap <= radius) = 1;
    else
        % 带羽化边缘的掩码
        % 使用sigmoid函数创建平滑过渡
        % 掩码中心区域值为1，外部区域值为0，边缘平滑过渡
        mask = 1 ./ (1 + exp((distanceMap - radius) / blurRadius));
    end
    
    % 确保数值在0-1范围内（对于羽化掩码可能会有小数值）
    mask = max(0, min(1, mask));
end