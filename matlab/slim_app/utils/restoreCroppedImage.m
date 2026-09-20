function restoredImage = restoreCroppedImage(Icropped, rect, origSize)
%RESTORECROPPEDIMAGE 将裁剪后的图像补零到原始大小
%   restoredImage = restoreCroppedImage(Icropped, rect, origSize) 将裁剪后的图像Icropped按照给定的裁剪矩形rect,
%   补零到与原始图像大小相同的大小，并保持图像相对位置不变。
%
%   输入参数:
%     Icropped - 裁剪后的图像
%     rect     - 裁剪矩形,格式为[x y width height]
%     origSize - 原始图像的大小,格式为[rows cols numChannels]
%
%   输出参数:
%     restoredImage - 将裁剪后的图像补零到原始大小后的结果图像

% 获取原始图像的大小
origRows = origSize(1);
origCols = origSize(2);
if length(origSize)<3
    numChannels = 1;
else
    numChannels = origSize(3);
end

% 获取裁剪图像的大小
[croppedRows, croppedCols, ~] = size(Icropped);

% 计算需要填充的行数和列数
topRows = rect(2); 
leftCols = rect(1);

% 创建一个与原始图像大小相同的全零矩阵
restoredImage = zeros(origRows, origCols, numChannels, 'like', Icropped);

% 将裁剪后的图像插入到适当的位置
restoredImage(topRows:topRows+croppedRows-1, leftCols:leftCols+croppedCols-1, :) = Icropped;

end
