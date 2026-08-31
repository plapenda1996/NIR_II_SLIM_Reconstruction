function forward_handle = forward_model(OTFs)
%FORWARD_MODEL_HANDLE 创建正向成像模型的函数句柄，使用循环卷积实现
% [FORWARD_HANDLE] = FORWARD_MODEL(OTFs, tformList) 返回基于OTFs和变换列表的前向成像函数句柄
%
% 参数:
%   OTFs - 4D数组，包含各深度各视角的3D OTF [height,width,numDepths,numViews]
%
% 返回值:
%   forward_handle - 前向成像函数句柄，接受重建图像，返回多视角投影
%

% 创建函数句柄
forward_handle = @(reconstructions) forward_projection(reconstructions, OTFs);

% 前向投影实现
function projections = forward_projection(reconstructions, OTFs)
    % 获取输入尺寸信息
    [nx,ny,nz] = size(reconstructions);
    % 计算3D FFT
    reconstructions = fftn(circshift(reconstructions, -floor([nx,ny,nz]/2)));
    % 计算3D卷积（广播到第四维度）
    projections = real(circshift(ifft2(ifft(reconstructions .* OTFs, [], 3)), floor([nx,ny,0,0]/2)));
    % 计算z切片（光场采样）
    projections = squeeze(projections(:,:,1,:));
end
end
