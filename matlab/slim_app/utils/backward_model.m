function backward_handle = backward_model(OTFs)
%BACKWARD_MODEL_HANDLE 创建反向成像模型的函数句柄，使用循环卷积实现
% [BACKWARD_HANDLE] = BACKWARD_MODEL(OTFs) 返回基于OTFs和变换列表的反向成像函数句柄
%
% 参数:
%   OTFs - 4D数组，包含各深度各视角的3D OTF [height,width,numDepths,numViews]
%
% 返回值:
%   backward_handle - 反向成像函数句柄，接受多视角投影，返回重建图像
%


% 创建函数句柄
backward_handle = @(projections) backward_projection(projections, OTFs);

% 反向投影实现
function reconstructions = backward_projection(projections, OTFs)
    % 获取输入尺寸信息
    [nx, ny, nViews] = size(projections);
    nz = size(OTFs,3);
    % 所有视图的z嵌入(embedding)操作(即切片的adjoint运算）
    mid_z = floor(nz/2);
    projections_emb = zeros(nx,ny,nz,nViews);
    projections_emb(:,:,mid_z,:) = reshape(projections, [nx, ny, 1, nViews]);
    % 所有扩展视图的 3D FFT计算
    projections_emb = fft(fft2(ifftshift(ifftshift(ifftshift(projections_emb,1),2),3)),[],3);
    % 所有扩展视图的共轭3D OTF（adjoint）卷积结果
    reconstructions = real(ifft2(ifft(projections_emb .* conj(OTFs),[],3)));
    reconstructions = fftshift(fftshift(fftshift(reconstructions,1),2),3);
    % 所有扩展视图平均
    reconstructions = mean(reconstructions,4);
end
end