function [psf_syn,psf_all_view] = warpPSFs(PSFs, outputSize, backward_rel_list)
% 生成所有深度相关的synthetic PSF集合
% 包括补零，warp
% 输入:
%   PSFs: 原始PSF数据，维度为[height, width, z, views]
%   backward_rel_list: 仿射变换obj，维度为[nViews, nDepths]
%   nDepths: 输出深度数量
%   nViews: 视图数量
% 输出:
%   all_depth_psfs: 所有深度相关的PSF，维度为[height, width, nDepths]

% PSF 补零到outputSize
psfSize = [size(PSFs,1),size(PSFs,2)];
padSize = outputSize - psfSize;
PSFs = padarray(PSFs,padSize,'post');
PSFs = circshift(PSFs, floor(outputSize/2)-floor(psfSize/2));

% 获取PSF尺寸
[nx, ny, nz, nViews] = size(PSFs);
refObj = imref2d([nx,ny]);

% 初始化输出结果
psf_syn = zeros(nx, ny, nz);
psf_all_view = zeros(nx,ny,nz,nViews);
% h= waitbar(0,'Starting warping PSFs...');
% 生成输出PSF的z切片
for z_idx = 1:nz
    % waitbar(z_idx/nz, h, sprintf('Slice %d/%d',z_idx,nz));
    for view_idx = 1:nViews
        % 获取对应的仿射变换
        T = backward_rel_list(view_idx, z_idx);
        % 应用仿射变换到PSF z切片
        psf_tmp = imwarp(PSFs(:, :, z_idx, view_idx), T, 'OutputView', refObj);
        psf_tmp = psf_tmp / sum(psf_tmp(:));
        psf_all_view(:,:,z_idx,view_idx) = psf_tmp;
    end
    % 对视图求平均
    psf_syn(:,:,z_idx) = mean(psf_all_view(:,:,z_idx,:), 4);
end
% close(h)
end