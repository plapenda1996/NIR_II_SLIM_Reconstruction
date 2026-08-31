function [PSF_interp,V_interp] = interpolate_PSFs(PSFs, lambda, pca_k)
% range: [0 1] for original
    % 获取patch尺寸
    [patchHeight, patchWidth, num_frame, num_view] = size(PSFs);
    num_depth = length(lambda);
    
    % 初始化重建的PSF
    PSF_interp = zeros(patchHeight, patchWidth, num_depth, num_view);
    
    % 对每个视角进行处理
    V_interp = zeros(num_view, num_depth, pca_k);
    for i = 1:num_view
        % 进行PCA分解
        [U, S, V] = svd(reshape(PSFs(:,:,:,i), [], size(PSFs, 3)), 'econ');
        V_pca = V(:, 1:pca_k);
        S_pca = S(1:pca_k, 1:pca_k);
        U_pca = U(:, 1:pca_k);

        % 进行插值
        for j = 1:pca_k
            V_interp(i,:,j) = interp1(linspace(0, 1, num_frame), V_pca(:, j), lambda, 'spline', 'extrap');
        end

        % 重建PSF
        X_recon = real(U_pca * S_pca * squeeze(V_interp(i,:,:))');
        X_recon(X_recon < 0) = 0;
        PSF_interp(:,:,:,i) = reshape(X_recon, patchHeight, patchWidth, num_depth);
        PSF_interp(:,:,:,i) = PSF_interp(:,:,:,i) / sum(PSF_interp(:,:,:,i), 'all');
    end
end