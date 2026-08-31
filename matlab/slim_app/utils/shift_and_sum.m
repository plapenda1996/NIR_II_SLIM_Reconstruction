function [final_mean, final_variance ] = shift_and_sum(im_data, backward_list, batch_size)
%WARP_AND_AVERAGE_ITERATIVE 利用GPU加速对图像数据进行仿射变换，并计算样本均值和样本方差
%
%   [final_mean, final_variance] = warp_and_average_iterative(im_data, backward_list, nDepths)
%
%   输入:
%       im_data       - 3D或4D图像数据
%                       3D: [x, y, nViews]
%                       4D: [x, y, nSamples, nViews]，其中样本维度为第三维度
%       backward_list - 仿射变换列表，尺寸为 [nViews x nDepths]，
%                       注意：若为 cell 数组，则用 backward_list{i,j} 索引
%       nDepths       - 深度层数，即需要生成的变换结果数量
%
%   输出:
%       final_mean     - 变换结果的样本均值（归一化后，除以整体最大值）
%       final_variance - 变换结果的样本方差（归一化后）
%
%   说明:
%       对于 3D 输入，直接计算 ssResult 后归一化，输出均值；样本方差设为全零。
%       对于 4D 输入，采用迭代方法处理每个样本，避免占用过大内存，同时利用 online 算法计算逐像素的均值和方差。
%

dims = ndims(im_data);


% 设置批次大小，依据 GPU 内存情况进行调整
if nargin<3
    batch_size = 10;  % 可根据需要调整
end

if dims == 3
    % 3D情况：im_data尺寸为 [x, y, nViews]
    nViews = size(backward_list, 1);
    nDepths = size(backward_list, 2);
    if ~isa(im_data, 'gpuArray')
        im_data = gpuArray(im_data);
    end
    
    ssResult = zeros([size(im_data(:,:,1)), nDepths], 'like', im_data);
    for j = 1:nDepths
        im_sum = zeros([size(im_data(:,:,1)), nViews], 'like', im_data);
        for i = 1:nViews
            im_mov = im_data(:,:,i);
            % 若 backward_list 为 cell 数组，则使用 backward_list{i,j}
            im_sum(:,:,i) = imwarp(im_mov, backward_list(i,j), 'OutputView', imref2d(size(im_mov)));
        end
        ssResult(:,:,j) = mean(im_sum, 3);
    end
    % 归一化处理
    ssResult = ssResult / max(ssResult(:));
    final_mean = gather(ssResult);
    final_variance = zeros(size(final_mean), 'like', final_mean);
    
elseif dims == 4
    % 4D情况：im_data尺寸为 [x, y, nSamples, nViews]
    nSamples = size(im_data, 3);
    nViews = size(backward_list, 1);
    nDepths = size(backward_list, 2);
    
    if ~isa(im_data, 'gpuArray')
        im_data = gpuArray(im_data);
    end

    % 获取图像尺寸
    [nx, ny, ~, ~] = size(im_data);

    % 初始化迭代统计量：running_mean 和 M2，尺寸为 [nx, ny, nDepths]
    count = 0;
    running_mean = zeros(nx, ny, nDepths, 'like', im_data);
    M2 = zeros(nx, ny, nDepths, 'like', im_data);

    h = waitbar(0, 'Processing samples...');
    % 分批处理样本
    for batch_start = 1:batch_size:nSamples
        batch_end = min(batch_start + batch_size - 1, nSamples);
        curBatchSize = batch_end - batch_start + 1;

        % 对当前批次，预分配 ss_batch，尺寸为 [nx, ny, nDepths, curBatchSize]
        ss_batch = zeros(nx, ny, nDepths, curBatchSize, 'like', im_data);

        % 对每个深度 j，批量处理所有当前批次样本
        for j = 1:nDepths
            warped_sum_batch = zeros(nx, ny, curBatchSize, 'like', im_data);
            for i = 1:nViews
                % 提取当前批次在视角 i 的图像，尺寸为 [nx, ny, curBatchSize]
                im_stack = im_data(:, :, batch_start:batch_end, i);
                % 利用 imwarp 对当前批次所有样本进行相同的 2-D 变换
                warped_stack = imwarp(im_stack, backward_list(i, j), 'OutputView', imref2d([nx, ny]));
                warped_sum_batch = warped_sum_batch + warped_stack;
            end
            % 对所有视角取平均
            ss_batch(:, :, j, :) = warped_sum_batch / nViews;
        end

        % 对当前批次中的每个样本进行迭代更新统计量
        for b = 1:curBatchSize
            sample_ss = ss_batch(:, :, :, b);  % 尺寸 [nx, ny, nDepths]
            count = count + 1;
            if count == 1
                running_mean = sample_ss;
            else
                delta = sample_ss - running_mean;
                running_mean = running_mean + delta / count;
                M2 = M2 + delta .* (sample_ss - running_mean);
            end
        end
        wait(gpuDevice);
        waitbar(batch_end / nSamples, h, sprintf('%d /%d Samples',batch_end,nSamples));
    end
    

    % 计算无偏样本方差（当 count>1 时）
    if count > 1
        running_variance = M2 / (count - 1);
    else
        running_variance = zeros(size(running_mean), 'like', running_mean);
    end

    % 归一化处理：使用 running_mean 全局最大值作为归一化因子
    norm_factor = max(running_mean(:));
    final_mean = gather(running_mean / norm_factor);
    final_variance = gather(running_variance / norm_factor);
    wait(gpuDevice);
    close(h)
else
    error('im_data 必须为 3D 或 4D 数组.');
end

end
