function tform_interp = interp_transformations(tform_list, lambda, method)
    % tform_list: 输入的 transformation 矩阵列表
    % lambda: 归一化深度值
    % method: 插值方法
    if nargin<3
        method = 'spline';
    end
    
    num_view = size(tform_list, 1);
    num_frame = size(tform_list, 2);
    num_depth = length(lambda);
    xi = linspace(0,1,num_frame);
    tform_interp = repmat(affinetform2d(),num_view,num_depth);

    for i = 1:num_view
        % 提取 transformation 矩阵的元素
        elements = zeros(num_frame, 9);
        for j = 1:num_frame
            elements(j, :) = tform_list(i, j).A(:);
        end
        % interpolate matrix elements
        elements_interp = zeros(num_depth, 9);
        for j=1:9
            elements_interp(:,j) = interp1(xi,elements(:,j),lambda,method,'extrap');
        end
        
        % Construct the matrix of independent variables
        for j=1:num_depth
            % Reconstruct the transformation matrices
            A = reshape(elements_interp(j,:), 3, 3);
            % cascade init tforms
            tform_interp(i,j) = affinetform2d(A);
        end
    end
end