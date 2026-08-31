function J = deconvlucy_gpu(varargin)
%DECONVLUCY_GPU GPU加速的Lucy-Richardson反卷积算法
%   功能与DECONVLUCY相同，但使用GPU加速计算
%
%   J = DECONVLUCY_GPU(I,FORWARD_HANDLE,BACKWARD_HANDLE) 使用传入的卷积函数执行反卷积
%   J = DECONVLUCY_GPU(I,FORWARD_HANDLE,BACKWARD_HANDLE,NUMIT) 指定迭代次数NUMIT
%   J = DECONVLUCY_GPU(I,FORWARD_HANDLE,BACKWARD_HANDLE,NUMIT,DAMPAR) 指定阻尼参数DAMPAR
%   J = DECONVLUCY_GPU(I,FORWARD_HANDLE,BACKWARD_HANDLE,NUMIT,DAMPAR,WEIGHT) 指定权重WEIGHT
%   J = DECONVLUCY_GPU(I,FORWARD_HANDLE,BACKWARD_HANDLE,NUMIT,DAMPAR,WEIGHT,READOUT) 指定读出噪声READOUT
%
%   参数说明:
%   - I: 输入图像
%   - FORWARD_HANDLE/BACKWARD_HANDLE: 前向/伴随卷积函数句柄
%   - NUMIT: 迭代次数，默认为10
%   - DAMPAR: 阻尼参数，默认为0
%   - WEIGHT: 权重图像，默认为全1
%   - READOUT: 读出噪声，默认为0
%

% 解析输入参数
[J, forward_handle, backward_handle, NUMIT, DAMPAR, READOUT, WEIGHT, sizeI, classI] = parse_inputs(varargin{:});

% 将输入数据移至GPU（如果尚未在GPU上）
[J, DAMPAR, READOUT, WEIGHT] = ensure_gpu_arrays(J, DAMPAR, READOUT, WEIGHT);

% 准备迭代参数
wI = max(WEIGHT.*(READOUT + J{1}), 0);

% 计算尺度因子 - 使用forward_handle的伴随模式
scale = backward_handle(WEIGHT) + sqrt(eps);
clear WEIGHT;

DAMPAR22 = (DAMPAR.^2)/2;

% Lucy-Richardson迭代
lambda = 2*any(J{4}(:)~=0);
for k = lambda + (1:NUMIT)   
    % 预测下一次迭代的图像
    if k > 2
        lambda = (J{4}(:,1).'*J{4}(:,2))/(J{4}(:,2).'*J{4}(:,2) + eps);
        lambda = max(min(lambda, 1), 0);
    end
    Y = max(J{2} + lambda*(J{2} - J{3}), 0);
    
    % 计算LR估计的核心部分 - 使用forward前向模式
    CC = corelucy_gpu(Y, forward_handle, DAMPAR22, wI, READOUT);
    
    % 确定下一次迭代的图像并应用正约束 - 使用伴随模式
    J{3} = J{2};
    J{2} = max(Y.*backward_handle(CC)./scale, 0);
    clear CC;
    J{4} = [J{2}(:)-Y(:) J{4}(:,1)];
end

clear wI scale Y;

% 将结果转换回原始图像类型并输出
num = 1 + strcmp(classI{1}, 'notcell');

% 将结果从GPU传回CPU
J{num} = gather(J{num});
if ~strcmp(classI{2}, 'double')
    J{num} = images.internal.changeClass(classI{2}, J{num});
end

if num == 2
    J = J{2};
end

% GPU加速的内核Lucy-Richardson算子
function f = corelucy_gpu(Y, forward_handle, DAMPAR22, wI, READOUT)
    % 使用前向模型计算卷积 - 使用forward_handle的前向模式
    ReBlurred = forward_handle(Y);
    
    % 下一步的估计
    ReBlurred = ReBlurred + READOUT;
    ReBlurred(ReBlurred == 0) = eps;
    AnEstim = wI./ReBlurred + eps;
    
    % 阻尼处理（如果需要）
    if DAMPAR22 == 0
        ImRatio = AnEstim;
    else
        gm = 10;
        g = (wI.*log(complex(AnEstim)) + ReBlurred - wI)./DAMPAR22;
        g = min(g, 1);
        G = (g.^(gm-1)).*(gm-(gm-1)*g);
        ImRatio = 1 + G.*(AnEstim - 1);
    end
    
    % 返回结果
    f = ImRatio;

% 确保数据在GPU上
function [J, DAMPAR, READOUT, WEIGHT] = ensure_gpu_arrays(J, DAMPAR, READOUT, WEIGHT)
    if ~isa(J{1}, 'gpuArray')
        J{1} = gpuArray(J{1});
    end
    if ~isa(J{2}, 'gpuArray')
        J{2} = gpuArray(J{2});
    end
    if ~isa(J{3}, 'gpuArray') && numel(J{3}) > 1
        J{3} = gpuArray(J{3});
    end
    if ~isa(J{4}, 'gpuArray') && numel(J{4}) > 1
        J{4} = gpuArray(J{4});
    end
    if ~isa(DAMPAR, 'gpuArray') && numel(DAMPAR) > 1
        DAMPAR = gpuArray(DAMPAR);
    end
    if ~isa(READOUT, 'gpuArray') && numel(READOUT) > 1
        READOUT = gpuArray(READOUT);
    end
    if ~isa(WEIGHT, 'gpuArray')
        WEIGHT = gpuArray(WEIGHT);
    end

% 输入参数解析函数
function [J, forward_handle, backward_handle, NUMIT, DAMPAR, READOUT, WEIGHT, sizeI, classI] = parse_inputs(varargin)
    % 检查输入参数数量
    narginchk(2, 7);
    
    % 默认参数值
    NUMIT = 10;
    DAMPAR = 0;
    READOUT = 0;
    
    % 处理输入图像
    if iscell(varargin{1})
        classI{1} = 'cell';
        J = varargin{1};
    else
        classI{1} = 'notcell';
        J{1} = varargin{1};
    end
    
    % 检查图像类型
    if isa(J{1}, 'gpuArray')
        origClass = class(gather(J{1}));
        classI{2} = origClass;
    else
        classI{2} = class(J{1});
    end
    
    % 验证图像
    if isa(J{1}, 'gpuArray')
        J1_check = gather(J{1});
    else
        J1_check = J{1};
    end
    validateattributes(J1_check, {'uint8', 'uint16', 'double', 'int16', 'single'}, ...
        {'real', 'nonempty', 'finite'}, mfilename, 'I', 1);
    
    if length(J{1}) < 2
        error('输入图像必须至少有2个元素');
    elseif ~isa(J{1}, 'double')
        if isa(J{1}, 'gpuArray')
            J{1} = im2double(gather(J{1}));
            J{1} = gpuArray(single(J{1}));
        else
            J{1} = im2double(J{1});
        end
    end

    % 获取卷积函数句柄
    forward_handle = varargin{2};
    backward_handle = varargin{3};
    % 设置J单元格数组
    len = length(J);
    if len == 1
        J{2} = max(backward_handle(J{1}), 0);
        J{3} = 0;
        J{4} = zeros(numel(J{2}), 2, 'like', J{1});
    elseif len ~= 4
        error('输入单元格必须有1或4个元素');
    else
        if ~all([isa(J{2}, 'double') || isa(J{2}, 'gpuArray'), ...
                isa(J{3}, 'double') || isa(J{3}, 'gpuArray'), ...
                isa(J{4}, 'double') || isa(J{4}, 'gpuArray')])
            error('输入图像单元格元素必须是double或gpuArray');
        end
    end
    
    % 处理其他数值参数
    if nargin >= 4
        NUMIT = varargin{4};
        % validateattributes(NUMIT, {'double'}, {'scalar', 'positive', 'finite'}, ...
        %     mfilename, 'NUMIT', 4);
    end
    
    if nargin >= 5
        DAMPAR = varargin{5};
        % validateattributes(DAMPAR, {'double'}, {'finite'}, mfilename, 'DAMPAR', 5);
    end
    
    if nargin >= 6
        WEIGHT = varargin{6};
        % validateattributes(WEIGHT, {'double'}, {'finite'}, mfilename, 'WEIGHT', 6);
    else
        WEIGHT = [];
    end
    
    if nargin >= 7
        READOUT = varargin{7};
        % validateattributes(READOUT, {'double'}, {'finite'}, mfilename, 'READOUT', 7);
    end
    
    % 获取图像尺寸
    if isa(J{1}, 'gpuArray')
        sizeI = size(gather(J{1}));
    else
        sizeI = size(J{1});
    end
    
    % 设置默认WEIGHT
    if isempty(WEIGHT)
        if isa(J{1}, 'gpuArray')
            WEIGHT = ones(sizeI, 'gpuArray');
        else
            WEIGHT = ones(sizeI);
        end
    end
    
    % 验证DAMPAR，READOUT和WEIGHT的大小
    if numel(DAMPAR) > 1 && ~isequal(size(DAMPAR), sizeI)
        error('DAMPAR必须与图像尺寸相同或是标量');
    end
    
    if numel(READOUT) > 1 && ~isequal(size(READOUT), sizeI)
        error('READOUT必须与图像尺寸相同或是标量');
    end
    
    if numel(WEIGHT) > 1 && ~isequal(size(WEIGHT), sizeI)
        error('WEIGHT必须与图像尺寸相同或是标量');
    elseif isscalar(WEIGHT)
        WEIGHT = repmat(WEIGHT, sizeI);
    end