function addScaleBar(ax, scaleLength_px, labelText, fontSize, barHeight_rel, position_rel)
% addScaleBar(ax, scaleLength_px, labelText, fontSize, barHeight_rel, position_rel)
%
% 在指定坐标轴 (ax) 内绘制比例尺。
%
% 参数:
%   ax            : 句柄, 目标坐标轴
%   scaleLength_px: 数值, 比例尺的长度 (像素单位)
%   labelText     : 字符串, 比例尺标签 (例如 '50 px')
%   fontSize      : 数值, 标签字体大小 (可选, 默认 12)
%   barHeight_rel : 数值, 比例尺的高度 (坐标轴高度比例, 默认 0.01)
%   position_rel  : 1x2 数组, 比例尺左下角位置 [x, y] (归一化坐标)
%
% 示例:  
%   figure; ax = axes; imagesc(ax, rand(200, 300)); axis image;
%   addScaleBar(ax, 50, '50 px');
%
%   addScaleBar(ax, 100, '100 px', 12, 0.02, [0.8, 0.1]);

    % 参数默认值
    if nargin < 6, position_rel = [0.6, 0.7]; end
    if nargin < 5, barHeight_rel = 0.03; end
    if nargin < 4, fontSize = 12; end

    % 确保输入的 ax 是有效的坐标轴句柄
    if ~isgraphics(ax, 'axes')
        error('Input ax must be a valid axes handle.');
    end

    hold(ax, 'on'); % 保持当前坐标轴内容

    % 获取坐标轴的 X 和 Y 范围
    x_limits = ax.XLim;
    y_limits = ax.YLim;

    % 图像宽度和高度
    imageWidth_px = diff(x_limits); % X 方向的像素范围
    barHeight = barHeight_rel * diff(y_limits); % 相对坐标系的高度

    % 将 scaleLength_px 转换为坐标轴的数据单位
    scaleLength_data = scaleLength_px / imageWidth_px * diff(x_limits);

    % 计算比例尺的起始和结束位置
    x_start = x_limits(1) + position_rel(1) * diff(x_limits); % 左下角 x 坐标
    y_start = y_limits(1) + position_rel(2) * diff(y_limits); % 左下角 y 坐标
    x_end = x_start + scaleLength_data;

    % 绘制比例尺（矩形）
    fill(ax, [x_start, x_end, x_end, x_start], ...
              [y_start, y_start, y_start + barHeight, y_start + barHeight], ...
              'r', 'EdgeColor', 'none'); % 黑色矩形

    % 添加标签（居中）
    text(ax, x_start + scaleLength_data / 2, y_start - barHeight * 5, labelText, ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
         'FontSize', fontSize, 'Color', 'r');

    hold(ax, 'off'); % 释放 hold
end
