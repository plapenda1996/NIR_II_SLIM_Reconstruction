function cropped_img = crop_image(img, crop_ratio, aspect_ratio)
    % 输入:
    %   - img: 原始图片，可以是灰度图或彩色图
    %   - crop_ratio: 裁剪比例，表示裁剪框面积与原图面积的比例
    %   - aspect_ratio: 裁剪框长宽比，表示宽度与高度的比例（可选，默认与原图保持一致）
    %
    % 输出:
    %   - cropped_img: 裁剪后的图片

    % 获取图片尺寸
    [img_height, img_width, ~] = size(img);

    % 如果未提供 aspect_ratio，则使用原图的长宽比
    if nargin < 3
        aspect_ratio = img_width / img_height;
    end

    % 计算裁剪框的面积
    crop_area = img_height * img_width * crop_ratio;

    % 根据长宽比计算裁剪框的宽度和高度
    crop_height = round(sqrt(crop_area / aspect_ratio));
    crop_width = round(crop_height * aspect_ratio);

    % 确保裁剪框不超出图片范围
    crop_height = min(crop_height, img_height);
    crop_width = min(crop_width, img_width);

    % 计算裁剪框的左上角坐标
    xmin = floor((img_width - crop_width) / 2);
    ymin = floor((img_height - crop_height) / 2);

    % 对图片进行裁剪
    cropped_img = img(ymin+1:ymin+crop_height, xmin+1:xmin+crop_width, :);
end