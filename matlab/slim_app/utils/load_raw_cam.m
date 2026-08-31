function im = load_raw_cam(folderpath,filename,startIndex, numImages)
if nargin<3
    startIndex = 1;
end
% 获取文件信息
fullname = fullfile(folderpath, filename);
fileInfo = dir(fullname);
fileSize = fileInfo.bytes;

% Replace these with image's actual specifications
width = 640;    % Image width
height = 512;    % Image height
numChannels = 1; 
bytesPerPixel = 2; % for uint16

% 计算图片数量
pixelsPerImage = width * height * numChannels;
bytesPerImage = pixelsPerImage * bytesPerPixel;
if nargin<4
    numImages = fileSize / bytesPerImage;
end
startOffset = (startIndex - 1) * bytesPerImage;

% 读取所有数据
totalPixels = pixelsPerImage * numImages;
fileID = fopen(fullname, 'r');
fseek(fileID, startOffset, 'bof');
% 读取大块数据
im = fread(fileID, totalPixels, 'int16=>int16');
fclose(fileID);

% 重塑为3D数组 [height, width, numImages]
im = reshape(im, [width, height, numImages]);
im = permute(im, [2, 1, 3]); % 转置每张图片

% 处理（如果需要）
offset = 400;
im = im + offset;
im(im < 0) = 0;

% 处理边缘（对每张图片）
for i = 1:numImages
    im(1,:,i) = im(2,:,i);
    im(end,:,i) = im(end-1,:,i);
    im(:,1,i) = im(:,2,i);
    im(:,end,i) = im(:,end-1,i);
end