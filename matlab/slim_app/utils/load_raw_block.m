function im_stack = load_raw_block(folderpath, filename)
    fullname = fullfile(folderpath, filename);
    fileInfo = dir(fullname);
    fileSize = fileInfo.bytes;
    
    % Replace these with image's actual specifications
    width = 640;        % Image width
    height = 512;       % Image height
    numChannels = 1;
    bytesPerPixel = 2;  % for uint16
    dataType = 'int16';
    
    % Calculate number of images
    pixelsPerImage = width * height * numChannels;
    bytesPerImage = pixelsPerImage * bytesPerPixel;
    numImages = fileSize / bytesPerImage;
    chunkSize = 2000;   % Process 2000 images at a time
    
    % Pre-allocate output array
    im_stack = zeros(height, width, numImages, dataType);
    
    % Initialize progress bar
    totalChunks = ceil(numImages / chunkSize);
    if totalChunks>1
        h = waitbar(0, sprintf('Loading 0 of %d images...', numImages), 'Name', 'Image Loader');
        startTime = tic;
    end
    
    for i = 1:totalChunks
        chunkStart = (i-1) * chunkSize + 1;
        chunkEnd = min(chunkStart + chunkSize - 1, numImages);
        currentChunkSize = chunkEnd - chunkStart + 1;
        
        % Load current chunk
        im_stack(:,:,chunkStart:chunkEnd) = load_raw_cam(folderpath, filename, chunkStart, currentChunkSize);
        
        if totalChunks>1
            % Update progress bar
            progress = i / totalChunks;
            elapsed = toc(startTime);

            if i > 1
                remaining = elapsed * (1 - progress) / progress;
                msg = sprintf('Loading %d-%d of %d images (%.0f sec remaining)', ...
                    chunkStart, chunkEnd, numImages, remaining);
            else
                msg = sprintf('Loading %d-%d of %d images', ...
                    chunkStart, chunkEnd, numImages);
            end
            waitbar(progress, h, msg);
        end
    end
    
    % Close progress bar
    if totalChunks>1
        close(h);
    end
end