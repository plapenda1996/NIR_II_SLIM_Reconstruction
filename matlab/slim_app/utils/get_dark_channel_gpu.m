function dark_channel = get_dark_channel_gpu(image, win_size)
    image = single(image);                          % 确保单精度
    [m, n, ch] = size(image);

    if ch == 3                                     % 彩色 → 先取每像素最小 RGB
        image = min(image, [], 3);
    end

    win_size  = max(3, min(win_size, min(m,n)/2));
    pad       = floor(win_size/2);

    padded    = padarray(image,[pad pad],Inf,'both');
    d_in      = gpuArray(padded);
    d_out     = gpuArray.zeros(m, n, 'single');

    k = parallel.gpu.CUDAKernel('dark_channel_kernel.ptx','dark_channel_kernel.cu');
    k.ThreadBlockSize = [16 16 1];
    k.GridSize       = [ceil(m/16) ceil(n/16)];

    d_out = feval(k, d_in, d_out, int32(m), int32(n), int32(win_size), int32(pad));

    dark_channel = gather(d_out);                   % 不要再裁边
end