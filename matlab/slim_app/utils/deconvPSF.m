function psf = deconvPSF(psf,dia_x,dia_y,iter)
upsample_factor = 2;
patch_size = size(psf);
psf = imresize(psf,upsample_factor);
patch_size_up = size(psf);
% 
% % 计算 psf 的质心
% [x, y] = meshgrid(1:patch_size_up(2), 1:patch_size_up(1));
% total_mass = sum(psf(:));
% center_x = sum(sum(x .* psf)) / total_mass;
% center_y = sum(sum(y .* psf)) / total_mass;
center_x = ceil(patch_size_up(1)/2);
center_y = ceil(patch_size_up(1)/2);
% 生成椭圆 mask
[xx, yy] = meshgrid(1:patch_size_up(2), 1:patch_size_up(1));
mask = double( ((xx - center_x) / (dia_x*upsample_factor / 2)).^2 + ((yy - center_y) / (dia_y*upsample_factor / 2)).^2 <= 1);
psf = deconvlucy(edgetaper(psf, crop_image(mask,0.15)), mask, iter);

psf = imresize(psf,patch_size);
psf = psf/sum(psf,'all');
