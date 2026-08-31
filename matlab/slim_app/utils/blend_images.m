function im_blend = blend_images(im,opacity)
if nargin<2
    opacity = 0.7;
end
cm = hsv(size(im,3));
im_blend = zeros(size(im,1),size(im,2),3,size(im,3));
for i=1:size(im,3)
    im_blend(:,:,:,i) = imblend(colorpict([size(im,1),size(im,2)],cm(i,:)),im(:,:,i),1,'multiply');
end
im_blend = mergedown(im_blend,opacity,'screen');
end