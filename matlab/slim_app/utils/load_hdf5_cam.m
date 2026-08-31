function image_stack = load_hdf5_cam(folderpath,filename)
fullname = fullfile(folderpath, filename);
info = h5info(fullname);
image_datasets = {info.Datasets.Name};
image_datasets = image_datasets(startsWith(image_datasets, 'Image_') & ~endsWith(image_datasets, '_Metadata'));
numbers = cellfun(@(x) str2double(x(7:end)), image_datasets);
[~, sorted_indices] = sort(numbers);
sorted_image_datasets = image_datasets(sorted_indices);
num_images = numel(sorted_image_datasets);
image_stack = cell(1, num_images);
offset = 400;
for i = 1:num_images
    dataset_name = ['/' sorted_image_datasets{i}];  % 添加前导斜杠
    im = h5read(fullname, dataset_name);
    % offset
    im = im + offset;
    im(im<0) = 0;
    % crop edges
    im(1,:,:) = im(2,:,:);
    im(end,:,:) = im(end-1,:,:);
    im(:,1,:) = im(:,2,:);
    im(:,end,:) = im(:,end-1,:);

    image_stack{i} = im';
end
image_stack = cat(3, image_stack{:});

end