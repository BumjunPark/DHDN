clear;
close all;
clc;

%% Train Set

val_folder_origin = 'E:\data\Kodak\Kodak_origin';

savepath_gt = 'gt_kodak_c.mat';
savepath_noisy = 'noisy_kodak_nl50_c.mat';

nl = 50;

data_val = zeros(512, 768, 3, 24, 'single');
label_val = zeros(512, 768, 3, 24, 'single');

count1 = 0;

list_origin = dir(val_folder_origin);
n = length(list_origin);
for i = 3 : n
    file_origin = strcat(val_folder_origin, '\', list_origin(i).name);    
    
    origin = imread(file_origin);    
    origin = single(origin)/255;
    count1=count1+1;
    [hei, wid, c] = size(origin);
%     if hei>wid
%         origin=imrotate(origin,90);
%     end
%     origin = origin(1:320, 1:480, :);
    label_val(:, :, :, count1) = origin;
    data_val(:, :, :, count1) = origin + nl/255*randn(size(origin));    
    
     display(100*i/(n-2));display('percent complete(val)');
end    

data_val(data_val<0)=0;
data_val(data_val>1)=1;

save(savepath_gt, 'label_val');
save(savepath_noisy, 'data_val');
