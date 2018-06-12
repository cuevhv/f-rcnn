a = dir('label_files/*.txt');
I_dir = dir('/home/hanz/Documents/2018_lib/datasets/branches_botanic_garden/');

len_a = length(a);
for i=1:len_a
    complete_dir = [pwd, '/label_files/', a(i).name];
    I_name = strsplit(a(i).name,'.');
    %disp(I_name{1})
    I = imread(['/home/hanz/Documents/2018_lib/datasets/branches_botanic_garden/', I_name{1},'.jpeg']);
    imwrite(I,sprintf('YOLO_format/%s.png',I_name{1}),'png');
    I_size = size(I);
    I_h = I_size(1); 
    I_w = I_size(2);
    I_d = I_size(3);
    fid = fopen(['YOLO_format/',I_name{1},'.txt'], 'wt');
    wid = fopen(['label_files/',I_name{1},'.txt'], 'r');
    tline = fgetl(wid);
    count = 1;
    while ischar(tline)
        lbl_val = strsplit(tline, ';');
        absolute_x = abs(str2double(lbl_val{1}));
        absolute_y = abs(str2double(lbl_val{2}));
        absolute_w = abs(str2double(lbl_val{3}));
        absolute_h = abs(str2double(lbl_val{4}));
        
        lbl_val = strsplit(tline, ';');
        fprintf(fid, sprintf('0 %.6f %.6f %.6f %.6f\n',(absolute_x-absolute_w)/2*I_w, (absolute_y-absolute_h)/2*/I_h, absolute_w/I_w, absolute_h/I_h));
        disp(sprintf('0 %.6f %.6f %.6f %.6f\n',absolute_x/I_w, absolute_y/I_h, absolute_w/I_w, absolute_h/I_h))
        tline = fgetl(wid);
        count = count + 1;
    end
    fclose(fid);
end
