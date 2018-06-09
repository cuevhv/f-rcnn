a = dir('label_files/*.txt');
I_dir = dir('/Users/hanzcuevas/Documents/2018/datasets/wageningen_bush/');

len_a = length(a);
for i=1:len_a
    complete_dir = [pwd, '/label_files/', a(i).name];
    I_name = strsplit(a(i).name,'.');
    %disp(I_name{1})
    I = imread(['/Users/hanzcuevas/Documents/2018/datasets/wageningen_bush/', I_name{1},'.jpeg']);
    I_size = size(I);
    I_h = I_size(1); 
    I_w = I_size(2);
    I_d = I_size(3);
    fid = fopen(['VOC_format/',I_name{1},'.xml'], 'wt');
    fprintf(fid, ['<annotation>', '\n']);
    fprintf(fid, sprintf('\t <folder>VOC_format</folder> \n'));
    fprintf(fid, ['\t <filename>',I_name{1},'</filename> \n']);
    fprintf(fid, sprintf('\t <source> \n'));
    fprintf(fid, sprintf('\t \t <database>Royal Garden Edinburgh</database>\n'));
    fprintf(fid, sprintf('\t </source> \n'));
    fprintf(fid, sprintf('\t <size> \n'));
    fprintf(fid, sprintf('\t \t <width>%d</width> \n',I_w));
    fprintf(fid, sprintf('\t \t <height>%d</height> \n',I_h));
    fprintf(fid, sprintf('\t \t <depth>%d</depth> \n',I_d));
    fprintf(fid, sprintf('\t </size> \n'));
    fprintf(fid, sprintf('\t <segmented>0</segmented> \n'));
    wid = fopen(['label_files/',I_name{1},'.txt'], 'r');
    tline = fgetl(wid);
    count = 1;
    while ischar(tline)
        lbl_val = strsplit(tline, ';');
        fprintf(fid, sprintf('\t <object> \n'));
        fprintf(fid, sprintf('\t \t <name>branch_%d</name> \n',count));
        fprintf(fid, sprintf('\t \t <pose>Unspecified</pose> \n'));
        fprintf(fid, sprintf('\t \t <truncated>0</truncated> \n'));
        fprintf(fid, sprintf('\t \t <difficult>0</difficult> \n'));
        fprintf(fid, sprintf('\t \t <bndbox> \n'));
        fprintf(fid, sprintf('\t \t \t <xmin>%s</xmin> \n',lbl_val{1}));
        fprintf(fid, sprintf('\t \t \t <ymin>%s</ymin> \n',lbl_val{2}));
        fprintf(fid, sprintf('\t \t \t <xmax>%d</xmax> \n',str2double(lbl_val{1})+str2double(lbl_val{3})));
        fprintf(fid, sprintf('\t \t \t <ymax>%d</ymax> \n',str2double(lbl_val{2})+str2double(lbl_val{4})));
        fprintf(fid, sprintf('\t \t </bndbox> \n'));
        fprintf(fid, sprintf('\t <object> \n'));
        fprintf(fid, sprintf('\t \t <vectors> \n'));
        fprintf(fid, sprintf('\t \t \t <direction_vector> \n'));
        fprintf(fid, sprintf('\t \t \t \t <x_start>%d</x_start> \n',str2double(lbl_val{5})));
        fprintf(fid, sprintf('\t \t \t \t <y_start>%d</y_start> \n',str2double(lbl_val{6})));
        fprintf(fid, sprintf('\t \t \t \t <x_end>%d</x_end> \n',str2double(lbl_val{7})));
        fprintf(fid, sprintf('\t \t \t \t <y_end>%d</y_end> \n',str2double(lbl_val{8})));
        fprintf(fid, sprintf('\t \t \t </direction_vector> \n'));
        fprintf(fid, sprintf('\t \t \t <radius_vector> \n'));
        fprintf(fid, sprintf('\t \t \t \t <x_start>%d</x_start> \n',str2double(lbl_val{5})));
        fprintf(fid, sprintf('\t \t \t \t <y_start>%d</y_start> \n',str2double(lbl_val{6})));
        fprintf(fid, sprintf('\t \t \t \t <x_end>%d</x_end> \n',str2double(lbl_val{9})));
        fprintf(fid, sprintf('\t \t \t \t <y_end>%d</y_end> \n',str2double(lbl_val{10})));
        fprintf(fid, sprintf('\t \t \t </radius_vectors> \n'));
        fprintf(fid, sprintf('\t \t </vectors> \n'));
        disp(tline)
        tline = fgetl(wid);
        count = count + 1;
    end
    
    fprintf(fid, ['</annotation>', '\n']);
    fclose(fid);
end