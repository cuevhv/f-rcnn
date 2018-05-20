function varargout = label_trees(varargin)
% LABEL_TREES MATLAB code for label_trees.fig
%      LABEL_TREES, by itself, creates a new LABEL_TREES or raises the existing
%      singleton*.
%
%      H = LABEL_TREES returns the handle to a new LABEL_TREES or the handle to
%      the existing singleton*.
%
%      LABEL_TREES('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in LABEL_TREES.M with the given input arguments.
%
%      LABEL_TREES('Property','Value',...) creates a new LABEL_TREES or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before label_trees_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to label_trees_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help label_trees

% Last Modified by GUIDE v2.5 20-May-2018 09:45:19

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @label_trees_OpeningFcn, ...
                   'gui_OutputFcn',  @label_trees_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before label_trees is made visible.
function label_trees_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to label_trees (see VARARGIN)

handles.pth = '';
handles.current_I = [];
handles.imsize = [];
handles.file_name_edit = '';
% Choose default command line output for label_trees
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes label_trees wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = label_trees_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in files_listbox.
function files_listbox_Callback(hObject, eventdata, handles)
% hObject    handle to files_listbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns files_listbox contents as cell array
%        contents{get(hObject,'Value')} returns selected item from files_listbox
contents = cellstr(get(hObject,'String'));
selected_file = contents{get(hObject,'Value')};
file_path = [handles.pth, selected_file];
I = imread(file_path);
handles.imsize = size(I);
handles.current_I = I;
imshow(I)
disp(handles.pth)
guidata(hObject, handles)
splt = strsplit(selected_file,'.');
new_f = ['label_files/',splt{1},'.txt'];
handles.file_name_edit = new_f;

files_lines = {};
count = 1;
fid = fopen(new_f, 'a+');
fclose(fid);
fid = fopen(new_f);
%formatSpec = '%s';
%A = fscanf(fid,formatSpec, [Inf, 1])
line_ex = fgetl(fid);
files_lines{count} = line_ex;
while ischar(line_ex)   
    line_ex = fgetl(fid);
    count = count + 1;
    files_lines{count} = line_ex;
end
%count = 1;
fclose(fid);
handles.drawn_listbox.String = files_lines(1,1:end-1);
disp(files_lines(1,1:end-1))  
%handles.drawn_listbox.String = f_name_c;
guidata(hObject, handles)

if isempty(handles.drawn_listbox.String)
    sze = 0;
else
    sze = size(handles.drawn_listbox.String,1);
end

if sze > 0
    for i = 1:sze
        cur_v = handles.drawn_listbox.String{i};
        splt = strsplit(cur_v,';');
        rec = [str2double(splt{1}), str2double(splt{2}), str2double(splt{3}), str2double(splt{4})];
        rectangle('Position',rec,'EdgeColor','b','LineWidth',3)
        line([str2double(splt{5}), str2double(splt{7})], [str2double(splt{6}), str2double(splt{8})],'Color','r','LineWidth',3)
        line([str2double(splt{5}), str2double(splt{9})], [str2double(splt{6}), str2double(splt{10})],'Color','r','LineWidth',3)
    end
end

% --- Executes during object creation, after setting all properties.
function files_listbox_CreateFcn(hObject, eventdata, handles)
% hObject    handle to files_listbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in load_files_pushbutton.
function load_files_pushbutton_Callback(hObject, eventdata, handles)
% hObject    handle to load_files_pushbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
pth = get(handles.edit1 , 'String');
handles.pth = pth;
pth_content = dir(pth);
n_files = size(pth_content,1);
f_name_c = cell(n_files,1);
for i = 1:n_files
    file_name = pth_content(i).name;
    %disp(file_name)
    f_name_c{i} =  file_name;
end
handles.files_listbox.String = f_name_c;
%handles.drawn_listbox.String = f_name_c;
disp(handles.pth)
guidata(hObject, handles)
if exist('label_files', 'dir') ~= 7
    mkdir label_files
end
    

% --- Executes on selection change in drawn_listbox.
function drawn_listbox_Callback(hObject, eventdata, handles)
% hObject    handle to drawn_listbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns drawn_listbox contents as cell array
%        contents{get(hObject,'Value')} returns selected item from drawn_listbox


% --- Executes during object creation, after setting all properties.
function drawn_listbox_CreateFcn(hObject, eventdata, handles)
% hObject    handle to drawn_listbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in save_pushbutton.
function save_pushbutton_Callback(hObject, eventdata, handles)
% hObject    handle to save_pushbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in rectangle_pushbutton.
function rectangle_pushbutton_Callback(hObject, eventdata, handles)
% hObject    handle to rectangle_pushbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
rect = getrect;
rect = round(rect);
if rect(1) < 1
    if rect(1) < 0
        rect(3) = rect(3)+rect(1)-1;
    else
        rect(3) = rect(3)-(1-rect(1));
    end
    rect(1) = 1;
end
if rect(2) < 1
    if rect(2) < 0
        rect(4) = rect(4)+rect(2)-1;
    else
        rect(4) = rect(4)-(1-rect(2));
    end
    rect(2) = 1;
end

if rect(4) > handles.imsize(1)
    rect(4) = handles.imsize(1);
end
if rect(3) > handles.imsize(2)
    rect(3) = handles.imsize(2);
end

%[x,y] = getpts;
if isempty(handles.drawn_listbox.String)
    sze = 0;
else
    sze = size(handles.drawn_listbox.String,1);
end

rectangle('Position',rect,'EdgeColor','b','LineWidth',3)
[xi,yi] = getpts;
xi = round(xi);
yi = round(yi);
line([xi(1), xi(2)], [yi(1), yi(2)],'Color','r','LineWidth',3)
line([xi(1), xi(3)], [yi(1), yi(3)],'Color','r','LineWidth',3)
handles.drawn_listbox.String{sze+1} = sprintf('%d; %d; %d; %d; %d; %d; %d; %d; %d; %d', rect(1), rect(2), rect(3), rect(4), xi(1), yi(1), xi(2), yi(2), xi(3), yi(3));
disp(handles.imsize)

fid = fopen(handles.file_name_edit, 'wt');
for ii = 1:sze+1
    fprintf(fid, [handles.drawn_listbox.String{ii}, '\n']);
end
fclose(fid);

%handles.rawn_listbox.String = 
% --- Executes on button press in delete_pushbutton.
function delete_pushbutton_Callback(hObject, eventdata, handles)
contents = cellstr(get(handles.drawn_listbox,'String'));
selected_file = contents{get(handles.drawn_listbox,'Value')};
disp(selected_file)
IndexC = strfind(contents, selected_file);
Index = find(not(cellfun('isempty', IndexC)));
contents(Index) = [];
set(handles.drawn_listbox, 'String', contents) 
imshow(handles.current_I)

if isempty(handles.drawn_listbox.String)
    sze = 0;
else
    sze = size(handles.drawn_listbox.String,1);
end

fid = fopen(handles.file_name_edit, 'wt');
if sze > 0
    for i = 1:sze
        cur_v = handles.drawn_listbox.String{i};
        fprintf(fid, [cur_v, '\n']);
        splt = strsplit(cur_v,';');
        rec = [str2double(splt{1}), str2double(splt{2}), str2double(splt{3}), str2double(splt{4})];
        rectangle('Position',rec,'EdgeColor','b','LineWidth',3)
        line([str2double(splt{5}), str2double(splt{7})], [str2double(splt{6}), str2double(splt{8})],'Color','r','LineWidth',3)
        line([str2double(splt{5}), str2double(splt{9})], [str2double(splt{6}), str2double(splt{10})],'Color','r','LineWidth',3)
    end
end

fclose(fid);



% hObject    handle to delete_pushbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
