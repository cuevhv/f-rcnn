function varargout = BasicGUI(varargin)
% BASICGUI MATLAB code for BasicGUI.fig
%      BASICGUI, by itself, creates a new BASICGUI or raises the existing
%      singleton*.
%
%      H = BASICGUI returns the handle to a new BASICGUI or the handle to
%      the existing singleton*.
%
%      BASICGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in BASICGUI.M with the given input arguments.
%
%      BASICGUI('Property','Value',...) creates a new BASICGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before BasicGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to BasicGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help BasicGUI

% Last Modified by GUIDE v2.5 17-May-2018 10:42:25

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @BasicGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @BasicGUI_OutputFcn, ...
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


% --- Executes just before BasicGUI is made visible.
function BasicGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to BasicGUI (see VARARGIN)

handles.peaks = peaks(35);
handles.membrane = membrane;
[x,y] = meshgrid(-8:.5:8);
r = sqrt(x.^2+y.^2)+eps;
sinc = sin(r)./r;
handles.sinc = sinc;
handles.currentData = handles.peaks;
surf(handles.currentData)

% Choose default command line output for BasicGUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes BasicGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = BasicGUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1
str = get(handles.popupmenu1, 'String');
val = get(handles.popupmenu1, 'Value');

%disp(aa{1})
%disp(class(str(val)))
switch str{val}
    case 'peaks'
        handles.currentData = handles.peaks;
    case 'membrane'
        handles.currentData = handles.membrane;
    case 'sinc'
        handles.currentData = handles.sinc;
end

guidata(hObject, handles)



% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in surf_push.
function surf_push_Callback(hObject, eventdata, handles)
% hObject    handle to surf_push (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
surf(handles.currentData);

% --- Executes on button press in mesh_push.
function mesh_push_Callback(hObject, eventdata, handles)
% hObject    handle to mesh_push (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
mesh(handles.currentData);

% --- Executes on button press in contour_push.
function contour_push_Callback(hObject, eventdata, handles)
% hObject    handle to contour_push (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
contour(handles.currentData);
