visualizing_anchors
xp = 8+16*7; 
yp = -8-16*6;
wp = 8;
hp = 8;


pred_x = 0.25*wp+xp;
pred_y = 0.25*hp+yp;
pred_w = exp(0.54)*wp;
pred_h = exp(0.54)*hp;


[pred_x, pred_y, pred_w, pred_h]
h1 = rectangle('Position',[pred_x-pred_w/2 pred_y-pred_h/2 pred_w pred_h], 'EdgeColor', [0 0.5 0], 'LineWidth', 3);


% xp = 8+16*6; 
% yp = -8-16*6;
% wp = 32;
% hp = 32;
% 
% 
% pred_x = 0.25*wp+xp;
% pred_y = 0.25*hp+yp;
% pred_w = exp(0.54)*wp;
% pred_h = exp(0.54)*hp;
% 
% 
% [pred_x, pred_y, pred_w, pred_h]
% h1 = rectangle('Position',[pred_x-pred_w/2 pred_y-pred_h/2 pred_w pred_h], 'EdgeColor', [0 0 0.5], 'LineWidth', 3);