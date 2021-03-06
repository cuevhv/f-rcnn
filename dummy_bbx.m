close all
rectangle('Position',[0 0 10 10])
%BBox_target
x = 3;
y = 3;
w = 2;
h = 2;

x = 6;
y = 4;
w = 8;
h = 4;
rectangle('Position',[x-w/2 y-h/2 w h], 'EdgeColor', 'b')


%BBox_predict
xp = 4;
yp = 4;
wp = 6;
hp = 6;
rectangle('Position',[xp-wp/2 yp-hp/2 wp hp], 'EdgeColor', 'g')

%% Center 
%axis([0 2 0 2])
vx = 0
vy = 0
fx = 1/2.2361
fy = 2/2.2361
line([vx + x, fx + x], [vy + y, fy + y], 'Color', 'b')

% predicted vector
vxp = 0
vyp = 0
fxp = 1
fyp = 1
line([vxp + xp, fxp + xp], [vyp + yp, fyp + yp], 'Color', 'g')

%% NOTE
% They use the difference (pred_bbx, anchor)
% and also the difference (ground_bbx, anchor)
%%
t_xp = (xp-xp)/wp;
t_x = (x-xp)/wp;
t_yp = (yp-yp)/hp;
t_y = (y-yp)/hp;
t_wp = log(wp/wp);
t_w = log(w/wp);
t_hp = log(hp/hp);
t_h = log(h/hp);

t_vxp = (fxp-fxp)/wp;
t_vx = (fx-fxp)/wp;
t_vyp = (fyp-fyp)/hp;
t_vy = (fy-fyp)/hp;

pred_t = [t_xp, t_yp, t_wp, t_hp, t_vxp, t_vyp];
%pred_t = [0.52, 0.32, 0.2, 0.5];
pause(2)
%%
for k = 1:200
    ls = [pred_t(1)-t_x, pred_t(2)-t_y, pred_t(3)-t_w, pred_t(4)-t_h, pred_t(5)-t_vx, pred_t(6)-t_vy];
    for i = 1:length(pred_t)
        if abs(ls(i)) < 1
            dif = 2*ls(i);
        else
            if ls(i) > 0
                dif = ls(i)/abs(ls(i));
            else
                dif = 0;
            end
        end
        pred_t(i) = pred_t(i) -0.1*dif;
    end
    pred_x = pred_t(1)*wp+xp;
    pred_y = pred_t(2)*hp+yp;
    pred_w = exp(pred_t(3))*wp;
    pred_h = exp(pred_t(4))*hp;
    
    pred_vx = pred_t(5)*wp+fxp;
    pred_vy = pred_t(6)*hp+fyp;
    [pred_x, pred_y, pred_w, pred_h pred_vx, pred_vy]
    h1 = rectangle('Position',[pred_x-pred_w/2 pred_y-pred_h/2 pred_w pred_h], 'EdgeColor', 'r');
    h2 = line([pred_x, pred_vx + pred_x], [pred_y, pred_vy + pred_y], 'Color', 'r');
    pause(0.3)
    delete(h1)
    delete(h2)
    
end

%% This works

% dif = 0;
% for k = 1:5
%     ls = t_xp-t_x
%     if abs(ls) < 1
%         dif = 2*ls;
%     else
%         if ls >0
%             dif = ls/abs(ls);
%         else
%             dif = 0;
%         end
%     end
%     t_xp = t_xp - 0.1*dif
%     t_x = t_x
%     predicted_x = t_xp*wp+xp
% end

