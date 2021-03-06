
rectangle('Position',[0 -224 224 224])
axis([0 250 -250 0])
for i = 0:16:224-16
    for j = 0:16:224
        rectangle('Position',[i -j 16 16], 'EdgeColor', 'r')
    end
end

close all
sprintf('Visualizing grid cell in red')
rectangle('Position',[0 -224 224 224])
%axis([0 250 -250 0])
for i = 0:16:224-16
    c_c = i + 16/2;
    for j = 16:16:224
        rectangle('Position',[i -j 16 16], 'LineStyle', ':')
        c_r = -j + 16/2;
        rectangle('Position',[c_c-1 c_r-1 2 2], 'EdgeColor', 'g')
        rectangle('Position',[c_c c_r 1 1], 'EdgeColor', 'b')
        if j == 112 && i == 112
            for sz = [8, 16, 32]
                if sz == 8
                    col = 'r';
                end
                if sz == 16
                    col = 'b';
                end
                if sz == 32
                    col = 'm';
                end
                rectangle('Position',[c_c-sz/2 c_r-sz/2 sz sz], 'EdgeColor', col)
                rectangle('Position',[c_c-sz*1.5/2 c_r-sz/1.5/2 sz*1.5 sz/1.5], 'EdgeColor', col)
                rectangle('Position',[c_c-sz/1.5/2 c_r-sz*1.5/2 sz/1.5 sz*1.5], 'EdgeColor', col)
                %rectangle('Position',[c_c-8*0.5/2 c_r-8/0.5/2 8*0.5 8/0.5], 'EdgeColor', 'r')
            end
        end
    end
end

