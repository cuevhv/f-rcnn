
rectangle('Position',[0 -224 224 224])
axis([0 250 -250 0])
for i = 0:16:224-16
    for j = 0:16:224
        rectangle('Position',[i -j 16 16], 'EdgeColor', 'r')
    end
end
