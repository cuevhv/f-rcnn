network = "vgg16"
bbx_size = [8, 16, 32]
bbx_ratio = [0.5, 1, 1.5]
bbx_image = list([[]])
count1 = 0
count2 = 0
if network == "vgg16":
    cell_size = 16
    row_size = 224
    col_size = 224
    centers = []
    for i in range (1, row_size+1, 16):
        c_row = i+cell_size/2-1
        #centers[0].append(c_row)
        bbx_image[count1].append(c_row)
        for j in range (1, col_size+1, 16):
            c_col = j+cell_size/2-1
            for sz in [8, 16, 32]:
                centers.append([c_col, c_row, sz, sz])
                pass
                #print "sz", sz
            #print c_row, c_col, j+cell_size-1
    print(centers)

d = {}
d[(8,8)] = 8
