from scipy.io import loadmat


# CUB200-2011
with open('./CUB_200_2011/images.txt', 'r') as src:
	srclines = src.readlines()

with open('./CUB_200_2011/cub_train.txt', 'w') as tf:
	for line in srclines:
		i, fname = line.strip().split()
		label = int(fname.split('.', 1)[0])
		if label <= 100:
			print('images/{},{}'.format(fname, label-1), file=tf) 
      
with open('./CUB_200_2011/cub_test.txt', 'w') as tf:
	for line in srclines:
		i, fname = line.strip().split()
		label = int(fname.split('.', 1)[0])
		if label > 100:
			print('images/{},{}'.format(fname, label-1), file=tf) 
      

# Cars196
file = loadmat('./CARS196/cars_annos.mat')
annos = file['annotations']

with open('./CARS196/cars_train.txt', 'w') as tf:
    for i in range(16185):
        if annos[0,i][-2] <= 98:
            print('{},{}'.format(annos[0,i][0][0], annos[0,i][-2][0][0]-1), file=tf)

with open('./CARS196/cars_test.txt', 'w') as tf:
    for i in range(16185):
        if annos[0,i][-2] > 98:
            print('{},{}'.format(annos[0,i][0][0], annos[0,i][-2][0][0]-1), file=tf)


# SOP
with open('./SOP/Stanford_Online_Products/Ebay_train.txt', 'r') as src:
    srclines = src.readlines()

with open('./SOP/sop_train.txt', 'w') as tf:
    for i in range(1, len(srclines)):
        line = srclines[i]
        line_split = line.strip().split(' ')
        cls_id = str(int(line_split[1]) - 1)
        img_path = 'Stanford_Online_Products/'+line_split[3]
        print(img_path+','+cls_id, file=tf)

with open('./SOP/Stanford_Online_Products/Ebay_test.txt', 'r') as src:
    srclines = src.readlines()

with open('./SOP/sop_test.txt', 'w') as tf:
    for i in range(1, len(srclines)):
        line = srclines[i]
        line_split = line.strip().split(' ')
        cls_id = str(int(line_split[1]) - 1)
        img_path = 'Stanford_Online_Products/'+line_split[3]
        print(img_path+','+cls_id, file=tf)


# In-Shop
with open('./Inshop/list_eval_partition.txt', 'r') as file_to_read:
    lines = file_to_read.readlines()

with open('./Inshop/inshop_train.txt', 'w') as tf: 
    cls_name2idx = {}
    cls_num = 0
    for line in lines:
        words = line.strip().split()
        if len(words)==3:
            if words[-1]=='train':
                path = words[0]
                cls_name = words[1]
                if cls_name not in cls_name2idx.keys():
                    cls_name2idx[cls_name] = cls_num
                    cls_num += 1
                print('{},{}'.format(path, cls_name2idx[cls_name]), file=tf)

with open('./Inshop/inshop_query.txt', 'w') as tf: 
    test_cls_name2idx = {}
    cls_num = 0
    for line in lines:
        words = line.strip().split()
        if len(words)==3:
            if words[-1]=='query':
                path = words[0]
                cls_name = words[1]
                if cls_name not in test_cls_name2idx.keys():
                    test_cls_name2idx[cls_name] = cls_num
                    cls_num += 1
                print('{},{}'.format(path, test_cls_name2idx[cls_name]), file=tf)

with open('./Inshop/inshop_gallery.txt', 'w') as tf: 
    for line in lines:
        words = line.strip().split()
        if len(words)==3:
            if words[-1]=='gallery':
                path = words[0]
                cls_name = words[1]
                if cls_name not in test_cls_name2idx.keys():
                    print('error!')
                    break
                print('{},{}'.format(path, test_cls_name2idx[cls_name]), file=tf)





