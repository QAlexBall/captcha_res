import os
from PIL import Image
for x in os.listdir('./train/'):
	print(x)
	if x == 'mappings.txt':
		break

	N = 2
	# 缩放图片(原图扩大N倍)
	for i in range(0, 9):
		path = './train/' + str(x) + '/' + str(i) + '.jpg'
		image = Image.open(path)
		size  = (45*N, 45*N)
		image = image.resize(size)
		image.save(path)
	image = Image.open('./train/' + str(x) + '/' + str(x) + '.jpg')
	size = (150*N, 45*N)
	image = image.resize(size)
	image.save('./train/' + str(x) + '/' + str(x) + '.jpg')
	
	# 裁剪样本
	for i in range(1, 5):
		img = Image.open('./train/' + str(x) + '/' + str(x) + '.jpg')
		img_crop = img.crop((0 + ((150*N)/4)*(i - 1), 0, ((150*N)/4)*i, 45*N))
		img_crop.save('./train/' + str(x) + '/' + str(x) + str(i) + '.jpg')


