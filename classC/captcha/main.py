import os
test_class = 'data3.pyc'
print('please enter image path:')
image_path = input()
cmd = 'python ' + str(test_class) + ' ' + str(image_path)
os.system(cmd)

