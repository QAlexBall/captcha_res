'''
	Author: QAlexBall
	Description: using sift for captcha
'''
import os
import cv2
import numpy as np

with open('loggings.txt', 'w') as f:
    f.truncate()bvnvbn
with open('mappings_test.txt', 'w') as f:
    f.truncate()
for x in os.listdir('./train/'):
    if x == 'mappings_test.txt':
        break
    print(str(x), end=',')
    maps = open("mappings_test.txt", "a")
    string1 = str(x) + ','
    maps.write(string1)
    for j in range(1, 5):
        imgname2 = './train/' + str(x) + '/' + str(x) + str(j) + '.jpg'
        num_sift = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(0, 9):
            imgname1 = './train/' + str(x) + '/' + str(i) + '.jpg'
            
            ## (1) prepare data
            img1 = cv2.imread(imgname1)
            img2 = cv2.imread(imgname2)
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            ## (2) Create SIFT object
            sift = cv2.xfeatures2d.SIFT_create()
            ## (3) Create flann matcher
            matcher = cv2.FlannBasedMatcher(dict(algorithm = 1, trees = 5), {})
            ## (4) Detect keypoints and compute keypointer descriptors
            kpts1, descs1 = sift.detectAndCompute(gray1, None)
            kpts2, descs2 = sift.detectAndCompute(gray2, None)
            ## (5) knnMatch to get Top2
            matches = matcher.knnMatch(descs1, descs2, 2)
            # Sort by their distance.
            matches = sorted(matches, key = lambda x:x[0].distance)
            
            ## (6) Ratio test, to get good matches.
            good = [m1 for (m1, m2) in matches if m1.distance < 0.75 * m2.distance]
            num_sift[i] = len(good)
    
        # find the max_num's index
        max_sift = num_sift.index(max(num_sift))
        print(max_sift, end='')
        string2 = str(max_sift)
        maps.write(string2)
        f = open("loggings.txt", 'a')
        string = str(x) + str(j)+ str(num_sift) + \
        'max:' + str(max_sift) +'\n'
        f.write(string)
    f.write('\n')
    maps.write('\n')
    print()
maps.close()
f.close()

   
   




