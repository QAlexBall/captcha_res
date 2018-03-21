import os
N = 2000
f1 = open("mappings_test.txt", 'r')
f2 = open("./train/mappings.txt", 'r')
count = 0
for i in range(0, N):
	if f1.readline() == f2.readline():
		count = count + 1
print(count/N)
f1.close()
f2.close()

f1 = open("mappings_test.txt", 'r')
f2 = open("./train/mappings.txt", 'r')
count = 0
# print(f1.read(5), f1.read(4), f1.read(6), f1.read(4))
# print(f2.read(5), f2.read(4), f2.read(6), f2.read(4))
for i in range(0, N):
	if i == 0:
		f1.read(5)
		f2.read(5)
	else:
		f1.read(6)
		f2.read(6)
	for i in range(0, 4):
		if f1.read(1) == f2.read(1):
			count = count + 1

print(count/(N*4))
f1.close()
f2.close()