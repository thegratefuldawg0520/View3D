import numpy as np
from collections import OrderedDict

norm = np.random.normal
outfile = open('levelingData.txt','w')

x = OrderedDict({
	1:0.000,
	2:3.246,
	3:1.843,
	4:1.627,
	5:2.034,
	6:2.891,
	7:0.757,
	8:0.328
})

x0 = {}
xnorms = {}

for i in x.keys():
	
	xnorms[i] = norm(0.0,0.001)
	x0[i] = x[i] + xnorms[i]
	
outfile.write('dx\n')
outfile.write(str(xnorms))

outfile.write('\n\nx0\n')
outfile.write(str(x0))

l = OrderedDict([
	('1-2',x[2] - x[1]),
	('2-3',x[3] - x[2]),
	('3-4',x[4] - x[3]),
	('4-8',x[8] - x[4]),
	('8-1',x[1] - x[8]),
	('1-7',x[7] - x[1]),
	('7-8',x[8] - x[7]),
	('8-6',x[6] - x[8]),
	('6-7',x[7] - x[6]),
	('7-5',x[5] - x[7]),
	('5-6',x[6] - x[5]),
	('6-4',x[4] - x[6]),
	('4-5',x[5] - x[4]),
	('5-2',x[2] - x[5]),
	('2-7',x[7] - x[2]),
	('7-5',x[5] - x[7])
])

outfile.write('\n\nl\n')
outfile.write(str(l))

lnorms = {}
l0 = OrderedDict()

for i in l.keys():
	
	lnorms[i] = norm(0.0,0.002)
	l0[i] = l[i] + lnorms[i]

print l0

xi = OrderedDict({
	1:0.000,
	5:2.034
})

xi[2] = xi[1] + l0['1-2']
xi[3] = xi[2] + l0['2-3']
xi[4] = xi[3] + l0['3-4']
xi[8] = xi[4] + l0['4-8']
xi[6] = xi[8] + l0['8-6']
xi[7] = xi[6] + l0['6-7']


outfile.write('\n\ndl\n')
outfile.write(str(lnorms))

outfile.write('\n\nl0\n')
outfile.write(str(l0))
	
outfile.write('\n\nxi\n')
outfile.write(str(xi))
