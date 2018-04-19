import gtsam
import numpy as np
from collections import OrderedDict

data = open('levelingData.txt').readlines()

x = OrderedDict()
l = OrderedDict()

for i in range(len(data)):
	
	if data[i][0:2] == 'xi':
		i+=1
		xString = data[i].split('[')[1].split(']')[0]
		xVals = xString[1:-1].split('), (')
		
		for elem in xVals:
			
			val = elem.split(',')
			x[val[0]] = val[1]

	elif data[i][0:2] == 'l0':
		i+=1
		lString = data[i].split('[')[1].split(']')[0]
		lVals = lString[1:-1].split('), (')

		for elem in lVals:

			val = elem.split(',')
			l[val[0]] = val[1]

parameters = gtsam.ISAM2Params()
parameters.relinearize_threshold = 0.01
parameters.relinearize_skip = 1
isam = gtsam.ISAM2(parameters)			

graph = gtsam.NonlinearFactorGraph()
initialEstimate = gtsam.Values()

priorNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.000, 0.0, 0.0]))
graph.add(gtsam.PriorFactorPose2(1, gtsam.Pose2(0.000, 0.0, 0.0), priorNoise))
initialEstimate.insert(1, gtsam.Pose2(0.0, 0.0, 0.0))

priorNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.000, 0.0, 0.0]))
graph.add(gtsam.PriorFactorPose2(5, gtsam.Pose2(2.034, 0.0, 0.0), priorNoise))
initialEstimate.insert(5, gtsam.Pose2(2.034, 0.0, 0.0))

isam.update(graph,initialEstimate)

initialEstimate.clear()

model = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.002, 0.0, 0.0]))

added = [1,5]

for lkey in l.keys():
	
	if not int(lkey[1]) in added:
		
		initialEstimate.insert(int(lkey[1]), gtsam.Pose2(float(x[lkey[1]]),0.0,0.0))
		added.append(int(lkey[1]))
		
	if not int(lkey[3]) in added:
		
		initialEstimate.insert(int(lkey[3]), gtsam.Pose2(float(x[lkey[3]]),0.0,0.0))
		added.append(int(lkey[3]))
		
	graph.add(gtsam.BetweenFactorPose2(int(lkey[1]), int(lkey[3]), gtsam.Pose2(float(l[lkey]), 0, 0), model))
	isam.update(graph, initialEstimate)
	currentEstimate = isam.calculate_estimate()
	print added
	
	for val in added:
		
		print val
		print currentEstimate.atPose2(val).x()
	
	print "*************************"
	input()
	graph.resize(0)
	initialEstimate.clear()
	
