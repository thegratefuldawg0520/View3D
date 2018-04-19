from __future__ import print_function
import gtsam as gt
import numpy as np

def Vector3(x, y, z): return np.array([x, y, z])

graph = gt.NonlinearFactorGraph()


priorNoise = gt.noiseModel.Diagonal.Sigmas(np.array([0.000, 0.0, 0.0]))
graph.add(gt.PriorFactorPose2(1, gt.Pose2(0.000, 0.0, 0.0), priorNoise))

priorNoise = gt.noiseModel.Diagonal.Sigmas(np.array([0.000, 0.0, 0.0]))
graph.add(gt.PriorFactorPose2(5, gt.Pose2(2.034, 0.0, 0.0), priorNoise))

model = gt.noiseModel.Diagonal.Sigmas(Vector3(0.002, 0.0, 0.0))

graph.add(gt.BetweenFactorPose2(1, 2, gt.Pose2(3.2456, 0.0, 0.0), model))
graph.add(gt.BetweenFactorPose2(2, 3, gt.Pose2(-1.4013, 0.0, 0.0), model))
graph.add(gt.BetweenFactorPose2(3, 5, gt.Pose2(0.1929, 0.0, 0.0), model))
graph.add(gt.BetweenFactorPose2(5, 7, gt.Pose2(-1.2761, 0.0, 0.0), model))
graph.add(gt.BetweenFactorPose2(7, 8, gt.Pose2(-0.4302, 0.0, 0.0), model))
graph.add(gt.BetweenFactorPose2(8, 2, gt.Pose2(2.9161, 0.0, 0.0), model))
graph.add(gt.BetweenFactorPose2(2, 6, gt.Pose2(-0.3506, 0.0, 0.0), model))
graph.add(gt.BetweenFactorPose2(6, 7, gt.Pose2(-2.1365, 0.0, 0.0), model))
graph.add(gt.BetweenFactorPose2(1, 8, gt.Pose2(0.3261, 0.0, 0.0), model))
graph.add(gt.BetweenFactorPose2(8, 6, gt.Pose2(2.5653, 0.0, 0.0), model))
graph.add(gt.BetweenFactorPose2(5, 6, gt.Pose2(0.8577, 0.0, 0.0), model))
graph.add(gt.BetweenFactorPose2(4, 5, gt.Pose2(0.4074, 0.0, 0.0), model))
graph.add(gt.BetweenFactorPose2(3, 4, gt.Pose2(-0.2153, 0.0, 0.0), model))
graph.add(gt.BetweenFactorPose2(3, 6, gt.Pose2(1.0458, 0.0, 0.0), model))

graph.print("\nFactor Graph\n")

initialEstimate = gt.Values()
initialEstimate.insert(1, gt.Pose2(0.0, 0.0, 0.0))
initialEstimate.insert(2, gt.Pose2(3.2456, 0.0, 0.0))
initialEstimate.insert(3, gt.Pose2(1.8443, 0.0, 0.0))
initialEstimate.insert(4, gt.Pose2(1.6266, 0.0, 0.0))
initialEstimate.insert(5, gt.Pose2(2.034, 0.0, 0.0))
initialEstimate.insert(6, gt.Pose2(2.8917, 0.0, 0.0))
initialEstimate.insert(7, gt.Pose2(0.7579, 0.0, 0.0))
initialEstimate.insert(8, gt.Pose2(0.3261, 0.0, 0.0))
initialEstimate.print("\nInitial Estimate:\n")  # print

parameters = gt.LevenbergMarquardtParams()
parameters.relativeErrorTol = 1e-5
parameters.maxIterations = 100
optimizer = gt.LevenbergMarquardtOptimizer(graph, initialEstimate, parameters)
result = optimizer.optimize()
result.print("Final Result:\n")

