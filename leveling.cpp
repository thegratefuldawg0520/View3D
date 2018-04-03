//
// Created by doopy on 15/07/16.
//
#include <fstream>
#include <gtsam/base/FastVector.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/NonlinearConjugateGradientOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>


using namespace std;
using namespace gtsam;
typedef boost::shared_ptr<Factor> sharedFactor;

int main(int argc, char** argv) {

    // Create a factor graph
    NonlinearFactorGraph graph;

    //ifstream fs("/home/doopy/Documents/gtsamSimulationData/datasets/noisyTestData.txt");
    //ifstream noiseFile("/home/doopy/Documents/gtsamSimulationData/datasets/uncertainties.txt");
    ofstream resultsFile("/home/doopy/Documents/View3D/View3D_0_1/gtsam/results.txt");
    // Create the keys we need for this leveling loop
    // x's correspond to the station coordinates
    // l's correspond to height differences between stations
    static Symbol x1('x',1), x2('x',2), x3('x',3), x4('x',4), x5('x',5), x6('x',6), x7('x',7), x8('x',8);

    // Pose2 will be used to model the station coordinates. This is in order to use the between factor to model the
    // height differences. Add a prior on stations x1 and x5. A prior factor consists of a mean and a noise
    // model (covariance matrix)
    Pose2 x1o(0.000, 0.0, 0.0);
    noiseModel::Diagonal::shared_ptr x1Noise = noiseModel::Diagonal::Sigmas(Vector3(0.000, 0.0, 0.0));
    graph.emplace_shared<PriorFactor<Pose2> >(x1, x1o, x1Noise); // add directly to graph

    Pose2 x5o(2.034, 0.0, 0.0);
    noiseModel::Diagonal::shared_ptr x5Noise = noiseModel::Diagonal::Sigmas(Vector3(0.000, 0.0, 0.0));
    graph.emplace_shared<PriorFactor<Pose2> >(x5, x5o, x5Noise); // add directly to graph

    Values initialEstimate;
    initialEstimate.insert(x1, Pose2(0.0, 0.0, 0.0));
    initialEstimate.insert(x2, Pose2(3.2456, 0.0, 0.0));
    initialEstimate.insert(x3, Pose2(1.8443, 0.0, 0.0));
    initialEstimate.insert(x4, Pose2(1.6266, 0.0, 0.0));
    initialEstimate.insert(x5, Pose2(2.034, 0.0, 0.0));
    initialEstimate.insert(x6, Pose2(2.8917, 0.0, 0.0));
    initialEstimate.insert(x7, Pose2(0.7579, 0.0, 0.0));
    initialEstimate.insert(x8, Pose2(0.3261, 0.0, 0.0));

    // Odometry Factors are used to represent height differences in the level network
    Pose2 l1(3.2456, 0.0, 0.0); // create a measurement for both factors (the same in this case)
    noiseModel::Diagonal::shared_ptr l1Noise = noiseModel::Diagonal::Sigmas(Vector3(0.002, 0.000, 0.000)); // 20cm std on x,y, 0.1 rad on theta
    graph.emplace_shared<BetweenFactor<Pose2> >(x1, x2, l1, l1Noise);

    Pose2 l2(-1.4013, 0.0, 0.0); // create a measurement for both factors (the same in this case)
    noiseModel::Diagonal::shared_ptr l2Noise = noiseModel::Diagonal::Sigmas(Vector3(0.002, 0.000, 0.000)); // 20cm std on x,y, 0.1 rad on theta
    graph.emplace_shared<BetweenFactor<Pose2> >(x2, x3, l2, l2Noise);

    Pose2 l3(2.9161, 0.0, 0.0); // create a measurement for both factors (the same in this case)
    noiseModel::Diagonal::shared_ptr l3Noise = noiseModel::Diagonal::Sigmas(Vector3(0.002, 0.000, 0.000)); // 20cm std on x,y, 0.1 rad on theta
    graph.emplace_shared<BetweenFactor<Pose2> >(x8, x2, l3, l3Noise);

    Pose2 l4(0.3261, 0.0, 0.0); // create a measurement for both factors (the same in this case)
    noiseModel::Diagonal::shared_ptr l4Noise = noiseModel::Diagonal::Sigmas(Vector3(0.002, 0.000, 0.000)); // 20cm std on x,y, 0.1 rad on theta
    graph.emplace_shared<BetweenFactor<Pose2> >(x1, x8, l4, l4Noise);

    Pose2 l5(2.5653, 0.0, 0.0); // create a measurement for both factors (the same in this case)
    noiseModel::Diagonal::shared_ptr l5Noise = noiseModel::Diagonal::Sigmas(Vector3(0.002, 0.000, 0.000)); // 20cm std on x,y, 0.1 rad on theta
    graph.emplace_shared<BetweenFactor<Pose2> >(x8, x6, l5, l5Noise);

    Pose2 l6(-0.4302, 0.0, 0.0); // create a measurement for both factors (the same in this case)
    noiseModel::Diagonal::shared_ptr l6Noise = noiseModel::Diagonal::Sigmas(Vector3(0.002, 0.000, 0.000)); // 20cm std on x,y, 0.1 rad on theta
    graph.emplace_shared<BetweenFactor<Pose2> >(x7, x8, l6, l6Noise);

    Pose2 l7(-1.2761, 0.0, 0.0); // create a measurement for both factors (the same in this case)
    noiseModel::Diagonal::shared_ptr l7Noise = noiseModel::Diagonal::Sigmas(Vector3(0.002, 0.000, 0.000)); // 20cm std on x,y, 0.1 rad on theta
    graph.emplace_shared<BetweenFactor<Pose2> >(x5, x7, l7, l7Noise);

    Pose2 l8(-2.1365, 0.0, 0.0); // create a measurement for both factors (the same in this case)
    noiseModel::Diagonal::shared_ptr l8Noise = noiseModel::Diagonal::Sigmas(Vector3(0.002, 0.000, 0.000)); // 20cm std on x,y, 0.1 rad on theta
    graph.emplace_shared<BetweenFactor<Pose2> >(x6, x7, l8, l8Noise);

    Pose2 l9(0.8577, 0.0, 0.0); // create a measurement for both factors (the same in this case)
    noiseModel::Diagonal::shared_ptr l9Noise = noiseModel::Diagonal::Sigmas(Vector3(0.002, 0.000, 0.000)); // 20cm std on x,y, 0.1 rad on theta
    graph.emplace_shared<BetweenFactor<Pose2> >(x5, x6, l9, l9Noise);

    Pose2 l10(0.1929, 0.0, 0.0); // create a measurement for both factors (the same in this case)
    noiseModel::Diagonal::shared_ptr l10Noise = noiseModel::Diagonal::Sigmas(Vector3(0.002, 0.000, 0.000)); // 20cm std on x,y, 0.1 rad on theta
    graph.emplace_shared<BetweenFactor<Pose2> >(x3, x5, l10, l10Noise);

    Pose2 l11(0.4074, 0.0, 0.0); // create a measurement for both factors (the same in this case)
    noiseModel::Diagonal::shared_ptr l11Noise = noiseModel::Diagonal::Sigmas(Vector3(0.002, 0.000, 0.000)); // 20cm std on x,y, 0.1 rad on theta
    graph.emplace_shared<BetweenFactor<Pose2> >(x4, x5, l11, l11Noise);

    Pose2 l12(-0.2153, 0.0, 0.0); // create a measurement for both factors (the same in this case)
    noiseModel::Diagonal::shared_ptr l12Noise = noiseModel::Diagonal::Sigmas(Vector3(0.002, 0.000, 0.000)); // 20cm std on x,y, 0.1 rad on theta
    graph.emplace_shared<BetweenFactor<Pose2> >(x3, x4, l12, l12Noise);

    Pose2 l13(1.0458, 0.0, 0.0); // create a measurement for both factors (the same in this case)
    noiseModel::Diagonal::shared_ptr l13Noise = noiseModel::Diagonal::Sigmas(Vector3(0.002, 0.000, 0.000)); // 20cm std on x,y, 0.1 rad on theta
    graph.emplace_shared<BetweenFactor<Pose2> >(x3, x6, l13, l13Noise);

    Pose2 l14(-0.3506, 0.0, 0.0); // create a measurement for both factors (the same in this case)
    noiseModel::Diagonal::shared_ptr l14Noise = noiseModel::Diagonal::Sigmas(Vector3(0.002, 0.000, 0.000)); // 20cm std on x,y, 0.1 rad on theta
    graph.emplace_shared<BetweenFactor<Pose2> >(x2, x6, l14, l14Noise);

    // Print
    graph.print("Factor Graph:\n");

    // Optimize using Levenberg-Marquardt optimization. The optimizer
    // accepts an optional set of configuration parameters, controlling
    // things like convergence criteria, the type of linear system solver
    // to use, and the amount of information displayed during optimization.
    // Here we will use the default set of parameters.  See the
    // documentation for the full set of parameters.

    const Values result = LevenbergMarquardtOptimizer(graph,initialEstimate).optimize();
    result.print("Final Result LM:\n");

    //for(Values::const_iterator keyi = result.begin(); keyi != result.end(); ++keyi)
    //{
        //const Pose2* p = dynamic_cast<const Pose2*>(&result.at(<Symbol>(&keyi)));
        //cout << "hello" << endl;
    //}
    //const Pose2* p = dynamic_cast<const Pose2*>(&result.at(x1));
    //cout << p->t() << endl;
    //resultsFile << p->t() << " " << p->r().theta() << endl;
    // Calculate and print marginal covariances for all variables
    Marginals marginals(graph, result);
    print(marginals.marginalCovariance(x1), "x1 covariance");
    print(marginals.marginalCovariance(x2), "x2 covariance");
    print(marginals.marginalCovariance(x3), "x3 covariance");
    print(marginals.marginalCovariance(x4), "x4 covariance");
    print(marginals.marginalCovariance(x5), "x5 covariance");
    print(marginals.marginalCovariance(x6), "x6 covariance");
    print(marginals.marginalCovariance(x7), "x7 covariance");
    print(marginals.marginalCovariance(x8), "x8 covariance");

    return 0;
}

