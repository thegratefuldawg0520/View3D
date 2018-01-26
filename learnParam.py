import tensorflow as tf

#Create a tensorflow session
sess = tf.Session()

#Define the model variables for sift
#nfeatures = tf.Variable([0], dtype=tf.float32)
nOctaveLayers = tf.Variable([3], dtype=tf.float32)
contrastThreshold = tf.Variable([0.04], dtype=tf.float32)
edgeThreshold = tf.Variable([10], dtype=tf.float32)
sigma = tf.Variable([1.6], dtype=tf.float32)

x = tf.placeholder(tf.float32)

#Construct the linear model
linear_model = (nOctaveLayers + contrastThreshold + edgeThreshold + sigma)*x

#Initialize all variables to default values
init = tf.global_variables_initializer()

#Initialize the computation graph
sess.run(init)

#Create a placeholder for the variable y
y = tf.placeholder(tf.float32)
fn = (nOctaveLayers + contrastThreshold + edgeThreshold + sigma)*y

#Define the minimum sum of squares loss function
squared_deltas = tf.square(linear_model - fn)
loss = tf.reduce_sum(squared_deltas)

#Define the gradient descent optimizer and specify it to minimize the loss 
#function
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

for i in range(10):
  sess.run(train, {x: [1, 1, 1, 1], y: [0, 1, 0, 1]})

print(sess.run([nOctaveLayers,contrastThreshold,edgeThreshold,sigma]))

/home/doopy/Documents/View3D/View3D_0_1/Glacier/img/EP-00-00019_0044_0002.JPG

fundamentals.append(fundamental('/home/dennis/Documents/View3D/DJI01435.JPG','/home/dennis/Documents/View3D/DJI01436.JPG',params))