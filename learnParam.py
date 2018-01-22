import tensorflow as tf

#Create a tensorflow session
sess = tf.Session()

#Define the model variables for sift
nfeatures = tf.Variable([0], dtype=tf.float32)
nOctaveLayers = tf.Variable([3], dtype=tf.float32)
contrastThreshold = tf.Variable([0.04], dtype=tf.float32)
edgeThreshold = tf.Variable([10], dtype=tf.int8)
sigma = tf.Variable([1.6], dtype=tf.float32)

tf.placeholder(tf.float32)

#Construct the linear model
linear_model = W*x + b

#Initialize all variables to default values
init = tf.global_variables_initializer()

#Initialize the computation graph
sess.run(init)

#Create a placeholder for the variable y
y = tf.placeholder(tf.float32)

#Define the minimum sum of squares loss function
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

#Define the gradient descent optimizer and specify it to minimize the loss 
#function
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

for i in range(1000):
  sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))

