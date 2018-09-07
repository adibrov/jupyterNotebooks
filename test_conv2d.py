import tensorflow as tf

batch_size= 10
dimX = 250
dimY = 850

tf.reset_default_graph()

device = "/device:GPU:0"
with tf.device(device):
    inp = tf.ones(shape=[batch_size,dimX,dimY,1])
    d_w = tf.get_variable("kernel_weights", [1, 1, 1, 1],initializer=tf.constant_initializer(1))
    out = tf.nn.conv2d(input=inp, filter=d_w, strides=[1, 1, 1, 1], padding='SAME')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

outA = sess.run(out)
print(outA[batch_size-1,:,:,0].mean())