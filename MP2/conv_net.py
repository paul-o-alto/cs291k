import tensorflow as tf

from data_utils import load_CIFAR100

X_train, y_train, X_test, y_test = load_CIFAR100('./cifar-100-batches-py/')

# This stuff probably needs to go elsewhere
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_val   -= mean_image
X_test  -= mean_image
X_train = X_train.reshape(num_training, -1)
X_val   = X_val.reshape(num_training,   -1)
X_test  = X_test.reshape(num_training,  -1)

images_placeholder = tf.placeholder(tf.float32, shape=(None,input_size), name='images')
labels_placeholder = tf.placeholder(tf.int64, shape=(None), name='labels')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, 
                        strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Organize 1st-layer
with tf.name_scope('firstlayer-conv'): 
    """ MP1 Hidden layer (for reference)
    h_w = tf.Variable(tf.truncated_normal([input_size, hidden_size], 
                                          stddev=1.0/math.sqrt(float(input_size))), 
                      name='hidden_weights')
    h_b = tf.Variable(tf.zeros([hidden_size]), 
                      name='hidden_biases')
    hidden = tf.nn.relu(tf.matmul(images_placeholder, h_w) + h_b)
    """

    W_convl = tf.Variable(tf.truncated_normal(shape=[5,5,3,4], stddev=1e-4))
    b_convl = tf.constant(0.1, [4]
    h_convl = tf.nn.relu(conv2d(x_image, W_convl) + b_convl)


with tf.name_scope('secondlayer-maxpool')
    """MP1 Softmax linear (for reference) 
    s_w = tf.Variable(tf.truncated_normal([hidden_size, output_size], 
                                          stddev=1.0/math.sqrt(float(hidden_size))), 
                      name='softmax_weights')
    s_b = tf.Variable(tf.zeros([output_size]), 
                      name='softmax_biases')
    logits = tf.matmul(hidden, s_w) + s_b
    """

    h_pool2 = max_pool_2x2(h_convl) 


# Correct loss for convnet?
with tf.name_scope('loss'):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy with_logits(
                              logits, 
                              labels_placeholder, 
                              name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    regularizers = tf.nn.l2_loss(h_w) + tf.nn.l2_loss(h_b)+tf.nn.l2_loss(s_w)+tf.nn.l2_loss(s_b)
    loss += reg*regularizers

td.scalar_summary('loss', loss) # Does this go here?

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)
 
sess = tf.InteractiveSession(graph=graph)
sess.run(tf.initialize_all_variables())

summary_op = tf.merge_all_summaries() # Do these go here? or earlier?
summary_writer = tf.train.SummaryWriter(log_dir, graph_def=sess.graph_def)

for iteration in iterations:

    offset = (step*batch_size) % (y_train.shape[0] -batch_size)
    batch_data   = X_train[offset:(offset + batch_size), :]
    batch_labels = y_train[offset:(offset + batch_size)]

    feeddict = {images_placeholder:batch_data,
            labels_placeholder:batch_labels}

    _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

    summary_str = sess.run(summary_op, feed_dict=feed_dict)
    summary_writer.add_summary(summary_str, step)

correct = tf.nn.in_top_k(logits, labels_placeholder, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.int32))
feed_dict = {images_placeholder:X_train, labels_placeholder:y_train}
precision = sess.run(accuracy, feed_dict=feed_dict)


# JUST FOR PROJECT?
#saver = tf.train.saver()
#saver.save(sess, os.path.join(log_dir, 'checkpoint'), global_step=step+1)

