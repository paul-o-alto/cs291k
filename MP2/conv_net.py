import tensorflow as tf
import numpy as np
import math

from data_utils import *

input_size = 32*32*3
batch_size  = 100
num_classes = 20

images_placeholder = tf.placeholder(tf.float32, 
                                    shape=(batch_size, 32, 32, 3),
                                    name='images')
labels_placeholder = tf.placeholder(#tf.int32, 
                                    tf.float32, 
                                    shape=(batch_size,#),
                                           num_classes), 
                                    name='labels')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, 
                        strides=[1, 1, 1, 1], 
                        padding='SAME')

def max_pool(x, k):
    return tf.nn.max_pool(x, 
                          ksize=[1, k, k, 1], 
                          strides=[1, k, k, 1], 
                          padding='SAME')



# Organize 1st-layer
with tf.name_scope('firstlayer-conv'): 

    #images_reshaped = tf.reshape(images_placeholder,
    #                             shape=[-1, 32, 32, 3])
    W_convl = tf.Variable(# 5x5 pixel patch, depth 3, 4 feature maps
                  tf.truncated_normal(shape=[5,5,3,4], 
                                      stddev=1e-4))
    b_convl = tf.constant(0.1, shape=[4]) # 4 feature maps
    h_convl = tf.nn.relu(conv2d(#images_reshaped, 
                                images_placeholder,
                                W_convl) + b_convl)


with tf.name_scope('secondlayer-maxpool'):

    h_pool2 = max_pool(h_convl, 2) 

with tf.name_scope('thirdlayer-conv'):

    W_convl3 = tf.Variable(# 5x5 pixel patch, depth 4, 6 feature maps
                   tf.truncated_normal(shape=[5,5,4,6], 
                                       stddev=1e-4))
    b_convl3 = tf.constant(0.1, shape=[6]) # 6 feature maps? 
    h_convl3 = tf.nn.relu(conv2d(h_pool2, W_convl3) + b_convl3)

with tf.name_scope('fourthlayer-maxpool'):
    
    h_pool4 = max_pool(h_convl3, 2)


with tf.name_scope('fully-connected-hidden'):

    h_w = tf.Variable(
              tf.truncated_normal(
                  [8*8*W_convl3.get_shape().as_list()[-1], 1024],
                  stddev=1.0/math.sqrt(float(input_size))),
              name='hidden_weights')

    h_pool4 = tf.reshape(h_pool4, [-1, h_w.get_shape().as_list()[0]])
    h_b = tf.Variable(tf.zeros([1024]),
                      name='hidden_biases')
    hidden = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_pool4, 
                                                 h_w), h_b))

with tf.name_scope('fully-connected-mlp'):
    
    p_w = tf.Variable(
              tf.truncated_normal( 
                  [1024, num_classes],
                  stddev=1.0),           #/math.sqrt(float(8*8))), 
              name='output_weights')
    p_b = tf.Variable(tf.zeros([num_classes]),
                      name='output_biases')
    logits = tf.add(tf.matmul(hidden, p_w), p_b)


# Utility function to convert batch labels to one hot encoding
# matrix
#with tf.name_scope('make_one_hot'):
#    sparse_labels = tf.reshape(labels_placeholder, [-1, 1])
#    num_labels  = tf.shape(sparse_labels)[0]
#    indices  = tf.reshape(tf.range(0, num_labels, 1), [-1, 1])
#    concated = tf.concat(1, [indices, sparse_labels])
#    outshape = tf.pack([num_labels, num_classes])
#    labels   = tf.sparse_to_dense(concated,
#                                  outshape, 1.0, 0.0)

# Correct loss for convnet?
with tf.name_scope('loss'):
    # Need sparse_ version in order to not have to deal with the 
    # one hot encoding ourselves (hopefully)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                              logits, 
                              #labels, 
                              labels_placeholder, 
                              name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    regularizers  = tf.nn.l2_loss(W_convl) + tf.nn.l2_loss(b_convl)
    regularizers += tf.nn.l2_loss(W_convl3) + tf.nn.l2_loss(b_convl3)
    regularizers += tf.nn.l2_loss(h_w) + tf.nn.l2_loss(h_b)
    regularizers += tf.nn.l2_loss(p_w) + tf.nn.l2_loss(p_b)
    reg = 0.1
    loss += reg*regularizers

tf.scalar_summary('loss', loss) # Does this go here?

learning_rate = 1e-3 
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)
 

max_iters = 100000
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables()) 

    #summary_op = tf.merge_all_summaries() # Do these go here?
    #summary_writer = tf.train.SummaryWriter("./log_temp", #log_dir, 
    #                                        graph_def=sess.graph_def)

    X_train, y_train, X_test, y_test = load_CIFAR100(
                                           './cifar-100-batches-py/')
    
    num_train = 49000
    num_val   = 1000
    num_test  = 10000
   
    # This stuff probably needs to go elsewhere
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    #X_val   -= mean_image
    X_test  -= mean_image
    #X_train = X_train.reshape(num_train, -1)
    #X_val   = X_val.reshape(num_val,   -1)
    #X_test  = X_test.reshape(num_test,  -1)


    # Utility function to convert batch labels to one hot encoding
    # matrix (numpy)
    def make_one_hot(labels_dense):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    step = 1
    while step*batch_size < max_iters:

        offset = (step*batch_size) % (y_train.shape[0] - batch_size)
        batch_data   = X_train[offset:(offset + batch_size), :]
        batch_labels = y_train[offset:(offset + batch_size)]
        batch_labels = make_one_hot(batch_labels)
        print batch_labels
  
        feed_dict = {images_placeholder:batch_data,
                     labels_placeholder:batch_labels}

        _, loss_value = sess.run([train_op, loss],feed_dict=feed_dict)

        #summary_str = sess.run(summary_op, feed_dict=feed_dict)
        #summary_writer.add_summary(summary_str, step)
        print "Iteration "+str(step*batch_size)
        print "Loss= %d" % loss_value
        step += 1

    # Check test set
    correct = tf.nn.in_top_k(logits, labels_placeholder, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.int32))
    feed_dict = {images_placeholder:X_test, 
                 labels_placeholder:y_test}
    precision = sess.run(accuracy, feed_dict=feed_dict)


# JUST FOR PROJECT?
#saver = tf.train.saver()
#saver.save(sess, os.path.join(log_dir, 'checkpoint'), global_step=step+1)

