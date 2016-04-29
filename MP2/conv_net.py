import tensorflow as tf
import numpy as np
import math

from data_utils import *

log_dir = "./log_dir"

input_size = 32*32*3
batch_size  = 100
num_classes = 20
dropout = False
dropout_prob = 0.95
learning_rate = 1e-3
reg_strength = 0.1
max_iters = 500000

images_placeholder = tf.placeholder(tf.float32, 
                                    shape=(batch_size, 32, 32, 3),
                                    name='images')
labels_placeholder = tf.placeholder(tf.float32, 
                                    shape=(batch_size,
                                           num_classes), 
                                    name='labels')
keep_prob = tf.placeholder(tf.float32) # for dropout

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
with tf.name_scope('first-conv-layer'): 

    W_convl1 = tf.Variable(# 5x5 pixel patch, depth 3, 4 feature maps
                  tf.truncated_normal(shape=[5,5,3,4], 
                                      stddev=1e-4))
    b_convl1 = tf.constant(0.1, shape=[4]) # 4 feature maps
    h_convl1 = tf.nn.relu(conv2d(images_placeholder,
                                 W_convl1) + b_convl1)


with tf.name_scope('first-maxpool-layer'):

    h_pool1 = max_pool(h_convl1, 2)
    if dropout:
        h_pool1 = tf.nn.dropout(h_pool1, dropout_prob) 

with tf.name_scope('second-conv-layer'):

    W_convl2 = tf.Variable(# 5x5 pixel patch, depth 4, 6 feature maps
                   tf.truncated_normal(shape=[5,5,4,6], 
                                       stddev=1e-4))
    b_convl2 = tf.constant(0.1, shape=[6]) 
    h_convl2 = tf.nn.relu(conv2d(h_pool1, W_convl2) + b_convl2)

with tf.name_scope('second-maxpool-layer'):
    
    h_pool2 = max_pool(h_convl2, 2)
    
    if dropout:
        tf.nn.dropout(h_pool2, dropout_prob)


with tf.name_scope('fully-connected-hidden'):

    h_w = tf.Variable(
              tf.truncated_normal(
                  [8*8*W_convl2.get_shape().as_list()[-1], 1024],
                  stddev=1.0/math.sqrt(float(input_size))),
              name='hidden_weights')

    h_pool2 = tf.reshape(h_pool2, [-1, h_w.get_shape().as_list()[0]])
    h_b = tf.Variable(tf.zeros([1024]),
                      name='hidden_biases')
    hidden = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_pool2, 
                                                 h_w), h_b))
    if dropout:
        hidden = tf.nn.dropout(hidden, dropout_prob)

with tf.name_scope('fully-connected-mlp'):
    
    p_w = tf.Variable(
              tf.truncated_normal( 
                  [1024, num_classes],
                  stddev=1.0),            
              name='output_weights')
    p_b = tf.Variable(tf.zeros([num_classes]),
                      name='output_biases')
    logits = tf.add(tf.matmul(hidden, p_w), p_b)



with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                              logits, 
                              #labels, 
                              labels_placeholder, 
                              name='xentropy')
    cost = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    regularizers  = tf.nn.l2_loss(W_convl1) + tf.nn.l2_loss(b_convl1)
    regularizers += tf.nn.l2_loss(W_convl2) + tf.nn.l2_loss(b_convl2)
    regularizers += tf.nn.l2_loss(h_w) + tf.nn.l2_loss(h_b)
    regularizers += tf.nn.l2_loss(p_w) + tf.nn.l2_loss(p_b)
    cost += reg_strength*regularizers

tf.scalar_summary('loss', cost)

optimizer = tf.train.GradientDescentOptimizer(
                learning_rate).minimize(cost)

# Evaluate
correct_pred = tf.equal(tf.argmax(logits, 1), 
                        tf.argmax(labels_placeholder, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
 

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
    display_step = 2
    while step*batch_size < max_iters:

        offset = (step*batch_size) % (y_train.shape[0] - batch_size)
        batch_data   = X_train[offset:(offset + batch_size), :]
        batch_labels = y_train[offset:(offset + batch_size)]
        batch_labels = make_one_hot(batch_labels)

        sess.run(optimizer,
                 feed_dict={images_placeholder:batch_data,
                            labels_placeholder:batch_labels,                                      keep_prob:dropout_prob})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, 
                          feed_dict={images_placeholder: batch_data, 
                                     labels_placeholder: batch_labels,                                     keep_prob: 1.})
            # Calculate batch loss
            loss = sess.run(cost, 
                          feed_dict={images_placeholder: batch_data, 
                                     labels_placeholder: batch_labels,                                     keep_prob: 1.})
            print "".join(["Iter ", str(step*batch_size),
                           ", Minibatch Loss= ",
                           "{:.6f}".format(loss),
                           ", Training Accuracy= ",
                           "{:.6f}".format(acc)])
            
        #summary_str = sess.run(summary_op, feed_dict=feed_dict)
        #summary_writer.add_summary(summary_str, step)
        step += 1

        #saver = tf.train.Saver()
        #saver.save(sess, os.path.join(log_dir, 'checkpoint'),
        #           global_step=step+1)

    # Check test set
    #correct = tf.nn.in_top_k(logits, labels_placeholder, 1)
    #accuracy = tf.reduce_mean(tf.cast(correct, tf.int32))
    step = 1
    while step*batch_size < 10000:

        offset = (step*batch_size) % (y_test.shape[0] - batch_size)
        batch_data   = X_test[offset:(offset + batch_size), :]
        batch_labels = y_test[offset:(offset + batch_size)]
        feed_dict = {images_placeholder:batch_data, 
                     labels_placeholder:make_one_hot(batch_labels),
                     keep_prob: 1.}
        precision = sess.run(accuracy, feed_dict=feed_dict)
        print "Test Accuracy= {:.6f}".format(precision)

#saver = tf.train.saver()
#saver.save(sess, os.path.join(log_dir, 'checkpoint'), 
#           global_step=step+1)

