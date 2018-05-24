# -*- coding: utf-8 -*-
#   Project name : RNN_WS
#   Edit with PyCharm
#   Create by simengzhao at 2018/5/14 下午7:04
#   南京大学软件学院 Nanjing University Software Institute
#
from Constant import *

import tensorflow as tf


def activation_summary(x):
    """
    为定义的操作添加总结，加入方便生成模型和流程图
    :param x:
    :return:
    """
    tensorname = x.op.name
    tf.summary.histogram(tensorname + "/activation", x)
    tf.summary.scalar(tensorname + '/sparsity', tf.nn.zero_fraction(x))
def get_variable_with_wd(name,shape,stdv=0.2,wd=None):
    init = tf.truncated_normal_initializer(stddev=stdv)
    var = get_variable_on_device(name, shape, initlizer=init)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name="weight_loss")
        tf.add_to_collection('losses', weight_decay)
    return var

def get_variable_on_device(name,shape,initlizer):
    with tf.device(DEFAULT_DEVICE):
        return tf.get_variable(name,shape,initializer=initlizer)


def model(input_data,shift_data,length):
    embedding = tf.get_variable('embedding',initializer=tf.random_uniform([VEC_SIZE,UNIT_NUM],-1.0,1.0))
    inputs = tf.nn.embedding_lookup(embedding,input_data)

    #inputs = tf.cast(input_data,tf.float32)

    rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=UNIT_NUM,forget_bias=0.5)
    rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell,output_keep_prob=0.5)

    rnn_cell2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=UNIT_NUM,forget_bias=0.5)
    rnn_cell2 = tf.nn.rnn_cell.DropoutWrapper(rnn_cell2,output_keep_prob=0.5)

    cells = [rnn_cell,rnn_cell2,rnn_cell2]

    cells = tf.nn.rnn_cell.MultiRNNCell(cells)
    if shift_data is not None:
        init_state = cells.zero_state(BATCH_SIZE, dtype=tf.float32)
    else:
        init_state = cells.  zero_state(1, dtype=tf.float32)
    output,state = tf.nn.dynamic_rnn(cells,inputs,initial_state=init_state,sequence_length=length)

    output = tf.reshape(output,[-1,UNIT_NUM])

    weights = tf.Variable(tf.truncated_normal(shape = [UNIT_NUM,VEC_SIZE]))
    bias = tf.Variable(tf.zeros(shape=[VEC_SIZE]))
    logits = tf.nn.bias_add(tf.matmul(output,weights),bias=bias)
    op_list = {}
    op_list['init_state'] = init_state
    op_list['output'] = output
    op_list['last_state'] = state
    if shift_data is not None:
        labels = tf.one_hot(tf.reshape(shift_data,[-1]),depth=VEC_SIZE)
        result_l = tf.nn.softmax(logits=logits,axis=1)
        result_l = tf.argmax(result_l,1,output_type=tf.int32)
        result_l = result_l-tf.reshape(shift_data,[-1])
        result_l = tf.count_nonzero(result_l)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels,logits=logits)
        total_loss = tf.reduce_mean(loss)
        train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(total_loss)
        op_list['train_op'] = train_op
        op_list['total_loss'] = total_loss
        op_list['loss'] =  loss
        op_list['result_false'] = result_l
    else:
        prediction = tf.nn.softmax(logits)
        op_list['prediction'] = prediction

    return  op_list
def CNNmodel(input_vec,label):
    with tf.variable_scope('CONV_PART'):
         with tf.variable_scope('conv'):
             kernels = get_variable_with_wd(name='kernels',shape=[KERNEL_WIDTH,KERNEL_HEIGHT,1,KERNEL_NUM],)
             bias = tf.get_variable(name='bias',shape=[KERNEL_NUM],initializer=tf.constant_initializer(0.0))
             cov = tf.nn.conv2d(input_vec,kernels,[1,1,1,1],padding='VALID')
             conv = tf.nn.bias_add(cov,bias)
         activation_summary(conv)
         c_shape = conv.get_shape()
         pooling = tf.nn.max_pool(conv,[1,c_shape[1],c_shape[2],1],[1,2,2,1],'VALID')
         norm1 = tf.nn.l2_normalize(pooling,3)
         dropout = tf.nn.dropout(norm1,CNN_KEEP_PROB)

         with tf.variable_scope('ann'):
             d_shape = dropout.get_shape()
             ann_in = tf.reshape(dropout, [d_shape[0], -1])
             weight = get_variable_with_wd(name='weight_in', shape=[d_shape[1], CNN_LAYER_SIZE],wd=0.01)
             bias = tf.get_variable(name='bias_in', shape=[CNN_LAYER_SIZE], initializer=tf.constant_initializer())
             ann_out = tf.matmul(ann_in,weight)+bias
             ann_in = tf.nn.relu(ann_out,name='ann_ENTRY')
             activation_summary(ann_in)
             for i in range(CNN_LAYER):
                 weight = get_variable_with_wd(name='weight%d' % i, shape=[CNN_LAYER_SIZE, CNN_LAYER_SIZE],wd=0.01)
                 bias = tf.get_variable(name='bias%d' % i, shape=[CNN_LAYER_SIZE],
                                        initializer=tf.constant_initializer())
                 ann_out = tf.matmul(ann_in, weight) + bias
                 ann_out = tf.nn.relu(ann_out, name='ann_OUT_%d'%i)
                 activation_summary(ann_in)
         with tf.variable_scope('softmax_linear'):
             weight = get_variable_with_wd(name='weight',shape=[CNN_LAYER_SIZE,CNN_OUT_SIZE],wd=0.01)
             bias = tf.get_variable(name='bias',shape=[CNN_OUT_SIZE])
             softmax = tf.matmul(ann_out,weight)+bias

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label,logits=softmax)
    loss_rm = tf.reduce_mean(loss)
    tf.add_to_collection('losses',loss_rm)
    loss_sum = tf.add_n(tf.get_collection('losses'),name='total_loss')
    activation_summary(loss_sum)
    output_la = tf.nn.softmax(softmax)
    ops = {
        'total_loss':loss_sum,
        'output':output_la
    }
    train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss_sum)
    ops['train_op'] = train_op
    return ops




