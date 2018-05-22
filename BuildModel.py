# -*- coding: utf-8 -*-
#   Project name : RNN_WS
#   Edit with PyCharm
#   Create by simengzhao at 2018/5/14 下午7:04
#   南京大学软件学院 Nanjing University Software Institute
#
from Constant import *

import tensorflow as tf

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
