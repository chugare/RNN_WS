# -*- coding: utf-8 -*-
#   Project name : RNN_WS
#   Edit with PyCharm
#   Create by simengzhao at 2018/5/15 上午10:54
#   南京大学软件学院 Nanjing University Software Institute
#
import itertools
import sys
import tensorflow as tf
import numpy as np
import DataPreparation,BuildModel,Constant,one_hot,PCA,DrawPlot
import os
tf.app.flags.DEFINE_integer('batch_size',Constant.BATCH_SIZE,'batch size')
tf.app.flags.DEFINE_integer('epochs',50,'train epochs')
tf.app.flags.DEFINE_float('learning_rate',Constant.LEARNING_RATE,'learning rate')
tf.app.flags.DEFINE_string('checkpoint_dir',os.path.abspath('./checkpoint'),'checkpoint save path')
tf.app.flags.DEFINE_string('checkpoint_dir_cnn',os.path.abspath('./checkpoint_cnn'),'cnn checkpoint save path')
tf.app.flags.DEFINE_string('file_path',os.path.abspath('./data'),'file name of data')
FLAGS = tf.app.flags.FLAGS
def padArray_oh(list,length):
    newarray = []
    for subarr in list:
        k = length - subarr.shape[0]
        karry = np.zeros(shape=[k,Constant.VEC_SIZE],dtype=np.int32)
        subarr = np.concatenate((subarr,karry))
        newarray.append(subarr)
    return np.array(newarray)
def padArray(list,length):
    newarray = []
    for subarr in list:
        k = length - subarr.shape[0]
        karry = np.zeros(shape=[k],dtype=np.int32)
        subarr = np.concatenate((subarr,karry))
        newarray.append(subarr)
    return np.array(newarray)
def run_training ():
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.mkdir(FLAGS.checkpoint_dir)
    input_data = tf.placeholder(tf.int32,shape=[FLAGS.batch_size,None])
    shift_data = tf.placeholder(tf.int32,shape=[FLAGS.batch_size,None])
    length_data = tf.placeholder(tf.int32,shape=[FLAGS.batch_size])
    ops = BuildModel.model(input_data,shift_data,length_data)
    saver = tf.train.Saver(tf.global_variables())
    gv_init_op = tf.global_variables_initializer()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(gv_init_op)
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

        start_epoch = 0
        if checkpoint:
            saver.restore(sess,checkpoint)
            print('[INFO] 从上一次的检查点:\t%s开始继续训练任务'%checkpoint)
            start_epoch += int(checkpoint.split('-')[-1])
        try:
            for epoch in range(start_epoch,FLAGS.epochs):
                #data_generator = DataPreparation.sentence_gen('content_law_labeled.txt')
                data_generator = DataPreparation.sentence_gen('test.txt',True)
                try:
                    count = 0
                    while True:
                        input_list = []
                        shift_list = []
                        length_l = []
                        max = 0
                        total_len = 0
                        for i in range(FLAGS.batch_size):
                            r1,r2 = next(data_generator)
                            input_list.append(r1)
                            shift_list.append(r2)
                            length_l.append(r1.shape[0])
                            total_len += r1.shape[0]
                            if r1.shape[0]>max:
                                max = r1.shape[0]
                        input_arr = padArray(input_list,max)
                        shift_arr = padArray(shift_list,max)
                        loss,state,_,f_count = sess.run([ops['total_loss'],
                                         ops['last_state'],
                                         ops['train_op'],ops['result_false']], feed_dict={input_data: input_arr, shift_data: shift_arr,length_data:length_l})
                        count+= 1
                        acc = float(total_len-f_count)/total_len
                        if count%1 == 0:
                            print('[INFO] Epoch: %d \tBatch: %d training loss: %.6f accuracy: %.4f'%(epoch,count,loss,acc))


                except StopIteration:
                    print('[INFO] Epoch %d 结束，对运行状态进行保存...' % epoch)
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'rnnws'), global_step=epoch)
                    pass
        except KeyboardInterrupt:
            print('[INFO] 手动暂停实验，将对运行状态进行保存...')
            saver.save(sess,os.path.join(FLAGS.checkpoint_dir,'rnnws'),global_step=epoch)
            print('[INFO] 保存完成')
def run_generate():
    with tf.Session() as sess:
        input_data = tf.placeholder(dtype=tf.int32,shape=[1,None])
        ops = BuildModel.model(input_data,None,[1])
        saver = tf.train.Saver(tf.global_variables())
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if checkpoint:
            saver.restore(sess,checkpoint)
        ohEncoder = one_hot.OneHotEncoder()
        first = ohEncoder.get_code('1')
        predict,state = sess.run([ops['prediction'],ops['last_state']],feed_dict={
            input_data:np.array([[first]])
        })
        word = ohEncoder.get_word(np.argmax(predict))
        while word!='EOD':
            sys.stdout.write(word)
            predict, state = sess.run([ops['prediction'], ops['last_state']], feed_dict={
                input_data: np.array([[ohEncoder.get_code(word)]]),
                ops['init_state']:state
            })
            word = ohEncoder.get_word(np.argmax(predict))
        print('')

def testRelation():
    # 映射空间，测试文本相似性的判断依据
    with tf.Session() as sess:
        tg = DataPreparation.TupleGenerator()
        ohEncoder = one_hot.OneHotEncoder()
        generator =  tg.tuple_gen('content_law_labeled.txt')
        input_data = tf.placeholder(tf.int32, shape=[1, None])
        length_data = tf.placeholder(tf.int32,shape=[1])
        ops = BuildModel.model(input_data,None,length=length_data)
        saver = tf.train.Saver(tf.global_variables())
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

        lstmcells = []
        for i in range(3):
            lstmcells.append({
                'c':[],
                'h':[]
            })
        if checkpoint:
            saver.restore(sess, checkpoint)
        ALL_NUM =1000
        for count in range(ALL_NUM):

            content = next(generator)
            enter_data = ohEncoder.one_hot_single(content,True)
            state,predict = sess.run([ops['last_state'],ops['prediction']],feed_dict={
                input_data:[enter_data],
                length_data:[len(enter_data)]
            })
            for c in range(3):
                lstmcells[c]['c'].append(np.array(state[c][0]))
                lstmcells[c]['h'].append(np.array(state[c][0]))
            if count%10 ==0:
                print('[INFO] Getting %d \tst word\'s vector...'%count)
        vec = lstmcells[1]['c']
        vec = np.array(vec)
        lddata,recon=PCA.PCA(np.mat(vec),2)
        points,labels = tg.generateLabel(lddata)
        DrawPlot.drawScatter(points,labels)
        outfile = open('result.txt','w',encoding='utf-8')
        for i in range(ALL_NUM):
            outfile.write('%f\t%f\n'%(lddata[i,0],lddata[i,1]))

def CNN_TRAIN():
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.mkdir(FLAGS.checkpoint_dir)
    input_data = tf.placeholder(tf.int32,shape=[FLAGS.batch_size,None])
    shift_data = tf.placeholder(tf.int32,shape=[FLAGS.batch_size,None])
    length_data = tf.placeholder(tf.int32,shape=[FLAGS.batch_size])
    ops = BuildModel.model(input_data,shift_data,length_data)
    cnn_enter_data = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,None,Constant.KERNEL_HEIGHT])
    cnn_label = tf.placeholder(tf.int32,shape=[FLAGS.batch_size])
    cnn_train = BuildModel.CNNmodel(cnn_enter_data,cnn_label)
    saver = tf.train.Saver(tf.global_variables())
    # gv_init_op = tf.global_variables_initializer()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

        start_epoch = 0
        if checkpoint:
            saver.restore(sess,checkpoint)
            print('[INFO] 从上一次的检查点:\t%s开始继续训练任务'%checkpoint)
            start_epoch += int(checkpoint.split('-')[-1])
        try:
            for epoch in range(start_epoch,FLAGS.epochs):
                #data_generator = DataPreparation.sentence_gen('content_law_labeled.txt')
                data_generator = DataPreparation.sentence_gen('test.txt',True)
                try:
                    count = 0
                    while True:
                        input_list = []
                        shift_list = []
                        length_l = []
                        max = 0
                        total_len = 0
                        for i in range(FLAGS.batch_size):
                            r1,r2 = next(data_generator)
                            input_list.append(r1)
                            shift_list.append(r2)
                            length_l.append(r1.shape[0])
                            total_len += r1.shape[0]
                            if r1.shape[0]>max:
                                max = r1.shape[0]
                        input_arr = padArray(input_list,max)
                        shift_arr = padArray(shift_list,max)
                        loss,state,_,f_count = sess.run([ops['total_loss'],
                                         ops['last_state'],
                                         ops['train_op'],ops['result_false']], feed_dict={input_data: input_arr, shift_data: shift_arr,length_data:length_l})
                        count+= 1
                        acc = float(total_len-f_count)/total_len
                        if count%1 == 0:
                            print('[INFO] Epoch: %d \tBatch: %d training loss: %.6f accuracy: %.4f'%(epoch,count,loss,acc))


                except StopIteration:
                    print('[INFO] Epoch %d 结束，对运行状态进行保存...' % epoch)
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'rnnws'), global_step=epoch)
                    pass
        except KeyboardInterrupt:
            print('[INFO] 手动暂停实验，将对运行状态进行保存...')
            saver.save(sess,os.path.join(FLAGS.checkpoint_dir,'rnnws'),global_step=epoch)
            print('[INFO] 保存完成')

testRelation()