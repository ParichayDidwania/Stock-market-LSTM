import tensorflow as tf
import numpy as np
from reading_tf import read_dataset

def model(data):

    #data_2 = tf.layers.batch_normalization(data)

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(16,forget_bias=1.0)

    out,_= tf.contrib.rnn.static_rnn(lstm_cell,data,dtype = tf.float32 )

    out =  tf.layers.dense(out[-1],1,activation = 'sigmoid')

    out = tf.reshape(out,[tf.shape(out)[0]])

    return out

def train_model(data, epoch = 10):
    x = tf.placeholder(tf.float32,[None,1,4],name = 'p1')
    y = tf.placeholder(tf.float32,[None],name = 'p2')
    infer_dataset = tf.data.Dataset.from_tensor_slices((x,y))
    infer_dataset = infer_dataset.batch(100)

    iterator = tf.data.Iterator.from_structure(output_types = data.output_types, output_shapes= data.output_shapes)
    next_params,label = iterator.get_next()
    next_params = tf.unstack(next_params,num = 1,axis=1)
    
    #print(next_params)
    logits = model(next_params)
    print(logits)
   

    train_op = iterator.make_initializer(data, name = 'train_op')
    inf_op = iterator.make_initializer(infer_dataset, name = 'inf_op')

    
    with tf.name_scope('loss'):
        loss = tf.losses.mean_squared_error(labels = label, predictions = logits)
        optimizer = tf.train.AdamOptimizer(0.05).minimize(loss)

        tf.summary.scalar('loss',loss)
        merge = tf.summary.merge_all()

        saver = tf.train.Saver()
        

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        
        writer = tf.summary.FileWriter('./graphs',sess.graph)
        sess.run(init_op)
        j=0
        for i in range(epoch):
            sess.run(train_op)
            while(True):            
                try:
                    
                    j+=1
                    if(j%100==0):
                        l,_ = sess.run([loss,optimizer])
                        print("Iteration : ",j," Loss : ",l)
                    else:
                        summ,_ = sess.run([merge,optimizer])
                        writer.add_summary(summ,j)
                        
                except tf.errors.OutOfRangeError:
                    break

        saver.save(sess,'./model/stock')
        

path = './lstm.tfrecords'
dataset = read_dataset(path)
train_model(dataset)
        







