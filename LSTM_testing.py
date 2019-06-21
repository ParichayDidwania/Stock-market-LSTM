import tensorflow as tf
import numpy as np

x_inp = np.array([[125.24/1584.44,119.98/1578.13,119.94/1549.94,125.54/1600.93]])
y_inp = np.random.rand(1)

x_inp = np.reshape(x_inp,[1,1,4])


with tf.Session() as sess:
    saver = tf.train.import_meta_graph("./model/stock.meta")
    saver.restore(sess,tf.train.latest_checkpoint('./model/'))
    graph = tf.get_default_graph()
    op = graph.get_operation_by_name('inf_op')
    x = graph.get_tensor_by_name('p1:0')
    y = graph.get_tensor_by_name('p2:0')
    pred = graph.get_tensor_by_name('Reshape:0')
    sess.run(op,feed_dict={x:x_inp,y:y_inp})
    print(sess.run(pred)*859643400)

