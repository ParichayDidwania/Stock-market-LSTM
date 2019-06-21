import tensorflow as tf
import pandas as pd
import csv
import numpy as np

dir = './prices.csv'
tf_dir = './lstm.tfrecords'

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

writer = tf.python_io.TFRecordWriter(tf_dir)
with open(dir) as f:
  x = csv.reader(f)
  j=0
  for i in x:           
    #print(i[1:30])
    #print(i[-1])
    if(j!=0):
      z = []
      for k in i[2:6]:
        z.append(float(k))  

      feature={
        'open' : _floats_feature(z[0]),
        'close' : _floats_feature(z[1]),
        'low' : _floats_feature(z[2]),
        'high' : _floats_feature(z[3]),
        'vol' : _floats_feature(float(i[-1]))                
      }

      example = tf.train.Example(features=tf.train.Features(feature=feature))
      writer.write(example.SerializeToString())
      
    j+=1
    
      

writer.close()
print(j)