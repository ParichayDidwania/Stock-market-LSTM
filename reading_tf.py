import tensorflow as tf
import numpy as np

def parse(serialized):
    
    feature = {
        'open' : tf.FixedLenFeature([],tf.float32),
        'close': tf.FixedLenFeature([],tf.float32),
        'low'  : tf.FixedLenFeature([],tf.float32),
        'high' : tf.FixedLenFeature([],tf.float32),
        'vol'  : tf.FixedLenFeature([],tf.float32)
    }

    example = tf.parse_single_example(serialized = serialized , features = feature)
    
    opn = example['open']/1584.44  #divided by max values for normalisation
    opn = tf.cast(opn,tf.float32)

    close = example['close']/1578.13
    close = tf.cast(close,tf.float32)

    low = example['low']/1549.94
    low = tf.cast(low,tf.float32)

    high = example['high']/1600.93
    high = tf.cast(high,tf.float32)

    label = example['vol']/859643400
    label = tf.cast(label,tf.float32)

    x = tf.stack([[opn,close,low,high]])
    

    return x,label

def read_dataset(filename,batch_size = 500):

    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse,16)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    return dataset

