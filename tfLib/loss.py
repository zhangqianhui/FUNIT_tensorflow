from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def getfeature_matching_loss(feature1, feature2):
    return tf.reduce_mean(tf.abs(
        tf.reduce_mean(feature1, axis=[1, 2]) - tf.reduce_mean(feature2, axis=[1, 2])))

def classification_loss(logits, labels) :
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                          (labels=labels, logits=logits))
    return loss

def cosine(f1, f2):
    f1_norm = tf.nn.l2_normalize(f1, dim=0)
    f2_norm = tf.nn.l2_normalize(f2, dim=0)
    return tf.losses.cosine_distance(f1_norm, f2_norm, dim=0)