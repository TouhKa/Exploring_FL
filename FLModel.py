import tensorflow as tf
class FLModel(tf.keras.Model):
  def __init__(self, number_of_classes):
    super(FLModel, self).__init__()
    tf.keras.backend.set_floatx('float64')
    initializer = tf.keras.initializers.Zeros()
    self.dense1 = tf.keras.layers.Dense(500, activation=tf.nn.relu, kernel_initializer=initializer)
    self.dense2 = tf.keras.layers.Dense(number_of_classes, kernel_initializer=initializer, activation=tf.nn.softmax)

  def call(self, inputs):
    x = self.dense1(inputs)
    return self.dense2(x)