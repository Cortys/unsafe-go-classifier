import tensorflow as tf

def platt_scaling(pred_logits, true_labels, epochs=50):
  T = tf.Variable(initial_value=1.5, name="T")
  pred_logits = tf.constant(pred_logits)
  loss = lambda: tf.keras.losses.sparse_categorical_crossentropy(
    true_labels, pred_logits / T, from_logits=True)
  opt = tf.keras.optimizers.SGD(learning_rate=.01)

  for _ in range(epochs): opt.minimize(loss, [T])

  return T.numpy()
