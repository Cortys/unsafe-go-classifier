import tensorflow as tf
import numpy as np

def pred_sorted_cumsum(pred_logits):
  pred = tf.nn.softmax(pred_logits).numpy()
  pred_pi = np.argsort(pred)[:, ::-1]
  pred_sorted = np.take_along_axis(pred, pred_pi, axis=1)
  pred_cumsum = np.cumsum(pred_sorted, -1)
  return pred_sorted, pred_cumsum, pred_pi

def adaptive_calibration_scores(pred_logits, true_labels):
  pred_sorted, pred_cumsum, pred_pi = pred_sorted_cumsum(pred_logits)
  pred_pi_inv = np.argsort(pred_pi)
  true_cumsumidx = np.take_along_axis(
      pred_pi_inv, np.expand_dims(true_labels, 1), axis=1)
  U = np.random.random_sample((pred_sorted.shape[0], 1))
  pred_cumsum = np.pad(pred_cumsum, ((0, 0), (1, 0)), "constant")
  scores = np.take_along_axis(pred_cumsum, true_cumsumidx, axis=1)
  scores += np.take_along_axis(pred_sorted, true_cumsumidx, axis=1) * U
  scores = np.squeeze(scores)
  return scores

def qhat(scores, alpha=0.1):
  n = scores.shape[0]
  q = np.ceil((n + 1) * (1 - alpha)) / n
  return np.quantile(scores, q, method="higher")

def adaptive_sets(pred_logits, qhat):
  _, pred_cumsum, pred_pi = pred_sorted_cumsum(pred_logits)
  sizes = np.argmax(pred_cumsum > qhat, axis=1)
  pred_sets = [pred_pi[i][:(sizes[i] + 1)] for i in range(sizes.shape[0])]
  return pred_sets

def topk_sets(pred_logits, k=3):
  return np.argsort(pred_logits)[:, -k:]
