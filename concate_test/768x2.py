import jsonlines
import numpy as np
import tensorflow as tf
import keras.backend as K
def precision(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2
    return precision

def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score
def confu(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  #true_positives
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))           #TP+FP
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    return c1
def recall(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  #true_positives
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))           #TP+FP
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))           #TP+FN
                                                        #Tn=all*acc-c1



    print('TP:  ',c1)
    print('FP:  ', c2-c1)
    print('FN ', c3-c1)
    # print('Tn  ', 3073*)



    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3



    return recall


alist=[]
with jsonlines.open('分割jason/lin1.jsonl') as reader:
    for obj in reader:
        # print(obj['features'][0]['layers'][0]['values'])
        alist.append(obj['features'][0]['layers'][0]['values'])
cc=np.array(alist)
blist=[]
with jsonlines.open('分割jason/lin2.jsonl') as reader2:
    for obj2 in reader2:
        blist.append(obj2['features'][0]['layers'][0]['values'])
dd=np.array(blist)
# cc=np.delete(cc,14621,0)
# dd=np.delete(dd,14621,0)
# final=np.concatenate((cc, dd), axis=1)
final=cc*dd
# trainData=np.split(final, [11547], axis=0)[0]
# devData=np.split(final, [11547], axis=0)[1]
# print(np.shape(trainData))
# print(np.shape(devData))
print(np.shape(cc))
print(np.shape(dd))

label=open('ptack標點完整.txt','r')
label_list=[]
for i in label:

    labels=int(i.split('\t')[0])
    label_list.append(labels)
    labels_array = np.array(label_list)

    # count=count+1
    # if  count<=11547:
    #     label_a.append(labels)
    #     labels_a=np.array(label_a)
    # else:
    #     label_b.append(labels)
    #     labels_b = np.array(label_b)
print(np.shape(labels_array))
# print(label_a)
# print(np.shape(labels_b))
# print(label_b)



from tensorflow import keras



model = keras.Sequential()

model.add(keras.layers.Dense(300,input_dim=768, activation=tf.nn.relu))                        #１６個unit的全鏈接
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))                      #單個輸出，sigmoid激活
model.summary()                                                                         #各種層

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy',f1_score, precision, recall,confu])                 #優化器的損失函數

x_val = final[11548:]
partial_x_train = final[:11548]
y_val = labels_array[11548:]
partial_y_train = labels_array[:11548]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=10,
                    batch_size=16,
                    validation_data=(x_val, y_val),
                    verbose=1)














# def create_initializer(initializer_range=0.02):
#   """Creates a `truncated_normal_initializer` with the given range."""
#   return tf.truncated_normal_initializer(stddev=initializer_range)
#
#
# def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
#                  labels, num_labels, use_one_hot_embeddings):
#   """Creates a classification model."""
#   # model = modeling.BertModel(
#   #     config=bert_config,
#   #     is_training=is_training,
#   #     input_ids=input_ids,
#   #     input_mask=input_mask,
#   #     token_type_ids=segment_ids,
#   #     use_one_hot_embeddings=use_one_hot_embeddings)
#
#   # In the demo, we are doing a simple classification task on the entire
#   # segment.
#   #
#   # If you want to use the token-level output, use model.get_sequence_output()
#   # instead.
#   first_token_tensor = final
#   output_layer = tf.layers.dense(first_token_tensor,hidden_size=1536,activation=tf.tanh,kernel_initializer=create_initializer(initializer_range=0.02))
#
#   hidden_size = output_layer.shape[-1].value
#
#   output_weights = tf.get_variable(
#       "output_weights", [num_labels, hidden_size],
#       initializer=tf.truncated_normal_initializer(stddev=0.02))
#
#   output_bias = tf.get_variable(
#       "output_bias", [num_labels], initializer=tf.zeros_initializer())
#
#   with tf.variable_scope("loss"):
#     if is_training:
#       # I.e., 0.1 dropout
#       output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
#
#     logits = tf.matmul(output_layer, output_weights, transpose_b=True)
#     logits = tf.nn.bias_add(logits, output_bias)
#     probabilities = tf.nn.softmax(logits, axis=-1)
#     log_probs = tf.nn.log_softmax(logits, axis=-1)
#
#     one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
#
#     per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
#     loss = tf.reduce_mean(per_example_loss)
#
#     return (loss, per_example_loss, logits, probabilities)



# first_token_tensor=final
#
# # config=copy.deepcopy(config)
#
# def create_initializer(initializer_range=0.02):
#   """Creates a `truncated_normal_initializer` with the given range."""
#   return tf.truncated_normal_initializer(stddev=initializer_range)
#
# def create_model(labels):
#
#     output_layer = tf.layers.dense(first_token_tensor,hidden_size=1536,activation=tf.tanh,kernel_initializer=create_initializer(initializer_range=0.02))
#     hidden_size = output_layer.shape[-1].value
#     num_labels=14622
#     output_weights = tf.get_variable("output_weights", [num_labels, hidden_size],initializer=tf.truncated_normal_initializer(stddev=0.02))
#     output_bias = tf.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())
#     # is_training = (mode == tf.estimator.ModeKeys.TRAIN)
#
#     with tf.variable_scope("loss"):
#         if mode == tf.estimator.ModeKeys.TRAIN:
#           # I.e., 0.1 dropout
#           output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
#
#         logits = tf.matmul(output_layer, output_weights, transpose_b=True)
#         logits = tf.nn.bias_add(logits, output_bias)
#         probabilities = tf.nn.softmax(logits, axis=-1)
#         log_probs = tf.nn.log_softmax(logits, axis=-1)
#
#         one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
#
#         per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
#         loss = tf.reduce_mean(per_example_loss)
#
#         return (loss, per_example_loss, logits, probabilities)
#
# def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
#                      num_train_steps, num_warmup_steps, use_tpu,
#                      use_one_hot_embeddings):
#   """Returns `model_fn` closure for TPUEstimator."""
#
#   def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
#     """The `model_fn` for TPUEstimator."""
#
#     tf.logging.info("*** Features ***")
#     for name in sorted(features.keys()):
#       tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
#
#     input_ids = features["input_ids"]
#     input_mask = features["input_mask"]
#     segment_ids = features["segment_ids"]
#     label_ids = features["label_ids"]
#     is_real_example = None
#     if "is_real_example" in features:
#       is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
#     else:
#       is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
#
#     is_training = (mode == tf.estimator.ModeKeys.TRAIN)
#
#     (total_loss, per_example_loss, logits, probabilities) = create_model(
#         bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
#         num_labels, use_one_hot_embeddings)
#
#     tvars = tf.trainable_variables()
#     initialized_variable_names = {}
#     scaffold_fn = None
#     if init_checkpoint:
#       (assignment_map, initialized_variable_names
#       ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
#       if use_tpu:
#
#         def tpu_scaffold():
#           tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
#           return tf.train.Scaffold()
#
#         scaffold_fn = tpu_scaffold
#       else:
#         tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
#
#     tf.logging.info("**** Trainable Variables ****")
#     for var in tvars:
#       init_string = ""
#       if var.name in initialized_variable_names:
#         init_string = ", *INIT_FROM_CKPT*"
#       tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
#                       init_string)
#
#     output_spec = None
#     if mode == tf.estimator.ModeKeys.TRAIN:
#
#       train_op = optimization.create_optimizer(
#           total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
#
#       output_spec = tf.contrib.tpu.TPUEstimatorSpec(
#           mode=mode,
#           loss=total_loss,
#           train_op=train_op,
#           scaffold_fn=scaffold_fn)
#     elif mode == tf.estimator.ModeKeys.EVAL:
#
#       def metric_fn(per_example_loss, label_ids, logits, is_real_example):
#         predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
#         accuracy = tf.metrics.accuracy(
#             labels=label_ids, predictions=predictions, weights=is_real_example)
#         loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
#         return {
#             "eval_accuracy": accuracy,
#             "eval_loss": loss,
#         }
#
#
#
#
#
#
#




