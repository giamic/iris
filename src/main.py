import tensorflow as tf


def parse_csv(line):
    data = tf.decode_csv(line, [[]] * 5)
    features = tf.stack(data[:4])
    labels = tf.one_hot(tf.cast(data[4], tf.int32), 3)
    return features, labels


training_iterator = tf.data.TextLineDataset("../data/train.csv").map(parse_csv).shuffle(200).repeat().batch(
    32).make_one_shot_iterator()
validation_iterator = tf.data.TextLineDataset("../data/validation.csv").map(parse_csv).shuffle(200).repeat().batch(
    32).make_one_shot_iterator()

handle = tf.placeholder(tf.string, shape=[])
x, y_ = tf.data.Iterator.from_string_handle(handle, training_iterator.output_types, training_iterator.output_shapes).get_next()

with tf.name_scope("network") as scope:  # definition of the neural network

    a1 = tf.layers.dense(x, 10, activation=tf.nn.relu)
    a2 = tf.layers.dense(a1, 20, activation=tf.nn.relu)
    a3 = tf.layers.dense(a2, 10, activation=tf.nn.relu)
    y = tf.layers.dense(a3, 3, activation=tf.nn.softmax)
    tf.summary.histogram("activation_1", a1)
    tf.summary.histogram("activation_2", a2)
    tf.summary.histogram("activation_3", a3)
    tf.summary.histogram("probs", y)

with tf.name_scope("training") as scope:  # training step
    loss = tf.losses.softmax_cross_entropy(y_, y)
    train_op = tf.train.AdamOptimizer(0.001).minimize(loss, global_step=tf.train.get_or_create_global_step())
    tf.summary.scalar("loss", loss)

with tf.name_scope('evaluation') as scope:  # evaluation of the results
    predictions = tf.argmax(y, 1, name="predictions")
    labels = tf.argmax(y_, 1, name="labels")
    correct_prediction = tf.equal(labels, predictions)  # boolean tensor that say if we did good
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    cm = tf.confusion_matrix(labels, predictions)
    tf.summary.tensor_summary("prediction", predictions)
    tf.summary.scalar("accuracy", accuracy)

with tf.Session() as sess:
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name.replace(":", "_"), var)
    merged_summary = tf.summary.merge_all()
    folder = '../models/model1/'
    train_writer = tf.summary.FileWriter(folder + 'train', sess.graph)
    test_writer = tf.summary.FileWriter(folder + 'test')

    tf.global_variables_initializer().run()

    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())

    N = 801
    for n in range(N):
        if n % 50 == 0:  # see what's going on every 50 steps
            print("step {} of {}, global_step set to {}".format(n, N - 1, sess.run(tf.train.get_global_step())))
            summary, acc, c = sess.run([merged_summary, y_, y, accuracy, cm, x], feed_dict={handle: validation_handle})
            test_writer.add_summary(summary, global_step=sess.run(tf.train.get_global_step()))
            print(c, "\n", acc)
        else:  # train
            summary, _ = sess.run([merged_summary, train_op], feed_dict={handle: training_handle})
            train_writer.add_summary(summary, global_step=sess.run(tf.train.get_global_step()))
