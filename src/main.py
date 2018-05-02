import tensorflow as tf

COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
DEFAULTS = [[0.0]] * (len(COLUMNS) - 1) + [[""]]


def parse_csv(line):
    data = tf.decode_csv(line, DEFAULTS)
    features = tf.stack(data[:len(COLUMNS) - 1])
    labels = data[len(COLUMNS) - 1]
    return features, labels


trn_data = tf.data.TextLineDataset("../data/train.csv")
evl_data = tf.data.TextLineDataset("../data/test.csv")

train_data, train_targets = trn_data.map(parse_csv).shuffle(200).repeat().batch(16).make_one_shot_iterator().get_next()
eval_data, eval_targets = evl_data.map(parse_csv).shuffle(200).repeat().batch(16).make_one_shot_iterator().get_next()

# training_handle = sess.run(training_iterator.string_handle())  # Maybe study this next?
# validation_handle = sess.run(validation_iterator.string_handle())

x = tf.placeholder(tf.float32, [None, 4], name="input")
y_ = tf.placeholder(tf.int64, [None, ], name="target")

a1 = tf.layers.dense(x, 10, activation=tf.nn.sigmoid)
a2 = tf.layers.dense(a1, 20, activation=tf.nn.sigmoid)
a3 = tf.layers.dense(a2, 10, activation=tf.nn.sigmoid)
y = tf.layers.dense(a3, 3, activation=tf.nn.softmax)
# y = tf.layers.dense(x, 3, activation=tf.nn.softmax)

with tf.name_scope("training") as scope:
    loss = tf.losses.softmax_cross_entropy(tf.one_hot(y_, 3), y)
    tf.summary.scalar("loss", loss)
    optimizer = tf.train.AdamOptimizer(0.001)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
    # train_op = tf.train.GradientDescentOptimizer(0.001, name="Adam").minimize(loss, global_step=tf.train.create_global_step())

with tf.name_scope('summaries') as scope:
    predictions = tf.argmax(y, 1, name="predictions")
    correct_prediction = tf.equal(y_, predictions)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    cm = tf.confusion_matrix(y_, predictions)
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

    N = 801
    for n in range(N):
        if n % 50 == 0:
            print("step {} of {}, global_step set to {}".format(n, N - 1, sess.run(tf.train.get_global_step())))
            summary, acc, c = sess.run([merged_summary, accuracy, cm],
                                       feed_dict={x: sess.run(eval_data), y_: sess.run(eval_targets)})
            print(c, acc)
            test_writer.add_summary(summary, global_step=sess.run(tf.train.get_global_step()))
        else:
            summary, _ = sess.run([merged_summary, train_op],
                                  feed_dict={x: sess.run(train_data), y_: sess.run(train_targets)})
            train_writer.add_summary(summary, global_step=sess.run(tf.train.get_global_step()))
