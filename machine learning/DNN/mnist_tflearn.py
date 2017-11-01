from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    mnist = input_data.read_data_sets("/tmp/data/")

    X_train = mnist.train.images
    X_test = mnist.test.images
    y_train = mnist.train.labels.astype("int")
    y_test = mnist.test.labels.astype("int")

    config = tf.contrib.learn.RunConfig(tf_random_seed=42)  # not shown in the config

    feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
    dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], n_classes=10,
                                             feature_columns=feature_cols, config=config)
    dnn_clf = tf.contrib.learn.SKCompat(dnn_clf)  # if TensorFlow >= 1.1
    dnn_clf.fit(X_train, y_train, batch_size=50, steps=40000)


    y_pred = dnn_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred['classes'])
    print(accuracy)
