import glob
import tensorflow as tf
import os

# : nnet files
# https://github.com/NeuralNetworkVerification/Marabou/blob/master/resources/nnet/acasxu/

# : nnet file specification
# https://github.com/sisl/NNet/blob/master/README.md


def build_dnn(weights, act_func='relu', last_activation=True):
    model = tf.keras.models.Sequential()
    input_size = weights[0][0].shape[1]
    model.add(tf.keras.layers.InputLayer(input_size))
    for i, (k, b) in enumerate(weights):
        model.add(tf.keras.layers.Dense(b.shape[0]))
        model.layers[-1].kernel.assign(tf.cast(tf.transpose(k), tf.float32))
        model.layers[-1].bias.assign(tf.cast(b, tf.float32))
        if i < len(weights) - 1:
            model.add(tf.keras.layers.Activation(act_func))
        elif last_activation:
            model.add(tf.keras.layers.Activation('softmax'))
    return model


def read_acasxu_weights(filepath):
    weights = []

    with open(filepath) as f:
        line = f.readline().strip()
        while line.startswith("//"):
            line = f.readline().strip()
        n_layer, input_size, output_size, max_layer_size = [
            int(v) for v in line.split(',')[:4]]
        line = f.readline().strip()
        layer_sizes = [int(v) for v in line.split(',')[:-1]]

        f.readline()
        line = f.readline().strip()
        _minimums = [float(v) for v in line.split(',')[:-1]]
        line = f.readline().strip()
        _maximums = [float(v) for v in line.split(',')[:-1]]
        line = f.readline().strip()
        _means = [float(v) for v in line.split(',')[:-1]]
        line = f.readline().strip()
        _stds = [float(v) for v in line.split(',')[:-1]]

        for next_layer_size in layer_sizes[1:]:
            kernel, bias = [], []
            for i_row in range(next_layer_size):
                line = f.readline().strip()
                row = [float(v) for v in line.split(',')[:-1]]
                kernel.append(row)
            for i_row in range(next_layer_size):
                line = f.readline().strip()
                row = [float(v) for v in line.split(',')[:-1]]
                bias.append(row[0])
            kernel = tf.constant(kernel)
            bias = tf.constant(bias)

            weights.append((kernel, bias))

    return weights


if __name__ == "__main__":
    for nnet_path in glob.glob("acasxu_nnet/*.nnet"):
        tf_keras_path = os.path.join(
            "acasxu_tf_keras",
            f'{os.path.basename(nnet_path)[:-4]}.h5'
        )
        acasxu_tf_keras = build_dnn(read_acasxu_weights(nnet_path))
        tf.keras.models.save_model(acasxu_tf_keras, tf_keras_path)
