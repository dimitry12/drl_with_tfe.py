from .context import cartpole_ppo_tf
import tensorflow as tf
import pandas as pd


def sparse_categorical_accuracy_with_logits(y_true, y_pred_logits):
    # convert dense predictions to labels
    y_pred = tf.nn.softmax(y_pred_logits)
    y_pred_labels = tf.argmax(y_pred, axis=-1)
    y_pred_labels = tf.cast(y_pred_labels, tf.keras.backend.floatx())
    y_true = tf.cast(y_true, tf.keras.backend.floatx())
    return tf.cast(tf.equal(y_true, y_pred_labels), tf.keras.backend.floatx())


def test_policy_model_trainable_count():
    model = cartpole_ppo_tf.P_and_V_Model(
        classes=1, mlp_extra_layers=1, mlp_units=64)
    model(tf.constant([[1.0]*10]))
    assert model.count_params() == 4994


def test_policy_model_training():
    model = cartpole_ppo_tf.P_and_V_Model(
        classes=3, mlp_extra_layers=1, mlp_units=10, heads='p')
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss=tf.losses.sparse_softmax_cross_entropy,
                  metrics=[sparse_categorical_accuracy_with_logits])

    TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
    TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    def get_dataset(file):
        dataset = tf.contrib.data.CsvDataset(file, record_defaults=[
            tf.keras.backend.floatx()]*4+[tf.int64], header=True)
        # stack into single Tensor before batching
        dataset = dataset.map(
            lambda a, b, c, d, e: (tf.stack([a, b, c, d]), e))
        dataset = dataset.batch(32).repeat()
        return dataset

    model.fit(get_dataset(train_path), epochs=20, steps_per_epoch=50)
    eval_result = model.evaluate(get_dataset(test_path), steps=30)
    assert eval_result[1] > .9

    return model


def build_p_and_v_model(*, flat_input_size, classes, mlp_extra_layers=cartpole_ppo_tf.MLP_EXTRA_LAYERS, mlp_units=cartpole_ppo_tf.MLP_UNITS):
    latent_shared = tf.keras.Sequential(
        [tf.keras.layers.Dense(mlp_units, input_shape=(flat_input_size,), activation='relu')] +
        [tf.keras.layers.Dense(mlp_units, activation='relu')]*mlp_extra_layers
    )

    policy_logits = tf.keras.Sequential([
        latent_shared,
        tf.keras.layers.Dense(classes, activation=None)
    ])

    value_logit = tf.keras.Sequential([
        latent_shared,
        tf.keras.layers.Dense(1, activation=None)
    ])

    return policy_logits, value_logit


def test_shared_weights():
    p_logits, v_logit = build_p_and_v_model(
        flat_input_size=4, classes=3, mlp_extra_layers=1, mlp_units=10)
    v_sampled_weight = v_logit.layers[0].get_weights()[0][0][0]
    p_sampled_weight = p_logits.layers[0].get_weights()[0][0][0]

    assert v_sampled_weight == p_sampled_weight

    w = v_logit.layers[0].get_weights()
    w[0][0][0] += 1
    v_logit.layers[0].set_weights(w)

    new_v_sampled_weight = v_logit.layers[0].get_weights()[0][0][0]
    new_p_sampled_weight = p_logits.layers[0].get_weights()[0][0][0]

    assert new_v_sampled_weight == new_p_sampled_weight
    assert v_sampled_weight != new_v_sampled_weight
