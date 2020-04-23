import numpy as np
import pandas as pd

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, GRU, Input, AveragePooling1D, TimeDistributed, Concatenate
from tensorflow.keras import Model, regularizers

from tensorflow.keras.optimizers import Nadam
# %%


def build_keras_model(sequence_length, n_features_aud,
                      n_features_vid,
                      pool_size=10,
                      n_neurons_gru=64,
                      n_neurons_hid_aud=50, n_neurons_hid_vid=60,
                      dropout_rate=0,
                      rec_dropout_rate=0,
                      training_mode=False,
                      rec_l2=0,
                      ker_l2=0):
    """Build and compile a keras model."""
    # audio inputs
    inputs_aud = Input(
        shape=(sequence_length, n_features_aud), name='inputs_aud')
    inputs_aud_d = TimeDistributed(
        Dropout(0.0))(inputs_aud, training=training_mode)
    aud_hid_1 = TimeDistributed(
        Dense(n_neurons_hid_aud, activation='relu',
              kernel_regularizer=regularizers.l2(ker_l2)))(inputs_aud_d)
    # aud_hid_2 = TimeDistributed(
    #    Dropout(0.0))(aud_hid_1, training=training_mode)
    pooled_aud = AveragePooling1D(
        pool_size=pool_size, padding='same')(aud_hid_1)

    # video inputs
    inputs_vid = Input(
        shape=(sequence_length, n_features_vid), name='inputs_vid')
    inputs_vid_d = TimeDistributed(
        Dropout(0.0))(inputs_vid, training=training_mode)
    vid_hid_1 = TimeDistributed(
        Dense(n_neurons_hid_vid, activation='relu',
              kernel_regularizer=regularizers.l2(ker_l2)))(inputs_vid_d)
    # vid_hid_2 = TimeDistributed(
    #    Dropout(0.0))(vid_hid_1, training=training_mode)
    pooled_vid = AveragePooling1D(
        pool_size=pool_size, padding='same')(vid_hid_1)

    # concatenate tensors along feature dim
    fusion_in = Concatenate(axis=-1)([pooled_aud,
                                      pooled_vid])
    fusion_out = TimeDistributed(
        Dropout(0.0))(fusion_in, training=training_mode)
    fusion_out = TimeDistributed(
        Dense(n_neurons_gru, activation='tanh',
              kernel_regularizer=regularizers.l2(ker_l2)))(fusion_out)

    # dropout_1 = keras.layers.TimeDistributed(Dropout(dropout_rate))(hidden_1)
    hidden_2 = GRU(n_neurons_gru,
                   return_sequences=True,
                   kernel_regularizer=regularizers.l2(ker_l2),
                   recurrent_regularizer=regularizers.l2(rec_l2),
                   dropout=dropout_rate,
                   recurrent_dropout=rec_dropout_rate,
                   )(fusion_out, training=training_mode)
    hidden_3 = GRU(n_neurons_gru,
                   return_sequences=True,
                   dropout=dropout_rate,
                   recurrent_dropout=rec_dropout_rate,
                   kernel_regularizer=regularizers.l2(ker_l2),
                   recurrent_regularizer=regularizers.l2(rec_l2),

                   )(hidden_2, training=training_mode)

    # prediction regression:
    # regression arousal
    pred_reg = GRU(1, return_sequences=True,
                   activation=None,
                   name='pred_reg'
                   )(hidden_3)
    model = Model(inputs=[inputs_aud, inputs_vid],
                  outputs=[pred_reg])

    model.compile(
        optimizer=Nadam(),
        loss=neg_ccc,
        metrics=['mae', neg_ccc])
    return model

# %%


def neg_ccc(y_true, y_pred):
    """Lin's Concordance correlation coefficient.

    The concordance correlation coefficient is the correlation between two
    variables that fall on the 45 degree line through the origin.
    It is a product of
    - precision (Pearson correlation coefficient) and
    - accuracy (closeness to 45 degree line)

    Interpretation:
    - `rho_c =  1` : perfect agreement
    - `rho_c =  0` : no agreement
    - `rho_c = -1` : perfect disagreement

    Args:
    - y_true: ground truth
    - y_pred: predicted values

    Returns:
    - concordance correlation coefficient (float)
    """

    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)
    sample_means_y_true = K.mean(y_true, axis=0)  # (, m_instances)
    sample_means_y_pred = K.mean(y_pred, axis=0)

    sample_vars_y_true = K.var(y_true, axis=0)
    sample_vars_y_pred = K.var(y_pred, axis=0)

    sample_covs = K.mean(
        (y_true - sample_means_y_true) * (y_pred - sample_means_y_pred),
        axis=0)

    ccc = 2 * sample_covs / (
        sample_vars_y_true + sample_vars_y_pred + (
            (sample_means_y_true - sample_means_y_pred) ** 2
        ))
    # ccc = K.mean(ccc)
    return 1 - ccc


def prepare_labels(y, sequence_length, pool_size):
    """Take df of 2d labels and prepare for keras.

    Reshape 3d.
    Regression labels:  AveragePooling
    """
    label_smoother = AveragePooling1D(
        pool_size=pool_size, padding='valid')
    if (type(y) == pd.Series) | (type(y) == pd.DataFrame):
        y = y.to_numpy()
    label_3d = y.reshape(
        -1, sequence_length, 1)
    label_3d_smooth = K.eval(label_smoother.call(label_3d))
    return label_3d_smooth


def prepare_X(X: pd.DataFrame, filter: str, seq_length: int):
    """Filter features from modality and reshape for feeding to keras.

    Inputs:
    X -- all training data
    filter -- str to match columns of one modality
    seq_length -- 2nd dim of tensor

    Outputs:
    X_3d -- data of matched modality in 3d shape
    (m_instances, timesteps, features)
    """
    X_3d = X.filter(regex=filter, axis=1).to_numpy()
    # last dim: feature of modality
    X_3d = X_3d.reshape(-1, seq_length, X_3d.shape[-1])
    return X_3d


# %%
def ccc(y_true, y_pred,
        sample_weight=None,
        multioutput='uniform_average'):
    """Concordance correlation coefficient.

    The concordance correlation coefficient is a measure of inter-rater
    agreement.
    It measures the deviation of the relationship between predicted and true
    values from the 45 degree angle.
    Read more:
    https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    Original paper: Lawrence, I., and Kuei Lin. "A concordance correlation
    coefficient to evaluate reproducibility." Biometrics (1989): 255-268.
    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    Returns

    -------
    loss : A float in the range [-1,1]. A value of 1 indicates perfect
    agreement
    between the true and the predicted values.
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    sample_means_y_true = np.mean(y_true, axis=0)  # (, m_instances)
    sample_means_y_pred = np.mean(y_pred, axis=0)

    sample_vars_y_true = np.var(y_true, axis=0)
    sample_vars_y_pred = np.var(y_pred, axis=0)

    sample_covs = np.mean(
        (y_true - sample_means_y_true) * (y_pred - sample_means_y_pred),
        axis=0)

    ccc = 2 * sample_covs / (
        sample_vars_y_true + sample_vars_y_pred + (
            (sample_means_y_true - sample_means_y_pred) ** 2
        ))
    ccc = np.mean(ccc)
    return ccc


# %%
if __name__ == '__main__':
    #  test np ccc implementation
    y_true = np.random.random(size=(100, 2, 1)) * 2
    # perfect agreement, should yield neg_ccc=-1
    y_pred = y_true
    a = ccc(y_true, y_pred)
    a
    # %% test keras 1 - ccc implementation

    #
    y_true.shape
    #
    y_true_k = K.variable(y_true)
    y_pred_k = K.variable(y_pred)
    ccc_keras = K.eval(neg_ccc(y_true_k, y_pred_k))

    # %%
    ccc_keras
    # %% now random data, should yield ccc=0
    y_true = np.random.random((1000, 100))
    y_pred = (np.random.random((1000, 100)) - 0.5)

    y_true_k = K.variable(y_true)
    y_pred_k = K.variable(y_pred)
    ccc_keras = K.eval(neg_ccc(y_true_k, y_pred_k))
    # %%
    ccc_keras
