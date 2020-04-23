
# from tensorflow import set_random_seed
from numpy.random import seed
from pathlib import Path
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skopt.plots import plot_evaluations, plot_convergence


from datetime import datetime

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, GRU, Input, Activation, AveragePooling1D, TimeDistributed
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

from custom_keras.helpers_keras import neg_ccc, prepare_labels, build_keras_model, prepare_X
from sklearn.model_selection import KFold

from skopt import gp_minimize, dump, load
from skopt.utils import use_named_args
from skopt.space import Real, Integer, Categorical
# %% get todays date to avoid overriding skopt results
dateTimeObj = datetime.now()
today_str = dateTimeObj.strftime("%d_%b_%y")

seed(1)

# %% Constants
PATH_SKOPT_RESULTS = Path('skopt', 'skopt_results_{0}.pkl'.format(today_str))


PATH_START_WEIGHTS = 'initial_weights.h5'
INDEX_COLS = ['participant', 'sequence', 'sample']
PATH_X_LABELLED = Path('AVEC2016', 'x_labelled.csv')
PATH_Y_LABELLED = Path('AVEC2016', 'y_labelled.csv')
PATH_X_POOL = Path('AVEC2016', 'x_pool.csv')
PATH_Y_POOL = Path('AVEC2016', 'y_pool.csv')


# %% global variables
X_labelled = pd.read_csv(PATH_X_LABELLED, index_col=INDEX_COLS)
y_labelled = pd.read_csv(PATH_Y_LABELLED, index_col=INDEX_COLS)
X_pool = pd.read_csv(PATH_X_POOL, index_col=INDEX_COLS)
y_pool = pd.read_csv(PATH_Y_POOL, index_col=INDEX_COLS)

runs_skopt = 0
N_FEATURES_AUD = X_labelled.filter(regex='aud', axis=1).shape[-1]
N_FEATURES_VID = X_labelled.filter(regex='vid', axis=1).shape[-1]
pool_size = 10
SEQUENCE_LENGTH = 100

# %%
X_labelled.head()
# %% search space
space = [Categorical([8, 16, 32], name='batch_size'),
         Real(0.1, 0.4, name='dropout'),
         Real(0.0, 0.5, name='rec_dropout'),
         Real(1e-06, 1e-04, prior='log-uniform', name='rec_l2'),
         Real(1e-06, 1e-04, prior='log-uniform', name='kernel_l2'),
         Integer(42, 44, name='n_neurons_hid_aud'),
         Integer(90, 180, name='n_neurons_hid_vid')

         ]
# %%
early_stopper = EarlyStopping(
    monitor='val_pred_reg_arou_loss', mode='min',
    min_delta=0.001, patience=70, verbose=0,
)
# %% Start hyperparameters:
x_0 = [32, 0.12, 0.08, 2e-06, 2e-06, 44, 120]


# %% fitness function
@use_named_args(space)
def fitness(batch_size, dropout, rec_dropout, rec_l2, kernel_l2,
            n_neurons_hid_aud, n_neurons_hid_vid):
    """Tune Hyperparameters."""
    # printing during hyper param tuning
    global runs_skopt
    runs_skopt += 1
    print('Run skopt:{}'.format(runs_skopt))
    print('hyperparameters: ')
    # prepare y and X from X_labelled for audio and video mod:
    label_dict = prepare_labels(y_labelled, SEQUENCE_LENGTH, pool_size)
    X_aud_3d = prepare_X(X_labelled, 'aud', SEQUENCE_LENGTH)
    X_vid_3d = prepare_X(X_labelled, 'vid', SEQUENCE_LENGTH)

    # Use Keras to train the model.
    # Create the neural network with these hyper-parameters.
    model = build_keras_model(SEQUENCE_LENGTH, N_FEATURES_AUD, N_FEATURES_VID,
                              dropout_rate=dropout,
                              rec_dropout_rate=rec_dropout,
                              rec_l2=rec_l2,
                              ker_l2=kernel_l2,
                              pool_size=pool_size,
                              n_neurons_hid_aud=n_neurons_hid_aud,
                              n_neurons_hid_vid=n_neurons_hid_vid)
    model.save_weights(PATH_START_WEIGHTS)
    # k-fold cross validation:
    val_losses = []
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in kf.split(
            X_aud_3d):
        # split x
        # split x
        x_train_aud_temp, x_val_aud_temp = X_aud_3d[train_index
                                                    ], X_aud_3d[test_index]
        x_train_vid_temp, x_val_vid_temp = X_vid_3d[train_index
                                                    ], X_vid_3d[test_index]
        # split labels
        y_arou = label_dict['y_arousal']
        y_valence = label_dict['y_valence']
        y_arou_train, y_arou_val = y_arou[
            train_index], y_arou[test_index]
        y_val_train, y_val_val = y_valence[
            train_index], y_valence[test_index]

        # store results of training in history object:
        # load initial weights on new run of k-fold:
        model.load_weights(PATH_START_WEIGHTS)
        history = model.fit([x_train_aud_temp, x_train_vid_temp],
                            [y_arou_train, y_val_train],
                            epochs=200,
                            batch_size=batch_size,
                            validation_data=([x_val_aud_temp, x_val_vid_temp],
                                             [y_arou_val, y_val_val]),
                            callbacks=[early_stopper],
                            verbose=0)
        # minimize mean validation loss without l2 regularization terms
        val_losses.append(
            (history.history["val_pred_reg_arou_loss"][
                -1] + history.history["val_pred_reg_val_loss"][-1]) / 2)
    # delete keras model and clear session to avoid adding
    # different models to the same tf graph:
    del model
    K.clear_session()
    # return mean loss on k held-out sets.
    return statistics.mean(val_losses)


# %%
res_gp = gp_minimize(fitness, space, n_calls=120, random_state=42,
                     x0=x_0)
# %%
# save hyperparams and results
dump(res_gp, PATH_SKOPT_RESULTS)

# %%
# res_gp = load('skopt/skopt_results_06_Feb_20.pkl')

# %%
res_gp.fun

res_gp.x

# res_gp.func_vals

# %% Model standalone:

tensor_board = TensorBoard(log_dir='./logs', histogram_freq=0,
                           write_graph=True,
                           write_grads=False, write_images=False)
# %%


# %%
# %%
X_aud_3d = prepare_X(X_labelled, 'aud', SEQUENCE_LENGTH)
X_vid_3d = prepare_X(X_labelled, 'vid', SEQUENCE_LENGTH)
# %%
X_vid_3d.shape
# %%
label_dict = prepare_labels(y_labelled, SEQUENCE_LENGTH, pool_size)

X_3d = X_labelled.to_numpy().reshape(-1, SEQUENCE_LENGTH,
                                     X_labelled.shape[-1])

model = build_keras_model(SEQUENCE_LENGTH, N_FEATURES_AUD, N_FEATURES_VID,
                          pool_size=pool_size,
                          n_neurons_gru=64,
                          n_neurons_hid_aud=44,
                          n_neurons_hid_vid=100,
                          dropout_rate=0.38, rec_dropout_rate=0.04,
                          rec_l2=0,
                          ker_l2=0)
model.save_weights(PATH_START_WEIGHTS)
# %%
model.summary()
# %%
X_pool.dtypes
# %%
val_losses = []
kf = KFold(n_splits=2, random_state=42, shuffle=True)
for train_index, test_index in kf.split(
        X_aud_3d):
    # split x
    x_train_aud_temp, x_val_aud_temp = X_aud_3d[train_index
                                                ], X_aud_3d[test_index]
    x_train_vid_temp, x_val_vid_temp = X_vid_3d[train_index
                                                ], X_vid_3d[test_index]
    # split labels
    y_arou = label_dict['y_arousal']
    y_valence = label_dict['y_valence']
    y_arou_train, y_arou_val = y_arou[
        train_index], y_arou[test_index]
    y_val_train, y_val_val = y_valence[
        train_index], y_valence[test_index]

    # store results of training in history object:
    # load initial weights on new run of k-fold:
    model.load_weights(PATH_START_WEIGHTS)
    history = model.fit([x_train_aud_temp, x_train_vid_temp],
                        [y_arou_train, y_val_train],
                        epochs=120,
                        batch_size=8,
                        validation_data=([x_val_aud_temp, x_val_vid_temp],
                                         [y_arou_val, y_val_val]),
                        callbacks=[tensor_board],
                        verbose=0)
    val_losses.append(history.history["val_loss"][-1])
