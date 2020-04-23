# %%
from sklearn.covariance import MinCovDet
import tensorflow as tf
from pathlib import Path
import pandas as pd
import numpy as np
from custom_keras.helpers_keras import neg_ccc, prepare_labels, build_keras_model, prepare_X
from active_gru.my_active_learner import ActiveGru, RandomGru, BaseGru, UncertaintyGru
# from numpy.random import seed
from tensorflow.keras import backend as K
import seaborn as sns
import matplotlib.pyplot as plt
# seed(1)
import os
import pickle
import re
from datetime import datetime
import sys
import time
import gc
from pympler import tracker, refbrowser
import multiprocessing


# %%
dateTimeObj = datetime.now()
time_str = dateTimeObj.strftime("%H:%M_%d_%b_%y")
# %% markdown
# Seed for comparable results:
# %%
tf.random.set_seed(22)
# %% markdown
# # Constants
# %%
INDEX_COLS = ['participant', 'sequence', 'sample']
PATH_X_LABELLED = Path('AVEC2016', 'x_labelled.csv')
PATH_Y_LABELLED = Path('AVEC2016', 'y_labelled.csv')
PATH_X_POOL = Path('AVEC2016', 'x_pool.csv')
PATH_Y_POOL = Path('AVEC2016', 'y_pool.csv')
PATH_X_TEST = Path('AVEC2016', 'x_test.csv')
PATH_Y_TEST = Path('AVEC2016', 'y_test.csv')
PATH_INITIAL_WEIGHTS = os.path.join('experiment', 'init_weights_experiment.h5')
PATH_HISTORY_RANDOM = Path('experiment', 'hist_rand_{}.pkl'.format(time_str))
PATH_HISTORY_ACTIVE = Path('experiment', 'hist_active_{}.pkl'.format(time_str))
PATH_HISTORY_UNCER = Path('experiment', 'hist_uncer_{}.pkl'.format(time_str))

SEQUENCE_LENGTH = 375
SEQ_PER_QUERY = 1
RUNS_EXP = 5
# %% global variables
X_labelled = pd.read_csv(PATH_X_LABELLED, index_col=INDEX_COLS)
y_labelled = pd.read_csv(PATH_Y_LABELLED, index_col=INDEX_COLS)
X_pool = pd.read_csv(PATH_X_POOL, index_col=INDEX_COLS)
y_pool = pd.read_csv(PATH_Y_POOL, index_col=INDEX_COLS)
X_test = pd.read_csv(PATH_X_TEST, index_col=INDEX_COLS)
y_test = pd.read_csv(PATH_Y_TEST, index_col=INDEX_COLS)
# %% What is the MAE and RMSE of always predicting 0
np.sqrt(np.mean((y_test - y_test.mean(axis=0))**2))  # RMSE
np.mean(np.abs((y_test - y_test.mean(axis=0))))  # MAE
# %%
X_labelled.shape[0] / 375
# %%
N_FEATURES_VID = X_labelled.filter(regex='vid', axis=1).shape[-1]
N_FEATURES_AUD = X_labelled.filter(regex='aud', axis=1).shape[-1]
pool_size = 25
# %% markdown
# # Helper Functions
# %% One label start with arousal
y_labelled = y_labelled['y_arousal']
y_pool = y_pool['y_arousal']
y_test = y_test['y_arousal']
# %%


def exp_run(learner: BaseGru, seq_per_query: int):
    """
    Perform experiment with one kind of active learner.

    Inputs:
    seq_per_query -- how many sequences get queried at once

    Output:
    histories -- list of df, each df contains performance of one experiment run


    The active learning loop. As long as there are samples in X_pool, do:
    1. query sequences
    2. Train model on all available labelled data
    3. Evaluate model on test set

    Due to high variance in performance (especially of the random active learner),
    repeat process n times and average performance.
    """
    query_counter = 0
    while learner.x_pool.shape[0] > 0:
        query_counter += 1
        # query sequences
        x_labelled_new, y_labelled_new = learner.query_sequences(seq_per_query)
        # train model
        # learner.train_on_batch(
        #    x_labelled_new, y_labelled_new, epochs=1, batch_size=8)
        learner.train_x_labelled(epochs=2)
        learner.evaluate_on_test_set()
        if query_counter % 1 == 0:
            print('querries: {}'.format(query_counter))
    # append performance of current run to list

    return learner.history
# %%


def load_pickle(path: str):
    return pickle.load(open(path, 'rb'))


def dump_pickle(history: object, path: str):
    pickle.dump(history, open(path, "wb"))


# %% initial weights
"""
model = build_keras_model(SEQUENCE_LENGTH,
                          N_FEATURES_AUD,
                          N_FEATURES_VID,
                          pool_size=pool_size,
                          n_neurons_gru=64,
                          n_neurons_hid_aud=44,
                          n_neurons_hid_vid=100,
                          dropout_rate=0.35,
                          rec_dropout_rate=0.04,
                          rec_l2=0,
                          ker_l2=0)

model.save_weights(PATH_INITIAL_WEIGHTS)
"""
# %% markdown
# # Random active Learner
# %% markdown
# These will be start weights for all experiments:
# %%
# model.save_weights(PATH_INITIAL_WEIGHTS, save_format='h5')

# %% markdown
# For comparison: MAE of always predicting the mean
# %%
# (np.abs(y_test - y_test.mean())).mean()
# %% markdown
# Baseline method: Randomly choose sequences.
# %% markdown
# The active learning loop. As long as there are samples in X_pool, do: <br>
# 1. query sequences
# 2. train model on new sequences
# <br>2.1 With lower frequency: Train model on all available labelled data
# 3. Evaluate model on test set
#
# Handled in function exp_run(). <br>
# Due to high variance in performance (especially of the random active learner)
# , repeat process 5 times and average performance:
# %%


def exp_random():
    histories_random = []
    for i in range(RUNS_EXP):
        print('run: {0}'.format(i + 1))

        model = build_keras_model(SEQUENCE_LENGTH,
                                  N_FEATURES_AUD,
                                  N_FEATURES_VID,
                                  pool_size=pool_size,
                                  n_neurons_gru=64,
                                  n_neurons_hid_aud=44,
                                  n_neurons_hid_vid=100,
                                  dropout_rate=0.38,
                                  rec_dropout_rate=0.04,
                                  rec_l2=0,
                                  ker_l2=0)
        # new active learner object
        random_learner = RandomGru(X_pool, y_pool, model, SEQUENCE_LENGTH,
                                   pool_size, X_test, y_test, X_labelled,
                                   y_labelled)
        # reset weights of model
        random_learner.model.load_weights(PATH_INITIAL_WEIGHTS)
        # train on x_labelled at t_0:
        random_learner.train_x_labelled(epochs=20)
        # one run, i.e. empty X_pool completely:
        history = exp_run(random_learner, SEQ_PER_QUERY)
        # save history
        histories_random.append(history.copy())
        del random_learner
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    return histories_random


# %%
histories_random = exp_random()
# %%

# %%
dump_pickle(histories_random, PATH_HISTORY_RANDOM)


# %% markdown
# # Uncertainty & Outlier based active learner
# %% markdown
# Prepare models:
# %%

# %% markdown
# The active learning loop. As long as there are samples in X_pool, do: <br>
# 1. query sequences
# 2. train model on new sequences
# <br>2.1 With lower frequency: Train model on all available labelled data
# 3. Evaluate model on test set
# %%
histories_active = []
for i in range(RUNS_EXP):
    model_active = build_keras_model(SEQUENCE_LENGTH,
                                     N_FEATURES_AUD,
                                     N_FEATURES_VID,
                                     pool_size=pool_size,
                                     n_neurons_gru=64,
                                     n_neurons_hid_aud=44,
                                     n_neurons_hid_vid=100,
                                     dropout_rate=0.38,
                                     rec_dropout_rate=0.04,
                                     rec_l2=1e-06,
                                     ker_l2=1e-04)
    # %% markdown
    # Clone of above model, but with dropout at prediction time
    # for uncertainty estimation:
    # %%
    model_active_dropout = build_keras_model(SEQUENCE_LENGTH,
                                             N_FEATURES_AUD,
                                             N_FEATURES_VID,
                                             pool_size=pool_size,
                                             n_neurons_gru=64,
                                             n_neurons_hid_aud=44,
                                             n_neurons_hid_vid=100,
                                             dropout_rate=0.4,
                                             rec_dropout_rate=0.05,
                                             rec_l2=1e-06,
                                             ker_l2=1e-04,
                                             training_mode=True)
    print('run: {0}'.format(i + 1))
    # new active learner object
    active_learner = ActiveGru(X_pool, y_pool, model_active,
                               model_active_dropout, SEQUENCE_LENGTH,
                               pool_size, X_test, y_test, X_labelled,
                               y_labelled)
    # reset weights of model
    active_learner.model.load_weights(PATH_INITIAL_WEIGHTS)
    active_learner.model_dropout_test.load_weights(PATH_INITIAL_WEIGHTS)
    # train on x_labelled at t_0:
    active_learner.train_x_labelled(epochs=20)
    # one run, i.e. empty X_pool completely:
    history = exp_run(active_learner, SEQ_PER_QUERY)
    # save history
    histories_active.append(history.copy())
    del active_learner
    del model_active
    del model_active_dropout
    tf.keras.backend.clear_session()
    gc.collect()
    time.sleep(10)

# %%
dump_pickle(histories_active, PATH_HISTORY_ACTIVE)


# %% markdown
# # Uncertainty only active Learner
# %%


def exp_uncertainty():
    histories_uncert = []
    for i in range(RUNS_EXP):
        print('run: {0}'.format(i + 1))
        model_uncert = build_keras_model(SEQUENCE_LENGTH,
                                         N_FEATURES_AUD,
                                         N_FEATURES_VID,
                                         pool_size=pool_size,
                                         n_neurons_gru=64,
                                         n_neurons_hid_aud=44,
                                         n_neurons_hid_vid=100,
                                         dropout_rate=0.35,
                                         rec_dropout_rate=0.04,
                                         rec_l2=0,
                                         ker_l2=0)

        model_uncert_drop = build_keras_model(SEQUENCE_LENGTH,
                                              N_FEATURES_AUD,
                                              N_FEATURES_VID,
                                              pool_size=pool_size,
                                              n_neurons_gru=64,
                                              n_neurons_hid_aud=44,
                                              n_neurons_hid_vid=100,
                                              dropout_rate=0.35,
                                              rec_dropout_rate=0.04,
                                              rec_l2=0,
                                              ker_l2=0,
                                              training_mode=True)
        # new uncert learner object
        uncert_learner = UncertaintyGru(X_pool, y_pool, model_uncert,
                                        model_uncert_drop, SEQUENCE_LENGTH,
                                        pool_size, X_test, y_test, X_labelled,
                                        y_labelled, t=50)

        # reset weights of model
        uncert_learner.model.load_weights(PATH_INITIAL_WEIGHTS)
        uncert_learner.model_dropout_test.load_weights(PATH_INITIAL_WEIGHTS)
        # train on x_labelled at t_0:
        uncert_learner.train_x_labelled(epochs=20)
        # one run, i.e. empty X_pool completely:
        history = exp_run(uncert_learner, SEQ_PER_QUERY)
        # save history
        histories_uncert.append(history)
        del model_uncert
        del model_uncert_drop
        del uncert_learner
        tf.keras.backend.clear_session()
        gc.collect()
        time.sleep(10)
    return histories_uncert


# %%
histories_uncert = exp_uncertainty()
# %%
dump_pickle(histories_uncert, PATH_HISTORY_UNCER)


# %% markdown
# # Plots
# %%
# %%
histories_random = load_pickle('experiment/valence/hist_val_rand_10.pkl')
histories_uncert = load_pickle(
    'experiment/valence/hist_val_uncer_10.pkl')

# %%
fig, ax = plt.subplots(1, 2, figsize=(30, 15))
sns.lineplot(x=histories_random.index, ci='sd', y='ccc',
             data=histories_random, ax=ax[0], color='blue', label='random')

# sns.lineplot(x=histories_active.index, ci='sd', y='ccc',
#             data=histories_active, ax=ax[0], color='green', label='active')
sns.lineplot(x=histories_uncert.index, ci='sd', y='ccc',
             data=histories_uncert, ax=ax[0], color='orange',
             label='MC dropout')

ax[0].legend()
ax[0].set(title='CCC valence', xlabel='queried sequences')
ax[0].xaxis.set_major_locator(plt.MaxNLocator())
# sns.lineplot(x=histories_active.index, ci='sd', y='mae',
#             data=histories_active, ax=ax[1], color='green', label='active')
sns.lineplot(x=histories_uncert.index, ci='sd', y='mae',
             data=histories_uncert, ax=ax[1], color='orange',
             label='MC dropout')
sns.lineplot(x=histories_random.index, ci='sd', y='mae',
             data=histories_random, ax=ax[1], color='blue', label='random')
ax[1].legend()
ax[1].set(title='MAE valence', xlabel='queried sequences')
ax[1].xaxis.set_major_locator(plt.MaxNLocator())

# %%
fig.savefig('experiment_uncer_valence.png')
