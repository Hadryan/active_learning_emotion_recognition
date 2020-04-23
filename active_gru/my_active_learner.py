from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from scipy.stats import rankdata
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from sklearn.covariance import MinCovDet

from custom_keras.helpers_keras import prepare_labels, ccc, prepare_X
# %%


def dummy_func(x):
    """For getting seq index only."""
    return 1


class BaseGru(ABC):
    def __init__(self, x_pool: pd.DataFrame, y_pool: pd.DataFrame,
                 model: Model,
                 sequence_length: int,
                 n_pool: int,
                 x_test: pd.DataFrame,
                 y_test: pd.DataFrame,
                 x_labelled: pd.DataFrame,
                 y_labelled: pd.DataFrame,
                 ):
        """
        Inputs:
        x_pool -- the data of X_pool at t_0.
        y_pool -- the lables of X_pool at t_0.
        model -- a keras rnn Regression model
        sequence_length -- number of samples per sequence
        n_pool - size of temporal pooling window
        x_labelled -- the data of X_labelled at t_0
        y_ labelled  the labels of X_labelled at t_0.
        """
        self.x_pool = x_pool
        self.x_labelled = self._init_x_labelled(x_labelled)
        self.y_pool = y_pool
        self.y_labelled = y_labelled
        self.model = model
        self.n_features = self.x_pool.shape[-1]
        self.sequence_length = sequence_length
        self.n_pool = n_pool
        self.x_test = x_test
        self.y_test = y_test
        self.queries = 0
        self.queried_seq_tot = 0
        self.history = self._init_history()

    def _init_x_labelled(self, x_labelled):
        """Initialize df for x_labelled."""
        if x_labelled is not None:
            return x_labelled
        else:
            # empty df of same structure as x_pool
            cols = self.x_pool.columns
            return pd.DataFrame(columns=cols)

    def train_on_batch(self, x, y, epochs=20, batch_size=8):
        """Train model on batch.

        x -- usually the instances that just entered X_labelled and left X_pool
        y -- corresponding labels

        """
        self._train(x, y, epochs, batch_size)

    def train_x_labelled(self, epochs=20, batch_size=8):
        """Train model on X_labelled.

        Train model on all available data.
        """
        self._train(self.x_labelled, self.y_labelled, epochs, batch_size)

    def _train(self, x, y, epochs, batch_size):
        """Train model on x and y.

        Internal helper method.
        """
        # prepare y and X:
        y_3d = prepare_labels(
            y, self.sequence_length, self.n_pool)
        X_aud_3d = prepare_X(x, 'aud', self.sequence_length)
        X_vid_3d = prepare_X(x, 'vid', self.sequence_length)
        # fit model
        self.model.fit(
            [X_aud_3d, X_vid_3d], [y_3d
                                   ],
            batch_size=batch_size, epochs=epochs, verbose=0)

    def _init_history(self):
        """Init the df storing performance on test set."""
        cols = ['mae', 'ccc', 'cum_queried_seqs']
        data = np.zeros(
            (self.x_pool.shape[0] // self.sequence_length + 1, len(cols)))
        return pd.DataFrame(data=data, columns=cols)

    def _grow_labels(self, y):
        """Bring the predictions back to the size of the labels in test set.

        Invert mean pooling.
        """
        # init nans with final size, grow time axis
        y_big = np.full(
            (y.shape[0], y.shape[1] * self.n_pool, y.shape[-1]), np.nan)
        # put predictions into new y_big
        y_big[:, ::self.n_pool, :] = y
        y_big = y_big.squeeze(-1)
        # use ffill from pd df for imputation
        df = pd.DataFrame(data=y_big)
        labels_big = df.fillna(method='ffill', axis=1).to_numpy()
        return labels_big

    def _mae(self, y_true, y_pred):
        """Return mae between two arrays."""
        return np.mean(np.abs(y_pred - y_true))

    def evaluate_on_test_set(self):
        """
        Evaluate model on test set.

        Add relevant information to history dataframe.
        """
        # get the labels
        y_3d = self.y_test.to_numpy().reshape(
            (-1, self.sequence_length))
        # get predictions
        X_aud_3d = prepare_X(self.x_test, 'aud', self.sequence_length)
        X_vid_3d = prepare_X(self.x_test, 'vid', self.sequence_length)
        preds = self.model.predict([X_aud_3d, X_vid_3d])

        # grow predictions to original size
        preds_grown = self._grow_labels(preds)
        # compute metrics
        # ccc
        self.history.loc[self.queries, 'ccc'] = ccc(
            y_3d, preds_grown)

        # mae
        self.history.loc[self.queries, 'mae'] = self._mae(
            y_3d, preds_grown)

        # count queried sequences
        self.history.loc[self.queries,
                         'cum_queried_seqs'] = self.queried_seq_tot

    def plot_performance(self):
        """Plot performance on test set."""
        fig, ax = plt.subplots(3, 2, sharex='col', figsize=(20, 10))
        cols = self.history.drop('cum_queried_seqs', axis=1).columns
        for i, col in enumerate(cols):
            j = i // 2
            k = i % 2
            if k == 0:  # first column mae
                ylabel = 'Mean absolute error'
                y_lim = [0., 0.2]
            else:  # 2nd column CCC
                ylabel = 'CCC'
                y_lim = [-1, 1]
            ax[j, k].plot(
                self.history.loc[:self.queries,
                                 'cum_queried_seqs'].to_numpy(),
                self.history.loc[:self.queries,
                                 col].to_numpy(), 'gx-')
            ax[j, k].set(title=col, xlabel='queried sequences', ylabel=ylabel,
                         ylim=y_lim)
        fig.tight_layout()
        return fig

    def query_sequences(self, k_queried_seq: int):
        """Return top k queried sequences with their labels.

        Access the top ranked sequences from X_pool. Append them to X_labelled,
        remove them from X_pool and also return them for training the network.

        Inputs:
        k_queried_seq -- number of sequences to query

        Outputs:
        x_labelled_new -- df, containing new training data
        y_labelled_new -- df, containig lables of x_labelled_new
        """
        # compute rank of each sequence:
        self._comp_seq_scores()
        self._comp_seq_rank()
        # get implicit indexes of first n_queried_seq sequences, in other words
        # the indexes of the queried sequences.
        # if there are not enough samples left, query fewer
        # (last query affected only):
        if self.x_pool.shape[0] < k_queried_seq * self.sequence_length:
            k_queried_seq = int(self.x_pool.shape[0] / self.sequence_length)
        # sort sequences by mean geometric rank
        idx_seq = np.argpartition(
            self.sequence_scores['mean_pos'].to_numpy(), k_queried_seq - 1)

        # get the explicit index (participant and sequence) of the queried
        # sequences:
        idx_queried_sequences = self.sequence_scores.iloc[
            idx_seq[:k_queried_seq], :].index
        idx_unqueried_sequences = self.sequence_scores.iloc[
            idx_seq[k_queried_seq:], :].index
        #  use explicit indexes to filter x_pool:
        # the use of unstack and stack allows fancy indexing, although
        # the sample index is missing in the sequence scores
        # filter x_pool
        x_labelled_new = self.x_pool.unstack(
            level=-1).loc[idx_queried_sequences, :].stack()
        self.x_pool = self.x_pool.unstack(
        ).loc[idx_unqueried_sequences, :].stack()
        # filter y_pool
        y_labelled_new = self.y_pool.unstack(
        ).loc[idx_queried_sequences, :].stack()
        self.y_pool = self.y_pool.unstack(
        ).loc[idx_unqueried_sequences, :].stack()
        # filter sequence_scores
        self.sequence_scores = self.sequence_scores.loc[
            idx_unqueried_sequences, :]

        # append x_labelled_new and y_labelled_new to x_labelled, y_labelled
        self.x_labelled = self.x_labelled.append(x_labelled_new)
        self.y_labelled = self.y_labelled.append(y_labelled_new)

        self.queried_seq_tot += k_queried_seq
        self.queries += 1

        return x_labelled_new, y_labelled_new


class RandomGru(BaseGru):
    def query_sequences(self, k_queried_seq: int):
        """Randomly query k sequences.

        Return the queried sequences, remove them from X_pool (and y_pool) and
        append them to X_labelled (and y_labelled)

        Inputs:
        k -- how many sequences to query

        Outputs:
        x_labelled_new -- df, containing new training data
        y_labelled_new -- df, containig lables of x_labelled_new
        """
        # Query fewer sequences, if X_pool is almost empty:
        if self.x_pool.shape[0] < k_queried_seq * self.sequence_length:
            k_queried_seq = int(self.x_pool.shape[0] / self.sequence_length)
        idx_seq = self.x_pool.index.droplevel('sample')
        # %%
        unique_seqs = idx_seq.unique()
        n_unique_seqs = len(unique_seqs)
        # implicit index of chosen seq, sampling without replacement
        rng = np.random.default_rng()
        idx_iloc = rng.choice(n_unique_seqs, k_queried_seq, replace=False)
        # explicit index of chosen seq
        idx_loc = unique_seqs[idx_iloc]
        # filter x_pool
        x_labelled_new = self.x_pool.unstack().loc[idx_loc, :].stack().copy()
        self.x_pool = self.x_pool.unstack().drop(
            idx_loc, axis=0).stack()
        # filter y_pool
        y_labelled_new = self.y_pool.unstack().loc[idx_loc, :].stack().copy()
        self.y_pool = self.y_pool.unstack().drop(idx_loc, axis=0).stack()

        # append x_labelled_new and y_labelled_new to x_labelled, y_labelled
        self.x_labelled = self.x_labelled.append(x_labelled_new,)
        self.y_labelled = self.y_labelled.append(y_labelled_new,)
        # update counter vars
        self.queried_seq_tot += k_queried_seq
        self.queries += 1

        return x_labelled_new, y_labelled_new


class ActiveGru(BaseGru):
    def __init__(self, x_pool: pd.DataFrame, y_pool: pd.DataFrame,
                 model: Model,
                 model_dropout_test: Model,
                 sequence_length: int,
                 n_pool: int,
                 x_test: pd.DataFrame,
                 y_test: pd.DataFrame,
                 x_labelled: pd.DataFrame,
                 y_labelled: pd.DataFrame,
                 t: int = 50,
                 ):
        """
        Initialize active GRU learner.

        Inputs:
        x_pool -- the data of X_pool at t_0.
        y_pool -- the lables of X_pool at t_0.
        model -- a keras rnn Regression model
        model_dropout_test -- a clone of model, but with dropout also during
        prediction on new data
        sequence_length -- number of samples per sequence
        n_pool -- size of temporal pooling window
        x_labelled -- the data of X_labelled at t_0
        y_ labelled -- the labels of X_labelled at t_0.
        t -- how many predictions to make for uncertainty score determination
        """
        super().__init__(x_pool, y_pool, model, sequence_length, n_pool,
                         x_test,
                         y_test, x_labelled, y_labelled
                         )
        self.model_dropout_test = model_dropout_test
        self.sequence_scores = self._init_seq_scores()
        self.t = t
        # fit Cov estimator to labels
        self.cov_est_labels = MinCovDet().fit(
            self.y_labelled.to_numpy().reshape(-1, 1))

    def _init_seq_scores(self):
        """Initialize df for seq_score of x_pool.

        Seq scores refers to the uncertainty and outlier score of each
        sequence.
        """
        idx = self.x_pool.groupby(
            ['participant', 'sequence']).apply(dummy_func).index
        cols = ['uncertainty',
                'outlier', 'mean_pos']
        data = np.full((idx.shape[0],
                        len(cols)), 0, dtype=np.float32)
        return pd.DataFrame(data=data, columns=cols, index=idx)

    def _comp_seq_scores(self):
        """Compute uncertainty and outlier scores per sequence."""
        # outlier scores:
        # use model to make predictions
        X_aud_3d = prepare_X(self.x_pool, 'aud', self.sequence_length)
        X_vid_3d = prepare_X(self.x_pool, 'vid', self.sequence_length)
        pred = self.model.predict([X_aud_3d, X_vid_3d])
        # compute outlier score of predicted labels: mahalanobis distance
        # to the mean
        pred_flattened = pred.reshape(-1, 1)
        outl = self.cov_est_labels.mahalanobis(
            pred_flattened).reshape(pred.shape)  # reshape outlier scores
        # mean per sequence (axis 1)
        self.sequence_scores['outlier'] = outl.mean(axis=1)

        # uncertainty scores
        # predictions: m_instances * labels_per_sequence *
        # number of predictions (t) * different_labels_predicted
        predictions = np.zeros(
            (self.sequence_scores.shape[0],
             int(self.sequence_length // self.n_pool),
             self.t))
        for j in range(self.t):
            pred_curr = self.model_dropout_test.predict(
                [X_aud_3d, X_vid_3d])
            predictions[:, :, j] = pred_curr.squeeze(axis=-1)
            # drop the last axis, which is just 1 anyway
        # variance along the axis 2 (different dropout predictions)
        # mean per sequence (axis 1)
        pred_var = np.mean(np.var(predictions, axis=2), axis=1)
        self.sequence_scores['uncertainty'] = pred_var
        return

    def _comp_seq_rank(self, ):
        """Compute geometric mean rank of sequences of x_pool.

        Based upon given outlier and uncertainty scores.

        Low ranks (1,2 ...) are better, i.e. queried sooner.
        """
        # rank sequences by uncertainty
        # use -, as we want high values to have a good (low) rank.
        # Given that scipy.rankdata ranks in ascending order
        cols = ['uncertainty', 'outlier', ]
        for i, col in enumerate(cols):
            self.sequence_scores['{}_rank'.format(col)] = rankdata(
                -(self.sequence_scores[col].to_numpy()),
                method='average')

        # compute geometric mean of the ranks
        # filter rank columns:
        rank_data = self.sequence_scores.filter(
            regex=("_rank"), axis=1).to_numpy()
        # rank sequenes
        self.sequence_scores['mean_pos'] = np.prod(
            rank_data, axis=1) ** (1 / rank_data.shape[1])
        return

    def _train(self, x, y, epochs, batch_size):
        """Train model on x and y.

        Internal helper method.
        """
        super()._train(x, y, epochs, batch_size)
        # copy weights to dropout model:
        self.model_dropout_test.set_weights(self.model.get_weights())


class UncertaintyGru(ActiveGru):

    def _comp_seq_scores(self):
        """Compute uncertainty scores per sequence."""
        # outlier scores:
        # use model to make predictions
        X_aud_3d = prepare_X(self.x_pool, 'aud', self.sequence_length)
        X_vid_3d = prepare_X(self.x_pool, 'vid', self.sequence_length)

        # uncertainty scores
        # predictions: m_instances * labels_per_sequence *
        # number of predictions (t) * different_labels_predicted
        predictions = np.zeros(
            (self.sequence_scores.shape[0],
             int(self.sequence_length // self.n_pool),
             self.t,))
        for j in range(self.t):
            pred_curr = self.model_dropout_test.predict(
                [X_aud_3d, X_vid_3d])
            predictions[:, :, j] = pred_curr.squeeze(axis=-1)
            # drop the last axis, which is just of dim 1 anyway
            # variance along the axis 2 (different dropout predictions)
            # mean per sequence (axis 1)
        pred_var = np.mean(np.var(predictions, axis=2), axis=1)
        self.sequence_scores['uncertainty'] = pred_var
        return predictions

    def _comp_seq_rank(self, ):
        """Compute geometric mean rank of sequences of x_pool.

        Based upon given uncertainty scores.

        Low ranks (1,2 ...) are better, i.e. queried sooner.
        """
        # rank sequences by uncertainty
        # use -, as we want high values to have a good (low) rank.
        # Given that scipy.rankdata ranks in ascending order
        cols = ['uncertainty']
        for i, col in enumerate(cols):
            self.sequence_scores['{}_rank'.format(col)] = rankdata(
                -(self.sequence_scores[col].to_numpy()),
                method='average')

        # compute geometric mean of the ranks
        # filter rank columns:
        rank_data = self.sequence_scores.filter(
            regex=("_rank"), axis=1).to_numpy()
        # rank sequenes
        self.sequence_scores['mean_pos'] = np.prod(
            rank_data, axis=1) ** (1 / rank_data.shape[1])
        return


class OutlierGru(BaseGru):
    def __init__(self, x_pool: pd.DataFrame, y_pool: pd.DataFrame,
                 model: Model,
                 sequence_length: int,
                 n_pool: int,
                 x_test: pd.DataFrame,
                 y_test: pd.DataFrame,
                 x_labelled: pd.DataFrame,
                 y_labelled: pd.DataFrame,
                 ):
        super().__init__(x_pool, y_pool, model, sequence_length, n_pool,
                         x_test,
                         y_test, x_labelled, y_labelled
                         )
        self.sequence_scores = self._init_seq_scores()
        # fit Cov estimator to labels
        self.cov_est_labels = MinCovDet().fit(
            self.y_labelled.to_numpy().reshape(-1, 1))

    def _init_seq_scores(self):
        """Initialize df for seq_score of x_pool.

        Seq scores refers to the uncertainty and outlier score of each
        sequence.
        """
        idx = self.x_pool.groupby(
            ['participant', 'sequence']).apply(dummy_func).index
        cols = [
            'outlier']
        data = np.full((idx.shape[0],
                        len(cols)), 0, dtype=np.float32)
        return pd.DataFrame(data=data, columns=cols, index=idx)

    def _comp_seq_scores(self):
        """Compute uncertainty and outlier scores per sequence."""
        # outlier scores:
        # use model to make predictions
        X_aud_3d = prepare_X(self.x_pool, 'aud', self.sequence_length)
        X_vid_3d = prepare_X(self.x_pool, 'vid', self.sequence_length)
        pred = self.model.predict([X_aud_3d, X_vid_3d])
        # compute outlier score of predicted labels: mahalanobis distance
        # to the mean
        pred_flattened = pred.reshape(-1, 1)
        outl = self.cov_est_labels.mahalanobis(
            pred_flattened).reshape(pred.shape)  # reshape outlier scores
        # mean per sequence (axis 1)
        self.sequence_scores['outlier'] = outl.mean(axis=1)
        return

    def _comp_seq_rank(self, ):
        """Compute geometric mean rank of sequences of x_pool.

        Based upon given uncertainty scores.

        Low ranks (1,2 ...) are better, i.e. queried sooner.
        """
        # rank sequences by uncertainty
        # use -, as we want high values to have a good (low) rank.
        # Given that scipy.rankdata ranks in ascending order
        cols = ['outlier']
        for i, col in enumerate(cols):
            self.sequence_scores['{}_rank'.format(col)] = rankdata(
                -(self.sequence_scores[col].to_numpy()),
                method='average')

        # compute geometric mean of the ranks
        # filter rank columns:
        rank_data = self.sequence_scores.filter(
            regex=("_rank"), axis=1).to_numpy()
        # rank sequenes
        self.sequence_scores['mean_pos'] = np.prod(
            rank_data, axis=1) ** (1 / rank_data.shape[1])
        return
