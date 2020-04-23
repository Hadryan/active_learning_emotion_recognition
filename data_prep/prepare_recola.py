from scipy.io import arff as arff_scipy
import arff
import pandas as pd
import numpy as np
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import re
import math

from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import MaxNLocator


def choose_random_sequences(n_sequences, X):
    """Randomly choose n sequences.

    Stratified to contain equal number of sequences per participant.
    Inputs:
    n_sequences -- number of sequences
    X -- df

    Returns:
    df_chosen -- df containing chosen sequences
    """
    chosen_seq = []
    seq_per_part = n_sequences // len(X.index.unique(level='participant'))
    # loop over participants
    for part, df in X.groupby('participant'):
        # highest sequence number for current participant
        # max_sequence = max(df.index.get_level_values('sequence'))
        max_sequence = 124
        # chosen sequences
        chosen_seq_num = np.random.randint(0, max_sequence + 1, seq_per_part)
        idx = pd.IndexSlice
        chosen_seq.append(df.loc[idx[:, :, chosen_seq_num], :])
    df_chosen = pd.concat(chosen_seq, axis=0)
    return df_chosen


def add_sequence_index(df, sequence_length):
    """Add index based upon sequence."""
    idx_seq = df.groupby('participant').apply(
        helper_sequence_index, sequence_length=sequence_length)
    return idx_seq.to_numpy().flatten()


def helper_sequence_index(df, sequence_length):
    """Helper func for add_sequence_index."""
    n_samples = df.shape[0]
    n_sequences = n_samples // sequence_length
    a = np.arange(1, n_samples) % n_sequences
    a = np.concatenate(
        [(np.full((n_samples % sequence_length), np.max(a) + 1)), a])
    return pd.Series(np.sort(a))


def get_labels(path):
    """
    Load the labels to dataframe.
    """
    parts_df = []
    for part in path.iterdir():
        data = arff.load(open(part))
        df_l = pd.DataFrame(data=data['data'], columns=[
                            'participant', 'sample', 'label'])
        df_l = df_l.set_index(keys=['participant', 'sample'])
        parts_df.append(df_l)
    parts_df_final = pd.concat(parts_df, axis=0)
    #labels = parts_np[:, 2].astype(np.float32)
    return parts_df_final


def load_data(path_part: Path, drop_frame_time=True):
    """Load arff file of one participant into pandas df.

    Arguments:
    path_part -- the path of one .arff data file
    drop_frame_time -- boolean, whether to keep frametime
                        in df after adding it to the index of df

    Return:
    df_part -- a pd df with the data of one arff file
    part_name -- string, Name of file, i.e. train_4

    """
    if type(path_part) == str:
        path_part = Path(path_part)
    data = arff_scipy.loadarff(path_part)
    # find out the participant
    reg_match = re.search('(train|dev|test)_[0-9]+', path_part.name)
    part_name = reg_match.group(0)
    # create multi-index: participant as first index, time as second index
    df_part = pd.DataFrame(data=data[0], columns=data[1])
    df_part = df_part.rename(
        columns={'frameTime': 'frametime'})  # naming consistent
    sample = df_part['frametime']
    assert(~(sample.isna().any()))  # assert no missing frameTimes
    iterables = [[part_name, ], sample]
    idx = pd.MultiIndex.from_product(
        iterables, names=['participant', 'sample'])
    df_part.set_index(idx, inplace=True)
    return df_part, part_name


def data_generator(path_parts: str, iter_test=False):
    """Yield the df of one participant for the current modality at a time.

    Arguments:
    path_parts -- dir of participant files for one modality
    iter_test -- boolean, whether to iterate train or test set files

    """
    participants = Path(path_parts)
    for df_part, part_name in (load_data(participant) for
                               participant in participants.iterdir()
                               if(('test' in participant.name) == iter_test)):
        yield df_part, part_name


def fit_scalers_to_train_data(data_path: str):
    """Iterate over training data to fit scaler.

    Arguments:
    data_path -- the root folder of the data as string

    """
    scalers = {}  # dict of statistics
    data_dir = Path(data_path)
    for mod_folder in data_dir.iterdir():  # loop over modality folders
        for reg_task in mod_folder.iterdir():  # loop over valence or arousal
            scaler_key = get_mod_task(reg_task)
            print(scaler_key)
            std_scaler = StandardScaler()
            # loop over participant
            for df_part, _ in data_generator(reg_task):
                # incrementally fit data, exclude frame time column
                std_scaler.partial_fit(df_part.drop('frametime', axis=1))
            scalers[scaler_key] = std_scaler
    return scalers


def scale_data(data_path: str, scalers: dict, root_folder='data',
               scale_test=False):
    """Scale data with previously fitted scalers.

    Also impute NanS with mean. Save as arff files to disk.

    Arguments:
    data_path -- the root folder of the data as string
    scalers -- dict with fitted scalers as values.
    scale_test -- whether to iterate train or test set.

    """
    data_dir = Path(data_path)
    for mod_folder in data_dir.iterdir():  # loop over modality folders
        for reg_task in mod_folder.iterdir():  # loop over valence or arousal
            scaler_key = get_mod_task(reg_task)
            std_scaler = scalers[scaler_key]
            # loop over participants, train or test set depending on boolean:
            for df_part,  part_name in data_generator(reg_task,
                                                      iter_test=scale_test):
                frame_time = df_part['frametime']
                df_part_num = df_part.drop('frametime', axis=1)
                np_part_sca = std_scaler.transform(
                    df_part_num)  # z-scores numpy array
                df_part_sca = pd.DataFrame(
                    np_part_sca, index=df_part.index,
                    columns=df_part_num.columns)
                # corresponds to imputing with mean as z-scores are used
                df_part_sca.fillna(value=0, inplace=True)
                df_part_sca['frametime'] = frame_time
                # path where to dump z-transformed arff file:
                new_root = root_folder + '_z'
                z_path = os.path.join(str(reg_task).replace(root_folder,
                                                            new_root),
                                      part_name + '.arff')
                # dump arff file
                attributes = [(c, 'REAL') for c in df_part_sca.columns.values]
                arff_dic = {
                    'attributes': attributes,
                    'data': df_part_sca.to_numpy(),
                    'relation': 'myrel',
                }
                with open(z_path, "w", encoding="utf-8") as f:
                    arff.dump(arff_dic, f)
    return  # nothing


def fit_pca_incremental(data_path_z: Path):
    """Incremetally fit PCA to provided data.

    PCAs are fitted seperately per modality.
    Return:
    pcas -- dict with modality a key and fitted pca object as value

    """
    if type(data_path_z) == str:
        data_path_z = Path(data_path_z)
    pcas = {}
    for mod_folder in data_path_z.iterdir():  # loop over modality folders
        for reg_task in mod_folder.iterdir():  # loop over valence or arousal
            pca_key = get_mod_task(reg_task)
            pca = IncrementalPCA(n_components=None)
            # loop over participant
            for df_part, _ in data_generator(reg_task):
                # incrementally fit data, exlude frame time column
                pca.partial_fit(df_part.drop('frametime', axis=1))
            pcas[pca_key] = pca
    return pcas


def pca_transform(data_path_z: Path, pcas_fitted: dict, comp_to_keep: dict):
    """Use fitted pca object to transform the data.

    Transform data seperately per modality.
    pcas_fitted --dict with
    """
    if type(data_path_z) == str:
        data_path_z = Path(data_path_z)
    df_mod_list = []
    for mod_folder in data_path_z.iterdir():  # loop over modality folders
        for reg_task in mod_folder.iterdir():  # loop over valence or arousal
            pca_key = get_mod_task(reg_task)
            n_components = comp_to_keep[pca_key]  # first n axes to projetc on
            pca = pcas_fitted[pca_key]
            df_parts_list = []  # list of dfs with same mod and task
            cols = ["{}th comp {}".format(i, pca_key)
                    for i in range(n_components)]
            for df_part, _ in data_generator(reg_task):
                # incrementally transform data, exlude frame time column
                # also only keep first n components:
                np_data = pca.transform(df_part.drop('frametime', axis=1))[
                    :, :n_components]
                df = pd.DataFrame(
                    data=np_data, index=df_part.index, columns=cols)
                df_parts_list.append(df)
            df_mod = pd.concat(df_parts_list, axis=0)
            df_mod_list.append(df_mod)
    df_final = pd.concat(df_mod_list, axis=1)
    return df_final


def plot_cum_explained_var(pca_fitted: IncrementalPCA, ax):
    """Plot the  percentage of explained variance over the pca components.

    Inputs:
    pca-fitted -- a fitted scikit learn pca object
    """
    var_ratio_cum = np.cumsum(pca_fitted.explained_variance_ratio_)
    ax.plot(
        np.arange(1, var_ratio_cum.shape[0] + 1), var_ratio_cum, 'o-g',
        markersize=0.2)
    ax.set(xlabel='nth component', ylabel='explained variance in %')
    # enforce integer ticks on x-axis
    nbins = min(20, pca_fitted.n_components_)
    ax.xaxis.set_major_locator(MaxNLocator(
        nbins=nbins,  integer=True))
    return ax


def get_mod_task(path_obj: Path):
    """Find out the modality and reg task by applying regex on path object.

    Arguments:
    path_obj -- the path to determine mod and task from

    Returns:
    mod_task -- String with modality and task information

    """
    # reg_ex = re.search(
    #    'features_[a-z]+/[a-z]+', str(path_obj), re.IGNORECASE)
    mod_task = '{0}_{1}'.format(path_obj.parts[2], path_obj.stem)
    return mod_task


def spot_files_with_nans(data_path: str):
    """Spot all arff files with at least one missing value.

    Arguments:
    data_path -- the root folder of the data as string

    Returns:
    missing_values -- a list with the arff files with NaNs as Strings

    """
    data_dir = Path(data_path)
    missing_values = []
    for mod_folder in data_dir.iterdir():  # loop over modality folders
        for reg_task in mod_folder.iterdir():
            for df_part, part_name in data_generator(reg_task):
                # any missing values per participant?
                if df_part.isna().any().any():
                    file_name = get_mod_task(reg_task) + part_name
                    missing_values.append(file_name)
    return missing_values
