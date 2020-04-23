# Active Learning on RECOLA
## important Notebooks/ scripts
- pipeline.ipynb: Preprocessing of data set for keras
- hyper_tuning_t0.py: Search Hyper parameters on X_labelled
- experiment.py: the global active learning experiment
- experiment_details_uncer.ipynb: Detailed plots for one run of the experiment

## important folders:
- AVEC2016: dataset
- data_prep: Data preprocessing including PCA, prepare data for keras
- active_gru: Class to implement active learning in my_active_learner.py
- custom keras: Model implementation
- experiment: contains the results and figures

## virtual env:
The code is tested with the following setup:
### available in conda:
- jupyter
- numpy 1.18.1
- pandas 0.25.3
- tensorflow-gpu 2.0 (or cpu)
- Keras
- h5py
- scikit-learn 0.22
- scikit-optimize 0.5.2
- seaborn
- matplotlib
These packages may be installed manually in conda/pip or from env_dependencies_cross_os.yml
### pip available only:
- liac-arff 2.4.0
Manual pip installation only

## The dataset:
Due to privacy concerns, the data is not included. It has to be added.
Video_appearance and audio features were used.
- Place the .arff files from features_audio/arousal in AVEC2016/features_audio/arousal
- Place the .arff files from  features_video_appearance/arousal in
AVEC2016/features_video_appearance/arousal
Labels:
- Place the folder gold_standard (with subfolders) in AVEC2016
