hipnn_LOAD_PNAS_MODEL_MD_and_SINGLEPOINT.ipynb - example of running simulations with HIPNN-SEQM model from https://doi.org/10.1073/pnas.2120333119 paper.
PNAS_model.pt - the model used in https://doi.org/10.1073/pnas.2120333119
hipnn_train_model.ipynb - example of pulling the data used in this https://doi.org/10.1073/pnas.2120333119 paper and training new models
hipnn_LOAD_NEW_MODEL_MD_and_SINGLEPOINT.ipynb - example of running simulations with newly trained models.

HIPNN needs to be modified:
1) Replace "seqm_modules.py" in hippynn/interfaces/pyseqm_interface/ with the file in this folder
2) Place "gen_par_full_model.py" in /hippynn/interfaces/pyseqm_interface/gen_par_full_model.py


Notes for training a new model:
1) Atoms in each molecule need to be sorted in descending order in training set, e.g. [8,6,1,1].
2) Zero-padding should be used to make all molecules of the same length, e.g. [[6,1,1,1,1], [8,1,1,0,0], [8,6,1,1,0]].
3) hipnn_train_model.ipynb pulls a ready-to-use training set in .npy format used in this paper https://doi.org/10.1073/pnas.2120333119.


NB:
1) Just one model file needs to be loaded to use the model from from https://doi.org/10.1073/pnas.2120333119 paper. See hipnn_LOAD_PNAS_MODEL_MD_and_SINGLEPOINT.ipynb
2) When training a new model, both model file and state_dict need to be loaded. See hipnn_LOAD_NEW_MODEL_MD_and_SINGLEPOINT.ipynb

Tested with torch '1.13.1'