hipnn_LOAD_PNAS_MODEL_MD_and_SINGLEPOINT.ipynb - example of running simulations with HIPNN-SEQM model from https://doi.org/10.1073/pnas.2120333119 paper.
PNAS_model.pt - the model used in https://doi.org/10.1073/pnas.2120333119
hipnn_train_model.ipynb - example of pulling the data used in this https://doi.org/10.1073/pnas.2120333119 paper and training new models
hipnn_LOAD_NEW_MODEL_MD_and_SINGLEPOINT.ipynb - example of running simulations with newly trained models.

HIPNN needs to be modified:
1) Replace "seqm_modules.py" in hippynn/interfaces/pyseqm_interface/ with the file in this folder
2) Place "gen_par_full_model.py" in /hippynn/interfaces/pyseqm_interface/gen_par_full_model.py

Tested with torch '1.13.1'