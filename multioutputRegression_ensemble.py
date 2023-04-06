import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error as mse
from pprint import pprint
from autosklearn.regression import AutoSklearnRegressor
import argparse


path = r'./ModelComparisons'
SEED = 2022

parser = argparse.ArgumentParser()
parser.add_argument("--time_left_for_this_task", type=int, default=3600)
parser.add_argument("--per_run_time_limit", type=int, default=360)
parser.add_argument("--memory_limit", type=int, default=8, help="Memory limit in GB")
parser.add_argument("--tmp_folder_name", type=str, default='tmp')
parser.add_argument("--n_jobs", type=int, default=1)
opt = parser.parse_args()
print(opt)

tmp_folder=os.path.join(path,'tmp',opt.tmp_folder_name)

# ============
# Data Loading

'''
Cmpds = pd.read_csv(os.path.join(path, 'Data', 'Data_Split.txt'), sep='\t', dtype=str)
Cmpds_train = Cmpds[Cmpds.Split_RandomPick == 'Training'].COMPOUND_NAME
Cmpds_train = set(Cmpds_train.to_list())
Cmpds_test = Cmpds[Cmpds.Split_RandomPick == 'Test'].COMPOUND_NAME
Cmpds_test = set(Cmpds_test.to_list())

CID2name = pd.read_csv(os.path.join(path, 'Data', 'SMILES_138cmpds.tsv'), sep='\t')
Cmpds_train = CID2name[CID2name.CompoundName.isin(Cmpds_train)].CID.to_list()
Cmpds_test = CID2name[CID2name.CompoundName.isin(Cmpds_test)].CID.to_list()

X = pd.read_csv(os.path.join(path, 'Data', 'MolDescriptors.tsv'), sep='\t', dtype=str, index=0)
X_train = X[X.index.isin(Cmpds_train)]
X_test = X[X.index.isin(Cmpds_test)]
'''

Y_train = pd.read_csv(os.path.join(path, 'Data', 'averageDataPerTreatment_TG_TrainingSet.tsv'), sep='\t')
treatments_train = Y_train[Y_train.columns[0:3]]
Y_train = Y_train[Y_train.columns[3:]]

Y_test = pd.read_csv(os.path.join(path, 'Data', 'averageDataPerTreatment_TG_TestSet.tsv'), sep='\t')
treatments_test = Y_test[Y_test.columns[0:3]]
Y_test = Y_test[Y_test.columns[3:]]

X = pd.read_csv(os.path.join(path, 'Data', 'MolDescriptors.tsv'), sep='\t', index_col=0)
CID2name = pd.read_csv(os.path.join(path, 'Data', 'SMILES_138cmpds.tsv'), sep='\t')

def Time(SACRIFICE_PERIOD):
    switcher = {
        '4 day': 3 / 28,
        '8 day': 7 / 28,
        '15 day': 14 / 28,
        '29 day': 28 / 28
    }
    return switcher.get(SACRIFICE_PERIOD, 'error')

def Dose(DOSE_LEVEL):
    switcher = {
        'Low': 0.1,
        'Middle': 0.3,
        'High': 1
    }
    return switcher.get(DOSE_LEVEL, 'error')

X_train = pd.DataFrame()
for i in range(len(treatments_train)):
    Stru = X[X.index.isin(CID2name[CID2name.CompoundName==treatments_train.COMPOUND_NAME[i]].CID)]
    Stru.insert(Stru.shape[1],'Time',Time(treatments_train.SACRI_PERIOD[i]))
    Stru.insert(Stru.shape[1], 'Dose', Dose(treatments_train.DOSE_LEVEL[i]))
    X_train = pd.concat([X_train, Stru], ignore_index=True)

X_test = pd.DataFrame()
for i in range(len(treatments_test)):
    Stru = X[X.index.isin(CID2name[CID2name.CompoundName==treatments_test.COMPOUND_NAME[i]].CID)]
    Stru.insert(Stru.shape[1],'Time',Time(treatments_test.SACRI_PERIOD[i]))
    Stru.insert(Stru.shape[1], 'Dose', Dose(treatments_test.DOSE_LEVEL[i]))
    X_test = pd.concat([X_test, Stru], ignore_index=True)


# ====================
# Build and fit models
# ====================
if __name__ == "__main__":
    automl = AutoSklearnRegressor(
        time_left_for_this_task=opt.time_left_for_this_task,
        per_run_time_limit=opt.per_run_time_limit,
        tmp_folder=tmp_folder,
        delete_tmp_folder_after_terminate=False,
        n_jobs=opt.n_jobs,
        memory_limit=opt.memory_limit*1024,
        resampling_strategy="cv",
        resampling_strategy_arguments={"folds": 5},
        seed=SEED,
        )
    automl.fit(X_train, Y_train, dataset_name=opt.tmp_folder_name)
    print(automl.sprint_statistics())  # Print statistics about the auto-sklearn run
    print(automl.leaderboard())  # view the models found by auto-sklearn
    pprint(automl.show_models(), indent=4)  # print the final ensemble constructed by auto-sklearn


    automl.refit(X_train.copy(), Y_train.copy())

    # Get the Score of the final ensemble...
    train_predictions = automl.predict(X_train)
    print("Train R2 score:", r2_score(Y_train, train_predictions))
    predictions = automl.predict(X_test)
    print("Test R2 score:", r2_score(Y_test, predictions))
    print("Test multioutput R2 score:", r2_score(Y_test, predictions, multioutput='raw_values'))
    print("Test MSE:", mse(Y_test, predictions))
    print("Test multioutput MSE:", mse(Y_test, predictions, multioutput='raw_values'))
    np.save(os.path.join(path, 'Results', 'Predictions_multioutputRegression.npy'), predictions)
    print(automl.get_configuration_space(X_train, Y_train))
