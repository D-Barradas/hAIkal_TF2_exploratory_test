# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf


# %%
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,matthews_corrcoef
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import sklearn


# %%
# print(f"PyTorch Version: {torch.__version__}")
print(f"Pandas  Version: {pd.__version__}")
# print(f"Snorkel Version: {snorkel.__version__}")
print(f"TF2 Version: {tf.__version__}")
print(f"sklearn Version: {sklearn.__version__}")


# %%
PDB_BM5 = [
'1EXB','1JTD','1M27','1RKE','2A1A','2GAF','2GTP','2VXT','2W9E',
'2X9A','2YVJ','3A4S','3AAA','BAAD','3AAD','3BIW','3BX7',
'3DAW','3EO1','3EOA','3F1P','3FN1','3G6D','3H11',
'3H2V','3HI6','3HMX','3K75','3L5W','3L89','3LVK','3MXW',
'BP57','CP57','3P57','3PC8','3R9A','3RVW','3S9D','3SZK',
'3V6Z','3VLB','4DN4','4FQI','4FZA','4G6J','4G6M','4GAM',
'4GXU','4H03','4HX3','4IZ7','4JCV','4LW4','4M76'
]

target_classifiers =  ['AP_DARS', 'AP_DDG_W', 'AP_DFIRE2', 'AP_GOAP_DF', 'AP_MPS', 'AP_PISA',
       'AP_dDFIRE', 'AlAr', 'ArAr', 'BSA_Apolar', 'CONSRANK_val', 'CP_BFKV',
       'CP_BT', 'CP_D1', 'CP_HLPL', 'CP_MJ2h', 'CP_MJ3h', 'CP_MJPL', 'CP_Qp',
       'CP_RMFCA', 'CP_RMFCEN1', 'CP_RMFCEN2', 'CP_SKOIP', 'CP_SKOb', 'CP_TB',
       'CP_TD', 'CP_TEl', 'CP_TS', 'CP_TSC', 'CP_ZS3DC_MIN', 'DDG_V',
       'PROPNSTS', 'PYDOCK_TOT', 'SIPPER', 'cips_AlAr', 
        'idx', 'RFC', 'TF2','NNC', 'label_binary'
                      ]
scoring_functions = ['CONSRANK_val', 'CP_HLPL', 'CP_MJ3h', 'DDG_V', 'CP_RMFCA', 'AP_GOAP_DF', 'NNC', 'RFC']


# %%
def save_metrics_results(model,x_test,y_test,tag):
    # target_names = ['Incorrect', 'Correct']

    # y_pred = model.predict_classes(x_test.to_numpy())
    y_pred = (model.predict(x_test.to_numpy()) > 0.5).astype("int32")
    cr = classification_report(y_true=y_test, y_pred=y_pred,output_dict=True,digits=4)
    mmc = matthews_corrcoef(y_true=y_test, y_pred=y_pred)
    # print (cr)
    acc = cr["accuracy"]
    rec_false = cr["False"]["recall"]
    rec_true  = cr["True"]["recall"]
    pres_false = cr["False"]["precision"]
    pres_true = cr["True"]["precision"]
    f1_false =  cr["False"]["f1-score"]
    f1_true =  cr["True"]["f1-score"]

    results = {
        "Accuracy": acc,
        "Recall_inc":rec_false,
        "Recall_cor":rec_true,
        "Precision_inc":pres_false ,
        "Precision_cor":pres_true,
        "F1_inc":f1_false,
        "F1_cor":f1_true,
        "MCC":mmc
    }
    mean_df = pd.DataFrame(data=results,index=[f"{tag}"])
    return mean_df.round(4)


# %%
# my_model_1 = tf.keras.models.load_model("../models/TF2_models_snorkel_trained_v1.h5")


# %%
# path2i = "../models/TF2_hp_models_snorkel_trained_v2.h5"
# my_model_2 = tf.keras.models.load_model(path2i)


# %%
def read_data() :
    path1 = "../data/BM5_analysis_balanced_data.csv" 
    path2 = "../data/Clean_dataframe_unbalanced_all_data_ccharppi_4_march_2020_complete_for_snorkel.csv"
    #path1 = "../data/Clean_dataframe_balanced_all_data_ccharppi_28_march_2020_complete.csv" 
    #path2 = "../data/Clean_dataframe_unbalanced_all_data_ccharppi_4_march_2020_complete_for_snorkel.csv"
    df_set_balanced = pd.read_csv(path1,dtype={'class_q': 'object'})
    df_set_unbalanced = pd.read_csv(path2,dtype={'class_q': 'object'})
    
    ### lets read the scorers set ## 
    path_3 = "../data/Scorers_set_analysis_balanced_data.csv"
    path_4 = "../data/Scorers_set_analysis_unbalanced_data.csv"
    # path_3 = "../data/Clean_dataframe_balanced_scorers_set_for_snorkel.csv"
    # path_4 = "../data/Clean_dataframe_unbalanced_scorers_set_for_snorkel.csv"
    df_scorers_set_balanced = pd.read_csv(path_3)
    df_scorers_set_unbalanced = pd.read_csv(path_4)
    
    df_set_balanced.rename(columns={'NIS Polar' :'NIS_Polar',
                                  'Nis Apolar':'Nis_Apolar',
                                  'BSA Apolar':'BSA_Apolar',
                                  'BSA Polar' :'BSA_Polar'},inplace=True)
    df_set_unbalanced.rename(columns={'NIS Polar' :'NIS_Polar',
                                  'Nis Apolar':'Nis_Apolar',
                                  'BSA Apolar':'BSA_Apolar',
                                  'BSA Polar' :'BSA_Polar'},inplace=True)
    
    df_scorers_set_balanced.rename(columns={'identification' :'idx',
                                  'binary_label':'label_binary'},inplace=True)
    df_scorers_set_unbalanced.rename(columns={'identification' :'idx',
                                  'binary_label':'label_binary'},inplace=True)
    
    df_set_balanced.set_index("Conf",inplace=True)
    df_set_unbalanced.set_index("Conf",inplace=True)
    
    df_scorers_set_balanced.set_index("Conf",inplace=True)
    df_scorers_set_unbalanced.set_index("Conf",inplace=True)
    df_scorers_set_balanced.dropna(inplace=True)
    df_scorers_set_unbalanced.dropna(inplace=True)
    
     ## Split into Training , validation and Testing Sets 
    x_train = df_set_balanced[~df_set_balanced["idx"].isin(PDB_BM5)]
    y_train = df_set_balanced[~df_set_balanced["idx"].isin(PDB_BM5)]["label_binary"].astype("bool")
    x_val = df_set_unbalanced[df_set_unbalanced["idx"].isin(PDB_BM5)]
    y_val = df_set_unbalanced[df_set_unbalanced["idx"].isin(PDB_BM5)]["label_binary"].astype("bool")
    x_test = df_scorers_set_unbalanced
    y_test = df_scorers_set_unbalanced["label_binary"].astype("bool")
    
    ## Select the features for the sets 
    x_train= x_train[scoring_functions] 
#     x_train= x_train.drop(['label_binary','TF2','idx'],axis=1)

    x_val  = x_val[scoring_functions]
#     x_val = x_val.drop(['label_binary','TF2','idx'],axis=1)
    
    x_test = x_test[scoring_functions]
#     x_test = x_test.drop(['label_binary','TF2','idx'],axis=1)
    
    ## Make a copy for handeling better 
    
    x_train = x_train.copy()
    y_train = y_train.copy()
    x_val = x_val.copy()
    y_val = y_val.copy()
    x_test = x_test.copy()
    y_test= y_test.copy()
    
    # scale 
    min_max_scaler = MinMaxScaler()
    for classifier in scoring_functions:
        if classifier != "RFC" or classifier != "NNC":
            x_train[classifier] = min_max_scaler.fit_transform(x_train[classifier].values.reshape(-1,1))
            x_val[classifier]  = min_max_scaler.transform(x_val[classifier].values.reshape(-1,1))
            x_test[classifier]  = min_max_scaler.transform(x_test[classifier].values.reshape(-1,1))
    return x_train, y_train,x_val , y_val, x_test , y_test


# %%
x_train, y_train,x_val , y_val, x_test , y_test = read_data()


# %%
#tf_v1_val = save_metrics_results(model=my_model_1,x_test=x_val , y_test=y_val , tag="Validation v1")
#tf_v2_val = save_metrics_results(model=my_model_2,x_test=x_val , y_test=y_val , tag="Validation v2")
#tf_v1_test = save_metrics_results(model=my_model_1,x_test=x_test , y_test=y_test , tag="Test_set v1")
#tf_v2_test = save_metrics_results(model=my_model_2,x_test=x_test, y_test=y_test , tag="Test_set v2")
#
#
## %%
#print (tf_v1_val)
#
#
## %%
#print (tf_v2_val)
#
#
## %%
#print(tf_v1_test)
#
#
## %%
#print(tf_v2_test)


# %%
models = [m for m in os.listdir("../models/") if m[-3:] == ".h5" ]
# print (models)

# %%
all_results_test , all_results_validation = [], [] 
for model in models :
    # print (model) 
    my_model_1 = tf.keras.models.load_model(f"../models/{model}")
    tf_test = save_metrics_results(model=my_model_1,x_test=x_test, y_test=y_test , tag=f"{model}")
    all_results_test.append(tf_test)
    tf_val = save_metrics_results(model=my_model_1,x_test=x_val, y_test=y_val , tag=f"{model}" )
    all_results_validation.append(tf_val)
df_all_results = pd.concat(all_results_test)
print (df_all_results.sort_values("Accuracy",ascending=False))
df_all_results = df_all_results.sort_values("Accuracy",ascending=False)
df_all_results.to_csv("../results/TF2_models_test_scorers_set_results_metrics.csv")


df_all_results = pd.concat(all_results_validation)
print (df_all_results.sort_values("Accuracy",ascending=False))
df_all_results = df_all_results.sort_values("Accuracy",ascending=False)
df_all_results.to_csv("../results/TF2_models_validation_metrics.csv")
