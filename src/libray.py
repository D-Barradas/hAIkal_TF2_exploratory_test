import pandas as pd
from sklearn.metrics import classification_report, matthews_corrcoef, confusion_matrix
import os
from sklearn.preprocessing import MinMaxScaler

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
# scoring_functions = ['CONSRANK_val', 'CP_HLPL', 'CP_MJ3h', 'DDG_V', 'CP_RMFCA', 'AP_GOAP_DF']

scoring_functions_gold = ['CONSRANK_val', 'CP_HLPL', 'CP_MJ3h', 'DDG_V', 'CP_RMFCA', 'AP_GOAP_DF', 'NNC', 'RFC']


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
    x_val_bal = df_set_balanced[df_set_balanced["idx"].isin(PDB_BM5)]
    y_val_bal = df_set_balanced[df_set_balanced["idx"].isin(PDB_BM5)]["label_binary"].astype("bool")
    x_test = df_scorers_set_unbalanced
    y_test = df_scorers_set_unbalanced["label_binary"].astype("bool")
    x_test_bal = df_scorers_set_balanced
    y_test_bal = df_scorers_set_balanced["label_binary"].astype("bool")
    
    x_ml_classifer_test = x_test[["RFC","NNC","TF2"]] 
    x_ml_classifer_test_bal = x_test_bal[["RFC","NNC","TF2"]] 
    x_ml_classifer = x_val[["RFC","NNC","TF2"]] 
    x_ml_classifer_bal = x_val_bal[["RFC","NNC","TF2"]] 

    ## Select the features for the sets 
    x_train= x_train[scoring_functions] 
#     x_train= x_train.drop(['label_binary','TF2','idx'],axis=1)

    x_val  = x_val[scoring_functions]
    x_val_bal  = x_val_bal[scoring_functions]
#     x_val = x_val.drop(['label_binary','TF2','idx'],axis=1)
    
    x_test = x_test[scoring_functions]
    x_test_bal = x_test_bal[scoring_functions]
#     x_test = x_test.drop(['label_binary','TF2','idx'],axis=1)
    
    ## Make a copy for handeling better 
    
    x_train = x_train.copy()
    y_train = y_train.copy()
    x_val = x_val.copy()
    y_val = y_val.copy()
    x_test = x_test.copy()
    y_test= y_test.copy()
    x_val_bal = x_val_bal.copy()
    y_val_bal = y_val_bal.copy()
    x_test_bal = x_test_bal.copy()
    y_test_bal= y_test_bal.copy()

    
    # scale 
    min_max_scaler = MinMaxScaler()
    for classifier in scoring_functions:
        if classifier != "RFC" or classifier != "NNC":
            x_train[classifier] = min_max_scaler.fit_transform(x_train[classifier].values.reshape(-1,1))
            x_val[classifier]  = min_max_scaler.transform(x_val[classifier].values.reshape(-1,1))
            x_test[classifier]  = min_max_scaler.transform(x_test[classifier].values.reshape(-1,1))
            x_val_bal[classifier]  = min_max_scaler.transform(x_val_bal[classifier].values.reshape(-1,1))
            x_test_bal[classifier]  = min_max_scaler.transform(x_test_bal[classifier].values.reshape(-1,1))
    return x_train, y_train, x_val , y_val, x_test , y_test , x_val_bal , y_val_bal , x_test_bal , y_test_bal, x_ml_classifer,x_ml_classifer_test, x_ml_classifer_bal, x_ml_classifer_test_bal
    # return x_train, y_train,x_val , y_val, x_test , y_test , x_val_bal , y_val_bal , x_test_bal , y_test_bal


def save_metrics_results(model,x_test,y_test,tag):
    # target_names = ['Incorrect', 'Correct']

    # y_pred = model.predict_classes(x_test.to_numpy())
    # y_pred = (model.predict(x_test.to_numpy()) > 0.5).astype("int32")
    y_pred = (model.predict(x_test.to_numpy()) > 0.5).astype("bool")

    cr = classification_report(y_true=y_test, y_pred=y_pred,output_dict=True,digits=4,zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=y_pred).ravel()
    # print (tn, fp, fn, tp)
    ### this depends on the opinion, having these values at zero, is that the model just returns one label 
    ### while MMC = 0 is basically random prediction , could be interpreted as bad classification
    ## I set the value of MMC to -1 beacuase is the worst case 
    if fp != 0 and tp != 0 :
        mmc = matthews_corrcoef(y_true=y_test, y_pred=y_pred)
    else : 
        mmc = -1 
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


def convert_pred(model , my_x, my_y, tag ):
    """ this funtions will get the columns neseray to create a dataframe to analize the success rate """
    df_pred = pd.DataFrame( model.predict(my_x.to_numpy()))
    df_pred = df_pred.set_index( my_x.index ) 
    label_binary = my_y
    idx = []

    if tag == "test":
        idx = [ x.split("_")[0] for x in my_x.index ]
        idx = pd.Series(idx, index=my_x.index)

    elif tag == "val":
        idx = [ x.split("_")[1] for x in my_x.index ]
        idx = pd.Series(idx, index=my_x.index)

    df_pred = pd.concat( [idx ,label_binary, df_pred ], axis=1)
    df_pred.columns = ["idx","label_binary","proba_true"]
    
    return df_pred


def get_success_rate(df,sel,tag):
    """this funtion read a pandas dataframe and that dataframe need the columns
    idx , proba_true , and the label_binary"""
    total_size = {"val":float(len(PDB_BM5)),"test":float(15)}
    
    tops = [1,10,100,1000]
    tcount = [0 for y in range(len(tops))]
    names = [ [] for y in range(len(tops))]
    ranks = []
    for m in df['idx'].unique():
#         print (m, end=" ")
        found = False
        count = 0
        selection = df[df["idx"]==m]
        selection= selection.sort_values(by=["proba_true"], ascending=False)
#         print (selection)
        for x in selection["label_binary"].values:
            count +=1
            if x != False:
#                 print (x,count)
                if not found:
                    rank = count
                    ranks.append(count)
                found = True
        if found:

            for z in range(len(tops)):
                if  rank < tops[z]+1:
                    tcount[z] += 1
                    if m not in names:
                        names[z].append(m)

    print (tag,tcount ,sel ) 
    results_rank = []
    for y in tcount:
        print ( round(float(y)/total_size[sel],4)*100 , end=" ")
        results_rank.append(round(float(y)/total_size[sel],4)*100)
    print()
    df_result =  pd.DataFrame(results_rank,index=[tops])
    df_result.columns = [f"{tag}"]
    # df_result = df_result.T
    return df_result.round(4)

def convert_pred(model , my_x, my_y, tag ):
    """ this funtions will get the columns neseray to create a dataframe to analize the success rate """
    df_pred = pd.DataFrame( model.predict(my_x.to_numpy()))
    df_pred = df_pred.set_index( my_x.index ) 
    label_binary = my_y
    idx = []

    if tag == "test":
        idx = [ x.split("_")[0] for x in my_x.index ]
        idx = pd.Series(idx, index=my_x.index)

    elif tag == "val":
        idx = [ x.split("_")[1] for x in my_x.index ]
        idx = pd.Series(idx, index=my_x.index)

    df_pred = pd.concat( [idx ,label_binary, df_pred ], axis=1)
    df_pred.columns = ["idx","label_binary","proba_true"]
    
    return df_pred

def read_data_gold_data() :
    """This funtion takes the data that is already standarized """
    path1 = "../data/snorkel_train_gold_set.csv" 
    path2 = "../data/Clean_dataframe_unbalanced_all_data_ccharppi_4_march_2020_complete_for_snorkel.csv"


    df_set_balanced = pd.read_csv(path1)
    df_set_unbalanced = pd.read_csv(path2,dtype={'class_q': 'object'})

 
    df_set_balanced.set_index("Conf",inplace=True)
    df_set_unbalanced.rename(columns={'NIS Polar' :'NIS_Polar',
                            'Nis Apolar':'Nis_Apolar',
                            'BSA Apolar':'BSA_Apolar',
                            'BSA Polar' :'BSA_Polar'},inplace=True)
    df_set_unbalanced.set_index("Conf",inplace=True) 
    
     ## Split into Training , validation and Testing Sets 
    x_train = df_set_balanced[~df_set_balanced["idx"].isin(PDB_BM5)]
    y_train = df_set_balanced[~df_set_balanced["idx"].isin(PDB_BM5)]["label_binary"].astype("bool")
    x_val = df_set_unbalanced[df_set_unbalanced["idx"].isin(PDB_BM5)]
    y_val = df_set_unbalanced[df_set_unbalanced["idx"].isin(PDB_BM5)]["label_binary"].astype("bool")
    
    ## Select the features for the sets 
    x_train= x_train[scoring_functions]
    x_val= x_val[scoring_functions]

    ## Make a copy for handling better 
    
    x_train = x_train.copy()
    y_train = y_train.copy()
    x_val = x_val.copy()
    y_val = y_val.copy()
    return x_train, y_train,x_val,y_val
