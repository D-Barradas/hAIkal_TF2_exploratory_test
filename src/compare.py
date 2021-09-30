import pandas as pd 
df1 = pd.read_csv("../data/BM5_analysis_unbalanced_data.csv")
df2 = pd.read_csv("../../ML_CONSRANK/data/Clean_dataframe_unbalanced_all_data_ccharppi_4_march_2020_complete_for_snorkel.csv",dtype={'class_q': 'object'})
df1.rename(columns={'NIS Polar' :'Nis_Polar',
                    'Nis Apolar':'Nis_Apolar',
                    'BSA Apolar':'BSA_Apolar',
                    'BSA Polar' :'BSA_Polar'},inplace=True)
df2.rename(columns={'NIS Polar' :'NIS_Polar',
                    'Nis Apolar':'Nis_Apolar',
                    'BSA Apolar':'BSA_Apolar',
                    'BSA Polar' :'BSA_Polar'},inplace=True)
df1.drop("CAT_PI",axis=1,inplace=True)
df3 = df2[df1.columns]

print (df1.head())
print 
print 
print (df3.head())
print 
print (df1.tail())
print
print
print (df3.tail())

