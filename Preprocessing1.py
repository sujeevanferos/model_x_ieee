import pandas as pd

path_in  = r"Dementia Prediction Dataset.csv"
path_out = r"Dementia Prediction Dataset_filtered.csv"

columns_to_keep = [
    "NACCID","NACCADC","PACKET","FORMVER","VISITMO","VISITDAY","VISITYR","NACCVNUM","NACCAVST","NACCNVST",
    "NACCDAYS","NACCFDYS","NACCCORE","NACCREAS","NACCREFR","BIRTHMO","BIRTHYR","SEX","HISPANIC","HISPOR",
    "HISPORX","RACE","RACEX","RACESEC","RACESECX","RACETER","RACETERX","PRIMLANG","PRIMLANX","EDUC",
    "MARISTAT","NACCLIVS","INDEPEND","RESIDENC","HANDED","INBIRMO","INBIRYR","INSEX","NEWINF","INHISP",
    "INHISPOR","INHISPOX","NACCNINR","INRACE","INRACEX","INRASEC","INRASECX","INRATER","INRATERX","INEDUC",
    "INRELTO","INRELTOX","INKNOWN","INLIVWTH","INVISITS","DCALLS","INRELY","NACCFAM","NACCMOM","NACCDAD",
    "NACCAM","NACCAMX","NACCAMS","NACCAMSX","NACCFM","NACCFMX","NACCFMS","NACCFMSX","NACCOM","NACCOMX",
    "NACCOMS","NACCOMSX","NACCFADM","NACCFFTD","TOBAC30","TOBAC100","SMOKYRS","PACKSPER","QUITSMOK",
    "ALCOCCAS","ALCFREQ","NACCAGEB","NACCAGE","NACCNIHR","NACCNOVS","NACCNURP","NACCNRMO","NACCNRDY","NACCNRYR","NACCUDSD"
]


df = pd.read_csv(path_in, low_memory=False)

present_cols = [c for c in columns_to_keep if c in df.columns]
missing_cols = [c for c in columns_to_keep if c not in df.columns]

df_filtered = df[present_cols].copy()
df_filtered.to_csv(path_out, index=False)

print("Number of columns after filtering:", len(df_filtered.columns))
if missing_cols:
    print("Missing columns (not found in source):", missing_cols)

