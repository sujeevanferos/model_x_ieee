import pandas as pd
import numpy as np

df = pd.read_csv(r"Dementia Prediction Dataset_filtered.csv")

# Apply replacements
df.replace(-4, np.nan, inplace=True)


replace_dict1 = {
    'NACCNRYR': [8888],
    'NACCNRDY': [88],
    'NACCNRMO': [88],
    'SMOKYRS': [88],
    'INHISPOR': [88],
    'RACETER': [88],
    'INRASEC': [88],
    'HISPOR': [88],
    'RACESEC': [88],
    'INRATER': [88],
    'NACCNOVS': [8],
    'INVISITS': [8],
    'ALCFREQ': [8],
    'QUITSMOK': [888],
    'PACKSPER': [8]
}


for col, values in replace_dict1.items():
    if col in df.columns:  
        df[col].replace(values, np.nan, inplace=True)

replace_dict2 = {
    'NACCNRYR': [9999],
    'NACCNRDY': [99],
    'NACCNRMO': [99],
    'SMOKYRS': [99],
    'INHISPOR': [99],
    'RACETER': [99],
    'INRASEC': [99],
    'HISPOR': [99],
    'RACESEC': [99],
    'INRATER': [99],
    'NACCNOVS': [9],
    'INVISITS': [9],
    'ALCFREQ': [9],
    'QUITSMOK': [999],
    'PACKSPER': [9],
    'NACCFAM': [9],
    'NACCMOM': [9],
    'NACCDAD': [9],
    'NACCAM': [9],
    'NACCAMS': [9],
    'NACCFM': [9],
    'NACCFMS': [9],
    'NACCOM': [9],
    'NACCOMS': [9],
    'HISPOR':[50],
    'RACE':[50]
}


for col, values in replace_dict2.items():
    if col in df.columns:  # only apply if column exists
        df[col].replace(values, np.nan, inplace=True)

#Drop unnecessary columns (cleaned names, removed trailing spaces)
columns_to_drop = [
    'NACCVNUM','NACCAVST','NACCDAYS','NACCNVST','NACCFDYS','NACCCORE','NACCREFR','NACCREAS','NACCADC',
    'PACKET','FORMVER','INKNOWN','INLIVWITH','INVISITS','INCALLS','INRELY','INRELTO','INRELTOX',
    'INBIRMO','INBIRYR','INSEX','INHISP','INHISPOR','INHISPOX','NACCNINR','INRACE','INRACEX',
    'INRASEC','INRASECX','INRATER','INRATERX','INEDUC','NACCNURP','NACCNRMO','NACCNRDY','NACCNRYR',
    'NACCNIHR','NACCNOVS','BIRTHMO','BIRTHYR','RACEX','RACESECX','RACETERX','PRIMLANG','PRIMLANX','NACCAMX','NACCAMSX','NACCFMX','NACCFMSX','NACCOMX','NACCOMSX'
]

df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

df.to_csv(r"Dementia Prediction Dataset_Preprocessed.csv", index=False)

print("Preprocessing complete. Cleaned dataset saved.")
