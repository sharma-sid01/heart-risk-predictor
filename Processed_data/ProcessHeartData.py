import pandas as pd
# Read the heart data data set file
df = pd.read_csv('heart_data.csv')
print(df.head())

#drop index, id columns                 
df.drop(df.iloc[:,0:2], inplace =True, axis =1)
print(df.head())
print("Number of records:", len(df))

#Translate age in days to age in years - divide by 365.25 and convert to int
ageInYrs =  df['age'] = (df['age']/ 365.25).astype(int)  
print(df.head())

#skip negative ap hi and lo values
df.drop(df[df['ap_hi'] < 20].index, inplace=True)
df.drop(df[df['ap_lo'] < 20].index, inplace=True)

print("Number of records after removing records with ap_hi and lo below 20:", len(df))
df.drop(df[df['ap_hi'] > 900].index, inplace=True)
df.drop(df[df['ap_lo'] > 910].index, inplace=True)

print("Number of records after removing records with ap_hi and lo above 900:", len(df))

#skip ap hi or lo values as 0
#df.drop(df[df['ap_hi'] == 0].index, inplace=True)
#df.drop(df[df['ap_lo'] == 0].index, inplace=True)
#print("Number of records:", len(df))



