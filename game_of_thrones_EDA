# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:26:49 2019

@author: lucas.barros

Assignment 2: Game of Thrones predictions
"""

#################################
# Basic libraries
#################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#################################
# Importing file
#################################

file = 'GOT_character_predictions.xlsx'

df = pd.read_excel(file)


##############################################################################
# EDA
##############################################################################

# showing all columns when called
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', None)

print(df.columns)

'''
Some column names are not clear on what they are or too complicated to type
everytime it is needed; I'm renaming some for covinience
'''


df = df.rename(index = str, columns ={'S.No': 'charnumber',
                                        'dateOfBirth': 'DOB',
                                        'book1_A_Game_Of_Thrones': 'book1',
                                        'book2_A_Clash_Of_Kings': 'book2',
                                        'book3_A_Storm_Of_Swords': 'book3',
                                        'book4_A_Feast_For_Crows': 'book4',
                                        'book5_A_Dance_with_Dragons': 'book5'
                                        })


print(df.info())



print(df.describe().round(2))
'''
The only variables that seem to be continuous are age, DOB, numDeadRelations,
popularity, the others are categorical/binary.
'''


df['isAlive'].describe()

df['isAlive'].value_counts()


#################################
# Flagging missing values
#################################


mv_bycolumn = df.isnull().sum()

print(mv_bycolumn)


#########################################################
# Creating new columns for the flagged missing values
#########################################################

'''
Creating columns for the missing values with 0 and 1s
'''

for col in df:
    if df[col].isnull().any():
        df['m_'+col] = df[col].isnull().astype(int)

df_dropped = df.dropna()


####################################
# Analysing the culture variable
####################################

df.culture.head()

# getting dummy variables for the cultures
dum_cult = pd.get_dummies(df[['culture']], dummy_na = True)

# analyzing the count of cultures
for col in dum_cult.iloc[:, :65]:

   count = dum_cult[col].value_counts()

   print(count)


'''
Westermen are decendent of Andals (have similar echnicity),
but since the lineage come from several generations on the past,
It is better to separete them.
'''

# filling NAs with unknown

fill = 'unknown'

df['culture'] = df['culture'].fillna(fill)


# Some culture have duplicates, aggregating them together


df['culture'][df['culture'].str.contains('Andal')] = 'Andal'

df['culture'][df['culture'].str.contains('Asshai')] = 'Asshai'

df['culture'][df['culture'].str.contains('Astapor')] = 'Astapor'

df['culture'][df['culture'].str.contains('Braavos')] = 'Braavos'

df['culture'][df['culture'].str.contains('Dorn')] = 'Dorne'

df['culture'][df['culture'].str.contains('Ghiscari')] = 'Ghiscari'

df['culture'][df['culture'].str.contains('Iron')] = 'Ironborn'
df['culture'][df['culture'].str.contains('iron')] = 'Ironborn'

df['culture'][df['culture'].str.contains('Lhazare')] = 'Lhazareen'

df['culture'][df['culture'].str.contains('Lyse')] = 'Lysene'

df['culture'][df['culture'].str.contains('Meereen')] = 'Meereen'

df['culture'][df['culture'].str.contains('orthmen')] = 'Northmen'

df['culture'][df['culture'].str.contains('Norvos')] = 'Norvos'

df['culture'][df['culture'].str.contains('Qarth')] = 'Qarth'

df['culture'][df['culture'].str.contains('Reach')] = 'Reach'

df['culture'][df['culture'].str.contains('River')] = 'Rivermen'

df['culture'][df['culture'].str.contains('Stormland')] = 'Stormland'

df['culture'][df['culture'].str.contains('Summer')] = 'Summer'

df['culture'][df['culture'].str.contains('Vale')] = 'Vale'

df['culture'][df['culture'].str.contains('Lyse')] = 'Lysene'

df['culture'][df['culture'].str.contains('ester')] = 'Westernmen'

'''
Free folks and windlings are actually the same people, just different
nomenclature.
'''
df['culture'][df['culture'].str.contains('Wilding')]= 'Windling'
df['culture'][df['culture'].str.contains('Free')]= 'Windling'
df['culture'][df['culture'].str.contains('free')]= 'Windling'


print(df['culture'][df['isAlive'] == 0].value_counts())
print(df['culture'][df['isAlive'] == 1].value_counts())

'''
Generally speaking, the inhabitants of the north of Westeros are the ones
that die the most. There is probably due to the number of wars in that region
plus what happens in the Great Wall.
'''


####################################
# Analysing the house variable
####################################

df.house.head()

# getting dummy variables for the cultures
dum_hou = pd.get_dummies(df[['house']], dummy_na = True)

#### analyzing the count of cultures
for col in dum_hou.iloc[:, :348]:

   count = dum_hou[col].value_counts()

   print(count)


# Filling NAs with unknown

fill = 'unknown'
df['house'] = df['house'].fillna(fill)


#### Some houses have duplicates, aggregating them together

df['house'][df['house'].str.contains('Lannister')] = 'Lannister'

df['house'][df['house'].str.contains('Baratheon')] = 'Baratheon'

df['house'][df['house'].str.contains('Brotherhood')] = 'Brotherhood without banners'

df['house'][df['house'].str.contains('Bolton')] = 'Bolton'

df['house'][df['house'].str.contains('Flint')] = 'Flint'

df['house'][df['house'].str.contains('Brune')] = 'Brune of Browhollow'

df['house'][df['house'].str.contains('Fossoway')] = 'Fossoway'

df['house'][df['house'].str.contains('Frey')] = 'Frey'

df['house'][df['house'].str.contains('Goodbrother')] = 'Goodbrother'

df['house'][df['house'].str.contains('House Harlaw')] = 'House Harlaw'

df['house'][df['house'].str.contains('Kenning')] = 'Kenning'

df['house'][df['house'].str.contains('Royce')] = 'Royce'

df['house'][df['house'].str.contains('Tyrell')] = 'Tyrell'

df['house'].value_counts()

print(df['house'][df['isAlive'] == 0].value_counts())
print(df['house'][df['isAlive'] == 1].value_counts())



'''
Night's Watch die the most, followed by obviously the Targaryen, and then 
Starks, Lannisters, Greyjoys, and Freys probably due to the war between the 
families.
'''

'''
According to my research, the most important families are Baratheon, Stark, 
Lannister, Arryn, Tyrell, Tully, Greyjoy, Martell, and Targaryen.
After the Red Wedding, House Frey became one of the most important. 
'''



##################################
# Analysing Title
##################################

print(df.title.value_counts().head(10))
print(df.title[df['isAlive'] == 1].value_counts().head(10))
df.title.isna().sum()

# filling NAs with unknown
fill = 'unknown'
df['title'] = df['title'].fillna(fill)

dum_title = pd.get_dummies(df[['title']], dummy_na = True)

df = pd.concat([df, dum_title], axis = 1)

'''
Higher titles of nobility seems to have a higher chance of surviving.
'''


##################################################
# Analysing Father, Mother, Heir, and Spouse
##################################################

# flagging missing values
print(df.father.isna().sum())
print(df.mother.isna().sum())
print(df.heir.isna().sum())
print(df.spouse.isna().sum())


# checking the distribution
print(df.father.value_counts())
print(df.mother.value_counts())
print(df.heir.value_counts())


# filling NAs with unknown
fill = 'unknown'
df['father'] = df['father'].fillna(fill)
df['mother'] = df['mother'].fillna(fill)
df['heir'] = df['heir'].fillna(fill)
df['spouse'] = df['spouse'].fillna(fill)


###################################################
# Analysing books
##################################################


# Flagging Missing Values
print(df.book1.isna().sum())
print(df.book2.isna().sum())
print(df.book3.isna().sum())
print(df.book4.isna().sum())
print(df.book5.isna().sum())

'''
no NAs
'''
print(df.book1.value_counts())
print(df.book2.value_counts())
print(df.book3.value_counts())
print(df.book4.value_counts())
print(df.book5.value_counts())


# Studying the relation between being in a book and being alive

'''There are not a lot of people alive in book1 since it tells a lot of 
stories about what happened in the past.
'''

 # Checking who appeared in all books, they are probably very significant.


df['all_books'] = (df['book1'] + df['book2'] + df['book3'] + df['book4'] +
                  df['book5'])

df['all_books'].value_counts()

# Doing a outlier for people who appeared in all books.

df['out_allbooks'] = 0
df['out_allbooks'] = df['all_books'][df['all_books'] == 5]
fill = 0
df['out_allbooks'] = df['out_allbooks'].fillna(fill)



# Flagging characters that didn't appear in any book.

df['no_books'] = 0

df.loc[ : , 'no_books'][df.loc[ : , 'all_books'] == 0] = 1

'''
Combining who appeared in different books might be significant to the 
final analysis
'''

df['book_4_5'] = 0
df['book_4_5'] = df['book4'] + df['book5']
df['book_4_5'][df['isAlive']== 1].value_counts()

df['book_1_5'] = 0
df['book_1_5'] = df['book1'] + df['book5']
df['book_1_5'][df['isAlive']== 1].value_counts()

df['book_3_n_5'] = 0
df['book_3_n_5'] = df['book3'] + df['book5']
df['book_3_n_5'][df['isAlive']== 1].value_counts()

df['book_2_3'] = 0  
df['book_2_3'] = df['book2'] + df['book3']
df['book_2_3'][df['isAlive']== 1].value_counts()

df['book_2_3'] = 0
df['book_2_3'] = df['book2'] + df['book3']
df['book_2_3'][df['isAlive']== 1].value_counts()

df['book_3_4_5'] = 0
df['book_3_4_5'] = df['book4'] + df['book5'] + df['book3'] 
df['book_3_4_5'][df['isAlive']== 1].value_counts()

'''
These combinations above shows who appeared in/or the selected books.
'''


print(np.corrcoef(x=df['isAlive'], y = df['book1']))

print(np.corrcoef(x=df['isAlive'], y = df['book2']))

print(np.corrcoef(x=df['isAlive'], y = df['book3']))

print(np.corrcoef(x=df['isAlive'], y = df['book4']))

print(np.corrcoef(x=df['isAlive'], y = df['book5']))

print(np.corrcoef(x=df['isAlive'], y = df['all_books']))

'''
The is a small correlation between being alive and the older the book,
although book4 has the highest correlation with being alive. Also the more the
person appeared the highest the probability of being alive.
'''


#################################################################
# Analysing If Mother, Father, Heir, and/or Spouse are alive
#################################################################

# Flagging missing Values
print(df.isAliveMother.isna().sum())
print(df.isAliveFather.isna().sum())
print(df.isAliveHeir.isna().sum())
print(df.isAliveSpouse.isna().sum())


'''
There are a lot of missing values, I'm assuming that if it is unknown that
their family is alive, the character is probably not important, hence I'm
inputing missing values with 0.
'''
# Filling NAs with unknown
fill = 0
df.isAliveMother = df.isAliveMother.fillna(fill)
df.isAliveFather = df.isAliveFather.fillna(fill)
df.isAliveHeir = df.isAliveHeir.fillna(fill)
df.isAliveSpouse = df.isAliveSpouse.fillna(fill)


###################################################
# Analysing if is Married and/or is Noble
###################################################

# Flagging missing Values
print(df.isMarried.isna().sum())
print(df.isNoble.isna().sum())

'''
No missing values
'''

# Checking the distribution of Married and Spouse
print(df.isMarried.value_counts())
print(df.isNoble.value_counts())

print(df['isMarried'][df['isAlive'] == 1].sum())
print(df['isNoble'][df['isAlive'] == 1].sum())

'''
69.2% of Married are alive
72.5% of Nobles are alive
'''

df['isMarried'][df['isMarried'] == 1 ][df['isNoble'] == 1][df['isAlive'] == 1].sum()

'''
183 are Married and are Noble
109 are Married, Noble, and are Alive
'''

'''
Creating a column for characters that are noble and married
'''
df['lucky'] = 0

df['lucky'] = df.loc[ : ,'isNoble'] + df.loc[: , 'isMarried']

df['lucky'] = df['lucky'].replace(1, 0)

df['lucky'] = df['lucky'].replace(2, 1)


#############################################################
# Analysing Age
#############################################################

# Flagging missing values for AGE

print(df.age.isna().sum())

'''
Droping the 2 extreme outliers
'''
df = df.drop(df.index[110])
df = df.drop(df.index[1349])

df.age.describe()


'''
Getting the age of the person, if he/she is alive and adding with the DOB it
will give us the current year of the dataset(which is 305). Also if we get the
oldest person alive and he his/hers DOB, we can assume that anyone that was 
born before that is dead. The oldest person alive was born in 208, so that
will be a threshold.

'''


'''
Creating a column with dummy 1 and 0 to if they are living the interval 
between 208 and 305.
'''

df['300year_vs_dob'] = 305 - df['DOB'] 

df['alive_by_age'] = 0

def conditions(df):
    if (df['age'] == df['300year_vs_dob']):
        return 0
    elif (df['age'] < df['300year_vs_dob']):
        return 1

df['alive_by_age'] = df.apply(conditions, axis=1)

print(df['alive_by_age'].sum())


# Filling the missing values with -1
df['300year_vs_dob'] = df['300year_vs_dob'].fillna(-1)


# Filling the missing value with -1 to 
fill = -1
df.alive_by_age = df.alive_by_age.fillna(fill)


# Filing the NA's with -1 to analyze the distribution afterwards
fill = -1
df['age'] = df['age'].fillna(fill)


# Creating a new colum without the Nas values of the age
df['out_age'] = df['age'][df['age'] != -1]
df['out_age'] = df['out_age'].fillna(0)


# Analysing the distribution of the ages
df_age = df.age.dropna()
fig, ax = plt.subplots(figsize=(20,10))
sns.distplot(df_age)
plt.show()

fig, ax = plt.subplots(figsize=(20,10))
sns.distplot(df.age)
plt.show()


# Filling NAs with the median
df['age'][df['age'] == -1] = 27


# Filling the NAs with the median to analyse the distribution afterwards
fill = -1
df['DOB'] = df['DOB'].fillna(fill)


# Creating a new colum without the NA values of the DOB
df['out_DOB'] = df['DOB'][df['DOB'] != -1]
df['out_DOB'] = df['out_DOB'].fillna(0)

df.DOB.describe()

df_DOB = df.DOB.dropna()
fig, ax = plt.subplots(figsize=(20,10))
sns.distplot(df_DOB)
plt.show()


# filling NAs with the median

df['DOB'][df['DOB'] == -1] = 268

''' 
Creating a new column with the sum of age and DOB, if the result != 305 then 
the character is not alive.
'''

df['out_year'] = df.DOB + df.age




##########################################################
# Analysing Number of dead relatives and popularity
##########################################################

# Flagging Missing Values
print(df.numDeadRelations.isna().sum())
print(df.popularity.isna().sum())


# distribution of dead relatives
print(df.numDeadRelations.value_counts())

# checking the correlation between dead relatives and being alive
np.corrcoef(x = df['numDeadRelations'] ,  y = df['isAlive'])

'''
It shows a very weak negative correlation between the number of dead relatives
and being alive

I'm creating a dummy variable for the number of read relatives, where if the
character has 0 dead relatives, it will flag as 1.
'''

dead_relations_zero = 0

df['out_deadrelations'] = 0

df.loc[ : , 'out_deadrelations'][df.loc[ : , 'numDeadRelations'] !=
                                                    dead_relations_zero] = 1


# Exploring the popularity 
print(df.popularity.describe())


# Analysing the distribution
fig, ax = plt.subplots(figsize=(20,10))
sns.distplot(df['popularity'])
plt.show()


sns.lmplot(x = 'popularity',
           y = 'isAlive',
           data = df
           )
plt.show()



# Checking the correlation with being alive

np.corrcoef(x = df['popularity'], y = df['isAlive'])


'''
I'm going to create a new column only with the most popular characters. 
Checking the distribution of according to the quantiles.
'''
df['popularity'].quantile([0.25,
                           0.50,
                           0.75,
                           0.80,
                           0.90,
                           0.95
                           ])

df_popularity = (df.loc[ : , ['name', 
                             'house',
                             'popularity',
                             'isAlive']]              
                 [df['popularity'] >= 0.3]
                 )
                  
print(df_popularity.describe())

print(np.corrcoef(x=df_popularity['popularity'],
                  y = df_popularity['isAlive']
                  ))



# Creating a new column only with characters >= 0.3 of popularity.

df['out_popular'] = 0
df['out_popular'][df['popularity'] >= 0.3] = 1

df_corr = df.loc[:, ['out_age', 'out_DOB', 'out_year', 'alive_by_age']
                                                            ].corr().round(2)


###############################################################################
# Dataset is ready for the models
###############################################################################

df.to_excel('got.xlsx')

