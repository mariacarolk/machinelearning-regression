import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def has_null(df: pd.DataFrame) -> bool:
    return df.isnull().any()

penguim = sns.load_dataset('penguins')

#examining our dataset using graphics
with sns.axes_style('whitegrid'):
    grafico = sns.pairplot(data=penguim, hue="sex", palette="pastel")
    plt.savefig("grafico1.png")

with sns.axes_style('whitegrid'):
    grafico = sns.pairplot(data=penguim, hue="species", palette="pastel")
    plt.savefig("grafico2.png")

with sns.axes_style('whitegrid'):
    grafico = sns.pairplot(data=penguim, hue="island", palette="pastel")
    plt.savefig("grafico3.png")

#adjust dataset NAN values :)

#use the mean for numeric attributes
med_bill_lenght = penguim['bill_length_mm'].mean()
med_bill_depth = penguim['bill_depth_mm'].mean()
med_flipper_lenght = penguim['flipper_length_mm'].mean()
med_body_mass = penguim['body_mass_g'].mean()

if has_null(df=penguim['bill_length_mm']):
    penguim['bill_length_mm'].fillna(value=med_bill_lenght, inplace=True)

if has_null(df=penguim['bill_depth_mm']):
    penguim['bill_depth_mm'].fillna(value=med_bill_depth, inplace=True)

if has_null(df=penguim['flipper_length_mm']):
    penguim['flipper_length_mm'].fillna(value=med_flipper_lenght, inplace=True)

if has_null(df=penguim['body_mass_g']):
    penguim['body_mass_g'].fillna(value=med_body_mass, inplace=True)

#drop lines with NAN values in categoric attributes
if has_null(df=penguim['species']):
    penguim.dropna(subset=['species'], inplace=True)

if has_null(df=penguim['island']):
    penguim.dropna(subset=['island'], inplace=True)

if has_null(df=penguim['sex']):
    penguim.dropna(subset=['sex'], inplace=True)

#standardize values

#numeric attributes
std_bill_lenght = penguim['bill_length_mm'].std()
std_bill_depth = penguim['bill_depth_mm'].std()
std_flipper_lenght = penguim['flipper_length_mm'].std()
std_body_mass = penguim['body_mass_g'].std()

penguim['bill_length_mm_std'] = penguim['bill_length_mm'].apply(lambda bill_length_mm: (bill_length_mm - med_bill_lenght) / std_bill_lenght)
penguim['bill_depth_mm_std'] = penguim['bill_depth_mm'].apply(lambda bill_depth_mm: (bill_depth_mm - med_bill_depth) / std_bill_depth)
penguim['flipper_length_mm_std'] = penguim['flipper_length_mm'].apply(lambda flipper_length_mm: (flipper_length_mm - med_flipper_lenght) / std_flipper_lenght)

#categoric attributes (one hot encoding)
penguim['species_adelie'] = penguim['species'].apply(lambda species : 1 if species == 'Adelie' else 0)
penguim['species_chinstrap'] = penguim['species'].apply(lambda species : 1 if species == 'Chinstrap' else 0)
penguim['species_gentoo'] = penguim['species'].apply(lambda species : 1 if species == 'Gentoo' else 0)
penguim['island_torgersen_nom'] = penguim['island'].apply(lambda island : 1 if island == 'Torgersen' else 0)
penguim['island_biscoe_nom'] = penguim['island'].apply(lambda island : 1 if island == 'Biscoe' else 0)
penguim['island_dream_nom'] = penguim['island'].apply(lambda island : 1 if island == 'Dream' else 0)
penguim['sex_male_nom'] = penguim['sex'].apply(lambda sex : 1 if sex == 'Male' else 0)
penguim['sex_female_nom'] = penguim['sex'].apply(lambda sex : 1 if sex == 'Female' else 0)

penguim.drop(columns=['species', 'island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'sex'], inplace=True)

#split data in train and test (2/3 proportion)
predictors_train, \
predictors_test, \
target_train, \
target_test = train_test_split(penguim.drop(['body_mass_g'], axis=1),
                                            penguim['body_mass_g'],
                                            test_size=0.33,
                                            random_state=123)

#Train a model of linear regression with train data (2/3)
model = LinearRegression()

model = model.fit(predictors_train, target_train)

print(model.__dict__)

a = model.coef_
b = model.intercept_

target_predicted = model.predict(predictors_test)
print(target_predicted)

rmse = np.sqrt(mean_squared_error(target_test, target_predicted))
print(rmse)

# #predict the weight of a new penguin with fixed characteristics
bill_length_mm = 38.2
bill_length_mm = (bill_length_mm - med_bill_lenght) / std_bill_lenght
bill_depth_mm = 18.1
bill_depth_mm = (bill_depth_mm - med_bill_depth) / std_bill_depth
flipper_length_mm = 185.0
flipper_length_mm = (flipper_length_mm - med_flipper_lenght) / std_flipper_lenght

new_penguim = np.array([1, 0, 0, 0, 1, 0, bill_length_mm, bill_depth_mm, flipper_length_mm, 0, 1])
target_predicted = model.predict(new_penguim.reshape(1, -1))
print(target_predicted)
