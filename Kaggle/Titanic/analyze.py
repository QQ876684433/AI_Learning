import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

import warnings

warnings.filterwarnings("ignore")

# %matplotlib inline 魔法函数，在IPython可以使用，模拟命令行输入，但是PyCharm中不受支持

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

sns.set_style('whitegrid')
train_data.head()

train_data.info()
print('=' * 40)
test_data.info()

train_data['Survived'].value_counts().plot.pie(autopct='%1.2f%%')

train_data.Embarked[train_data.Embarked.isnull()] = train_data.Embarked.dropna().mode().values

# replace missing value with U0
train_data['Cabin'] = train_data.Cabin.fillna('U0')  # train_data.Cabin[train_data.Cabin.isnull()]='U0'

# choose training data to predict age
age_df = train_data[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
age_df_notnull = age_df.loc[(train_data['Age'].notnull())]
age_df_isnull = age_df.loc[(train_data['Age'].isnull())]
X = age_df_notnull.values[:, 1:]
Y = age_df_notnull.values[:, 0]
# use RandomForestRegression to train data
RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
RFR.fit(X, Y)
predictAges = RFR.predict(age_df_isnull.values[:, 1:])
train_data.loc[train_data['Age'].isnull(), ['Age']] = predictAges

print("=" * 40)
train_data.info()

# 探究性别与生存的关系
print("=" * 40)
print(train_data.groupby(['Sex', 'Survived'])['Survived'].count())
train_data[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar()

# 探究船舱等级与生存的关系
print("=" * 40)
print(train_data.groupby(['Pclass', 'Survived'])['Pclass'].count())
train_data[['Pclass', 'Survived']].groupby(['Pclass']).mean().plot.bar()

print("=" * 40)
print(train_data.groupby(['Sex', 'Pclass', 'Survived'])['Survived'].count())
train_data[['Sex', 'Pclass', 'Survived']].groupby(['Pclass', 'Sex']).mean().plot.bar()

# 探究年龄与存活的关系
print('=' * 40)
fig, ax = plt.subplots(1, 2, figsize=(18, 8))
sns.violinplot('Pclass', 'Age', hue='Survived', data=train_data, split=True, ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0, 110, 10))

sns.violinplot("Sex", "Age", hue="Survived", data=train_data, split=True, ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0, 110, 10))

plt.show()

# 分析总体的年龄分布
print('=' * 40)
plt.figure(figsize=(12, 5))
plt.subplot(121)
train_data['Age'].hist(bins=70)
plt.xlabel('Age')
plt.ylabel('Num')

plt.subplot(122)
train_data.boxplot(column='Age', showfliers=False)
plt.show()

# 不同年龄下的生存和非生存的分布情况：
facet = sns.FacetGrid(train_data, hue='Survived', aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train_data['Age'].max()))
facet.add_legend()

# 不同年龄下的平均生存率：
fig, axis1 = plt.subplots(1, 1, figsize=(18, 4))
train_data['Age_int'] = train_data['Age'].astype(int)
average_age = train_data[['Age_int', 'Survived']].groupby(['Age_int'], as_index=False).mean()
sns.barplot(x='Age_int', y='Survived', data=average_age)

print(train_data['Age'].describe())

# 按照年龄划分儿童、少年、成年、老年
print('=' * 40)
bins = [0, 12, 18, 65, 100]
train_data['Age_group'] = pd.cut(train_data['Age'], bins)
by_age = train_data.groupby('Age_group')['Survived'].mean()
print(by_age)
by_age.plot(kind='bar')

# 探究称呼与存活的关系
print('=' * 40)
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(train_data['Title'], train_data['Sex'])
train_data[['Title', 'Survived']].groupby(['Title']).mean().plot.bar()

# 观察名字长度和生存率之间存在关系的可能
fig, axis1 = plt.subplots(1, 1, figsize=(18, 4))
train_data['Name_length'] = train_data['Name'].apply(len)
name_length = train_data[['Name_length', 'Survived']].groupby(['Name_length'], as_index=False).mean()
sns.barplot(x='Name_length', y='Survived', data=name_length)

# 有无兄弟姐妹和存活的关系
sibsp_df = train_data[train_data['SibSp'] != 0]
no_sibsp_df = train_data[train_data['SibSp'] == 0]

plt.figure(figsize=(10, 5))
plt.subplot(121)
sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct='%1.1f%%')
plt.xlabel('sibsp')

plt.subplot(122)
no_sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct='%1.1f%%')
plt.xlabel('no_sibsp')

plt.show()

# 有无父母子女和存活与否的关系
parch_df = train_data[train_data['Parch'] != 0]
no_parch_df = train_data[train_data['Parch'] == 0]

plt.figure(figsize=(10, 5))
plt.subplot(121)
parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct='%1.1f%%')
plt.xlabel('parch')

plt.subplot(122)
no_parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct='%1.1f%%')
plt.xlabel('no_parch')

plt.show()

# 亲友的人数和存活与否的关系
fig, ax = plt.subplots(1, 2, figsize=(18, 8))
train_data[['Parch', 'Survived']].groupby(['Parch']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Parch and Survived')
train_data[['SibSp', 'Survived']].groupby(['SibSp']).mean().plot.bar(ax=ax[1])
ax[1].set_title('SibSp and Survived')

train_data['Family_Size'] = train_data['Parch'] + train_data['SibSp'] + 1
train_data[['Family_Size', 'Survived']].groupby(['Family_Size']).mean().plot.bar()

# 票价分布与存活的关系
plt.figure(figsize=(10, 5))
train_data['Fare'].hist(bins=70)
train_data.boxplot(column='Fare', by='Pclass', showfliers=False)
plt.show()

print('=' * 40)
print(train_data['Fare'].describe())

# 生存与否与票价均值和方差的关系
fare_not_survived = train_data['Fare'][train_data['Survived'] == 0]
fare_survived = train_data['Fare'][train_data['Survived'] == 1]
average_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])
average_fare.plot(yerr=std_fare, kind='bar', legend=False)
plt.show()

# 船舱类型与存活的关系
# Replace missing values with "U0"
train_data.loc[train_data.Cabin.isnull(),'Cabin']='U0'
train_data['Has_Cabin']=train_data['Cabin'].apply(lambda x:0 if x=='U0' else 1)
train_data[['Has_Cabin','Survived']].groupby(['Has_Cabin']).mean().plot.bar()

# 对不同的船舱类型进行分析
train_data['CabinLetter']=train_data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
train_data['CabinLetter']=pd.factorize(train_data['CabinLetter'])[0]
train_data[['CabinLetter','Survived']].groupby(['CabinLetter']).mean().plot.bar()

# 港口和存活与否的关系
sns.countplot('Embarked',hue='Survived',data=train_data)
plt.title('Embarked and Survived')
sns.factorplot('Embarked','Survived',data=train_data,size=3,aspect=2)
plt.title('Embarked and Survived rate')
plt.show()
