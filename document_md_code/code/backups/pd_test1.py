import numpy as np
import pandas as pd
import matplotlib as plt
import math

titanic = pd.read_csv('train.csv')
print(titanic.describe())
print(titanic.shape)

# 1-性别数值化
titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 0
titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1
# print(titanic['Sex'].describe())


# print(titanic['Age'].describe())
# 统计空值
def null_count(column):
    column_null = pd.isnull(column)
    # print(column_null)
    null = column[column_null]
    print(len(null))
null_count(titanic['Age'])

# 2-填充年龄空值
titanic['Age']=titanic['Age'].fillna(titanic['Age'].mean()) # 用均值填充
print(titanic['Age'].describe())

print('-'*30)
# 3-将上船港口映射至数值
print(titanic['Embarked'].unique())
print('-'*30)
print(titanic['Embarked'].mode())
null_count(titanic['Embarked'])
print('-'*30)
print(titanic['Embarked'].describe())
print('-'*30)
# 3-1 填充港口空值
titanic['Embarked']=titanic['Embarked'].fillna('S')
# titanic['Embarked']=titanic['Embarked'].fillna(titanic['Embarked'].mode())
print(titanic['Embarked'].describe())

# 3-2 港口：S-0, C-1, Q-2
titanic.loc[titanic['Embarked'] == 'S','Embarked'] = 0
titanic.loc[titanic['Embarked'] == 'C','Embarked'] = 1
titanic.loc[titanic['Embarked'] == 'Q','Embarked'] = 2
print(titanic['Embarked'].describe())

print('-'*30)
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_validate
# # print(help(KFold))
# predictors = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
# alg = LinearRegression()
# kf = KFold(titanic.shape[0],random_state= 1 )
#
# predictions=[]
# for train,test in kf:
#     train_predictors=(titanic[predictors].iloc[train,:])
#     train_target=titanic['Survived'].iloc[train]
#     alg.fit(train_predictors,train_target)
#     test_predictions=alg.predict(titanic[predictors].iloc[test,:])
#     predictions.append(test_predictions)
# predictions=np.concatenate(predictions,axis=0) #在行的方向拼接两个数组
# # print(help(np.concatenate))
# predictions[predictions>0.5]=1
# predictions[predictions<=0.5]=0
# accuracy=len(predictions[predictions==titanic['Survived']])/(len(predictions)) #预测的情况与实际情况一致的数目/所有样本数
# print(accuracy) #结果：0.7833894500561167
#2. 用逻辑回归预测
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
alg = LogisticRegression(random_state=1,solver='liblinear')
scores = cross_val_score(alg, titanic[predictors], titanic['Survived'], cv=3)
print(scores.mean())  # 0.7901234567901234

# 3. 用随机森林改进模型
# from sklearn import model_selection
# from sklearn.ensemble import RandomForestClassifier
#
# predictors=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'] #用于预测的项,【有改进的空间】
# #n_estimators 森林中树的数目
# rf_alg=RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)
# kf=model_selection.KFold(titanic.shape[0], random_state=1)
# scores=model_selection.cross_val_score(rf_alg,titanic[predictors],titanic['Survived'],cv=kf)
# print(scores) #[0.78114478 0.82491582 0.84175084]
# print(scores.mean()) #0.8159371492704826



