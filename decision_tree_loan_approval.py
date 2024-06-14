import pandas as pd
df = pd.read_csv('/content/loan.csv')

from sklearn.preprocessing import LabelEncoder
age=LabelEncoder()
gender=LabelEncoder()
occupation=LabelEncoder()
education_level=LabelEncoder()
marital_status=LabelEncoder()
income=LabelEncoder()
credit_score=LabelEncoder()
loan_status=LabelEncoder()

df['occupation']= occupation.fit_transform(df['occupation'])
df['age']= occupation.fit_transform(df['age'])
df['gender']= occupation.fit_transform(df['gender'])
df['education_level']= occupation.fit_transform(df['education_level'])
df['marital_status']= occupation.fit_transform(df['marital_status'])
df['income']= occupation.fit_transform(df['income'])
df['credit_score']= occupation.fit_transform(df['credit_score'])
df['loan_status']= occupation.fit_transform(df['loan_status'])

features_cols = ['age', 'gender', 'occupation', 'education_level', 'marital_status', 'income', 'credit_score']
X = df[features_cols]
y = df.loan_status

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='gini')
clf.fit(X_train,y_train)

clf.predict(X_test)

from sklearn import tree
tree.plot_tree(clf)