

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the packages.
2.Analyse the data.
3.Use modelselection and Countvectorizer to preditct the values.
4.Find the accuracy and display the result.
 ```

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: MONIKA V
RegisterNumber:  212225100028
*/


import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding='latin -1')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)



```

## Output:
<img width="889" height="552" alt="Screenshot 2026-05-25 210005" src="https://github.com/user-attachments/assets/29b56de2-a405-4197-80e3-cf8c2db84a72" />

<img width="334" height="63" alt="Screenshot 2026-05-25 210017" src="https://github.com/user-attachments/assets/285a5d7a-c11b-49e4-87d5-1d3ac41a071e" />



<img width="331" height="125" alt="Screenshot 2026-05-25 210029" src="https://github.com/user-attachments/assets/f97df42d-e323-42a2-8c24-ef036567bcbf" />

<img width="409" height="147" alt="Screenshot 2026-05-25 210110" src="https://github.com/user-attachments/assets/17919db3-9df3-46ee-8835-c8830f843b4c" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
