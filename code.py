import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA as sklearnPCA
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

df = pd.read_csv('mushroom.csv')#reading mushroom csv file
#Creating a list of dictionaries to change the features provided in alphabets to numbers.
list = [{ 'class' : { 'p' : 1 , 'e' : 2 }},
        { 'cap-shape' : { 'b' : 1, 'c' : 2,'f' : 3, 'x' : 4, 'k' : 5, 's' : 6 }},
        { 'cap-surface' : { 'f' : 1, 'g' : 2, 'y' : 3, 's' : 4 }},
        { 'cap-color' : { 'n' : 1, 'b' : 2,'c' : 3, 'g' : 4, 'r' : 5, 'p' : 6, 'u' : 7, 'e':8, 'w':9, 'y':10 }},
        { 'bruises' : { 't' : 1, 'f' : 2 }},
        { 'odor' : { 'a' : 1, 'l' : 2,'c' : 3, 'y' : 4, 'f' : 5, 'm' : 6 , 'n': 6, 'p': 7 , 's': 8 }},
        { 'gill-attachment' : { 'a' : 1, 'd' : 2,'f' : 3, 'n' : 4 }},
        { 'gill-spacing' : { 'c' : 1, 'w' : 2,'d' : 3 }},
        { 'gill-size' : { 'b' : 1, 'n' : 2 }},
        { 'gill-color' : { 'k' : 1, 'n' : 2,'b' : 3, 'h' : 4, 'g' : 5, 'r' : 6 ,'o' : 7, 'p':8, 'u':9, 'e':10 ,'w': 11,'y': 12 }},
        { 'stalk-shape' : { 'e' : 1, 't' : 2 }},
        { 'stalk-root' : { 'b' : 1, 'c' : 2,'u' : 3, 'e' : 4, 'z' : 5, 'r' : 6 ,'?': 7}},
        { 'stalk-surface-above-ring' : { 'f' : 1, 'y' : 2,'k' : 3, 's' : 4 }},
        { 'stalk-surface-below-ring' : { 'f' : 1, 'y' : 2,'k' : 3, 's' : 4 }},
        { 'stalk-color-above-ring' :{ 'n' : 1, 'b' : 2,'c' : 3, 'g' : 4, 'o' : 5, 'p' : 6, 'e' : 7, 'w':8, 'y':9 }},
        { 'stalk-color-below-ring' : {'n' : 1, 'b' : 2,'c' : 3, 'g' : 4, 'o' : 5, 'p' : 6, 'e' : 7, 'w':8, 'y':9  }},
        { 'veil-type' : { 'p' : 1, 'u' : 2 }},
        { 'veil-color' : { 'n' : 1, 'o' : 2,'w' : 3, 'y' : 4 }},
        { 'ring-type' : { 'c' : 1, 'e' : 2,'f' : 3, 'l' : 4, 'n' : 5, 'p' : 6 ,'s': 7 , 'z': 8}},
        { 'ring-number' : { 'n' : 1, 'o' : 2,'t' : 3 }},
        { 'spore-print-color' : { 'k' : 1, 'n' : 2,'b' : 3, 'h' : 4, 'r' : 5, 'o' : 6, 'u' : 7, 'w':8, 'y':9 }},
        { 'population' : { 'a' : 1, 'c' : 2,'n' : 3, 's' : 4, 'v' : 5, 'y' : 6 }},
        { 'habitat' : { 'g' : 1, 'l' : 2,'m' : 3, 'p' : 4, 'u' : 5, 'w' : 6, 'd' : 7 }}]

#Using for loop to change the dataframe using the replace function.
for i in list:
    df = df.replace(i)
#converting any string or integer data to float.
df = df.astype(float)

X_plot = df.drop(['class','veil-type'],axis = 1)
y_plot = df['class']

X_norm = (X_plot - X_plot.min())/(X_plot.max() - X_plot.min())#rescaling the data in range [0,1].

#Using Principle Component Analysis (PCA), a method of dimensionality reduction to reduce this multidimensional data into 2-dimension.
pca = sklearnPCA(n_components=2) #2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(X_norm))

#Plotting the data using matplotlib.
plt.scatter(transformed[y_plot==1][0], transformed[y_plot==1][1], label='Poisonus', c='red')
plt.scatter(transformed[y_plot==2][0], transformed[y_plot==2][1], label='Edible', c='blue')
plt.title('Plot of Known Classified data')
plt.legend(loc = 2)
plt.show()

X = np.array(df.drop(['class'], 1))
X = preprocessing.scale(X)#preprocessing the data.
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)#splitting the give data into train_data and test_data.
#We will use train_data to train our algorithm and test it with test_data. Here size of test_data is 50% of given data.

clf = svm.SVC()#Using the Support Vector Machine Classifier to classify the data.

clf.fit(X_train, y_train)#training the algorithm using the train_data.

accuracy = clf.score(X_test,y_test)#Testing the algorithm on test_data.
#this accuracy is equal to (number of correct predictions/ total number of predictions).
print ('accuracy : ',accuracy)#printing the accuracy.

df_predict = pd.read_csv('mushroom_predict.csv')#reading mushroom_predict csv file

#Creating a list of dictionaries to change the features provided in alphabets to numbers.
list = [{ 'cap-shape' : { 'b' : 1, 'c' : 2,'f' : 3, 'x' : 4, 'k' : 5, 's' : 6 }},
        { 'cap-surface' : { 'f' : 1, 'g' : 2, 'y' : 3, 's' : 4 }},
        { 'cap-color' : { 'n' : 1, 'b' : 2,'c' : 3, 'g' : 4, 'r' : 5, 'p' : 6, 'u' : 7, 'e':8, 'w':9, 'y':10 }},
        { 'bruises' : { 't' : 1, 'f' : 2 }},
        { 'odor' : { 'a' : 1, 'l' : 2,'c' : 3, 'y' : 4, 'f' : 5, 'm' : 6 , 'n': 6, 'p': 7 , 's': 8 }},
        { 'gill-attachment' : { 'a' : 1, 'd' : 2,'f' : 3, 'n' : 4 }},
        { 'gill-spacing' : { 'c' : 1, 'w' : 2,'d' : 3 }},
        { 'gill-size' : { 'b' : 1, 'n' : 2 }},
        { 'gill-color' : { 'k' : 1, 'n' : 2,'b' : 3, 'h' : 4, 'g' : 5, 'r' : 6 ,'o' : 7, 'p':8, 'u':9, 'e':10 ,'w': 11,'y': 12 }},
        { 'stalk-shape' : { 'e' : 1, 't' : 2 }},
        { 'stalk-root' : { 'b' : 1, 'c' : 2,'u' : 3, 'e' : 4, 'z' : 5, 'r' : 6 ,'?': 7}},
        { 'stalk-surface-above-ring' : { 'f' : 1, 'y' : 2,'k' : 3, 's' : 4 }},
        { 'stalk-surface-below-ring' : { 'f' : 1, 'y' : 2,'k' : 3, 's' : 4 }},
        { 'stalk-color-above-ring' :{ 'n' : 1, 'b' : 2,'c' : 3, 'g' : 4, 'o' : 5, 'p' : 6, 'e' : 7, 'w':8, 'y':9 }},
        { 'stalk-color-below-ring' : {'n' : 1, 'b' : 2,'c' : 3, 'g' : 4, 'o' : 5, 'p' : 6, 'e' : 7, 'w':8, 'y':9  }},
        { 'veil-type' : { 'p' : 1, 'u' : 2 }},
        { 'veil-color' : { 'n' : 1, 'o' : 2,'w' : 3, 'y' : 4 }},
        { 'ring-type' : { 'c' : 1, 'e' : 2,'f' : 3, 'l' : 4, 'n' : 5, 'p' : 6 ,'s': 7 , 'z': 8}},
        { 'ring-number' : { 'n' : 1, 'o' : 2,'t' : 3 }},
        { 'spore-print-color' : { 'k' : 1, 'n' : 2,'b' : 3, 'h' : 4, 'r' : 5, 'o' : 6, 'u' : 7, 'w':8, 'y':9 }},
        { 'population' : { 'a' : 1, 'c' : 2,'n' : 3, 's' : 4, 'v' : 5, 'y' : 6 }},
        { 'habitat' : { 'g' : 1, 'l' : 2,'m' : 3, 'p' : 4, 'u' : 5, 'w' : 6, 'd' : 7 }}]

#Using for loop to change the dataframe using the replace function.
for i in list:
    df_predict = df_predict.replace(i)


df_predict = df_predict.astype(float)
X_predict = np.array(df_predict)
X_predict = preprocessing.scale(X_predict)#preprocessing the predict_data.

prediction = pd.DataFrame(clf.predict(X_predict))#predicting the class labels using the svm classifier and converting to dataframe.

df_predict['class'] = prediction#appending the predict_data with the predicted class.


X_predict_plot = df_predict.drop(['class','veil-type'],axis = 1)
y_predict_plot = df_predict['class']

X_norm = (X_predict_plot - X_predict_plot.min())/(X_predict_plot.max() - X_predict_plot.min())#rescaling the data in range [0,1]


#Using Principle Component Analysis (PCA), a method of dimensionality reduction to reduce this multidimensional data into 2-dimension.
pca = sklearnPCA(n_components=2) #2-dimensional PCA
transformed_predict = pd.DataFrame(pca.fit_transform(X_norm))

#Plotting the predicted data using matplotlib.
plt.title('Plot of Predicted Data')
plt.scatter(transformed_predict[y_predict_plot==1][0], transformed_predict[y_predict_plot==1][1], label='Poisonus', c='orange')
plt.scatter(transformed_predict[y_predict_plot==2][0], transformed_predict[y_predict_plot==2][1], label='Edible', c='cyan')
plt.legend(loc = 2)
plt.show()

#Plotting both given data and predicted data and comparing them.
plt.title('Plot of Known Data and Predicted Data')
plt.scatter(transformed[y_plot==1][0], transformed[y_plot==1][1], label='Poisonus', c='red')
plt.scatter(transformed[y_plot==2][0], transformed[y_plot==2][1], label='Edible', c='blue')
plt.scatter(transformed_predict[y_predict_plot==1][0], transformed_predict[y_predict_plot==1][1], label='Poisonus Predicted', c='orange')
plt.scatter(transformed_predict[y_predict_plot==2][0], transformed_predict[y_predict_plot==2][1], label='Edible predicted', c='cyan')
plt.legend(loc = 2)
plt.show()

exit(0)
