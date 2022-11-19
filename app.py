
import streamlit as st
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

st.set_option('deprecation.showPyplotGlobalUse', False)



def main():
    st.title("Application de Machine Learning pour la détection de fraude par carte de credit")
    st.subheader("Mondjehi Roland")
    
    @st.cache(persist=True) # permet de save en memoire les données pour faciliter leur utilisation
    # Fonction d'importation des données
    def load_data():
        data = pd.read_csv("creditcard.csv")
        return data
    
    # Affichage de la table de données
    df = load_data()
    df_sample = df.sample(100)
    #st.write(df_sample)
    
    if st.sidebar.checkbox("Afficher les données brutes", False):
        st.subheader("Jeu de données creditcard :  Echantillon de 100 observations")
        st.write(df_sample)
     
    seed = 123
    
    @st.cache(persist=True)
    def split_data(df):
        
        y = df['Class']
        X = df.drop('Class', axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = seed, test_size=0.2, stratify=y)
        
        return X_train, X_test, y_train, y_test
    
    X_train, X_test, y_train, y_test = split_data(df)
    
    class_names = ['T authentique', 'T frauduleuse']
    
    classifier = st.sidebar.selectbox(
         "Classificateur", 
         ("Random Forest", "SVM", "Logistic Regression") )
    
    #Analyse de la performance des modèles
    def plot_perf(graphes):
        
        if 'Confusion matrix' in graphes:
            st.subheader('Matrice de confusion')
            plot_confusion_matrix(model, X_test, y_test)
            st.pyplot()
            
        if 'ROC Curve' in graphes:
            st.subheader('Courbe ROC')
            plot_roc_curve(model, X_test, y_test)
            st.pyplot()
            
        if 'Precision-Recall Curve' in graphes:
            st.subheader('Courbe de Precision-Recall')
            plot_precision_recall_curve(model, X_test, y_test)
            st.pyplot()
            
            
                         
        
    # Random Forest
    
    if classifier == "Random Forest":
        st.sidebar.subheader("Hyperparamètres du modèle")
        n_estimators = st.sidebar.number_input("choisir le nombre d'arbres dans la forêt",  
                                               100, 1000, step=10)
        
        max_depth = st.sidebar.number_input("choisir la profondeur de l'arbre",  
                                                1, 20, step=1)
        
        bootstrap = st.sidebar.radio(
            "Echantillon Boostrap lors de la création d'arbres",
            ("True", "False")
            )
        
        graphes_perf = st.sidebar.multiselect(
            "Choisir un graphique de performance",
            ("Confusion matrix", "ROC Curve", "Precision-Recall Curve"))
        
        if st.sidebar.button("Execution", "Classify"):
            st.subheader("Random Forest Results")
            
            #initialisation d'un objet RandomForest
            
            model = RandomForestClassifier(
                n_estimators = n_estimators,
                max_depth=max_depth,
                bootstrap=bootstrap
                )
            #Entrainement  de l'algorithme
            model.fit(X_train, y_train)
            
            #Predictions
            y_pred = model.predict(X_test)
            
            # Metrics de performances
            
            accuracy = model.score(X_test, y_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            # Afficher les metrics dans les applications
            
            st.write("Accuracy : ", accuracy)
            st.write("Precision : ", precision)
            st.write("recall : ", recall)
            
            # Affichier les metrics de performance
            plot_perf(graphes_perf)
            
    
    # Regression logistic
    
    if classifier == "Logistic Regression":
        st.sidebar.subheader("Hyperparamètres du modèle")
        c = st.sidebar.number_input("Choisir la regularisation",  
                                               0.01, 10.0)
        
        n_max_iter = st.sidebar.number_input("Nombre maximal d'itération",  
                                                100, 1000, step=10)
        
        
        
        graphes_perf = st.sidebar.multiselect(
            "Choisir un graphique de performance",
            ("Confusion matrix", "ROC Curve", "Precision-Recall Curve"))
        
        if st.sidebar.button("Execution", "Classify"):
            st.subheader("Logistic Regression Results")
            
            #initialisation d'un objet LogisticRegression
            
            model = LogisticRegression(
                C = c,
                max_iter = n_max_iter,
                )
            #Entrainement  de l'algorithme
            model.fit(X_train, y_train)
            
            #Predictions
            y_pred = model.predict(X_test)
            
            # Metrics de performances
            
            accuracy = model.score(X_test, y_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            # Afficher les metrics dans les applications
            
            st.write("Accuracy : ", accuracy)
            st.write("Precision : ", precision)
            st.write("recall : ", recall)
            
            # Affichier les metrics de performance
            plot_perf(graphes_perf)
            
    
    # SVM
    
    if classifier == "SVM":
        st.sidebar.subheader("Hyperparamètres du modèle")
        c = st.sidebar.number_input("Choisir la regularisation",  0.01, 10.0)
        
        kernel = st.sidebar.radio("Choisir le Kernel",  ('rbf', 'linear'))
        
        gamma = st.sidebar.radio("Gamma", ('scale', 'auto'))
        
        
        graphes_perf = st.sidebar.multiselect(
            "Choisir un graphique de performance",
            ("Confusion matrix", "ROC Curve", "Precision-Recall Curve"))
        
        if st.sidebar.button("Execution", "Classify"):
            st.subheader("SVM Results")
            
            #initialisation d'un objet SVC
            
            model = SVC(
                C = c,
                kernel = kernel,
                gamma = gamma
                )
            #Entrainement  de l'algorithme
            model.fit(X_train, y_train)
            
            #Predictions
            y_pred = model.predict(X_test)
            
            # Metrics de performances
            
            accuracy = model.score(X_test, y_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            # Afficher les metrics dans les applications
            
            st.write("Accuracy : ", accuracy)
            st.write("Precision : ", precision)
            st.write("recall : ", recall)
            
            # Affichier les metrics de performance
            plot_perf(graphes_perf)
            
    
    
            
            
if __name__ == '__main__':
    main()