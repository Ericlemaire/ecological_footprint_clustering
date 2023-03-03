#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:02:17 2023

@author: lemaireeric
"""

import pandas as pd 
import numpy as np
import seaborn as sns 


import streamlit as st 

import matplotlib.pyplot as plt 
import plotly.express as px 

import geopandas as gpd
import contextily as ctx
import folium,mapclassify

import plotly.figure_factory as ff
from joblib import dump, load


import shap 
from streamlit_shap import st_shap
shap.initjs()


#import pydeck as pdk

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

st.markdown("# Introduction")

st.markdown("""
            Cette application a pour objectif de permettre aux utilisateurs d'explorer 
            les relations entre la satisfaction des populations et/ou leur niveau de développement 
            et les soubassements matériels ou écologiques de leurs formes de vie. 
            
            Dans la page "Exploration des données", l'utilisateur pourra ainsi explorer la tables des données, les princippales statistiques descriptives 
            s'y rattachant ; mais il pourra également visualiser les variables, sous forme d'histogramme, de cartes, ou leurs relations
            grâce à des nuages de points.
            
            Nous proposons également (Page : clustering) la possibilité d'utiliser un algorithme d'apprentissage non supervisé appelé KMeans afin de chercher 
            des regroupements possibles entre pays sur la base de leurs similarités. 
            
            Cette application est un travail en cours ou work in progress. Elle est susceptible
            de multiples améliorations. Aussi, vos idées et suggestions sont les bienvenues ! 
            """)



# Menu des pages 
pages = [
    "Exploration des données", 
    "Clustering",
    "Interprétation des clusters"
        ]
page = st.sidebar.radio("Aller vers", pages)



#pages introduction
if page == pages[0]:  
    st.markdown("## Exploration des données")
    
# Affichage du dataframe 
    df = pd.read_csv("eco_ft_clustering.csv")
    df = df.drop(["Unnamed: 0", "geometry"], axis = 1)
    st.dataframe(df)

# Explorer les statistiques de base en fonction d'une varaible sélectionnée
    option_metric = st.selectbox(
    'Quelle variable voulez-vous explorer ?',options = df.columns[3:])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("moyenne", df.mean().round(1)[option_metric])
    col2.metric("Max", df[option_metric].round().max())
    col3.metric("Min", df[option_metric].round().min())
    col4.metric("Écart-type", df.std()[option_metric].round(1))


# Affichage des statitistques de base 
    st.write("Résumé des principales statistiques")

    st.dataframe(df.describe())


    st.subheader("Cartographie")
    st.write("Explorer comment les différentes variables se répartissent géographiquement")




# Lecture de df avec géopandas
    
    gpd_df = gpd.read_file("eco_ft_clustering.json")
    #st.dataframe(gpd_df)
    carto_option = st.selectbox(
     'Avec quelle variable voulez-vous colorer le monde ?', options = gpd_df.columns)

 # Création et affichage de la carte
    fig = px.choropleth_mapbox(gpd_df,
                            geojson=gpd_df.geometry,
                            locations=gpd_df.index,
                            color=carto_option,
                            center={"lat": 20.5517, "lon": -5.7073},
                            mapbox_style="open-street-map",
                            zoom=0.5)

    st.plotly_chart(fig, use_container_width=False)
    



# Histogrammes 


    st.subheader("Distribution des variables")

 #if st.checkbox("Afficher les valeurs manquantes") :
  #   st.dataframe(df.isna().sum())
  
    st.write("Les hisogrammes permettent de représenter la forme de la distribution des variables.")
   
   # Bouton d'option pour les histogrammes
    option_hist_1 = st.selectbox(
   'Quelle variable voulez-vous visualiser en ordonnée ?', options = df.columns[1:])
    #option_hist_2 = st.selectbox(
   #'Comment voulez-vous grouper vos données?',options = ['Région du monde', 'Régime politique'])
    group_labels = [option_hist_1] # name of the dataset

   #hist_surv = ff.create_distplot(option_hist_1, group_labels)

# Création de l'histogramme 
    hist_surv = px.histogram(df, x = option_hist_1, nbins=20) #, color = option_hist_2,  barmode = "group")
 
    st.plotly_chart(hist_surv)




   
   
# Nuage de points 
   
    st.subheader("Visualiser les relations entre variables")
   
    st.markdown("Explorez les relations entre variables du jeu de données en choisissant les variables à faire apparaître en ordonnée, en abscisse, et en couleur. Jouer avec les variables permet de reprérer visuellement des corrélations. Vous pouvez ensuite tester votre intuition en réalisant le test de Pearson.")

    option_1 = st.selectbox(
    'Quelle variable voulez-vous visualiser en ordonnée pour le nuage de points?',options = df.columns[:-1])

    option_2 = st.selectbox(
   'Quelle variable voulez-vous visualiser en abscisse pour le nuage de points',options = df.columns[:-1])
   
   
    option_3 = st.selectbox(
   'Avec quelle variable voulez-vous colorer vos points?', options = df.columns[1:])
   
    fig = px.scatter(df, x = option_2, y= option_1, color = option_3, 
    hover_data=['Country'])        
    st.plotly_chart(fig)
    
   





    
    
    
if page == pages[1]: 
    st.markdown("## Clustering")
    
    df = pd.read_csv("eco_ft_clustering.csv")
    df = df.drop(["Unnamed: 0", "geometry"], axis = 1)
    df = df.dropna(axis=0)
    st.dataframe(df)
    
    scaler = MinMaxScaler()
    df_sc = scaler.fit_transform(df.iloc[: , 1:])

# Entraîment du KMeans et réalisation des graphiques pour les 4 métriques


    from sklearn.cluster import KMeans
    from sklearn import metrics
    #from sklearn.metrics import pairwise_distances
    #from sklearn.metrics import davies_bouldin_score


    inertias = []
    sils = []
    chs = []
    dbs = []


    sizes = range(2,15)
    for k in sizes :
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=40)
        kmeans.fit(df_sc)
        labels = kmeans.labels_
        inertias.append(kmeans.inertia_)
        sils.append(metrics.silhouette_score(df_sc, labels, metric='euclidean'))
        chs.append(metrics.calinski_harabasz_score(df_sc, labels))
        dbs.append(metrics.davies_bouldin_score(df_sc, labels))
  

# dataframe des Métriques

    Métriques_clustering = pd.DataFrame({"Inertie": inertias, "Silhouète":sils, "Calinski": chs, "davis": dbs, "k": sizes }).set_index("k")
    print(Métriques_clustering)


# Représentation graphique des métriques 
    st.markdown('Évaluation de l''''algorithme de clustering''')
    
    st.markdown('Que signifie les différentes métriques ?')
# Changer pour plotly
    fig, ax = plt.subplots(figsize=(6,4))
    Métriques_clustering.plot(ax= ax, subplots = True, layout=(2,2))
    st.pyplot(fig)
     
    
# Entraînement du KMeans en fonction du nombre de clusters sélectionné
    
    values = st.slider(
        'Select the number of clusters',
        2, 10)
    kmeans = KMeans(n_clusters=values, init='k-means++', random_state=40)
## Ajsutement 
    kmeans.fit(df_sc)
    y_kmeans = kmeans.predict(df_sc)

# Dataframe avec les centroïds
    clusters = pd.DataFrame(kmeans.cluster_centers_, columns = df.iloc[: , 1:].columns)
    st.dataframe(clusters)
    
# Représentation des centroïdes : "Le pays moyen du cluster" !
    fig, ax = plt.subplots(figsize=(4,2))
    df.assign(cluster =y_kmeans).groupby("cluster").mean().T.plot.bar(ax = ax)
    st.pyplot(fig)





# Entraînement d'un arbre de décision pour préparaer l'interprétation avec SHAP 

# SOURCES NON CONSULTÉES MAIS IMPORTANTES : https://towardsdatascience.com/explainable-ai-xai-with-shap-multi-class-classification-problem-64dd30f97cea
# SOURCES : https://towardsdatascience.com/how-to-make-clustering-explainable-1582390476cc

    clf=RandomForestClassifier()
    clf.fit(df_sc,y_kmeans)
    clf.score(df_sc,y_kmeans)
    
  
    
  
# Création de la matrice de confusion
  
    
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import plot_confusion_matrix

    y_true = y_kmeans
    y_pred = clf.predict(df_sc)
    confusion_matrix(y_true, y_pred)
    matrix = plot_confusion_matrix(clf, df_sc, y_true)  
    
    st.write(matrix)


# Interprétation 

# Avec arbre de décision : 
    
    
    from sklearn import tree


    clf_dt = tree.DecisionTreeClassifier(random_state=0)
    clf_dt = clf_dt.fit(df.iloc[:,1:],y_kmeans)
 
    tree.plot_tree(clf_dt, class_names = ["A","B","C"], feature_names = df.iloc[:,1:].columns );

    plt.show()





# SOURCES : https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Iris%20classification%20with%20scikit-learn.html
# SOURCE MEDIUM : https://towardsdatascience.com/explainable-ai-xai-with-shap-multi-class-classification-problem-64dd30f97cea


    import shap
    
    explainer= shap.TreeExplainer(clf)
    shap_values = explainer(df_sc).values

    #explainer = shap.KernelExplainer(clf_dt.predict_proba, df_clustering.iloc[0:,1:])
    shap_values = explainer.shap_values(df_sc)
    
    shap.initjs()


# SHAP DEPENDANCE PLOT 

    #ax.set_title(f"Influence de la variable '{option_dependance_plot}' de choix sur les prédictions")
    #shap.dependence_plot(option_dependance_plot, shap_values,df_sc, interaction_index = ??? , ax= ax,show=False)
    #st.pyplot( bbox_inches = 'tight')      


# SHAP FORCE PLOT 

    fig, ax = plt.subplots()
    shap.initjs()
    shap_plot_1 = shap.force_plot(explainer.expected_value[0], shap_values[0], df_sc)
    st.pyplot(shap_plot_1)




#pages 3
if page == pages[2]:  
    st.markdown("## Interprétation des clusters")




    df = pd.read_csv("eco_ft_clustering.csv")
    df = df.drop(["Unnamed: 0", "geometry"], axis = 1)
    df = df.dropna(axis=0)
    st.dataframe(df)

    scaler = MinMaxScaler()
    df_sc = scaler.fit_transform(df.iloc[: , 1:])
# Entraînement d'un arbre de décision pour préparaer l'interprétation avec SHAP 

# SOURCES NON CONSULTÉES MAIS IMPORTANTES : https://towardsdatascience.com/explainable-ai-xai-with-shap-multi-class-classification-problem-64dd30f97cea
# SOURCES : https://towardsdatascience.com/how-to-make-clustering-explainable-1582390476cc


    clf=RandomForestClassifier()
    clf.fit(df_sc,y_kmeans)
    clf.score(df_sc,y_kmeans)
    
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import plot_confusion_matrix

# Création de la matrice de confusion
    y_true = y_kmeans
    y_pred = clf.predict(df_sc)
    confusion_matrix(y_true, y_pred)
    matrix = plot_confusion_matrix(clf, df_sc, y_true)  
    
    st.pyplto(matrix)
