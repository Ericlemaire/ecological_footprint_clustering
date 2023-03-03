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
    st.image("AAdobeStock_250594518.jpeg",width=600)

# Affichage du dataframe avec les données relatives à l'empreinté cologiques et autres variables

    df = pd.read_csv("eco_ft_clustering.csv")
    df = df.drop(["Unnamed: 0", "geometry"], axis = 1)
    st.dataframe(df)

# Explorer les statistiques de base en fonction d'une variable sélectionnée
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




# Lecture de df avec géopandas pour la cartographie
    
    gpd_df = gpd.read_file("eco_ft_clustering.json")

# Création d'une boîte pour sélectionnner les variable à afficher dans la carte
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
    



# Histogrammes des distribution des variables

    st.subheader("Distribution des variables")

  
    st.write("Les hisogrammes permettent de représenter la forme de la distribution des variables.")
   
   # Bouton d'option pour les histogrammes
   
    option_hist_1 = st.selectbox(
   'Quelle variable voulez-vous visualiser en ordonnée ?', options = df.columns[1:])
   
    
    group_labels = [option_hist_1] # name of the dataset


# Création de l'histogramme 
    hist_surv = px.histogram(df, x = option_hist_1, nbins=20) #, color = option_hist_2,  barmode = "group")
 
    st.plotly_chart(hist_surv)



   
   
# Nuage de points permettant d'explorer les relations entre variables.
   
    st.subheader("Visualiser les relations entre variables")
   
    st.markdown("Explorez les relations entre variables du jeu de données en choisissant les variables à faire apparaître en ordonnée, en abscisse, et en couleur. Jouer avec les variables permet de reprérer visuellement des corrélations. Vous pouvez ensuite tester votre intuition en réalisant le test de Pearson.")

# Création des boîte permttant de slectionner les différentes variables à comparer

    option_1 = st.selectbox(
    'Quelle variable voulez-vous visualiser en ordonnée pour le nuage de points?',options = df.columns[:-1])

    option_2 = st.selectbox(
   'Quelle variable voulez-vous visualiser en abscisse pour le nuage de points',options = df.columns[:-1])
   
    option_3 = st.selectbox(
   'Avec quelle variable voulez-vous colorer vos points?', options = df.columns[1:])
   
# Création et affichage des nuages de points 
   
    fig = px.scatter(df, x = option_2, y= option_1, color = option_3, 
    hover_data=['Country'])        
    st.plotly_chart(fig)
    
   

    
    
    
if page == pages[1]: 
    st.markdown("## Clustering")

# Ouvertude du dataframe contenant les données pour le clustering

    df = pd.read_csv("eco_ft_clustering.csv")
    df = df.drop(["Unnamed: 0", "geometry"], axis = 1)
    df = df.dropna(axis=0)
    st.dataframe(df)
 
# Standardisation des données

    scaler = MinMaxScaler()
    df_sc = scaler.fit_transform(df.iloc[: , 1:])

# Entraîment du KMeans et réalisation des graphiques pour les 4 métriques

    from sklearn.cluster import KMeans
    from sklearn import metrics
    

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
    
    st.markdown("""
                
                """)
# Changer pour plotly
    fig, ax = plt.subplots(figsize=(6,4))
    Métriques_clustering.plot(ax= ax, subplots = True, layout=(2,2))
    st.pyplot(fig)
     
    
# Entraînement du KMeans en fonction du nombre de clusters sélectionné
    
# Slier permettant de sélectionner le nombre de clusters
    values = st.slider(
        'Select the number of clusters',
        2, 10)

# Instanciation du modèle de clsutering

    kmeans = KMeans(n_clusters=values, init='k-means++', random_state=40)

# Ajustement 
    kmeans.fit(df_sc)
    y_kmeans = kmeans.predict(df_sc)

# Dataframe avec les centroïds
    clusters = pd.DataFrame(kmeans.cluster_centers_, columns = df.iloc[: , 1:].columns)
    st.dataframe(clusters)
    
# Représentation des centroïdes : "Le pays moyen du cluster" !
    fig, ax = plt.subplots(figsize=(4,2))
    df.assign(cluster =y_kmeans).groupby("cluster").mean().T.plot.bar(ax = ax)
    st.pyplot(fig)





