# Visualizing the clusters geographically
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
#from chart_studio.plotly import plot, iplot
from plotly.offline import iplot
import warnings

# Temporarily suppress all warnings
warnings.filterwarnings("ignore")

data = dict(type = 'choropleth', 
           locations = happy_df_cluster["Country"],
           locationmode = 'country names',
           colorscale='RdYlGn',
           z = happy_df_cluster['cluster'], 
           text = happy_df_cluster["Country"],
           colorbar = {'title':'Clusters'})

layout = dict(
    title='Geographical Visualization of Clusters',
    geo=dict(
        showframe=True,
        projection={'type': 'azimuthal equal area'}
    ),
    # Adjust the height and width to make the map bigger
    height=800,
    width=1200   
)

choromap3 = go.Figure(data = [data], layout=layout)
iplot(choromap3)
