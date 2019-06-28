from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go
import random
import numpy as np
import pandas as pd


l= []
y= []
data= pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv")

data.shape
N= 53

c= ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]
c
len(c)
data['Rank']
dict(width=1, height=2)
# Each trace is a basic unit for plot.ly
# we can set name(name), coordinates(x, y, z), style(marker: color, size, opacity, line), annotation(test)
for i in range(int(N)):
    y.append((2000+i))
    trace0= go.Scatter(
        x= data['Rank'],
        y= data['Population']+(i*1000000),
        mode= 'markers',
        marker= dict(size= 14,
                    line= dict(width=1),
                    color= c[i],
                    opacity= 0.3
                   ),
        name= y[i],
        text= data['State']) # The hover text goes here... 
    l.append(trace0);

layout= go.Layout(
    title= 'Stats of USA States',
    hovermode= 'closest',
    xaxis= dict(
        title= 'Population',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Rank',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)
fig= go.Figure(data=l, layout=layout)
iplot(fig)