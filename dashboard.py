# Importing modules
import streamlit as st
from streamlit_extras.mandatory_date_range import date_range_picker
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go


# Setting page config
st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# Reading data
st.header('Shooting dashboard')
df = pd.read_excel('https://github.com/Savelyev-Ilya/shooting_dashboard/raw/refs/heads/master/shooting_data.xlsx')


# Creating sidebar
with st.sidebar:
    st.header("Filters")

    st.write('Date range')
    date_range = date_range_picker("Select a date period you want to analyze", 
                               default_start=df['date'].min(),
                               default_end=df['date'].max()
                               )

    st.write('Time grain')
    time_grain = st.selectbox(
        label="Select a time granularity",
        options=("Day", "Week", "Month", "Year"),
        index=0
    )

# Date range    
df = df[(df['date'] >= pd.Timestamp(date_range[0])) & (df['date'] < pd.Timestamp(date_range[1] + timedelta(days=1)))]


# Time grain
df_chart = df.groupby(['date', 'round_id']).agg({'result': 'sum', 'second_shot': 'sum', 'shot_id': 'count'}).reset_index()

if time_grain == 'Day':
    df_chart['date'] = df_chart['date'].dt.date
    df_chart = df_chart.groupby('date').agg({'result': ['mean', 'sum'], 'shot_id': 'sum'}).reset_index()
    df_chart.columns = ['date', 'avg_score', 'hits', 'shots']

elif time_grain == 'Week':
    df_chart['date'] = df_chart['date'].dt.to_period('W').apply(lambda r: r.start_time).dt.date
    df_chart = df_chart.groupby('date').agg({'result': ['mean', 'sum'], 'shot_id': 'sum'}).reset_index()
    df_chart.columns = ['date', 'avg_score', 'hits', 'shots']

elif time_grain == 'Month':
    df_chart['date'] = df_chart['date'].dt.to_period('M').apply(lambda r: r.start_time).dt.date
    df_chart = df_chart.groupby('date').agg({'result': ['mean', 'sum'], 'shot_id': 'sum'}).reset_index()
    df_chart.columns = ['date', 'avg_score', 'hits', 'shots']

elif time_grain == 'Year':
    df_chart['date'] = df_chart['date'].dt.to_period('Y').apply(lambda r: r.start_time).dt.date
    df_chart = df_chart.groupby('date').agg({'result': ['mean', 'sum'], 'shot_id': 'sum'}).reset_index()
    df_chart.columns = ['date', 'avg_score', 'hits', 'shots']

df_chart['avg_score'] = df_chart['avg_score'].round(1)


# Metrics calculation
avg_score = round(df['result'].sum() / len(df['round_id'].unique()), 1)

total_hits = df['result'].sum()

total_shots = df.shape[0]


# Metrics visualization
st.markdown('**Key metrics**')

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.metric(label="Average score", value=avg_score)

with col2:
    with st.container(border=True):
        st.metric(label="Total targets hit", value=total_hits)

with col3:
    with st.container(border=True):
        st.metric(label="Total shots made", value=total_shots)


# Average score over time chart
with st.container(border=True):

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_chart['date'], y=df_chart['avg_score'], mode='lines', line_color='orange'))
    fig.add_trace(go.Scatter(x=df_chart['date'], y=df_chart['avg_score'], mode='markers', marker=dict(color='white')))


    # Add annotations for each point
    y_align = (df_chart['avg_score'].max() - df_chart['avg_score'].min()) / 10

    for i in range(len(df_chart)):
        fig.add_annotation(
            x=df_chart['date'][i], 
            y=df_chart['avg_score'][i] + y_align, 
            text=df_chart['avg_score'][i],
            showarrow=False,
            font=dict(color='white')
        )

    # Update layout for better visualization
    fig.update_layout(
        title=dict(text="Average score over time", font=dict(color='white')),
        xaxis_title=dict(text=time_grain, font=dict(color='white')),
        yaxis_title=dict(text="Average score", font=dict(color='white')),
        showlegend=False,
        xaxis = dict(
            tickmode = 'array',
            tickvals = df_chart['date'],
            ticktext = df_chart['date'],
            color='white'
        ),
        yaxis = dict(
            tickmode = 'linear',
            tick0 = df_chart['avg_score'].min(),
            dtick = 0.2,
            color='white'
        ),
        paper_bgcolor='#0f1116',
        plot_bgcolor='#0f1116'
    )

    # config = {'staticPlot': True}

    # st.plotly_chart(fig, config=config)

    fig.write_image('fig.jpg', scale=2.0)

    st.image('fig.jpg')


# Results by shot number
df_shots = df.groupby('shot_number').agg({'result': 'mean'}).reset_index()

with st.container(border=True):

    fig1 = go.Figure()

    for i in range(25):

        if df_shots['result'][i] >= avg_score / 25:
            color = 'Green'
        elif df_shots['result'][i] >= avg_score / 25 - 0.05:
            color = 'Yellow'
        else:
            color = 'Red'

        fig1.add_shape(type='rect',
                        x0=i+0.65+(0.2 * (i // 5)), y0=0.05,
                        x1=i+1.45+(0.2 * (i // 5)), y1=0.35,
                        line=dict(
                            color='White', 
                        ),
                        line_width=0.5,
                        fillcolor=color
                    )

        shot_result = int(df_shots["result"][i] * 100)

        fig1.add_annotation(x=(i+1.35+i+0.65+(0.2 * (i // 5))+(0.2 * (i // 5)))/2,
                            y=0.4,
                            text=f'{shot_result}%',
                            showarrow=False,
                            font=dict(size=10)
                        )


    # Update layout for better visualization
    fig1.update_layout(
        title=f"Accuracy by shot number",
        showlegend=False,
        yaxis=dict(
            range=[0, 0.4],
            showgrid=False,
            visible=False,
        ),
        xaxis=dict(
            range=[0.5, 26.5],
            showgrid=False,
            visible=False
        ),
        height=250
    )

    config1 = {'staticPlot': True}

    st.plotly_chart(fig1, config=config1)


# Results by target direction
with st.container(border=True):

    col_directions, col_switch = st.columns([0.8, 0.2])

    with col_switch:
        is_detailed = st.toggle("Detalied report")

    with col_directions:
        fig2 = go.Figure()

        x_dirs = (-1.2, 0, 1.2)
        dirs_descriptions = ('Left', 'Straight', 'Right')

        for i in range(3):

            fig2.add_annotation(x=x_dirs[i],
                        y=1.25,
                        text=f'<b>{dirs_descriptions[i]}</b>',
                        showarrow=False,
                        font=dict(size=15)
                    ) 

        if is_detailed:
            df_directions = df.groupby(['direction_id']).agg({'result': 'mean'}).reset_index() 

            x_tuple = ([-1.5, -1.6], [-1.5, -1.8], [-1.5, -1.9], [-1.5, -1.9], [-1.5, -1.6], [-0.3, -0.3], [-0.3, -0.3], [-0.3, -0.3], [0.9, 1], [0.7, 1], [0.6, 1], [0.6, 1], [0.9, 1])
            y_tuple = ([1, 1.2], [0.75, 0.9], [0.55, 0.62], [0.35, 0.37], [0.05, 0.15], [0.8, 1.1], [0.4, 0.6], [0.1, 0.2], [1, 1.2], [0.75, 0.9], [0.55, 0.62], [0.35, 0.37], [0.05, 0.15])
            direction_descriptions = (["High<br>Short", 
                                       "High<br>Long", 
                                       "Mid<br>Long", 
                                       "Low<br>Long", 
                                       "Low<br>Short",
                                       "High",
                                       "Mid",
                                       "Low",
                                       "High<br>Short", 
                                       "High<br>Long", 
                                       "Mid<br>Long", 
                                       "Low<br>Long", 
                                       "Low<br>Short"])

            for i in range(len(x_tuple)):

                fig2.add_trace(go.Scatter(
                            x=x_tuple[i],
                            y=y_tuple[i],
                            mode="lines+markers",
                            marker=dict(
                                symbol="arrow",
                                size=15,
                                angleref="previous",
                                color='orange'
                                )
                            )
                )

                direction_result = round(df_directions["result"][i] * 100, 1)

                # Add accuracy percentage
                fig2.add_annotation(x=max(x_tuple[i])+0.2,
                                    y=sum(y_tuple[i])/2,
                                    text=f'{direction_result}%',
                                    showarrow=False,
                                    font=dict(size=12)
                                )   

                # Add direction description
                fig2.add_annotation(x=max(x_tuple[i])+0.5,
                                    y=sum(y_tuple[i])/2,
                                    text=direction_descriptions[i],
                                    showarrow=False,
                                    font=dict(size=12)
                                )                                            

            fig2.update_layout(
                title="Accuracy by target direction",
                showlegend=False,
                yaxis=dict(
                    range=[0, 1.3],
                    showgrid=False,
                    visible=False,
                ),
                xaxis=dict(
                    range=[-2, 2],
                    showgrid=False,
                    visible=False
                ),
                height=500
            )

        else:
            def direction(row):
                if row['direction_id'] <= 5:
                    return 0
                elif row['direction_id'] >= 9:
                    return 2
                else:
                    return 1
                
            df['direction'] = df.apply(direction, axis=1)
            df_directions = df.groupby(['direction']).agg({'result': 'mean'}).reset_index()

            x0 = -1.4
            for i in range(0, 3):
                x0 += 0.2

                fig2.add_trace(go.Scatter(
                            x=[0, i-1],
                            y=[0, 0.9],
                            mode="lines+markers",
                            marker=dict(
                                symbol="arrow",
                                size=15,
                                angleref="previous",
                                color='orange'
                                )
                            )
                )

                direction_result = round(df_directions["result"][i] * 100, 1)

                fig2.add_annotation(x=i+x0,
                                    y=1.05,
                                    text=f'{direction_result}%',
                                    showarrow=False,
                                    font=dict(size=20)
                                )

            fig2.update_layout(
                title="Accuracy by target direction",
                showlegend=False,
                yaxis=dict(
                    range=[0, 1.3],
                    showgrid=False,
                    visible=False,
                ),
                xaxis=dict(
                    range=[-2, 2],
                    showgrid=False,
                    visible=False
                ),
                height=350
            )

        config2 = {'staticPlot': True}

        st.plotly_chart(fig2, config=config2)