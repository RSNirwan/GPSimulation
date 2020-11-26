import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
import time

from gpcode import get_posterior_samples



st.title("Samples from a Gaussian Process Posterior")
progress_bar = st.progress(0)
frame_text = st.empty()
image = st.empty()

st.sidebar.title("Parameters")
N_samples = st.sidebar.slider("Number of samples", 1, 10, 3, 1)
kernel_l = st.sidebar.slider("Kernel lengthscale", 0.3, 1.0, 0.5, 0.1)
kernel_std = st.sidebar.slider("Kernel std", 0.1, 2.0, 1., 0.1)
kernel_name = st.sidebar.selectbox("Kernel function", 
        options=["Gaussian", "Matern32", "Exp"],
        index=0)
N_plot = st.sidebar.slider("Number of evaluation in x", 30, 80, 50, 5)
data_points_option =[(0.0, -0.3),
                (0.5, 0.2), 
                (2.2, 0.5),
                (3.0, -0.4), 
                (3.5, -0.6)]
data_points = st.sidebar.multiselect("Observed data points", 
        options=data_points_option,
        default=data_points_option)



T_plot = 150 #time discretization
x, y = np.array(data_points).T
x_plot, y_plots, mu, std = get_posterior_samples(x, y, N_samples, 
                        N_plot, T_plot, 
                        kernel_l, kernel_std, kernel_name)

xlim = (x_plot.min(), x_plot.max())
ylim = (y_plots.min(), y_plots.max())


def get_df(t):
    ys = y_plots[:, :, t].reshape(-1)
    xs = np.tile(x_plot, N_samples)
    error = np.tile(std, N_samples)
    df = pd.DataFrame([xs, ys], index=['x', 'y']).T
    df['sample'] = np.concatenate(
        [N_plot*[j] for j in range(N_samples)]
    )
    df['ymin'] = np.tile(mu, N_samples) - 2*error
    df['ymax'] = np.tile(mu, N_samples) + 2*error
    return df


for t in range(1,T_plot):
    progress_bar.progress((t+1.)/T_plot)
    frame_text.text(f'Frame {t+1}/{T_plot}')
    df = get_df(t)
    line = alt.Chart(df).mark_line().encode(
        alt.X('x', scale=alt.Scale(domain=xlim)),
        alt.Y('y', scale=alt.Scale(domain=ylim)),
        color='sample:N'
    )
    band = alt.Chart(df).mark_area(opacity=0.2).encode(
        x='x:Q',
        y='ymin:Q',
        y2='ymax:Q'
    )
    image.altair_chart(line + band)

    time.sleep(0.1)

st.button("Re-run")
