import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
import time

from gpcode import get_posterior_samples

def get_df(i):
    ys = y_plots[:, :, i].reshape(-1)
    xs = np.tile(x_plot, N_samples)
    error = np.tile(std, N_samples)
    df = pd.DataFrame([xs, ys], index=['x', 'y']).T
    df['sample'] = np.concatenate(
        [N_plot*[j] for j in range(N_samples)]
    )
    df['ymin'] = np.tile(mu, N_samples) - 2*error
    df['ymax'] = np.tile(mu, N_samples) + 2*error
    return df

N_samples, N_plot, T_plot = 3, 30, 300
x, y = np.array([(0, -0.3), (0.5, 0.2), (2.2, 0.5), (3, -0.4), (3.5, -0.6)]).T

st.title("Samples from a Gaussian Process Posterior")
progress_bar = st.progress(0)
frame_text = st.empty()
image = st.empty()

x_plot, y_plots, mu, std = get_posterior_samples(x, y, N_samples, N_plot, T_plot)

xlim = (x_plot.min(), x_plot.max())
ylim = (y_plots.min(), y_plots.max())

for i in range(1,T_plot):
    progress_bar.progress(i/T_plot)
    frame_text.text(f'Frame {i}/{T_plot}')
    df = get_df(i)
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
