#!/Users/angerami/Desktop/Materials/baseball-states/venv/bin/python3
"""Streamlit dashboard for exploring plate appearance data."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datasets import Dataset
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="PA Explorer", layout="wide")

# Column name mapping
COLUMN_LABELS = {
    'at_bat_number': 'At Bat Number',
    'num_pitches': 'Number of Pitches',
    'outs': 'Outs',
    'bases': 'Bases',
    'batter': 'Batter ID',
    'pitcher': 'Pitcher ID',
    'inning': 'Inning',
}

COLUMN_DESCRIPTIONS = {
    'outs': 'Outs at start of PA (0, 1, or 2)',
    'bases': 'Bases state (0-7): 0=Empty, 1=1st, 2=2nd, 3=1st&2nd, 4=3rd, 5=1st&3rd, 6=2nd&3rd, 7=Loaded',
    'at_bat_number': 'Sequential at-bat number within the game',
    'num_pitches': 'Total pitches thrown during this plate appearance',
    'batter': 'Unique identifier for the batter',
    'pitcher': 'Unique identifier for the pitcher',
    'inning': 'Inning number when PA occurred',
}

@st.cache_data
def load_data():
    """Load PA dataset and rename columns."""
    data_path = Path("data/plate_appearances")
    dataset = Dataset.load_from_disk(data_path)
    df = dataset.to_pandas()

    # Rename columns
    df = df.rename(columns={
        'initial_outs': 'outs',
        'initial_bases': 'bases'
    })

    return df

df = load_data()

# Sidebar
st.sidebar.header("Dataset Summary")
st.sidebar.metric("Total Rows", f"{len(df):,}")
st.sidebar.metric("Total Columns", len(df.columns))
st.sidebar.metric("Avg Pitches/PA", f"{df['num_pitches'].mean():.2f}")

# Main
st.title("Plate Appearance Explorer")

# Section 1: Column Histogram
st.header("Column Distribution")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Remove game_pk from options
numeric_cols = [col for col in numeric_cols if col != 'game_pk']

# Create display options
display_options = [COLUMN_LABELS.get(col, col) for col in numeric_cols]
selected_display = st.selectbox("Select column", display_options)

# Map back to internal column name
selected_col = numeric_cols[display_options.index(selected_display)]

# Show description
if selected_col in COLUMN_DESCRIPTIONS:
    st.caption(COLUMN_DESCRIPTIONS[selected_col])

log_scale_hist = st.radio("Y-axis scale", ["Linear", "Logarithmic"], horizontal=True, key="hist_scale")

fig = px.histogram(df, x=selected_col, nbins=50, title=f"Distribution of {selected_display}")
if log_scale_hist == "Logarithmic":
    fig.update_yaxes(type="log")
st.plotly_chart(fig, use_container_width=True)

# Section 2: State Transition Matrix
st.header("State Transition Heatmap")

log_scale_heatmap = st.radio("Color scale", ["Linear", "Logarithmic"], horizontal=True, key="heatmap_scale")
show_large_heatmap = st.checkbox("Show large version (2400x2400)", value=False)

# Build transition matrix: current state -> next state
# Group by game to maintain temporal order
transitions = []

for game_id in df['game_pk'].unique():
    game_df = df[df['game_pk'] == game_id].sort_values('at_bat_number')

    for i in range(len(game_df) - 1):
        curr = game_df.iloc[i]
        next_pa = game_df.iloc[i + 1]

        curr_state = (curr['outs'], curr['bases'])
        next_state = (next_pa['outs'], next_pa['bases'])

        transitions.append({
            'from_outs': curr_state[0],
            'from_bases': curr_state[1],
            'to_outs': next_state[0],
            'to_bases': next_state[1]
        })

trans_df = pd.DataFrame(transitions)

# Create 24x24 matrix (3 outs × 8 bases = 24 states)
state_counts = np.zeros((24, 24))

for _, row in trans_df.iterrows():
    from_idx = int(row['from_outs']) * 8 + int(row['from_bases'])
    to_idx = int(row['to_outs']) * 8 + int(row['to_bases'])
    state_counts[from_idx, to_idx] += 1

# Normalize rows to get probabilities
row_sums = state_counts.sum(axis=1, keepdims=True)
state_probs = np.divide(state_counts, row_sums, where=row_sums > 0)

# Apply log scale if selected
z_data = np.log10(state_probs + 1e-10) if log_scale_heatmap == "Logarithmic" else state_probs

# Create labels
bases_labels = ['Empty', '1st', '2nd', '1st&2nd', '3rd', '1st&3rd', '2nd&3rd', 'Loaded']
state_labels = []
for outs in range(3):
    for bases in range(8):
        state_labels.append(f"{outs}_{bases_labels[bases]}")

# Set size based on checkbox
height = 2400 if show_large_heatmap else 800
width = 2400 if show_large_heatmap else 800
text_size = 10 if show_large_heatmap else 6

fig = go.Figure(data=go.Heatmap(
    z=z_data,
    x=state_labels,
    y=state_labels,
    colorscale='Viridis',
    text=np.round(state_probs, 3),
    texttemplate='%{text}',
    textfont={"size": text_size},
    hovertemplate='From: %{y}<br>To: %{x}<br>Probability: %{text}<extra></extra>'
))

colorbar_title = "log10(Probability)" if log_scale_heatmap == "Logarithmic" else "Probability"

fig.update_layout(
    title="State Transition Probabilities (State i → State i+1)",
    xaxis_title="Next State (Outs_Bases)",
    yaxis_title="Current State (Outs_Bases)",
    height=height,
    width=width,
    xaxis={'tickangle': -45, 'tickfont': {'size': 8 if not show_large_heatmap else 12}},
    yaxis={'tickfont': {'size': 8 if not show_large_heatmap else 12}},
    coloraxis_colorbar={'title': colorbar_title}
)

st.plotly_chart(fig, use_container_width=True)
