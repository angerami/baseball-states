#!/Users/angerami/Desktop/Materials/baseball-states/venv/bin/python3
"""Streamlit dashboard for n-gram analysis of game state sequences."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datasets import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from nltk import ngrams
import sys
sys.path.append(str(Path(__file__).parent.parent))
from sequencer import SPECIAL_TOKENS, get_all_state_tokens

st.set_page_config(page_title="N-gram Explorer", layout="wide")

@st.cache_data
def load_sequences(dataset_type: str):
    """Load sequence dataset."""
    data_path = Path(f"data/sequences_{dataset_type}")
    dataset = Dataset.load_from_disk(data_path)
    df = dataset.to_pandas()
    # Parse sequences back to lists
    sequences = df['sequence'].tolist()
    return sequences

@st.cache_data
def compute_ngrams(_sequences_hash, sequences_list, n, skip_special):
    """Compute n-grams with caching."""
    all_ngrams = []
    for seq in sequences_list:
        if skip_special:
            special_token_values = set(SPECIAL_TOKENS.values())
            seq = [token for token in seq if token not in special_token_values]

        seq_ngrams = list(ngrams(seq, n))
        all_ngrams.extend(seq_ngrams)

    return Counter(all_ngrams)

@st.cache_data
def compute_transition_matrix(_sequences_hash, sequences_list):
    """Compute transition matrix with caching."""
    # Build ordered vocabulary
    ordered_vocab = []
    all_tokens = set(token for seq in sequences_list for token in seq)

    if SPECIAL_TOKENS['start_game'] in all_tokens:
        ordered_vocab.append(SPECIAL_TOKENS['start_game'])

    ordered_vocab.append(SPECIAL_TOKENS['start_inning'])

    state_tokens = get_all_state_tokens()
    for outs in range(3):
        for bases in range(8):
            token = state_tokens[outs * 8 + bases]
            ordered_vocab.append(token)

    ordered_vocab.append(SPECIAL_TOKENS['end_inning'])

    if SPECIAL_TOKENS['end_game'] in all_tokens:
        ordered_vocab.append(SPECIAL_TOKENS['end_game'])

    vocab_size = len(ordered_vocab)
    token_to_idx = {token: idx for idx, token in enumerate(ordered_vocab)}

    transition_counts = np.zeros((vocab_size, vocab_size))

    for seq in sequences_list:
        for i in range(len(seq) - 1):
            curr_token = seq[i]
            next_token = seq[i + 1]

            if curr_token in token_to_idx and next_token in token_to_idx:
                curr_idx = token_to_idx[curr_token]
                next_idx = token_to_idx[next_token]
                transition_counts[curr_idx, next_idx] += 1

    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_probs = np.divide(
        transition_counts, 
        row_sums, 
        out=np.zeros_like(transition_counts), 
        where=row_sums > 0
    )
    return transition_probs, ordered_vocab

# Sidebar
st.sidebar.header("Dataset Selection")
dataset_type = st.sidebar.radio("Sequence Level", ["inning", "game"])

sequences = load_sequences(dataset_type)
sequences_hash = hash(dataset_type)  # Simple hash based on dataset type

st.sidebar.metric("Total Sequences", f"{len(sequences):,}")
st.sidebar.metric("Avg Length", f"{np.mean([len(s) for s in sequences]):.1f}")

# Main
st.title("N-gram Sequence Explorer")

# Section 1: N-gram Distribution
st.header("N-gram Distribution")

col1, col2, col3 = st.columns(3)
with col1:
    n = st.selectbox("N-gram size", [1, 2, 3, 4, 5], index=1)
with col2:
    top_n = st.number_input("Top N to show", min_value=5, max_value=100, value=20, step=5)
with col3:
    skip_special = st.checkbox("Skip special tokens", value=False)

# Compute n-grams using cached function
ngram_counts = compute_ngrams(sequences_hash, sequences, n, skip_special)
most_common = ngram_counts.most_common(int(top_n))

# Create DataFrame for plotting
ngram_labels = [' → '.join(ng) for ng, _ in most_common]
counts = [count for _, count in most_common]

df_ngrams = pd.DataFrame({
    'n-gram': ngram_labels,
    'count': counts
})

fig = px.bar(
    df_ngrams,
    x='count',
    y='n-gram',
    orientation='h',
    title=f"Top {len(most_common)} {n}-grams",
    labels={'count': 'Frequency', 'n-gram': f'{n}-gram'},
    height=max(400, len(most_common) * 20)
)
fig.update_yaxes(categoryorder='total ascending')
st.plotly_chart(fig, use_container_width=True)

# Section 2: Transition Heatmap
st.header("Token Transition Heatmap")

log_scale = st.radio("Color scale", ["Linear", "Logarithmic"], horizontal=True)

# Use cached transition matrix computation
transition_probs, ordered_vocab = compute_transition_matrix(sequences_hash, sequences)

# Apply log scale if selected
if log_scale == "Logarithmic":
    # Find minimum non-zero probability for zero replacement
    non_zero_probs = transition_probs[transition_probs > 0]
    if len(non_zero_probs) > 0:
        min_nonzero = non_zero_probs.min()
        zero_replacement = 0.1 * min_nonzero
    else:
        zero_replacement = 1e-10  # fallback
    
    safe_probs = np.where(transition_probs > 0, transition_probs, zero_replacement)
    z_data = np.log10(safe_probs)
else:
    z_data = transition_probs

fig = go.Figure(data=go.Heatmap(
    z=z_data,
    x=ordered_vocab,
    y=ordered_vocab,
    colorscale='Spectral',
    hovertemplate='From: %{y}<br>To: %{x}<br>Probability: %{customdata:.4f}<extra></extra>',
    customdata=transition_probs
))

colorbar_title = "log10(Probability)" if log_scale == "Logarithmic" else "Probability"

fig.update_layout(
    title="Token → Next Token Transition Probabilities",
    xaxis_title="Next Token",
    yaxis_title="Current Token",
    height=1000,
    width=1000,
    xaxis={'tickangle': -90, 'tickfont': {'size': 7}},
    yaxis={'tickfont': {'size': 7}},
    coloraxis_colorbar={'title': colorbar_title}
)

st.plotly_chart(fig, use_container_width=True)

# Stats summary
st.subheader("Statistics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total tokens", f"{sum(len(s) for s in sequences):,}")
with col2:
    st.metric(f"Unique {n}-grams", f"{len(ngram_counts):,}")
with col3:
    st.metric("Vocabulary size", len(ordered_vocab))
