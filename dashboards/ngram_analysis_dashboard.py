#!/usr/bin/env python3
"""
N-gram Analysis Dashboard

Interactive dashboard for comparing n-gram statistics across multiple model checkpoints.
Loads pre-computed analysis artifacts (no need to re-run analysis).

Usage:
    streamlit run dashboards/ngram_analysis_dashboard.py
"""

import streamlit as st
import numpy as np
import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from baseball_states.tokenizer import GameStateTokenizer
from baseball_states.utils import format_state_symbols, parse_state


# ============================================================================
# Data Loading
# ============================================================================

@st.cache_data
def load_analysis(analysis_dir):
    """Load analysis artifacts from directory."""
    analysis_dir = Path(analysis_dir)

    # Load metadata
    with open(analysis_dir / "analysis_metadata.json", 'r') as f:
        metadata = json.load(f)

    # Load arrays
    arrays = np.load(analysis_dir / "analysis_arrays.npz")

    return metadata, dict(arrays)


def discover_analyses(base_dir="."):
    """Discover all analysis directories."""
    base_path = Path(base_dir)
    analyses = {}

    # Look for analysis_output* directories
    for path in base_path.glob("analysis_output*"):
        if path.is_dir() and (path / "analysis_metadata.json").exists():
            # Extract name from directory
            name = path.name.replace("analysis_output", "").replace("_", " ").strip()
            if not name:
                name = "trained"
            analyses[name] = str(path)

    return analyses


# ============================================================================
# Token Label Helpers
# ============================================================================

@st.cache_data
def get_token_labels():
    """Get human-readable labels for all tokens."""
    tokenizer = GameStateTokenizer()
    labels = []
    symbols = []

    for idx in range(tokenizer.vocab_size):
        token = tokenizer.id_to_token[idx]

        # Special tokens
        if token.startswith('<') and token.endswith('>'):
            labels.append(token)
            symbols.append(token)
        # State tokens
        elif token.startswith('OUT'):
            parsed = parse_state(token)
            if parsed:
                outs, bases = parsed
                label = f"O{outs}B{bases}"
                symbol = format_state_symbols(outs, bases)
                labels.append(label)
                symbols.append(symbol)
        # Outcome tokens
        else:
            labels.append(token)
            symbols.append(token)

    return labels, symbols


# ============================================================================
# Heatmap Formatting Helper
# ============================================================================

def format_bigram_heatmap(fig, vocab_size, use_latex=True):
    """
    Format a vocab x vocab bigram heatmap with gridlines and labels.

    Args:
        fig: Plotly figure with heatmap
        vocab_size: Size of vocabulary
        use_latex: Use latex notation for axis labels
    """
    # Get vocab structure dynamically from constants
    from baseball_states.constants import Vocabulary
    n_special = len(Vocabulary.SPECIAL)
    n_states = len(Vocabulary.STATE)
    n_outcomes = len(Vocabulary.OUTCOME)

    # Vocab structure:
    # Special: 0 to (n_special-1)
    # States: n_special to (n_special + n_states - 1)
    # Outcomes: (n_special + n_states) to (vocab_size - 1)

    # Add faint grid lines
    shapes = []
    gridline_positions = []

    # Major gridline after special tokens
    if n_special > 0:
        gridline_positions.append(n_special - 0.5)

    # Major gridline after state tokens (if outcomes exist)
    if n_outcomes > 0 and n_states > 0:
        gridline_positions.append(n_special + n_states - 0.5)

    for pos in gridline_positions:
        # Vertical line
        shapes.append(dict(
            type='line',
            x0=pos, x1=pos,
            y0=-0.5, y1=vocab_size-0.5,
            line=dict(color='rgba(128,128,128,0.3)', width=2)
        ))
        # Horizontal line
        shapes.append(dict(
            type='line',
            x0=-0.5, x1=vocab_size-0.5,
            y0=pos, y1=pos,
            line=dict(color='rgba(128,128,128,0.3)', width=2)
        ))

    # Add finer grid for state tokens (3x8 grid within states region)
    # States: 24 tokens organized as 3 outs x 8 bases
    # Group by outs: OUT0 (8 tokens), OUT1 (8 tokens), OUT2 (8 tokens)
    if n_states == 24:  # Only if we have the expected state structure
        state_start = n_special
        bases_per_out = 8
        # Gridlines between OUT0/OUT1 and OUT1/OUT2
        state_gridlines = [
            state_start + bases_per_out - 0.5,      # After OUT0
            state_start + 2 * bases_per_out - 0.5   # After OUT1
        ]
        state_end = state_start + n_states - 0.5

        for pos in state_gridlines:
            shapes.append(dict(
                type='line',
                x0=pos, x1=pos,
                y0=state_start - 0.5, y1=state_end,
                line=dict(color='rgba(100,100,100,0.2)', width=1)
            ))
            shapes.append(dict(
                type='line',
                x0=state_start - 0.5, x1=state_end,
                y0=pos, y1=pos,
                line=dict(color='rgba(100,100,100,0.2)', width=1)
            ))

    fig.update_layout(shapes=shapes)

    # Update axis labels
    if use_latex:
        fig.update_xaxes(title_text=r"$x_{i+1}$ (next token)")
        fig.update_yaxes(title_text=r"$x_i$ (current token)")

    return fig


# ============================================================================
# Plotting Functions - Category 1 (Overlayable)
# ============================================================================

def plot_unigram_comparison(analyses_data, log_scale=False):
    """Plot unigram distributions for multiple models."""
    fig = go.Figure()

    labels, symbols = get_token_labels()

    for name, (metadata, arrays) in analyses_data.items():
        vocab_size = metadata['vocab_size']
        x = list(range(vocab_size))

        # Add data distribution (only once)
        if name == list(analyses_data.keys())[0]:
            fig.add_trace(go.Bar(
                x=x,
                y=arrays['p_data_unigram'],
                name='Data',
                marker_color='steelblue',
                opacity=0.7,
                customdata=labels,
                hovertemplate='Token: %{customdata}<br>Prob: %{y:.4f}<extra></extra>'
            ))

        # Add model distribution
        fig.add_trace(go.Bar(
            x=x,
            y=arrays['p_model_unigram'],
            name=f'Model ({name})',
            opacity=0.7,
            customdata=labels,
            hovertemplate='Token: %{customdata}<br>Prob: %{y:.4f}<extra></extra>'
        ))

    fig.update_layout(
        title="Unigram Distribution Comparison",
        xaxis_title="Token ID",
        yaxis_title="Probability",
        yaxis_type="log" if log_scale else "linear",
        barmode='group',
        height=500,
        hovermode='x unified'
    )

    return fig


def plot_bigram_comparison(analyses_data, log_scale=False, top_n=100):
    """Plot bigram transition probabilities (flattened and sorted)."""
    fig = go.Figure()

    for name, (metadata, arrays) in analyses_data.items():
        # Flatten and sort by data probability
        p_data_flat = arrays['p_data_bigram_cond'].flatten()
        p_model_flat = arrays['p_model_bigram_cond'].flatten()

        sort_idx = np.argsort(p_data_flat)[::-1]
        p_data_sorted = p_data_flat[sort_idx]
        p_model_sorted = p_model_flat[sort_idx]

        # Add data distribution (only once)
        if name == list(analyses_data.keys())[0]:
            fig.add_trace(go.Scatter(
                x=list(range(top_n)),
                y=p_data_sorted[:top_n],
                mode='lines+markers',
                name='Data',
                line=dict(color='steelblue', width=2),
                marker=dict(size=4)
            ))

        # Add model distribution
        fig.add_trace(go.Scatter(
            x=list(range(top_n)),
            y=p_model_sorted[:top_n],
            mode='lines+markers',
            name=f'Model ({name})',
            line=dict(width=2),
            marker=dict(size=4)
        ))

    fig.update_layout(
        title=f"Top {top_n} Bigram Transitions (Sorted by Data Frequency)",
        xaxis_title="Transition Rank",
        yaxis_title=r"$P(x_{i+1} | x_i)$",
        yaxis_type="log" if log_scale else "linear",
        height=500,
        hovermode='x unified'
    )

    return fig


def plot_bigram_heatmap_2d(metadata, arrays, log_scale=False):
    """Plot 2D heatmap of bigram conditional probabilities."""
    vocab_size = metadata['vocab_size']
    labels, symbols = get_token_labels()

    # Create subplots for data, model, and difference
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Data', 'Model', 'Absolute Difference'),
        horizontal_spacing=0.08
    )

    p_data = arrays['p_data_bigram_cond']
    p_model = arrays['p_model_bigram_cond']

    # Data heatmap
    z_data = np.log10(p_data + 1e-10) if log_scale else p_data
    fig.add_trace(
        go.Heatmap(
            z=z_data,
            colorscale='Viridis',
            showscale=False,
            hovertemplate='From: %{y}<br>To: %{x}<br>P: %{z:.4f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Model heatmap
    z_model = np.log10(p_model + 1e-10) if log_scale else p_model
    fig.add_trace(
        go.Heatmap(
            z=z_model,
            colorscale='Viridis',
            showscale=False,
            hovertemplate='From: %{y}<br>To: %{x}<br>P: %{z:.4f}<extra></extra>'
        ),
        row=1, col=2
    )

    # Difference heatmap
    diff = np.abs(p_data - p_model)
    z_diff = np.log10(diff + 1e-10) if log_scale else diff
    fig.add_trace(
        go.Heatmap(
            z=z_diff,
            colorscale='Hot',
            showscale=True,
            colorbar=dict(title="log10(|Δ|)" if log_scale else "|Δ|"),
            hovertemplate='From: %{y}<br>To: %{x}<br>|Diff|: %{z:.4f}<extra></extra>'
        ),
        row=1, col=3
    )

    # Format all subplots
    for i in range(1, 4):
        fig = format_bigram_heatmap(fig, vocab_size, use_latex=True)

    fig.update_layout(
        title_text=f"Bigram Transition Probabilities: P(x_{{i+1}} | x_i)",
        height=600
    )

    return fig


def plot_divergence_comparison(analyses_data):
    """Bar chart comparing divergence metrics across models."""
    metrics_names = ['JS (unigram)', 'JS (bigram)', 'JS (trigram)', 'KL (trigram)']

    fig = go.Figure()

    for name, (metadata, arrays) in analyses_data.items():
        metrics = metadata['metrics']
        values = [
            metrics['js_unigram'],
            metrics['js_bigram'],
            metrics['js_trigram'],
            metrics['weighted_kl_trigram']
        ]

        fig.add_trace(go.Bar(
            x=metrics_names,
            y=values,
            name=name,
            text=[f"{v:.4f}" for v in values],
            textposition='auto',
        ))

    fig.update_layout(
        title="Divergence Metrics Comparison",
        xaxis_title="Metric",
        yaxis_title="Divergence",
        barmode='group',
        height=500,
        hovermode='x unified'
    )

    return fig


def plot_entropy_comparison(analyses_data):
    """Plot conditional entropy vs context length."""
    fig = go.Figure()

    for name, (metadata, arrays) in analyses_data.items():
        entropies = metadata['entropies']
        x = list(range(1, len(entropies) + 1))

        # Only plot data once (it's the same for all models)
        if name == list(analyses_data.keys())[0]:
            fig.add_trace(go.Scatter(
                x=x,
                y=entropies,
                mode='lines+markers',
                name='Data',
                line=dict(color='steelblue', width=3),
                marker=dict(size=8)
            ))
            break

    fig.update_layout(
        title="Conditional Entropy vs Context Length (Data)",
        xaxis_title="Context Length (n)",
        yaxis_title=r"$H[X_n | X_1...X_{n-1}]$ (bits)",
        height=500,
        hovermode='x unified'
    )

    return fig


# ============================================================================
# Plotting Functions - Category 2 (Heatmaps)
# ============================================================================

def plot_trigram_heatmaps(metadata, arrays, log_scale=False):
    """Side-by-side heatmaps for trigrams."""
    p_data_flat = arrays['p_data_flat']
    p_model_flat = arrays['p_model_flat']

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Data', 'Model', 'Absolute Difference'),
        horizontal_spacing=0.1
    )

    # Data
    z_data = np.log10(p_data_flat + 1e-10) if log_scale else p_data_flat
    fig.add_trace(
        go.Heatmap(z=z_data, colorscale='Viridis', showscale=False),
        row=1, col=1
    )

    # Model
    z_model = np.log10(p_model_flat + 1e-10) if log_scale else p_model_flat
    fig.add_trace(
        go.Heatmap(z=z_model, colorscale='Viridis', showscale=False),
        row=1, col=2
    )

    # Difference
    diff = np.abs(p_data_flat - p_model_flat)
    z_diff = np.log10(diff + 1e-10) if log_scale else diff
    fig.add_trace(
        go.Heatmap(z=z_diff, colorscale='Hot', showscale=True),
        row=1, col=3
    )

    fig.update_xaxes(title_text=r"$x_{i+2}$ (next token)")
    fig.update_yaxes(title_text=r"$(x_i, x_{i+1})$ history (flattened)")

    fig.update_layout(
        title_text=f"Trigram Conditional Probabilities: P(x_{{i+2}} | x_i, x_{{i+1}})",
        height=600
    )

    return fig


def plot_kl_heatmap(metadata, arrays, log_scale=False):
    """Heatmap of per-history KL divergence."""
    kl_per_history = arrays['m3_per_history']
    vocab_size = metadata['vocab_size']

    z = np.log10(kl_per_history + 1e-10) if log_scale else kl_per_history

    fig = go.Figure(data=go.Heatmap(
        z=z,
        colorscale='RdYlBu_r',  # Red for high divergence, blue for low
        colorbar=dict(title="log10(KL)" if log_scale else "KL")
    ))

    fig = format_bigram_heatmap(fig, vocab_size, use_latex=True)

    fig.update_layout(
        title=f"Per-History KL Divergence: KL(P_data || P_model)",
        height=600
    )

    return fig


# ============================================================================
# Main Dashboard
# ============================================================================

def main():
    st.set_page_config(
        page_title="N-gram Analysis Dashboard",
        page_icon="📊",
        layout="wide"
    )

    st.title("N-gram Analysis Dashboard")
    st.markdown("""
    Interactive comparison of n-gram statistics across multiple model checkpoints.
    This dashboard loads pre-computed analysis artifacts, so no need to re-run analysis!
    """)

    # Sidebar: Analysis selection
    st.sidebar.header("Analysis Selection")

    # Discover available analyses
    available_analyses = discover_analyses()

    if not available_analyses:
        st.error("No analysis directories found! Run `scripts/run_postanalysis.py` first.")
        return

    # Let user select which analyses to compare
    selected_analyses = st.sidebar.multiselect(
        "Select analyses to compare:",
        options=list(available_analyses.keys()),
        default=list(available_analyses.keys())
    )

    if not selected_analyses:
        st.warning("Please select at least one analysis to visualize.")
        return

    # Load selected analyses
    analyses_data = {}
    for name in selected_analyses:
        try:
            metadata, arrays = load_analysis(available_analyses[name])
            analyses_data[name] = (metadata, arrays)
        except Exception as e:
            st.sidebar.error(f"Error loading {name}: {e}")

    if not analyses_data:
        st.error("Could not load any analyses.")
        return

    # Sidebar: Display options
    st.sidebar.header("Display Options")
    log_scale = st.sidebar.checkbox("Use logarithmic scale", value=False)
    overlay_mode = st.sidebar.checkbox("Overlay plots (Category 1)", value=True)

    # Main content
    tabs = st.tabs([
        "Divergence Metrics",
        "Unigrams",
        "Bigrams",
        "Trigram Heatmaps",
        "Per-History KL",
        "Entropy Analysis"
    ])

    # Tab 1: Divergence Metrics
    with tabs[0]:
        st.header("Divergence Metrics Comparison")
        fig = plot_divergence_comparison(analyses_data)
        st.plotly_chart(fig, use_container_width=True)

        # Show metrics table
        st.subheader("Metrics Table")
        metrics_table = []
        for name, (metadata, arrays) in analyses_data.items():
            metrics = metadata['metrics']
            metrics_table.append({
                "Model": name,
                "JS (unigram)": f"{metrics['js_unigram']:.6f}",
                "JS (bigram)": f"{metrics['js_bigram']:.6f}",
                "JS (trigram)": f"{metrics['js_trigram']:.6f}",
                "Weighted KL": f"{metrics['weighted_kl_trigram']:.6f}"
            })
        st.table(metrics_table)

    # Tab 2: Unigrams
    with tabs[1]:
        st.header("Unigram Distribution")
        if overlay_mode and len(analyses_data) > 1:
            fig = plot_unigram_comparison(analyses_data, log_scale)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Show individual plots
            cols = st.columns(min(2, len(analyses_data)))
            for idx, (name, (metadata, arrays)) in enumerate(analyses_data.items()):
                with cols[idx % 2]:
                    vocab_size = metadata['vocab_size']
                    x = list(range(vocab_size))

                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=x, y=arrays['p_data_unigram'], name='Data'))
                    fig.add_trace(go.Bar(x=x, y=arrays['p_model_unigram'], name='Model'))
                    fig.update_layout(
                        title=f"Unigram - {name}",
                        yaxis_type="log" if log_scale else "linear",
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Bigrams
    with tabs[2]:
        st.header("Bigram Transitions")

        st.subheader("1D View: Sorted by Frequency")
        top_n = st.slider("Number of top transitions to show", 10, 500, 100, 10)

        if overlay_mode and len(analyses_data) > 1:
            fig = plot_bigram_comparison(analyses_data, log_scale, top_n)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Show individual plots
            for name, (metadata, arrays) in analyses_data.items():
                st.subheader(f"Model: {name}")
                fig = plot_bigram_comparison({name: (metadata, arrays)}, log_scale, top_n)
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("2D Heatmap View")
        for name, (metadata, arrays) in analyses_data.items():
            st.markdown(f"**Model: {name}**")
            fig = plot_bigram_heatmap_2d(metadata, arrays, log_scale)
            st.plotly_chart(fig, use_container_width=True)

    # Tab 4: Trigram Heatmaps
    with tabs[3]:
        st.header("Trigram Conditional Probabilities")
        st.info("Heatmaps are shown individually (cannot be overlayed).")

        for name, (metadata, arrays) in analyses_data.items():
            st.subheader(f"Model: {name}")
            fig = plot_trigram_heatmaps(metadata, arrays, log_scale)
            st.plotly_chart(fig, use_container_width=True)

    # Tab 5: Per-History KL
    with tabs[4]:
        st.header("Per-History KL Divergence")
        st.info("KL divergence heatmaps are shown individually.")

        for name, (metadata, arrays) in analyses_data.items():
            st.subheader(f"Model: {name}")
            fig = plot_kl_heatmap(metadata, arrays, log_scale)
            st.plotly_chart(fig, use_container_width=True)

    # Tab 6: Entropy Analysis
    with tabs[5]:
        st.header("Conditional Entropy Analysis")
        st.info("Entropy is computed from data only (same across all models).")

        fig = plot_entropy_comparison(analyses_data)
        st.plotly_chart(fig, use_container_width=True)

        # Show entropy reduction
        st.subheader("Information Gain from Additional Context")
        name = list(analyses_data.keys())[0]
        metadata, arrays = analyses_data[name]
        entropies = metadata['entropies']
        reductions = [entropies[i] - entropies[i+1] for i in range(len(entropies)-1)]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(2, len(entropies) + 1)),
            y=reductions,
            name="H[n-1] - H[n]"
        ))
        fig.update_layout(
            title="Information Gain from Additional Context",
            xaxis_title="Context Length (n)",
            yaxis_title="Entropy Reduction (bits)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
