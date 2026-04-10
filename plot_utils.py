"""
plot_utils.py — Shared Plotly utilities for all pipeline scripts.

Every plot is created with Plotly and saved as:
  - .png  (for reports, static viewing)
  - .json (for interactive dashboard rendering)

The dashboard loads .json files with st.plotly_chart(fig) for full
interactivity (zoom, hover, pan, download).

Usage in any script:
    from plot_utils import save_fig
    import plotly.graph_objects as go

    fig = go.Figure(...)
    save_fig(fig, output_dir / "my_plot")  # saves .png + .json
"""

import json
from pathlib import Path

try:
    import plotly.graph_objects as go
    import plotly.io as pio

    # Default template — clean, works in light and dark
    pio.templates.default = "plotly_white"

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("⚠️  plotly not installed. Run: pip install plotly kaleido")


# Consistent color palette across all plots
COLORS = {
    'observed':    "#272cba",
    'xgb':         '#2E86C1',
    'lgb':         '#1ABC9C',
    'ridge':       "#0DC8D6",
    'ensemble':    '#2E86C1',
    'lstm':        '#E74C3C',
    'hybrid':      '#27AE60',
    'ssp245':      '#8E44AD',
    'ssp585':      '#C0392B',
    'monsoon':     '#EF9F27',
    'dry':         '#3B8BD4',
    'baseflow':    '#3B8BD4',
    'directrunoff':'#E8593C',
    'primary':     '#2E86C1',
    'secondary':   '#E74C3C',
    'accent':      '#27AE60',
    'fill':        'rgba(46,134,193,0.15)',
}


def save_fig(fig, path_stem, width=1200, height=500):
    """
    Save a Plotly figure as both .png and .json.

    Args:
        fig: plotly.graph_objects.Figure
        path_stem: Path without extension (e.g. Path("outputs/my_plot"))
        width: image width in pixels
        height: image height in pixels
    """
    if not HAS_PLOTLY:
        return

    path_stem = Path(path_stem)

    # Update layout defaults
    fig.update_layout(
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=60, r=30, t=50, b=50),
        legend=dict(font=dict(size=11)),
        hovermode='x unified',
    )

    # Save JSON (for interactive dashboard)
    json_path = path_stem.with_suffix('.json')
    fig.write_json(str(json_path))

    # Save PNG (for reports)
    png_path = path_stem.with_suffix('.png')
    try:
        fig.write_image(str(png_path), width=width, height=height, scale=2)
    except Exception:
        # kaleido may not be installed — skip PNG, JSON is enough
        pass

    print(f"   📸 Saved: {path_stem.name} (.json + .png)")