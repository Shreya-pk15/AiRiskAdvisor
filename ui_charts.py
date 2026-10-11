"""
AI Project Intelligence & Risk Advisor — Shared Plotly Visualizations Helper
Styled consistently with transparent backgrounds, Inter typography, and semantic palette.
"""

from datetime import datetime, timedelta
import random
from typing import Any, Dict, List, Optional
import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
    import plotly.graph_objects as go


# ── Semantic Palette Constants ──
COLOR_CORAL = "#EF6B73"
COLOR_AMBER = "#F5B547"
COLOR_MINT = "#34C38F"
COLOR_VIOLET = "#8B7CF6"
COLOR_GREY = "#A3A9BF"
COLOR_GRID = "#ECE9FB"
COLOR_TEXT = "#1F2340"
COLOR_MUTED = "#7A7F9A"


def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    """Apply unified modern theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, sans-serif", color=COLOR_TEXT, size=12),
        margin=dict(l=20, r=20, t=30, b=20),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#ECE9FB",
            font=dict(family="Inter, sans-serif", size=12, color=COLOR_TEXT)
        )
    )
    return fig


def create_health_gauge(score: float, classification: str = "Healthy") -> go.Figure:
    """Create a modern semicircular gauge for project health score (0-100)."""
    score_val = float(score or 0)
    
    # Semantic color determination
    if score_val >= 80:
        bar_color = COLOR_MINT
    elif score_val >= 60:
        bar_color = COLOR_AMBER
    else:
        bar_color = COLOR_CORAL

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score_val,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "/100", 'font': {'size': 36, 'color': COLOR_TEXT, 'family': 'Inter'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': COLOR_GRID, 'tickfont': {'color': COLOR_MUTED, 'size': 10}},
            'bar': {'color': bar_color, 'thickness': 0.75},
            'bgcolor': "#EDEAFE",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 60], 'color': "rgba(239, 107, 115, 0.08)"},
                {'range': [60, 80], 'color': "rgba(245, 181, 71, 0.08)"},
                {'range': [80, 100], 'color': "rgba(52, 195, 143, 0.08)"}
            ],
            'threshold': {
                'line': {'color': bar_color, 'width': 4},
                'thickness': 0.8,
                'value': score_val
            }
        }
    ))

    fig.update_layout(height=200, margin=dict(l=15, r=15, t=10, b=10))
    return apply_plotly_theme(fig)


def create_risk_priority_matrix(risks: List[Dict[str, Any]], filter_level: str = "All") -> go.Figure:
    """
    Create Risk Priority Matrix (Probability vs Impact).
    X = Probability (1-5), Y = Impact (1-5) with soft hazard zones.
    """
    fig = go.Figure()

    # 1. Background Hazard Zones (Soft Coral top-right, Soft Mint bottom-left)
    fig.add_shape(
        type="rect", x0=3.0, y0=3.0, x1=5.2, y1=5.2,
        fillcolor="rgba(239, 107, 115, 0.12)", line=dict(width=0),
        layer="below"
    )
    fig.add_annotation(
        x=4.1, y=5.0, text="⚠️ Critical Hazard Zone",
        showarrow=False, font=dict(color=COLOR_CORAL, size=11, family="Inter"),
        opacity=0.85
    )

    fig.add_shape(
        type="rect", x0=0.8, y0=0.8, x1=3.0, y1=3.0,
        fillcolor="rgba(52, 195, 143, 0.12)", line=dict(width=0),
        layer="below"
    )
    fig.add_annotation(
        x=1.9, y=1.1, text="✓ Low Priority Zone",
        showarrow=False, font=dict(color=COLOR_MINT, size=11, family="Inter"),
        opacity=0.85
    )

    # Dashed threshold mid-lines
    fig.add_shape(type="line", x0=3.0, y0=0.8, x1=3.0, y1=5.2, line=dict(color=COLOR_GRID, width=1.5, dash="dash"))
    fig.add_shape(type="line", x0=0.8, y0=3.0, x1=5.2, y1=3.0, line=dict(color=COLOR_GRID, width=1.5, dash="dash"))

    if not risks:
        fig.add_annotation(text="No risks recorded to plot", showarrow=False, font=dict(color=COLOR_MUTED, size=13))
        fig.update_xaxes(range=[0.8, 5.2], visible=False)
        fig.update_yaxes(range=[0.8, 5.2], visible=False)
        return apply_plotly_theme(fig)

    # Process risks
    sev_map = {"HIGH": 4.5, "MEDIUM": 3.0, "LOW": 1.5}
    prob_map = {"HIGH": 4.2, "MEDIUM": 2.8, "LOW": 1.6}

    # Group risks by severity for legend/color separation
    categories = [
        ("High / Critical", COLOR_CORAL, "HIGH"),
        ("Medium", COLOR_AMBER, "MEDIUM"),
        ("Low", COLOR_MINT, "LOW")
    ]

    for label, color, sev_code in categories:
        xs, ys, texts, hovertexts, ids = [], [], [], [], []
        for idx, r in enumerate(risks, 1):
            r_sev = str(r.get("severity", "MEDIUM")).upper()
            if sev_code not in r_sev and not (sev_code == "HIGH" and "CRIT" in r_sev):
                if sev_code == "MEDIUM" and "MED" not in r_sev: continue
                if sev_code == "LOW" and "LOW" not in r_sev: continue

            if filter_level != "All" and filter_level.upper() not in r_sev:
                continue

            r_id = f"RSK-{idx:02d}"
            # Derive probability and impact if not explicitly provided
            base_p = float(r.get("probability", prob_map.get(r_sev, 3.0)))
            base_i = float(r.get("impact", sev_map.get(r_sev, 3.0)))
            
            # Subtle jitter so overlapping dots separate
            jitter_x = (random.random() - 0.5) * 0.35
            jitter_y = (random.random() - 0.5) * 0.35
            
            xs.append(base_p + jitter_x)
            ys.append(base_i + jitter_y)
            texts.append(r_id)
            desc = r.get("description", r.get("title", "Risk Item"))
            action = r.get("recommended_action", "Monitor")
            hovertexts.append(f"<b>{r_id}: {r.get('title', 'Risk')}</b><br>Severity: {r_sev}<br>{desc[:100]}...<br><i>Action: {action[:80]}</i>")
            ids.append(r_id)

        if xs:
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                text=texts,
                textposition="top center",
                textfont=dict(family="Inter", size=10, color=COLOR_TEXT, weight="bold"),
                hoverinfo="text",
                hovertext=hovertexts,
                name=label,
                marker=dict(
                    size=16,
                    color=color,
                    line=dict(color="#FFFFFF", width=2.5),
                    opacity=0.92
                )
            ))

    fig.update_layout(
        xaxis=dict(
            title=dict(text="Probability →", font=dict(size=12, color=COLOR_MUTED)),
            range=[0.8, 5.2],
            showgrid=True,
            gridcolor=COLOR_GRID,
            zeroline=False,
            tickvals=[1, 2, 3, 4, 5],
            ticktext=["Very Low", "Low", "Medium", "High", "Critical"]
        ),
        yaxis=dict(
            title=dict(text="Impact →", font=dict(size=12, color=COLOR_MUTED)),
            range=[0.8, 5.2],
            showgrid=True,
            gridcolor=COLOR_GRID,
            zeroline=False,
            tickvals=[1, 2, 3, 4, 5],
            ticktext=["Negligible", "Minor", "Moderate", "Major", "Severe"]
        ),
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return apply_plotly_theme(fig)


def create_blocker_urgency_matrix(blockers: List[Dict[str, Any]], filter_cat: str = "All") -> go.Figure:
    """
    Create Blocker Urgency Matrix (Days Open vs Impact).
    X = Days Open (or urgency score), Y = Impact (1-5) with 'Escalate now' zone.
    """
    fig = go.Figure()

    # Escalate Now Zone in top-right
    fig.add_shape(
        type="rect", x0=7.0, y0=3.0, x1=15.5, y1=5.2,
        fillcolor="rgba(239, 107, 115, 0.12)", line=dict(width=0),
        layer="below"
    )
    fig.add_annotation(
        x=11.2, y=5.0, text="🚨 Escalate Now",
        showarrow=False, font=dict(color=COLOR_CORAL, size=11, family="Inter"),
        opacity=0.85
    )

    # Dashed divider
    fig.add_shape(type="line", x0=7.0, y0=0.8, x1=7.0, y1=5.2, line=dict(color=COLOR_GRID, width=1.5, dash="dash"))
    fig.add_shape(type="line", x0=0.5, y0=3.0, x1=15.5, y1=3.0, line=dict(color=COLOR_GRID, width=1.5, dash="dash"))

    if not blockers:
        fig.add_annotation(text="No active blockers recorded", showarrow=False, font=dict(color=COLOR_MUTED, size=13))
        fig.update_xaxes(range=[0.5, 15.5], visible=False)
        fig.update_yaxes(range=[0.8, 5.2], visible=False)
        return apply_plotly_theme(fig)

    cat_colors = {
        "Technical": COLOR_CORAL,
        "Dependency": COLOR_AMBER,
        "Approval": COLOR_VIOLET,
        "Resource": "#5B8DEF",
        "Testing": COLOR_MINT
    }

    by_cat: Dict[str, Dict[str, list]] = {}
    for idx, b in enumerate(blockers, 1):
        cat = b.get("category", "Technical")
        if filter_cat != "All" and cat.lower() != filter_cat.lower():
            continue

        if cat not in by_cat:
            by_cat[cat] = {"x": [], "y": [], "text": [], "hover": []}

        b_id = f"BLK-{idx:02d}"
        days = float(b.get("days_open", 2.0 + (idx * 2.2) % 12))
        impact = float(b.get("impact_score", 3.0 + (idx % 3) * 0.8))
        jitter_x = (random.random() - 0.5) * 0.4
        jitter_y = (random.random() - 0.5) * 0.3

        by_cat[cat]["x"].append(days + jitter_x)
        by_cat[cat]["y"].append(impact + jitter_y)
        by_cat[cat]["text"].append(b_id)
        owner = b.get("owner", "Unassigned")
        title = b.get("title", "Blocker")
        by_cat[cat]["hover"].append(f"<b>{b_id}: {title}</b><br>Category: {cat}<br>Owner: {owner}<br>Days Active: ~{int(days)}d")

    for cat, data in by_cat.items():
        fig.add_trace(go.Scatter(
            x=data["x"],
            y=data["y"],
            mode="markers+text",
            text=data["text"],
            textposition="top center",
            textfont=dict(family="Inter", size=10, color=COLOR_TEXT, weight="bold"),
            hoverinfo="text",
            hovertext=data["hover"],
            name=cat,
            marker=dict(
                size=16,
                color=cat_colors.get(cat, COLOR_VIOLET),
                line=dict(color="#FFFFFF", width=2.5),
                opacity=0.92
            )
        ))

    fig.update_layout(
        xaxis=dict(
            title=dict(text="Days Open / Aging →", font=dict(size=12, color=COLOR_MUTED)),
            range=[0.5, 15.5],
            showgrid=True,
            gridcolor=COLOR_GRID,
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text="Impact Severity →", font=dict(size=12, color=COLOR_MUTED)),
            range=[0.8, 5.2],
            showgrid=True,
            gridcolor=COLOR_GRID,
            zeroline=False,
            tickvals=[1, 2, 3, 4, 5],
            ticktext=["Low", "Minor", "Medium", "High", "Critical"]
        ),
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return apply_plotly_theme(fig)


def create_scope_timeline(deliverables_or_milestones: List[Dict[str, Any]]) -> go.Figure:
    """
    Create Scope Timeline Chart.
    Each deliverable/milestone has a start dot, end dot, line between them, colored by status, with Today line.
    """
    fig = go.Figure()

    if not deliverables_or_milestones:
        fig.add_annotation(text="No scope timeline items to display", showarrow=False, font=dict(color=COLOR_MUTED, size=13))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return apply_plotly_theme(fig)

    base_date = datetime.now()
    items = []

    def _try_parse_date(value: Any) -> Optional[datetime]:
        """Try to parse a date string from various formats; return None on failure."""
        if not value:
            return None
        s = str(value).strip()
        if not s or s.lower() in ("none", "not specified", "n/a", "tbd", ""):
            return None
        # Try common formats
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y",
                    "%d %B %Y", "%d %b %Y", "%Y/%m/%d", "%b %Y", "%B %Y"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        # Try to extract a 4-digit year + month as fallback (e.g. "Sep 2025")
        import re as _re
        m = _re.search(r"(\b\w+ \d{4}\b|\d{4}-\d{2})", s)
        if m:
            for fmt in ("%B %Y", "%b %Y", "%Y-%m"):
                try:
                    return datetime.strptime(m.group(1), fmt)
                except ValueError:
                    continue
        return None

    for idx, item in enumerate(deliverables_or_milestones):
        name = item.get("name") or item.get("Deliverable") or item.get("Milestone") or f"Item {idx+1}"
        status = str(item.get("status", item.get("Status", "In Progress"))).title()
        owner = item.get("owner", item.get("assignee", "Team"))

        # Try to use document dates first; fall back to sequential offsets only when absent
        start_dt = (_try_parse_date(item.get("start_date"))
                    or _try_parse_date(item.get("target_date"))
                    or _try_parse_date(item.get("value"))
                    or base_date + timedelta(days=idx * 14))
        end_dt = (_try_parse_date(item.get("end_date"))
                  or _try_parse_date(item.get("deadline"))
                  or _try_parse_date(item.get("target_date"))
                  or start_dt + timedelta(days=14))

        # Ensure end >= start
        if end_dt < start_dt:
            end_dt = start_dt + timedelta(days=7)

        items.append({
            "name": name,
            "owner": owner,
            "status": status,
            "start": start_dt,
            "end": end_dt,
            "y": idx
        })

    # Sort in reverse so first item is at the top
    items = items[::-1]

    # Status color mapping
    def get_status_color(st_str: str) -> str:
        s = st_str.lower()
        if "done" in s or "complet" in s: return COLOR_MINT
        if "risk" in s or "delay" in s: return COLOR_CORAL
        if "prog" in s or "ident" in s: return COLOR_VIOLET
        return COLOR_AMBER

    y_names = []
    for it in items:
        color = get_status_color(it["status"])
        y_label = f"<b>{it['name']}</b><br><span style='font-size:10px;color:#7A7F9A;'>{it['owner']}</span>"
        y_names.append(y_label)

        # Connector Line
        fig.add_trace(go.Scatter(
            x=[it["start"], it["end"]],
            y=[y_label, y_label],
            mode="lines",
            line=dict(color=color, width=4),
            hoverinfo="skip",
            showlegend=False
        ))

        # Start Dot (small)
        fig.add_trace(go.Scatter(
            x=[it["start"]],
            y=[y_label],
            mode="markers",
            marker=dict(size=8, color=color, line=dict(color="#FFFFFF", width=1.5)),
            hoverinfo="text",
            hovertext=f"{it['name']} (Start: {it['start'].strftime('%b %d')})",
            showlegend=False
        ))

        # End Dot (large)
        fig.add_trace(go.Scatter(
            x=[it["end"]],
            y=[y_label],
            mode="markers",
            marker=dict(size=14, color=color, line=dict(color="#FFFFFF", width=2)),
            hoverinfo="text",
            hovertext=f"<b>{it['name']}</b><br>Status: {it['status']}<br>Due: {it['end'].strftime('%b %d')}<br>Owner: {it['owner']}",
            showlegend=False
        ))

    # "Today" dashed reference line
    fig.add_shape(
        type="line",
        x0=base_date, y0=-0.5,
        x1=base_date, y1=len(items) - 0.5,
        line=dict(color=COLOR_CORAL, width=2, dash="dash")
    )
    fig.add_annotation(
        x=base_date, y=len(items) - 0.5,
        text="📍 Today", showarrow=False,
        font=dict(color=COLOR_CORAL, size=11, family="Inter"),
        xanchor="left"
    )

    fig.update_layout(
        height=max(260, len(items) * 48),
        xaxis=dict(
            type="date",
            showgrid=True,
            gridcolor=COLOR_GRID,
            tickformat="%b %d",
            zeroline=False
        ),
        yaxis=dict(
            showgrid=False,
            autorange="reversed"
        ),
        margin=dict(l=140, r=20, t=20, b=20)
    )

    return apply_plotly_theme(fig)
