"""Reusable plot helpers shared across pages."""

import plotly.express as px
import plotly.graph_objects as go


def africa_project_map(countries_df, sites_df, title="ZOA proposals across Africa", scope="africa"):
    """Africa-only map: highlighted countries + hoverable project-site markers.

    ``countries_df`` — columns ``country`` (Natural-Earth name) and ``proposals``.
    ``sites_df``     — one row per site with ``lat``, ``lon``, ``site``, ``title``,
                       ``donor``, ``budget``, ``timeframe``, ``partners``,
                       ``sector``, ``beneficiaries`` (may be empty).
    ``scope``        — Plotly geo scope ("africa" or "world" when Asian countries are present).
    """
    fig = go.Figure()

    # Layer 1 — shade every country that has at least one project.
    if not countries_df.empty:
        zmax = int(countries_df["proposals"].max())
        fig.add_trace(
            go.Choropleth(
                locations=countries_df["country"],
                locationmode="country names",
                z=countries_df["proposals"],
                # Start the scale at a clearly-visible blue so even a single
                # proposal reads as "highlighted", not near-white.
                colorscale=[[0.0, "#9ecae1"], [1.0, "#08306b"]],
                zmin=0,
                zmax=zmax,
                marker_line_color="white",
                marker_line_width=0.6,
                showscale=False,
                hovertemplate="<b>%{location}</b><br>%{z} proposal(s)<extra></extra>",
            )
        )

    # Layer 2 — a marker per project site, rich hover card.
    if not sites_df.empty:
        hover = [
            "<br>".join(
                [
                    f"<b>{r.title}</b>",
                    f"📍 {r.site}",
                    f"🏛️ Donor: {r.donor}",
                    f"🗂️ Sector: {r.sector}",
                    f"💶 Budget: {r.budget}",
                    f"🗓️ {r.timeframe}",
                    f"🤝 {r.partners}",
                    f"👥 {r.beneficiaries}",
                ]
            )
            for r in sites_df.itertuples()
        ]
        fig.add_trace(
            go.Scattergeo(
                lon=sites_df["lon"],
                lat=sites_df["lat"],
                text=hover,
                mode="markers",
                marker=dict(
                    size=11,
                    color="#d6336c",
                    line=dict(width=1.2, color="white"),
                    opacity=0.9,
                ),
                hoverinfo="text",
                name="Project sites",
            )
        )

    fig.update_geos(
        scope=scope,
        showcountries=True,
        countrycolor="#cccccc",
        showland=True,
        landcolor="#f5f5f5",
        showocean=True,
        oceancolor="#eaf4fb",
        showframe=False,
    )
    fig.update_layout(
        title=title,
        margin=dict(t=40, l=0, r=0, b=0),
        height=620,
        showlegend=False,
    )
    return fig


def sector_bar(df, title="Indicators per sector"):
    """Horizontal bar of indicator count per sector, sorted descending."""
    counts = (
        df["Sector"].value_counts().rename_axis("Sector").reset_index(name="count")
    )
    fig = px.bar(
        counts.sort_values("count"),
        x="count",
        y="Sector",
        orientation="h",
        text="count",
        title=title,
        color="count",
        color_continuous_scale="Blues",
    )
    fig.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="indicators")
    return fig


def crosstab_heatmap(ct, title, color="Blues"):
    """Heatmap from a pandas crosstab."""
    fig = px.imshow(
        ct,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=color,
        title=title,
    )
    fig.update_layout(xaxis_title="", yaxis_title="")
    return fig
