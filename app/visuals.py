"""Dashboard visuals: an animated powerhouse flow diagram and weather charts.

The powerhouse is a self-contained HTML/CSS block (rendered in an iframe via
st.components.v1.html) that animates the live dispatch of a selected hour:
renewable sources (solar + wind) feed the load and charge the renewable battery
container; the battery drains into the load when renewable falls short; the backup
source only fires when the battery is empty and renewable cannot cover demand.
Arrow animations toggle on the actual dispatch numbers, so it reads like a real
power-house management screen rather than a static picture.
"""
import altair as alt
import pandas as pd

WEATHER_SERIES = {
    "temperature_2m_c": ("Temperature (°C)", "#e4572e"),
    "cloud_cover_pct": ("Cloud cover (%)", "#7f7f7f"),
    "relative_humidity_pct": ("Humidity (%)", "#17becf"),
    "solar_radiation_w_m2": ("Solar radiation (W/m²)", "#f4a300"),
    "wind_speed_10m_kmh": ("Wind speed (km/h)", "#2ca02c"),
    "precipitation_mm": ("Precipitation (mm)", "#1f77b4"),
}


def powerhouse_html(sc, capacity_kwh, soc_kwh, theme="dark"):
    """Color-coded animated single-hour powerhouse dispatch flow diagram.

    Color coding per energy path:
      - Solar -> Busbar: Gold (#f1c40f)
      - Wind -> Busbar: Emerald Green (#2ecc71)
      - Busbar -> Battery (Charge): Cyan (#00d2d3)
      - Battery -> Load (Discharge): Amber/Orange (#f39c12)
      - Backup -> Load: Crimson Red (#e74c3c)
    """
    soc_pct = 0.0 if capacity_kwh <= 0 else max(0.0, min(100.0, 100.0 * soc_kwh / capacity_kwh))
    charging = sc.get("battery_charge_kw", 0.0) > 0.5
    discharging = sc.get("battery_discharge_kw", 0.0) > 0.5
    backup_on = sc.get("backup_kw", 0.0) > 0.5
    solar_on = sc.get("solar_kw", 0.0) > 0.5
    wind_on = sc.get("wind_kw", 0.0) > 0.5
    renew_to_load = sc.get("renewable_used_kw", 0.0) > 0.5
    curtailed_on = sc.get("curtailed_kw", 0.0) > 0.5

    def cls(flag):
        return "on" if flag else "off"

    return f"""
<div class="ph">
  <style>
    .ph {{ --bg:#0b131e; --panel:#141e2b; --border:#233346; --text:#e6edf3; --dim:#8b98a5;
          font-family:'Segoe UI',system-ui,sans-serif; color:var(--text);
          background:var(--bg); border:1px solid var(--border); border-radius:16px; padding:18px; }}
    @media (prefers-color-scheme: light) {{
      .ph {{ --bg:#f8fafc; --panel:#ffffff; --border:#cbd5e1; --text:#0f172a; --dim:#64748b; }}
    }}
    .ph-head {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; }}
    .ph-head h4 {{ margin:0; font-size:15px; font-weight:700; color:var(--text); }}
    .ph-head .sub {{ font-size:12px; color:var(--dim); }}

    /* Layout Grid */
    .ph-grid {{ display:grid; grid-template-columns:1fr 0.9fr 1.1fr 1fr; gap:12px; align-items:center; }}
    .ph-col {{ display:flex; flex-direction:column; gap:12px; }}

    /* Nodes */
    .ph-node {{ background:var(--panel); border:1.5px solid var(--border); border-radius:12px;
                padding:10px 12px; text-align:center; position:relative; transition:all 0.3s ease; }}
    .ph-node .icon {{ font-size:22px; line-height:1; }}
    .ph-node .lbl {{ font-size:11px; color:var(--dim); margin-top:2px; font-weight:600; text-transform:uppercase; }}
    .ph-node .val {{ font-size:17px; font-weight:700; margin-top:2px; }}
    .ph-node .unit {{ font-size:11px; font-weight:400; color:var(--dim); }}

    .ph-node.solar.on {{ border-color:#f1c40f; box-shadow:0 0 10px rgba(241,196,15,0.25); }}
    .ph-node.wind.on  {{ border-color:#2ecc71; box-shadow:0 0 10px rgba(46,204,113,0.25); }}
    .ph-node.batt.on  {{ border-color:#00d2d3; box-shadow:0 0 10px rgba(0,210,211,0.25); }}
    .ph-node.load.on  {{ border-color:#e4572e; box-shadow:0 0 10px rgba(228,87,46,0.25); }}
    .ph-node.backup.on{{ border-color:#e74c3c; box-shadow:0 0 10px rgba(231,76,60,0.3); }}

    /* Central Busbar */
    .ph-busbar {{ background:#1e293b; border:2px solid #3b82f6; border-radius:10px; padding:12px 8px;
                 text-align:center; box-shadow:0 0 12px rgba(59,130,246,0.3); }}
    .ph-busbar .lbl {{ font-size:10px; color:#93c5fd; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; }}
    .ph-busbar .val {{ font-size:15px; font-weight:700; color:#ffffff; margin-top:2px; }}

    /* Battery Container Gauge */
    .ph-batt-gauge {{ height:76px; border-radius:8px; border:2px solid var(--border);
                     background:#091018; position:relative; overflow:hidden; margin-top:4px; }}
    .ph-fill {{ position:absolute; bottom:0; left:0; right:0; height:{soc_pct:.1f}%;
               background:linear-gradient(180deg,#00d2d3,#00a8a9); transition:height .8s ease; }}
    .ph-fill.low {{ background:linear-gradient(180deg,#f39c12,#d35400); }}
    .ph-fill.crit {{ background:linear-gradient(180deg,#e74c3c,#c0392b); }}
    .ph-soc-text {{ position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
                   font-size:15px; font-weight:700; text-shadow:0 1px 3px rgba(0,0,0,.8); }}

    /* Flow Lines & Animated Particles */
    .ph-flow {{ display:flex; flex-direction:column; gap:2px; position:relative; padding:2px 0; }}
    .ph-line {{ height:5px; background:var(--border); border-radius:3px; position:relative; overflow:hidden; }}
    .ph-line::after {{ content:''; position:absolute; top:0; left:-40%; width:40%; height:100%; opacity:0; border-radius:3px; }}

    .ph-flow.solar-flow.on .ph-line::after {{ background:#f1c40f; opacity:1; animation:ph-flow-r 1s linear infinite; }}
    .ph-flow.wind-flow.on  .ph-line::after {{ background:#2ecc71; opacity:1; animation:ph-flow-r 1s linear infinite; }}
    .ph-flow.charge-flow.on .ph-line::after {{ background:#00d2d3; opacity:1; animation:ph-flow-r 1s linear infinite; }}
    .ph-flow.dis-flow.on    .ph-line::after {{ background:#f39c12; opacity:1; animation:ph-flow-l 1s linear infinite; }}
    .ph-flow.backup-flow.on .ph-line::after {{ background:#e74c3c; opacity:1; animation:ph-flow-r 1s linear infinite; }}
    .ph-flow.curtail-flow.on .ph-line::after {{ background:#9b59b6; opacity:1; animation:ph-flow-r 1.2s linear infinite; }}

    @keyframes ph-flow-r {{ 0%{{left:-40%}} 100%{{left:100%}} }}
    @keyframes ph-flow-l {{ 0%{{left:100%}} 100%{{left:-40%}} }}

    .ph-tag {{ font-size:10px; font-weight:700; white-space:nowrap; display:flex; justify-content:space-between; }}
    .tag-solar {{ color:#f1c40f; }}
    .tag-wind  {{ color:#2ecc71; }}
    .tag-charge{{ color:#00d2d3; }}
    .tag-dis   {{ color:#f39c12; }}
    .tag-backup{{ color:#e74c3c; }}

    /* Legend Pills */
    .ph-legend {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; padding-top:10px; border-top:1px dashed var(--border); font-size:11px; }}
    .ph-pill {{ background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:2px 10px; color:var(--dim); }}
    .ph-pill b {{ color:var(--text); }}
  </style>

  <div class="ph-head">
    <h4>⚡ Powerhouse Dispatch Topology — {sc.get('datetime','')}</h4>
    <div class="sub">Live color-coded energy flows: Solar (Gold) | Wind (Green) | Battery Charge (Cyan) | Battery Discharge (Amber) | Backup (Red)</div>
  </div>

  <div class="ph-grid">
    <!-- COL 1: RENEWABLE GENERATORS -->
    <div class="ph-col">
      <div class="ph-node solar {cls(solar_on)}">
        <div class="icon">☀️</div>
        <div class="lbl">Solar Array</div>
        <div class="val">{sc.get('solar_kw',0):,.0f} <span class="unit">kW</span></div>
      </div>
      <div class="ph-flow solar-flow {cls(solar_on)}">
        <div class="ph-tag tag-solar"><span>☀️ Solar Flow</span> <span>{sc.get('solar_kw',0):,.0f} kW</span></div>
        <div class="ph-line"></div>
      </div>

      <div class="ph-node wind {cls(wind_on)}">
        <div class="icon">🌬️</div>
        <div class="lbl">Wind Array</div>
        <div class="val">{sc.get('wind_kw',0):,.0f} <span class="unit">kW</span></div>
      </div>
      <div class="ph-flow wind-flow {cls(wind_on)}">
        <div class="ph-tag tag-wind"><span>🌬️ Wind Flow</span> <span>{sc.get('wind_kw',0):,.0f} kW</span></div>
        <div class="ph-line"></div>
      </div>
    </div>

    <!-- COL 2: CENTRAL AC BUSBAR -->
    <div class="ph-col">
      <div class="ph-busbar">
        <div class="lbl">⚡ Central AC Busbar</div>
        <div class="val">{sc.get('renewable_kw',0):,.0f} kW</div>
        <div style="font-size:10px;color:#94a3b8;margin-top:2px">Total Clean Supply</div>
      </div>

      <div class="ph-flow charge-flow {cls(charging)}">
        <div class="ph-tag tag-charge"><span>🔋 Auto-Charge →</span> <span>{sc.get('battery_charge_kw',0):,.0f} kW</span></div>
        <div class="ph-line"></div>
      </div>
    </div>

    <!-- COL 3: BATTERY ENERGY STORAGE -->
    <div class="ph-col">
      <div class="ph-node batt {cls(charging or discharging)}">
        <div class="lbl">🔋 Battery BESS ({soc_pct:.0f}%)</div>
        <div class="ph-batt-gauge">
          <div class="ph-fill {'crit' if soc_pct<15 else ('low' if soc_pct<40 else '')}"></div>
          <div class="ph-soc-text">{soc_kwh:,.0f} / {capacity_kwh:,.0f} kWh</div>
        </div>
        <div class="val" style="font-size:13px;margin-top:4px">
          {f"⚡ Charging +{sc.get('battery_charge_kw',0):,.0f} kW" if charging else (f"🔋 Discharging -{sc.get('battery_discharge_kw',0):,.0f} kW" if discharging else "Standby")}
        </div>
      </div>

      <div class="ph-flow dis-flow {cls(discharging)}">
        <div class="ph-tag tag-dis"><span>← Discharge to Load</span> <span>{sc.get('battery_discharge_kw',0):,.0f} kW</span></div>
        <div class="ph-line"></div>
      </div>
    </div>

    <!-- COL 4: CONSUMPTION & BACKUP -->
    <div class="ph-col">
      <div class="ph-node load on">
        <div class="icon">🏭</div>
        <div class="lbl">Facility Demand</div>
        <div class="val">{sc.get('demand_kw',0):,.0f} <span class="unit">kW</span></div>
      </div>

      <div class="ph-node backup {cls(backup_on)}">
        <div class="icon">🔌</div>
        <div class="lbl">Grid / Diesel Backup</div>
        <div class="val">{sc.get('backup_kw',0):,.0f} <span class="unit">kW</span></div>
      </div>
      <div class="ph-flow backup-flow {cls(backup_on)}">
        <div class="ph-tag tag-backup"><span>🔌 Backup Flow</span> <span>{sc.get('backup_kw',0):,.0f} kW</span></div>
        <div class="ph-line"></div>
      </div>
    </div>
  </div>

  <div class="ph-legend">
    <span class="ph-pill">Direct Renewables: <b style="color:#2ecc71">{sc.get('renewable_used_kw',0):,.0f} kW</b></span>
    <span class="ph-pill">Stored to Battery: <b style="color:#00d2d3">{sc.get('battery_charge_kw',0):,.0f} kW</b></span>
    <span class="ph-pill">From Battery: <b style="color:#f39c12">{sc.get('battery_discharge_kw',0):,.0f} kW</b></span>
    <span class="ph-pill">Backup Grid/Diesel: <b style="color:#e74c3c">{sc.get('backup_kw',0):,.0f} kW</b></span>
    <span class="ph-pill">Curtailed: <b style="color:#9b59b6">{sc.get('curtailed_kw',0):,.0f} kW</b></span>
  </div>
</div>
"""


def weather_chart(frame, time_col="datetime", height=320):
    """Graph 3: Single combined zoomable chart for all weather attributes in distinct colors."""
    cols = [c for c in WEATHER_SERIES if c in frame.columns]
    if not cols or time_col not in frame.columns:
        return None
    long = frame[[time_col, *cols]].melt(id_vars=time_col, var_name="series", value_name="value")
    long["label"] = long["series"].map(lambda s: WEATHER_SERIES[s][0])
    domain = [WEATHER_SERIES[c][0] for c in cols]
    rng = [WEATHER_SERIES[c][1] for c in cols]
    return (alt.Chart(long)
            .mark_line(strokeWidth=2.2, interpolate="monotone")
            .encode(
                x=alt.X(f"{time_col}:T", title="Time Horizon"),
                y=alt.Y("value:Q", title="Weather Attribute Value", scale=alt.Scale(zero=False)),
                color=alt.Color("label:N", scale=alt.Scale(domain=domain, range=rng),
                                legend=alt.Legend(title="Weather Factor")),
                tooltip=[alt.Tooltip(f"{time_col}:T"), "label:N", alt.Tooltip("value:Q", format=",.1f")]
            )
            .properties(height=height).interactive())


def demand_breakdown_chart(frame, time_col="period", height=340):
    """Graph 2: 3-line breakdown chart showing Demand (Coral), Solar (Gold), Wind (Blue)."""
    need = ["demand_expected_kw", "solar_expected_kw", "wind_expected_kw"]
    if not all(c in frame.columns for c in need) or time_col not in frame.columns:
        return None
    long = frame[[time_col, *need]].melt(id_vars=time_col, var_name="series", value_name="kW")
    long["series"] = long["series"].map({
        "demand_expected_kw": "Demand",
        "solar_expected_kw": "Solar Power",
        "wind_expected_kw": "Wind Power",
    })
    colors = {"Demand": "#e4572e", "Solar Power": "#f2a71b", "Wind Power": "#1f77b4"}
    return (alt.Chart(long)
            .mark_line(strokeWidth=2.5, interpolate="monotone")
            .encode(
                x=alt.X(f"{time_col}:T", title="Time Horizon"),
                y=alt.Y("kW:Q", title="Power (kW)"),
                color=alt.Color("series:N", scale=alt.Scale(domain=list(colors.keys()), range=list(colors.values())),
                                legend=alt.Legend(title="Category")),
                tooltip=[alt.Tooltip(f"{time_col}:T"), "series:N", alt.Tooltip("kW:Q", format=",.0f")]
            )
            .properties(height=height).interactive())
