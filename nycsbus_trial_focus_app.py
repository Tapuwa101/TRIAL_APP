
import os, io, glob, importlib.util
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, date, timedelta

# Optional libs
try:
    import h3
except Exception:
    h3 = None

try:
    import folium
    from streamlit.components.v1 import html as st_html
except Exception:
    folium = None
    st_html = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

try:
    import pymannkendall as mk
except Exception:
    mk = None

# =============== APP CONFIG & THEME ===============
st.set_page_config(page_title='NYCSBUS Trial • H3 & Trends', layout='wide')
st.title('NYCSBUS Trial • H3 Binning, Time Buckets, Trends & Alarms (Focused)')
st.caption('This Streamlit app wraps the core logic from **nycsbus_trial_eda_august_17.py**: H3 hex bins, time buckets, trend tests, and simple alarms, with your CSVs.')

# =============== DYNAMIC IMPORT ===============
USER_CODE_PATH = '/mnt/data/nycsbus_trial_eda_august_17.py'
user_mod = None
if os.path.exists(USER_CODE_PATH):
    try:
        spec = importlib.util.spec_from_file_location('nycsbus_trial', USER_CODE_PATH)
        user_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user_mod)  # noqa: S102 (Streamlit safe)
        st.success('Loaded functions from nycsbus_trial_eda_august_17.py')
    except Exception as e:
        st.warning(f'Could not import your script; falling back to built-in functions. Error: {e}')

# =============== FALLBACKS (if user_mod missing) ===============
H3_RESOLUTION_DEFAULT = 9

def _lat_lon_to_h3(row, res=H3_RESOLUTION_DEFAULT):
    if h3 is None:
        return None
    lat = row.get('LATITUDE') if 'LATITUDE' in row else row.get('latitude')
    lon = row.get('LONGITUDE') if 'LONGITUDE' in row else row.get('longitude')
    if pd.isna(lat) or pd.isna(lon):
        return None
    try:
        return h3.latlng_to_cell(lat, lon, res)
    except Exception:
        try:
            return h3.geo_to_h3(lat, lon, res)
        except Exception:
            return None

def _get_h3_center(cell):
    if h3 is None or pd.isna(cell):
        return (np.nan, np.nan)
    try:
        lat, lon = h3.cell_to_latlng(cell)
        return (lat, lon)
    except Exception:
        try:
            lat, lon = h3.h3_to_geo(cell)
            return (lat, lon)
        except Exception:
            return (np.nan, np.nan)

def _assign_time_bin(dt):
    # Example: we’ll offer hour-of-day and day-of-week bins
    if pd.isna(dt):
        return {'hour': np.nan, 'dow': np.nan}
    dti = pd.to_datetime(dt, errors='coerce')
    return {'hour': dti.hour, 'dow': dti.day_name()[:3]}

def _time_bucket(dt, bucket='hour'):
    if pd.isna(dt):
        return np.nan
    dti = pd.to_datetime(dt, errors='coerce')
    if bucket == 'hour':
        return dti.hour
    if bucket == 'dow':
        return dti.day_name()[:3]
    if bucket == 'date':
        return dti.date()
    return dti

# Resolve functions to use (prefer user's if available)
lat_lon_to_h3 = getattr(user_mod, 'lat_lon_to_h3', _lat_lon_to_h3)
get_h3_center  = getattr(user_mod, 'get_h3_center', _get_h3_center)
assign_time_bin = getattr(user_mod, 'assign_time_bin', _assign_time_bin)
time_bucket = getattr(user_mod, 'time_bucket', _time_bucket)

# =============== SIDEBAR INPUTS ===============
st.sidebar.header('Data')
use_discover = st.sidebar.checkbox('Auto-discover CSVs', value=True)
uploads = st.sidebar.file_uploader('Upload crash CSVs', type=['csv'], accept_multiple_files=True)
h3_res = st.sidebar.slider('H3 resolution', 5, 10, H3_RESOLUTION_DEFAULT)
win = st.sidebar.slider('Alarm baseline window (days)', 7, 28, 14, 1)
zthr = st.sidebar.slider('Alarm z-score threshold', 1.0, 4.0, 2.0, 0.5)
min_count = st.sidebar.slider('Min count to alarm', 1, 10, 3, 1)

st.sidebar.header('Columns (optional override)')
date_col = st.sidebar.text_input('Date column name', value='CRASH DATE')
time_col = st.sidebar.text_input('Time column name', value='CRASH TIME')
lat_col  = st.sidebar.text_input('Latitude column name', value='LATITUDE')
lon_col  = st.sidebar.text_input('Longitude column name', value='LONGITUDE')
boro_col = st.sidebar.text_input('Borough column name (optional)', value='BOROUGH')

# =============== LOAD FILES ===============
def discover_paths():
    out = []
    for pat in ['./*.csv','../*.csv','/mnt/data/*.csv']:
        out.extend(glob.glob(pat))
    return [p for p in out if any(k in os.path.basename(p).lower() for k in ['crash','collisions','motor_vehicle'])]

sources = []
if uploads:
    sources = uploads
elif use_discover:
    sources = discover_paths()

dfs = []
names = []
for s in sources:
    try:
        if hasattr(s, 'read'):
            df0 = pd.read_csv(s, low_memory=False)
            names.append(getattr(s, 'name', 'uploaded.csv'))
        else:
            df0 = pd.read_csv(s, low_memory=False)
            names.append(os.path.basename(s))
        dfs.append(df0)
    except Exception as e:
        st.warning(f'Skipped a file: {e}')

if not dfs:
    st.info('Upload or auto-discover one or more crash CSVs to begin.')
    st.stop()

raw = pd.concat(dfs, ignore_index=True)

with st.expander('Preview raw columns'):
    st.write(pd.DataFrame({'columns': raw.columns.tolist()}))

# =============== CLEANING & NORMALIZATION ===============
def _norm(s): 
    import re; 
    return re.sub(r'[^a-z0-9]+','', str(s).lower())

colmap = {}
nl = {_norm(c): c for c in raw.columns}
def _pick(*cands):
    for c in cands:
        if _norm(c) in nl: return nl[_norm(c)]
    # partial
    for c in raw.columns:
        if any(_norm(cand) in _norm(c) for cand in cands):
            return c
    return None

c_date = _pick(date_col, 'crash date', 'crash_date', 'collision date', 'date')
c_time = _pick(time_col, 'crash time', 'time')
c_lat  = _pick(lat_col, 'latitude', 'lat', 'y')
c_lon  = _pick(lon_col, 'longitude', 'lon', 'lng', 'x')
c_boro = _pick(boro_col, 'borough', 'boro', 'county')

if not c_date:
    st.error('Could not locate a date column. Please set it under Sidebar → Columns.')
    st.stop()

df = raw.copy()
d = pd.to_datetime(df[c_date], errors='coerce')
if c_time and c_time in df:
    dt = pd.to_datetime(d.dt.date.astype(str) + ' ' + df[c_time].astype(str), errors='coerce')
else:
    dt = pd.to_datetime(d.dt.date.astype(str), errors='coerce')
df['crash_datetime'] = dt
df['crash_date'] = df['crash_datetime'].dt.date

# Copy lat/lon to expected names + uppercase expected by user's functions
if c_lat: df['LATITUDE'] = pd.to_numeric(df[c_lat], errors='coerce')
if c_lon: df['LONGITUDE'] = pd.to_numeric(df[c_lon], errors='coerce')
if c_boro: df['BOROUGH'] = df[c_boro]

# NYC bbox filter (only if coordinates present)
if 'LATITUDE' in df and 'LONGITUDE' in df:
    lat_ok = df['LATITUDE'].between(40.3, 41.1, inclusive='neither')
    lon_ok = df['LONGITUDE'].between(-74.5, -73.3, inclusive='neither')
    has_xy = df['LATITUDE'].notna() & df['LONGITUDE'].notna()
    df = df[(~has_xy) | (lat_ok & lon_ok)]

df = df.dropna(subset=['crash_datetime']).copy()
st.success(f'Cleaned rows: {len(df):,}')

# =============== H3 BINNING & CENTROIDS ===============
if h3 is None:
    st.warning('Package `h3` not available; H3 features disabled. Install `h3` to enable maps and per-cell counts.')
    df['h3'] = np.nan
else:
    df['h3'] = df.apply(lambda r: lat_lon_to_h3(r, res=h3_res) if callable(lat_lon_to_h3) else _lat_lon_to_h3(r, res=h3_res), axis=1)

# Compute centroids for cells (for mapping)
if h3 is not None:
    cell_centers = []
    for cell in df['h3'].dropna().unique():
        lat, lon = get_h3_center(cell) if callable(get_h3_center) else _get_h3_center(cell)
        cell_centers.append({'h3': cell, 'lat': lat, 'lon': lon})
    centers_df = pd.DataFrame(cell_centers)
else:
    centers_df = pd.DataFrame(columns=['h3','lat','lon'])

# =============== TIME BUCKETS ===============
bins = df['crash_datetime'].apply(lambda x: assign_time_bin(x) if callable(assign_time_bin) else _assign_time_bin(x))
df['hour'] = [b.get('hour', np.nan) for b in bins]
df['dow']  = [b.get('dow',  np.nan) for b in bins]

# =============== KPIs ===============
c1, c2, c3, c4 = st.columns(4)
c1.metric('Rows', f'{len(df):,}')
c2.metric('H3 cells', df['h3'].nunique() if 'h3' in df else 0)
c3.metric('Date span (days)', (pd.to_datetime(df['crash_date']).max() - pd.to_datetime(df['crash_date']).min()).days + 1)
c4.metric('With coordinates', int((df['LATITUDE'].notna() & df['LONGITUDE'].notna()).sum()) if 'LATITUDE' in df else 0)

st.markdown('---')
tab_data, tab_h3, tab_time, tab_trend, tab_alarms = st.tabs(['Data','H3 Map','Time Buckets','Trends (Mann-Kendall)','Alarms'])

# =============== DATA TAB ===============
with tab_data:
    st.subheader('Filtered data sample')
    st.dataframe(df.head(50))
    # Downloads
    buf = io.BytesIO(); df.to_csv(buf, index=False)
    st.download_button('⬇️ Download cleaned data', buf.getvalue(), file_name='cleaned_crashes.csv', mime='text/csv')

# =============== H3 MAP TAB ===============
with tab_h3:
    st.subheader('H3 cell density')
    if h3 is None or centers_df.empty or 'h3' not in df:
        st.info('H3 not available or no coordinates in the data.')
    else:
        # per-cell counts
        cell_counts = df.groupby('h3').size().rename('count').reset_index()
        m = centers_df.merge(cell_counts, on='h3', how='left').dropna(subset=['lat','lon'])
        # Build a Folium map for 'focus on your code' feel
        if folium is not None and st_html is not None:
            center_lat = m['lat'].mean() if len(m) else 40.73
            center_lon = m['lon'].mean() if len(m) else -73.94
            fmap = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='cartodbpositron')
            for _, r in m.iterrows():
                folium.CircleMarker(location=[r['lat'], r['lon']], radius=max(3, min(12, r['count']**0.5)),
                                    fill=True, fill_opacity=0.7, popup=f"{r['h3']} • {int(r['count'])} crashes").add_to(fmap)
            st_html(fmap._repr_html_(), height=520)
        else:
            st.warning('folium not available; showing table instead.')
            st.dataframe(m.sort_values('count', ascending=False))

# =============== TIME BUCKETS TAB ===============
with tab_time:
    st.subheader('Hourly and weekday patterns')
    hourly = df.groupby('hour').size().rename('count').reset_index()
    dow = df.groupby('dow').size().rename('count').reset_index()
    if px is not None:
        fig1 = px.bar(hourly, x='hour', y='count', title='Crashes by Hour')
        fig2 = px.bar(dow, x='dow', y='count', title='Crashes by Day of Week',
                      category_orders={'dow': ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']})
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.dataframe(hourly)
        st.dataframe(dow)

# =============== TRENDS TAB (Mann-Kendall) ===============
with tab_trend:
    st.subheader('Trend test by H3 cell (Mann–Kendall)')
    if mk is None:
        st.info('pymannkendall not installed. Add `pymannkendall` to requirements to enable this tab.')
    else:
        # select a cell
        cells = df['h3'].dropna().unique().tolist() if 'h3' in df else []
        if not cells:
            st.info('No H3 cells to test.')
        else:
            sel = st.selectbox('Select H3 cell', options=cells)
            g = df[df['h3'] == sel].groupby('crash_date').size().rename('count').reset_index().sort_values('crash_date')
            st.write(g.tail(15))
            if len(g) >= 10:
                res = mk.original_test(g['count'])
                st.write(f"Trend: {res.trend}, p={res.p:.4f}, tau={res.Tau:.3f}" if hasattr(res, 'Tau') else res)
            else:
                st.info('Need at least ~10 daily points for a stable test.')

# =============== SIMPLE ALARMS TAB ===============
with tab_alarms:
    st.subheader('Rolling z-score alarms (simple)')
    # per-cell daily counts
    if 'h3' not in df or df['h3'].isna().all():
        st.info('No H3 cells available for alarms.')
    else:
        cd = df.groupby(['h3','crash_date']).size().rename('count').reset_index().sort_values('crash_date')
        def _alarms(g, w=14, z=2.0, min_c=3):
            g = g.copy().sort_values('crash_date')
            s = pd.Series(g['count'].values)
            g['mu'] = s.rolling(w, min_periods=5).mean().shift(1)
            g['sd'] = s.rolling(w, min_periods=5).std(ddof=0).shift(1)
            g['z'] = (g['count'] - g['mu']) / g['sd']
            g['z'] = g['z'].replace([np.inf,-np.inf], np.nan)
            g['alarm'] = np.where((g['count']>=min_c) & (g['z']>=z), 'alarm', 'none')
            return g
        out = cd.groupby('h3', group_keys=False).apply(lambda g: _alarms(g, w=win, z=zthr, min_c=min_count))
        st.dataframe(out.sort_values(['crash_date','z'], ascending=[False, False]).reset_index(drop=True))

        # Next-day naive prediction
        sel = st.selectbox('Inspect a cell for next-day prediction', options=out['h3'].dropna().unique().tolist())
        gg = out[out['h3']==sel].sort_values('crash_date')
        if not gg.empty:
            base = gg['count'].tail(7).mean()
            mu = gg['mu'].iloc[-1] if pd.notna(gg['mu'].iloc[-1]) else base
            sd = gg['sd'].iloc[-1] if (pd.notna(gg['sd'].iloc[-1]) and gg['sd'].iloc[-1] > 0) else 1.0
            pred = base  # simple naive
            z = (pred - mu) / sd if sd else np.nan
            st.write({'pred_count': float(np.nan_to_num(pred)), 'pred_z': float(np.nan_to_num(z)), 'threshold': float(mu + zthr*sd)})
