# known_planets_viewer.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import matplotlib.pyplot as plt

from utils import format_star_id

class KnownPlanetsViewer:
    def __init__(self):
        self.data_loaded = False

    def search_star_direct(self, star_id: str, mission: str):
        """Search for a specific star directly via API"""
        star_id_clean = star_id.strip()
        try:
            if mission.lower() == "tess":
                return self._query_toi_by_tic_full(star_id_clean)
            elif mission.lower() == "kepler":
                return self._query_koi_by_kic(star_id_clean)
            else:
                st.error(f"Unknown mission: {mission}")
                return None
        except Exception as e:
            st.error(f"Search error: {e}")
            return None

    def _query_koi_by_kic(self, kic_id: str):
        """Query Kepler KOI table - works perfectly"""
        st.info(f"🔍 Querying NASA archive for KIC {kic_id}...")
        base_url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
        params = {
            'table': 'cumulative',
            'where': f'kepid={kic_id}',
            'format': 'csv',
            'select': 'kepoi_name,kepid,koi_period,koi_depth,koi_duration,koi_prad,koi_pdisposition,koi_srad,koi_smass,koi_kepmag,koi_model_snr'
        }
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = pd.read_csv(StringIO(response.text))

            if len(data) == 0:
                return None

            data.rename(columns={
                'koi_period': 'period',
                'koi_depth': 'depth',
                'koi_duration': 'duration',
                'koi_prad': 'planet_radius',
                'koi_pdisposition': 'disposition',
                'koi_srad': 'stellar_radius',
                'koi_smass': 'stellar_mass',
                'koi_kepmag': 'kepmag',
                'koi_model_snr': 'snr'
            }, inplace=True)

            st.success(f"✅ Found {len(data)} object(s)")
            return data

        except Exception as e:
            st.error(f"KOI Query failed: {e}")
            return None

    def _query_toi_by_tic_full(self, tic_id: str):
        """
        Query TESS TOI table robustly, with as many fields as possible.
        """
        st.info(f"🔍 Querying NASA archive for TIC {tic_id}...")

        base_url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
        # Use correct column names for TOI table
        select_cols = [
            'toipfx', 'tid', 'tfopwg_disp', 
            'pl_orbper', 'pl_trandep', 'pl_trandurh', 'pl_rade',
            'st_tmag', 'st_rad', 'st_teff'
        ]
        params = {
            'table': 'toi',
            'where': f'tid={tic_id}',
            'format': 'csv',
            'select': ','.join(select_cols)
        }
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            content = response.text

            if 'ERROR' in content or len(content) < 50:
                return None

            data = pd.read_csv(StringIO(content))
            if len(data) == 0:
                return None

            # Clean and align columns with correct TOI column names
            data.rename(columns={
                'toipfx': 'toi',
                'tid': 'tic_id',
                'tfopwg_disp': 'disposition',
                'pl_orbper': 'period',
                'pl_trandep': 'depth',
                'pl_trandurh': 'duration',  # Note: this is already in hours
                'pl_rade': 'planet_radius',
                'st_tmag': 'tess_mag',
                'st_rad': 'stellar_radius',
                'st_teff': 'stellar_teff'
            }, inplace=True)

            # Some TOIs may not have any planet param; add fallback columns
            for field in ['period', 'depth', 'duration', 'planet_radius']:
                if field not in data.columns:
                    data[field] = np.nan

            st.success(f"✅ Found {len(data)} object(s)")
            return data

        except Exception as e:
            st.error(f"TESS/TOI Query failed: {e}")
            return None

    def display_planet_data(self, results: pd.DataFrame, star_id: str, mission: str):
        """Display planet data"""
        if results is None or len(results) == 0:
            st.error(f"❌ No data found for {format_star_id(star_id, mission)}")
            if mission.lower() == "tess":
                st.warning("""
                ### 🔍 TIC ID Not in TOI Catalog
                The **TOI (TESS Objects of Interest)** catalog contains only **~7,000 stars**
                out of 200+ million TESS targets.
                """)
            else:
                st.info("No KOI data for this Kepler star. Try example KIC IDs from sidebar.")
            return

        st.success(f"✅ Found {len(results)} planet(s)/candidate(s) for {format_star_id(star_id, mission)}")
        for idx, planet in results.iterrows():
            self._display_single_planet(planet, mission, idx)

    def _display_single_planet(self, planet: pd.Series, mission: str, index: int):
        st.markdown("---")
        # Header and Names
        if mission.lower() == "tess":
            toi_val = planet.get('toi', index+1)
            planet_name = f"TOI {toi_val}"
        else:
            planet_name = planet.get('kepoi_name', f"KOI {index+1}")
        st.markdown(f"## 🪐 {planet_name}")

        disposition = planet.get('disposition', 'UNKNOWN')
        if mission.lower() == "tess":
            disp_map = {
                'CP': ('CONFIRMED PLANET', '🟢', 'success'),
                'PC': ('PLANET CANDIDATE', '🟡', 'warning'),
                'FP': ('FALSE POSITIVE', '❌', 'error'),
                'KP': ('KNOWN PLANET', '🟢', 'success'),
                'APC': ('AMBIGUOUS CANDIDATE', '🟠', 'warning')
            }
        else:
            disp_map = {
                'CONFIRMED': ('CONFIRMED PLANET', '🟢', 'success'),
                'CANDIDATE': ('PLANET CANDIDATE', '🟡', 'warning'),
                'FALSE POSITIVE': ('FALSE POSITIVE', '❌', 'error')
            }
        disp_label, emoji, disp_type = disp_map.get(disposition, ('UNKNOWN', '❓', 'info'))
        if disp_type == 'success':
            st.success(f"{emoji} **{disp_label}** (NASA Official)")
        elif disp_type == 'warning':
            st.warning(f"{emoji} **{disp_label}** (NASA Official)")
        else:
            st.error(f"{emoji} **{disp_label}** (NASA Official)")

        st.info("📚 Data from NASA Exoplanet Archive")

        # Planet parameters
        st.markdown("### 📊 Planet Parameters")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            period = planet.get('period', np.nan)
            st.metric("Orbital Period", f"{period:.3f} days" if pd.notna(period) else "N/A")
        with col2:
            depth = planet.get('depth', np.nan)
            st.metric("Transit Depth", f"{depth:.0f} ppm" if pd.notna(depth) else "N/A")
        with col3:
            duration = planet.get('duration', np.nan)
            st.metric("Duration", f"{duration:.2f} hours" if pd.notna(duration) else "N/A")
        with col4:
            radius = planet.get('planet_radius', np.nan)
            st.metric("Planet Radius", f"{radius:.2f} R⊕" if pd.notna(radius) else "N/A")

        # Stellar Parameters
        if mission.lower() == "kepler":
            st.markdown("### ⭐ Stellar Parameters")
            star_col1, star_col2, star_col3 = st.columns(3)
            with star_col1:
                st_rad = planet.get('stellar_radius', np.nan)
                st.metric("Stellar Radius", f"{st_rad:.2f} R☉" if pd.notna(st_rad) else "N/A")
            with star_col2:
                st_mass = planet.get('stellar_mass', np.nan)
                st.metric("Stellar Mass", f"{st_mass:.2f} M☉" if pd.notna(st_mass) else "N/A")
            with star_col3:
                mag = planet.get('kepmag', np.nan)
                st.metric("Kepler Magnitude", f"{mag:.2f}" if pd.notna(mag) else "N/A")
            snr = planet.get('snr', np.nan)
            if pd.notna(snr):
                st.markdown("### 📈 Signal Quality")
                st.metric("Signal-to-Noise Ratio", f"{snr:.1f}σ")
        elif mission.lower() == "tess":
            st.markdown("### ⭐ Stellar Parameters")
            star_col1, star_col2, star_col3 = st.columns(3)
            with star_col1:
                st_rad = planet.get('stellar_radius', np.nan)
                st.metric("Stellar Radius", f"{st_rad:.2f} R☉" if pd.notna(st_rad) else "N/A")
            with star_col2:
                st_teff = planet.get('stellar_teff', np.nan)
                st.metric("Stellar Temperature", f"{st_teff:.0f} K" if pd.notna(st_teff) else "N/A")
            with star_col3:
                tmag = planet.get('tess_mag', np.nan)
                st.metric("TESS Magnitude", f"{tmag:.2f}" if pd.notna(tmag) else "N/A")
        # Visualization
        period = planet.get('period', np.nan)
        depth = planet.get('depth', np.nan)
        if pd.notna(period) and pd.notna(depth) and period > 0 and depth > 0:
            st.markdown("### 📊 Transit Visualization")
            self._plot_transit(period, depth, mission)
        with st.expander("📋 **Complete Data Record**"):
            display_data = {k: v for k, v in planet.items() if pd.notna(v)}
            if display_data:
                df_display = pd.DataFrame(display_data, index=[0]).T
                df_display.columns = ['Value']
                st.dataframe(df_display, use_container_width=True)

    def _plot_transit(self, period: float, depth: float, mission: str):
        try:
            # Always treat depth as ppm for display simplicity
            depth_frac = depth / 1e6
            phase = np.linspace(-0.5, 0.5, 1000)
            flux = np.ones_like(phase)
            in_transit = np.abs(phase) < 0.025
            flux[in_transit] = 1 - depth_frac
            try:
                from scipy.ndimage import gaussian_filter1d
                flux = gaussian_filter1d(flux, sigma=10)
            except:
                pass
            noise = np.random.normal(0, depth_frac * 0.1, len(flux))
            flux_noisy = flux + noise
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(phase, flux_noisy, s=2, alpha=0.3, color='gray', label='Simulated data')
            ax.plot(phase, flux, lw=3, color='blue', label='Transit model')
            ax.set_xlabel("Phase", fontsize=12)
            ax.set_ylabel("Normalized Flux", fontsize=12)
            ax.set_title(f"Synthetic Transit (Period={period:.3f}d, Depth={depth:.0f}ppm)")
            ax.set_xlim(-0.5, 0.5)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.info("ℹ️ Synthetic visualization based on archived parameters")
        except Exception as e:
            st.warning(f"Transit plot failed: {e}")

def create_known_planets_interface():
    st.markdown('<h1 style="text-align:center;color:#1f77b4;">🔭 Known Exoplanets Lookup</h1>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;margin-bottom:2rem;color:#666;">'
                'Query NASA Exoplanet Archive</div>', unsafe_allow_html=True)
    with st.sidebar:
        st.header("🔍 Search Parameters")
        mission = st.selectbox("Select Mission", ["Kepler", "TESS"])
        if mission == "Kepler":
            default_id = "10593626"
            st.success("✅ **All KOIs Available**")
            examples = [
                ("10593626", "Kepler-11 (6 planets)"),
                ("11904151", "Kepler-10b"),
                ("10666592", "Kepler-22b"),
                ("6922244", "Kepler-10"),
                ("10024862", "Kepler-5b"),
            ]
        else:
            default_id = "279741379"
            st.warning("⚠️ **Only 7,000 TOIs**")
            st.info("Most TIC IDs won't be here!")
            examples = [
                ("279741379", "TOI-270"),
                ("460205581", "TOI-700"),
                ("307210830", "TOI-175"),
            ]
        star_id = st.text_input(
            f"Enter {'KIC' if mission=='Kepler' else 'TIC'} Number",
            default_id
        )
        search_button = st.button("🔎 Search Archive", use_container_width=True)
        st.markdown("---")
        st.markdown("#### 📚 Examples")
        for idx, (ex_id, desc) in enumerate(examples):
            if st.button(ex_id, key=f"known_{mission}_{idx}", help=desc, use_container_width=True):
                st.session_state.known_star = ex_id
                st.rerun()
    if "known_star" in st.session_state:
        star_id = st.session_state.pop("known_star")
        search_button = True
    if not search_button:
        st.markdown("## 📖 How This Works")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 📚 Known Planets Database

            **Kepler Mission:**
            - 4,000+ KOIs (Kepler Objects of Interest)
            - Comprehensive data
            - All parameters available

            **TESS Mission:**
            - Only 7,000 TOIs (out of 200M+ stars)
            - Limited to stars WITH detected candidates
            - Basic parameters only

            ### 🔍 What You Get

            - Official NASA classification
            - Planet parameters
            - Synthetic transit plot
            """)
        with col2:
            st.markdown("""
            ### 💡 Important for TESS

            **The TOI catalog is TINY!**

            Only ~7,000 stars out of 200+ million have TOI designations.

            **If your TIC ID isn't found:**

            ✅ **This is NORMAL**

            ✅ **Your star is valid**

            ✅ **Use "Classify New Planets" mode instead**

            That mode analyzes ANY TIC ID and can discover NEW candidates!

            ### 🎯 Recommendation

            **For Research:** Use "Classify New Planets"

            **For Learning:** Browse known TOIs here
            """)
        st.markdown("---")
        st.info("👈 Enter a star ID to search")
        return

    viewer = KnownPlanetsViewer()
    with st.spinner(f"Searching for {format_star_id(star_id, mission)}..."):
        results = viewer.search_star_direct(star_id, mission)
    viewer.display_planet_data(results, star_id, mission)
