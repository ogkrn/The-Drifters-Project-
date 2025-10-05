# lightcurve_generator.py - FIXED WITH GRAPHS

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
import lightkurve as lk
import requests
import time


class LightcurveGenerator:
    """
    Generate light curves with advanced detrending and visualization
    
    FIXED: 
    - Downloads ALL quarters when quarter=None
    - Shows before/after detrending graphs
    - Noise filtering visualization
    """
    
    def __init__(self):
        self.lightcurve = None
        self.raw_lightcurve = None
        self.raw_data = None
        self.star_id = None
        self.mission = None
        
    def generate_from_pixels(self, processed_pixels: Dict, star_id: str, mission: str) -> bool:
        """Generate light curve with visualization"""
        
        try:
            self.star_id = star_id
            self.mission = mission
            
            st.write("📊 *Extracting light curve from pixel data...*")
            
            # METHOD 1: Extract from TPF
            if processed_pixels and 'tpf' in processed_pixels:
                try:
                    st.info("🔄 Extracting light curve from Target Pixel File...")
                    tpf = processed_pixels['tpf']
                    
                    # Use pipeline aperture if available
                    if hasattr(tpf, "pipeline_mask") and tpf.pipeline_mask is not None:
                        ap_mask = tpf.pipeline_mask
                        st.info("✅ Using pipeline-optimized aperture")
                    else:
                        ap_mask = 'threshold'
                        st.info("ℹ Using threshold aperture")
                    
                    self.lightcurve = tpf.to_lightcurve(aperture_mask=ap_mask)
                    self.raw_lightcurve = self.lightcurve.copy()
                    
                    self.lightcurve = self.lightcurve.remove_nans()
                    
                    # Show RAW light curve
                    self._plot_raw_lightcurve()
                    
                    # Apply detrending with visualization
                    self._apply_detrending_with_viz()
                    
                    self.lightcurve = self.lightcurve.normalize()
                    
                    self.raw_data = processed_pixels
                    
                    st.success("✅ Light curve extracted from TPF!")
                    self._display_lightcurve_info()
                    self._plot_lightcurve()
                    return True
                    
                except Exception as tpf_error:
                    st.warning(f"TPF extraction failed: {tpf_error}")
                    st.info("Trying alternative method...")
            
            # METHOD 2: Download light curve directly
            st.info("🔄 Searching for pre-processed light curve...")
            
            try:
                # FIXED: Don't specify quarter/sector to get ALL data
                if mission.lower() == "kepler":
                    search_result = lk.search_lightcurve(f"KIC {star_id}", mission="Kepler")
                else:
                    search_result = lk.search_lightcurve(f"TIC {star_id}", mission="TESS")
                
                if len(search_result) == 0:
                    st.error(f"No light curve data found for {star_id}")
                    return False
                
                st.info(f"Found {len(search_result)} light curve file(s)")
                
                # OPTIMIZED: Download limited files for faster/more reliable downloads
                # Check if user wants fast mode or full mode
                use_fast_mode = st.session_state.get('use_fast_mode', True)
                
                if use_fast_mode:
                    max_files = min(4, len(search_result))
                    st.info(f"⚡ Fast Mode: Downloading {max_files}/{len(search_result)} files for quick analysis")
                else:
                    max_files = len(search_result)
                    st.info(f"📥 Full Mode: Downloading ALL {max_files} light curve files (this may take 1-2 minutes)")
                
                # Download with retry on MAST errors
                lc_collection = None
                max_retries = 2
                
                for attempt in range(max_retries):
                    try:
                        if max_files == len(search_result):
                            lc_collection = search_result.download_all()
                        else:
                            # Download limited subset
                            lc_list = []
                            for i in range(max_files):
                                lc = search_result[i].download()
                                if lc:
                                    lc_list.append(lc)
                            if lc_list:
                                from lightkurve.collections import LightCurveCollection
                                lc_collection = LightCurveCollection(lc_list)
                        break  # Success!
                        
                    except Exception as download_err:
                        if "tempdb" in str(download_err).lower() or "transaction" in str(download_err).lower():
                            if attempt < max_retries - 1:
                                st.warning(f"⚠️ NASA MAST server busy (attempt {attempt+1}/{max_retries}). Retrying in 3 seconds...")
                                time.sleep(3)
                            else:
                                st.error("❌ NASA MAST archive is currently overloaded. Please try again in a few minutes.")
                                st.info("💡 This is a NASA server issue, not a problem with your analysis.")
                                return False
                        else:
                            raise download_err
                
                if lc_collection is None or len(lc_collection) == 0:
                    st.error("Failed to download light curve data")
                    return False
                
                # Stitch multiple segments
                if len(lc_collection) > 1:
                    st.info(f"Stitching {len(lc_collection)} light curve segments together")
                    self.lightcurve = lc_collection.stitch()
                else:
                    self.lightcurve = lc_collection[0]
                
                self.raw_lightcurve = self.lightcurve.copy()
                
                # Preprocessing
                st.info("🔧 Preprocessing light curve...")
                self.lightcurve = self.lightcurve.remove_nans()
                self.lightcurve = self.lightcurve.remove_outliers()
                
                # Show RAW
                self._plot_raw_lightcurve()
                
                # Detrending with viz
                self._apply_detrending_with_viz()
                
                self.lightcurve = self.lightcurve.normalize()
                
                self.raw_data = processed_pixels
                
                st.success("✅ Light curve downloaded and processed!")
                self._display_lightcurve_info()
                self._plot_lightcurve()
                return True
                
            except Exception as e:
                st.error(f"Light curve generation failed: {e}")
                return False
        
        except Exception as e:
            st.error(f"❌ Light curve generation failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return False
    
    def _plot_raw_lightcurve(self):
        """Plot RAW light curve before any processing"""
        try:
            if self.raw_lightcurve is None:
                return
            
            st.markdown("### 📈 Raw Light Curve (Before Processing)")
            
            time_vals = self.raw_lightcurve.time.value if hasattr(self.raw_lightcurve.time, 'value') else np.array(self.raw_lightcurve.time)
            flux_vals = self.raw_lightcurve.flux.value if hasattr(self.raw_lightcurve.flux, 'value') else np.array(self.raw_lightcurve.flux)
            
            time_vals = np.asarray(time_vals, dtype=np.float64)
            flux_vals = np.asarray(flux_vals, dtype=np.float64)
            
            # Downsample for plotting
            if len(time_vals) > 20000:
                step = len(time_vals) // 20000
                time_vals = time_vals[::step]
                flux_vals = flux_vals[::step]
            
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.scatter(time_vals, flux_vals, s=0.5, alpha=0.6, color='gray', label='Raw data')
            ax.set_xlabel('Time (days)', fontsize=11, weight='bold')
            ax.set_ylabel('Raw Flux', fontsize=11, weight='bold')
            ax.set_title(f'{self.mission} Light Curve - {self.star_id} (RAW)', fontsize=12, weight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.warning(f"Raw plot failed: {e}")
    
    def _apply_detrending_with_viz(self):
        """Apply detrending and SHOW before/after comparison"""
        
        try:
            st.markdown("### 🔧 Detrending (Removing Stellar Variability)")
            st.info("Removing long-term trends while preserving short-term transit signals...")
            
            # Store before detrending
            flux_before = self.lightcurve.flux.value if hasattr(self.lightcurve.flux, 'value') else np.array(self.lightcurve.flux)
            flux_before = np.asarray(flux_before, dtype=float)
            
            try:
                from scipy.signal import savgol_filter
                
                # Mission-specific detrending parameters
                if self.mission and self.mission.lower() == "tess":
                    # TESS: Shorter window due to 27-day sectors
                    base_window = min(51, len(flux_before) // 20)
                    st.info("🛰️ Using TESS-optimized detrending (shorter window)")
                else:
                    # Kepler: Longer window for 90-day quarters
                    base_window = min(101, len(flux_before) // 10)
                    st.info("🔭 Using Kepler-optimized detrending")
                
                # Calculate window
                window = base_window
                if window < 5:
                    window = 5
                if window % 2 == 0:
                    window += 1
                
                if len(flux_before) > window:
                    st.write(f"  → Using Savitzky-Golay filter (window={window} points)")
                    
                    trend = savgol_filter(flux_before, window, polyorder=2)
                    flux_after = flux_before / trend
                    
                    from lightkurve import LightCurve
                    self.lightcurve = LightCurve(
                        time=self.lightcurve.time,
                        flux=flux_after
                    )
                    
                    # SHOW COMPARISON
                    self._plot_detrending_comparison(flux_before, trend, flux_after)
                    
                    st.success("✅ Stellar variability removed")
                    
                else:
                    st.info("ℹ Insufficient data for Savitzky-Golay")
                    
            except ImportError:
                st.info("ℹ scipy not available - using basic flatten")
                try:
                    self.lightcurve = self.lightcurve.flatten(window_length=101)
                    st.success("✅ Basic detrending applied")
                except:
                    st.info("ℹ Skipping detrending")
                    
        except Exception as e:
            st.warning(f"Detrending failed: {e}")
            st.info("ℹ Continuing without detrending")
    
    def _plot_detrending_comparison(self, flux_before, trend, flux_after):
        """Plot before/after detrending comparison"""
        try:
            time_vals = self.lightcurve.time.value if hasattr(self.lightcurve.time, 'value') else np.array(self.lightcurve.time)
            time_vals = np.asarray(time_vals, dtype=float)
            
            # Downsample
            if len(time_vals) > 10000:
                step = len(time_vals) // 10000
                time_vals = time_vals[::step]
                flux_before = flux_before[::step]
                trend = trend[::step]
                flux_after = flux_after[::step]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
            
            # Before
            ax1.scatter(time_vals, flux_before, s=0.5, alpha=0.5, color='gray', label='Raw')
            ax1.plot(time_vals, trend, color='red', lw=2, label='Detected trend')
            ax1.set_ylabel('Flux', fontsize=11, weight='bold')
            ax1.set_title('Before Detrending: Raw data + Stellar variability trend', fontsize=12, weight='bold')
            ax1.grid(alpha=0.3)
            ax1.legend()
            
            # After
            ax2.scatter(time_vals, flux_after, s=0.5, alpha=0.5, color='blue', label='Detrended')
            ax2.axhline(1.0, color='red', ls='--', alpha=0.5, label='Baseline')
            ax2.set_xlabel('Time (days)', fontsize=11, weight='bold')
            ax2.set_ylabel('Normalized Flux', fontsize=11, weight='bold')
            ax2.set_title('After Detrending: Trend removed, transits preserved', fontsize=12, weight='bold')
            ax2.grid(alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Show noise reduction stats
            std_before = np.std(flux_before)
            std_after = np.std(flux_after)
            improvement = (1 - std_after/std_before) * 100
            
            if improvement > 5:
                st.success(f"✅ Noise reduced by {improvement:.1f}%")
            
        except Exception as e:
            st.warning(f"Detrending plot failed: {e}")
    
    def _display_lightcurve_info(self):
        """Display light curve statistics"""
        try:
            n_points = len(self.lightcurve.time)
            
            if hasattr(self.lightcurve.time, 'value'):
                time_vals = self.lightcurve.time.value
            else:
                time_vals = np.array(self.lightcurve.time)
            
            time_span = float(np.ptp(time_vals))
            
            if hasattr(self.lightcurve.flux, 'value'):
                flux_vals = self.lightcurve.flux.value
            else:
                flux_vals = np.array(self.lightcurve.flux)
            
            flux_median = float(np.median(flux_vals))
            flux_std = float(np.std(flux_vals))
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("*Data Points:*")
                st.write(f"- Total: {n_points:,}")
                st.write(f"- Mission: {self.mission}")
            
            with col2:
                st.write("*Time Coverage:*")
                st.write(f"- Span: {time_span:.1f} days")
                st.write(f"- Years: {time_span/365:.2f}")
            
            with col3:
                st.write("*Data Quality:*")
                st.write(f"- Median flux: {flux_median:.6f}")
                st.write(f"- Noise: {flux_std:.6f}")
            
        except Exception as e:
            st.info(f"Info display: {e}")
    
    def _plot_lightcurve(self):
        """Plot final processed light curve"""
        
        if self.lightcurve is None:
            return
        
        try:
            st.markdown("### 📊 Final Processed Light Curve")
            
            time_vals = self.lightcurve.time.value if hasattr(self.lightcurve.time, 'value') else np.array(self.lightcurve.time)
            flux_vals = self.lightcurve.flux.value if hasattr(self.lightcurve.flux, 'value') else np.array(self.lightcurve.flux)
            
            time_vals = np.asarray(time_vals, dtype=np.float64)
            flux_vals = np.asarray(flux_vals, dtype=np.float64)
            
            # Downsample for plotting
            n_points = len(time_vals)
            if n_points > 15000:
                step = max(1, n_points // 15000)
                time_vals = time_vals[::step]
                flux_vals = flux_vals[::step]
            
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.scatter(time_vals, flux_vals, s=0.5, alpha=0.7, color='blue')
            ax.set_xlabel('Time (days)', fontsize=11, weight='bold')
            ax.set_ylabel('Normalized Flux', fontsize=11, weight='bold')
            ax.set_title(f'{self.mission} Light Curve - {self.star_id} (PROCESSED)', fontsize=12, weight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            flux_mean = np.mean(flux_vals)
            flux_std = np.std(flux_vals)
            ax.axhline(flux_mean, color='red', linestyle='--', alpha=0.5, label=f'Mean: {flux_mean:.6f}')
            ax.axhline(flux_mean + 2*flux_std, color='orange', linestyle=':', alpha=0.5, label='±2σ')
            ax.axhline(flux_mean - 2*flux_std, color='orange', linestyle=':', alpha=0.5)
            ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.warning(f"Plot failed: {e}")
    
    def get_lightcurve(self):
        """Get the generated light curve"""
        return self.lightcurve
    
    def reset(self):
        """Reset the generator state"""
        self.lightcurve = None
        self.raw_lightcurve = None
        self.raw_data = None
        self.star_id = None
        self.mission = None