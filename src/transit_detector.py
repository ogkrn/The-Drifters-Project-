# transit_detector.py - COMPLETE TLS WITH CANDIDATE-BASED CLASSIFICATION

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from transitleastsquares import transitleastsquares
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from utils import display_progress_step


class TransitDetector:
    """
    NASA-standard transit detector with TLS
    
    CLASSIFICATION PHILOSOPHY:
    - We identify CANDIDATES for professional follow-up
    - Only professional astronomers can confirm planets
    - Honest, scientifically responsible approach
    """

    def __init__(self):
        self.periodogram = None
        self.tls_results = None
        self.best_period = None
        self.max_power = 0.0
        self.sde = 0.0
        self.folded_lc = None
        self.transit_features = {}
        self.vetting_results = {}
        self.classification_result = None
        
        self.original_time = None
        self.original_flux = None
        self.stellar_params = {"radius": 1.0, "mass": 1.0}

    # ========== PERIOD SEARCH (TLS) ==========

    def run_periodogram_analysis(self, lc, min_period=0.3, max_period=None):
        """Run TLS periodogram analysis"""
        try:
            display_progress_step(4, 6, "🔍 Running TLS (Transit Least Squares) analysis...")

            # Extract time and flux
            t = lc.time.value if hasattr(lc.time, "value") else np.array(lc.time)
            f = lc.flux.value if hasattr(lc.flux, "value") else np.array(lc.flux)
            t = np.asarray(t, dtype=float)
            f = np.asarray(f, dtype=float)

            self.original_time = t.copy()
            self.original_flux = f.copy()

            # Clean data
            mask = np.isfinite(t) & np.isfinite(f)
            t, f = t[mask], f[mask]
            
            if t.size < 100:
                st.error(f"❌ Insufficient data: {t.size} points")
                return False

            # OPTIMIZATION: Bin data if too many points (faster TLS)
            if t.size > 50000:
                st.info(f"⚡ Binning {t.size:,} points to ~30,000 for faster computation...")
                bin_size = int(np.ceil(t.size / 30000))
                
                # Bin the data
                n_bins = len(t) // bin_size
                t_binned = np.array([np.mean(t[i*bin_size:(i+1)*bin_size]) for i in range(n_bins)])
                f_binned = np.array([np.mean(f[i*bin_size:(i+1)*bin_size]) for i in range(n_bins)])
                
                # Remove any NaNs from binning
                mask_binned = np.isfinite(t_binned) & np.isfinite(f_binned)
                t, f = t_binned[mask_binned], f_binned[mask_binned]
                
                st.success(f"✅ Binned to {t.size:,} points (preserves transit signals)")

            time_span = t.max() - t.min()
            
            if time_span < 2:
                st.error(f"❌ Data span too short: {time_span:.1f} days")
                return False

            st.info(f"📊 Data span: {time_span:.1f} days, {t.size:,} points")

            # Calculate max period
            if max_period is None:
                if time_span > 365:
                    max_period = min(500, time_span / 2.5)
                    st.info("🌍 Searching for habitable zone planets (up to 500 days)")
                elif time_span > 180:
                    max_period = min(200, time_span / 2.5)
                elif time_span > 90:
                    max_period = min(100, time_span / 2.5)
                else:
                    max_period = time_span / 2.5
            
            max_period = min(max_period, time_span / 2.0)
            
            st.info(f"🔎 Period range: {min_period:.2f} - {max_period:.2f} days")
            st.info("✨ Using TLS: Physically realistic limb-darkened transit shapes")

            # Run TLS
            tls_results = self._run_tls(t, f, min_period, max_period, time_span)
            
            if tls_results is None:
                st.error("❌ TLS failed")
                return False

            self.tls_results = tls_results
            
            # Extract best results
            best_period = float(tls_results.period)
            best_power = float(tls_results.power.max())
            sde = float(tls_results.SDE)

            st.success(f"🔍 TLS Detection:")
            st.write(f"  • Period: {best_period:.5f} days")
            st.write(f"  • SDE: {sde:.2f} (Signal Detection Efficiency)")
            st.write(f"  • Power: {best_power:.4f}")
            
            # Validate TLS depth
            tls_depth = float(tls_results.depth)
            if 0.00001 < tls_depth < 0.5:
                st.write(f"  • TLS Depth: {tls_depth*1e6:.1f} ppm")
            else:
                st.warning(f"  • TLS Depth: {tls_depth*1e6:.1f} ppm (will validate)")
            
            tls_duration = float(tls_results.duration * 24)
            st.write(f"  • Duration: {tls_duration:.2f} hours")

            # Check harmonics
            periods_to_check = tls_results.periods
            powers_to_check = tls_results.power
            
            for mult in [2, 3]:
                test_p = best_period * mult
                if test_p > max_period:
                    continue
                close_idx = np.argmin(np.abs(periods_to_check - test_p))
                if powers_to_check[close_idx] >= 0.7 * best_power:
                    st.warning(f"⚠ Harmonic at {mult}×P detected")

            self.best_period = best_period
            self.max_power = best_power
            self.sde = sde

            # Plot periodogram
            self._plot_periodogram(periods_to_check, powers_to_check, best_period, sde)

            # Store features
            self.transit_features.update({
                'period': self.best_period,
                'bls_power': self.max_power,
                'sde': self.sde,
                'tls_depth_raw': tls_depth,
                'tls_duration': tls_duration,
                'data_span_days': time_span,
                'num_data_points': int(len(self.original_time)),
            })

            st.success(f"✅ Best period: **{self.best_period:.5f} days**")
            
            if best_period < 1:
                st.info("🔥 Ultra-hot Jupiter (P < 1d)")
            elif best_period < 10:
                st.info("🔥 Hot planet (P < 10d)")
            elif best_period < 50:
                st.info("☀ Warm planet (10-50d)")
            else:
                st.info("🌍 Cool/Habitable zone planet (P > 50d)")
            
            return True

        except Exception as e:
            st.error(f"❌ TLS periodogram failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            return False

    def _run_tls(self, t, f, min_period, max_period, time_span):
        """Run TLS computation - OPTIMIZED FOR SPEED"""
        
        st.info("⚡ Running FAST TLS computation...")
        
        try:
            model = transitleastsquares(t, f)
            
            R_star = self.stellar_params.get("radius", 1.0)
            M_star = self.stellar_params.get("mass", 1.0)
            
            # Create progress tracking
            progress_bar = st.progress(0, text="Initializing TLS computation...")
            status_text = st.empty()
            
            # Estimate number of periods to test
            estimated_periods = int((max_period - min_period) / (min_period * 0.01))
            status_text.info(f"🔍 Searching ~{estimated_periods:,} period values from {min_period:.2f} to {max_period:.2f} days")
            
            # OPTIMIZED: Faster parameters for quick analysis
            # - oversampling_factor: 3 → 2 (2-3x faster, still accurate)
            # - duration_grid_step: 1.1 → 1.15 (fewer duration steps)
            # - use_threads: 2 (parallel computation)
            # - show_progress_bar: True (shows in terminal)
            
            import time
            start_time = time.time()
            
            # Show progress during computation (Streamlit UI)
            progress_bar.progress(0.1, text="⚙️ Computing transit models...")
            
            results = model.power(
                period_min=min_period,
                period_max=max_period,
                n_transits_min=2,
                R_star=R_star,
                M_star=M_star,
                oversampling_factor=2,        # Faster: 2 instead of 3
                duration_grid_step=1.15,      # Faster: 1.15 instead of 1.1
                use_threads=2,                # Parallel computation
                show_progress_bar=True        # Show progress bar in terminal
            )
            
            elapsed = time.time() - start_time
            progress_bar.progress(1.0, text=f"✅ TLS complete in {elapsed:.1f}s!")
            status_text.success(f"✅ Searched {len(results.periods):,} periods in {elapsed:.1f} seconds")
            
            if results.period == 0 or not np.isfinite(results.period):
                st.error("❌ TLS did not find a valid period")
                
                # DETAILED DIAGNOSTICS - Show exactly what TLS found
                st.markdown("### 🔍 Detailed Diagnostics")
                st.write("**What TLS actually found:**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    period_val = float(results.period) if hasattr(results, 'period') else 0
                    st.metric("Period", f"{period_val:.6f} days", 
                             help="0 or NaN means no valid period detected")
                
                with col2:
                    if hasattr(results, 'SDE'):
                        sde_val = float(results.SDE)
                        st.metric("SDE", f"{sde_val:.2f}", 
                                 delta=f"{sde_val - 7.1:.1f} below threshold" if sde_val < 7.1 else "OK",
                                 help="Signal Detection Efficiency - need ≥7.1")
                    else:
                        st.metric("SDE", "N/A")
                
                with col3:
                    if hasattr(results, 'power') and len(results.power) > 0:
                        power_val = float(results.power.max())
                        st.metric("Max Power", f"{power_val:.4f}",
                                 help="Highest power in periodogram")
                    else:
                        st.metric("Max Power", "N/A")
                
                st.markdown("---")
                st.info("**Why this failed:**")
                
                reasons = []
                if period_val == 0:
                    reasons.append("❌ **Period = 0 days** - Physically impossible! TLS found no convincing period")
                elif not np.isfinite(period_val):
                    reasons.append("❌ **Period = NaN (Not a Number)** - No valid period found")
                
                if hasattr(results, 'SDE'):
                    if sde_val < 5.0:
                        reasons.append(f"❌ **SDE too low: {sde_val:.2f}** - Need ≥5.0 (marginal) or ≥7.1 (confident)")
                    elif sde_val < 7.1:
                        reasons.append(f"⚠️ **SDE marginal: {sde_val:.2f}** - Below NASA threshold (7.1)")
                
                if hasattr(results, 'depth'):
                    depth_val = float(results.depth)
                    if depth_val <= 0:
                        reasons.append(f"❌ **No transit depth measured** - No dip in brightness detected")
                    elif depth_val > 0.15:
                        reasons.append(f"❌ **Depth too large: {depth_val*100:.1f}%** - Likely eclipsing binary, not planet")
                
                for reason in reasons:
                    st.write(reason)
                
                st.markdown("---")
                st.success("**This is NORMAL! ~99% of stars don't have detectable transiting planets.**")
                st.write("• No convincing periodic transit signal detected")
                st.write("• Most stars don't have planets that cross in front from our viewpoint")
                st.write("• Try confirmed planet examples like KIC 11904151 (Kepler-10b)")
                
                return None
            
            st.success(f"✅ TLS complete - tested {len(results.periods)} periods")
            
            return results
            
        except Exception as e:
            st.error(f"TLS computation error: {e}")
            import traceback
            st.code(traceback.format_exc())
            return None

    def _plot_periodogram(self, periods, powers, best_period, sde):
        """Plot TLS periodogram"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Top: Full periodogram
            ax1.plot(periods, powers, lw=1, alpha=0.8, color='steelblue', label='TLS Power')
            ax1.axvline(best_period, color="red", ls="--", lw=2.5, 
                       label=f"Best: {best_period:.4f} d (SDE={sde:.1f})")
            
            for mult in [2, 3]:
                hp = best_period * mult
                if hp <= periods.max():
                    ax1.axvline(hp, color="orange", ls=":", lw=1.5, alpha=0.6)
            
            ax1.set_xlabel("Period (days)", fontsize=11, weight='bold')
            ax1.set_ylabel("TLS Power", fontsize=11, weight='bold')
            ax1.set_title("Transit Least Squares Periodogram", fontsize=12, weight='bold')
            ax1.grid(alpha=0.3)
            ax1.legend(fontsize=9)
            
            # Bottom: Zoomed
            period_window = 0.2 * best_period
            zoom_mask = (periods > best_period - period_window) & (periods < best_period + period_window)
            
            if np.any(zoom_mask):
                ax2.plot(periods[zoom_mask], powers[zoom_mask], lw=2, color='steelblue')
                ax2.axvline(best_period, color="red", ls="--", lw=2)
                ax2.set_xlabel("Period (days)", fontsize=11, weight='bold')
                ax2.set_ylabel("TLS Power", fontsize=11, weight='bold')
                ax2.set_title(f"Zoomed View (±{period_window:.2f} days)", fontsize=11, weight='bold')
                ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.warning(f"Periodogram plot failed: {e}")

    # ========== FOLDING ==========

    def create_folded_lightcurve(self, lc):
        """Fold and analyze light curve"""
        try:
            display_progress_step(4, 6, "📊 Folding light curve...")
            
            if self.best_period is None:
                st.error("No period set")
                return False

            t = lc.time.value if hasattr(lc.time, "value") else np.array(lc.time)
            f = lc.flux.value if hasattr(lc.flux, "value") else np.array(lc.flux)
            t = np.asarray(t, dtype=float)
            f = np.asarray(f, dtype=float)
            
            m = np.isfinite(t) & np.isfinite(f)
            t, f = t[m], f[m]

            # Get T0 from TLS
            if self.tls_results is not None and hasattr(self.tls_results, 'T0'):
                t0 = float(self.tls_results.T0)
                st.success(f"✅ Using TLS-optimized T0 = {t0:.4f} days")
            else:
                t0 = float(t[0])
                st.info("ℹ Using fallback T0")

            # Fold
            phase = ((t - t0) / self.best_period) % 1.0
            phase = np.where(phase > 0.5, phase - 1.0, phase)
            order = np.argsort(phase)
            phase, f = phase[order], f[order]

            self.folded_lc = {"phase": phase, "flux": f, "t0": t0}

            # Extract features with robust measurement
            self._extract_transit_features_robust(phase, f)
            
            # Vetting
            self._run_vetting_checks()
            self._run_advanced_vetting_tests()
            
            # Plot
            self._plot_folded_lightcurve()

            return True

        except Exception as e:
            st.error(f"❌ Folding failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            return False

    def _extract_transit_features_robust(self, phase, flux):
        """Robust feature extraction with validation"""
        
        if not hasattr(self, 'stellar_params'):
            self.stellar_params = {"radius": 1.0, "mass": 1.0}
        
        st.markdown("### 📊 Measuring Transit Properties")
        
        # Calculate baseline
        oot_mask = np.abs(phase) > 0.25
        
        if np.sum(oot_mask) < 10:
            st.warning("⚠ Few out-of-transit points")
            baseline = float(np.median(flux))
            noise = float(np.std(flux))
        else:
            baseline = float(np.median(flux[oot_mask]))
            noise = float(np.std(flux[oot_mask]))
        
        noise = max(noise, 1e-6)
        
        st.write(f"**Baseline flux:** {baseline:.6f}")
        st.write(f"**Noise level:** {noise*1e6:.1f} ppm")
        
        # Expected duration
        expected_h = self._expected_duration(self.best_period)
        expected_width = (expected_h / 24.0) / self.best_period
        expected_width = min(expected_width, 0.15)
        
        st.write(f"**Expected duration:** {expected_h:.2f} hours (±{expected_width*100:.1f}% phase)")
        
        # MEASURE DEPTH FROM DATA - WITH MINIMUM DATA REQUIREMENTS
        best_depth = 0.0
        best_width = expected_width
        best_n_points = 0
        
        # Try multiple window sizes, from narrow to wide
        # LOWERED MINIMUM: Accept even 1 point if that's all we have
        for width_mult in [0.7, 1.0, 1.4, 2.0, 3.0]:  # Added wider windows
            test_width = expected_width * width_mult
            in_mask = np.abs(phase) < test_width
            n_in = np.sum(in_mask)
            
            # CRITICAL FIX: Accept minimum 1 point (was 3)
            if n_in < 1:
                continue
            
            in_flux = flux[in_mask]
            
            # Use mean for single point, median for multiple
            if n_in == 1:
                test_depth = baseline - in_flux[0]
            else:
                test_depth = baseline - np.median(in_flux)
            
            # Keep the measurement with most points (prefer more data)
            if n_in > best_n_points or (n_in == best_n_points and test_depth > best_depth):
                best_depth = test_depth
                best_width = test_width
                best_n_points = n_in
        
        depth_measured = best_depth / baseline if baseline > 0 else 0.0
        depth_measured = max(0.0, depth_measured)
        
        st.write(f"**Measured from data:**")
        st.write(f"  • In-transit points: {best_n_points}")
        st.write(f"  • Depth: {depth_measured*1e6:.1f} ppm")
        
        # Validate TLS depth - IMPROVED FOR LIMITED DATA
        tls_depth_raw = self.transit_features.get('tls_depth_raw', 0.0)
        use_tls_depth = False
        
        if tls_depth_raw > 0:
            st.write(f"**TLS reported depth:** {tls_depth_raw*1e6:.1f} ppm")
            
            # CRITICAL FIX: If measured depth is 0 or unrealistic, use TLS directly
            if depth_measured <= 0 or depth_measured > 0.2:
                if 0.00001 < tls_depth_raw < 0.1:
                    use_tls_depth = True
                    depth_final = tls_depth_raw
                    st.info(f"✅ Using TLS depth (measured depth was {depth_measured*1e6:.1f} ppm)")
                else:
                    st.warning(f"⚠ TLS depth unrealistic ({tls_depth_raw*1e6:.0f} ppm) - need better data")
                    depth_final = max(depth_measured, tls_depth_raw * 0.0001)  # Use small fraction as estimate
            elif 0.00001 < tls_depth_raw < 0.1:
                # Both measured and TLS are reasonable - compare them
                ratio = tls_depth_raw / max(depth_measured, 1e-6)
                if 0.5 < ratio < 2.0:
                    use_tls_depth = True
                    depth_final = tls_depth_raw
                    st.success(f"✅ TLS depth validated (ratio: {ratio:.2f})")
                else:
                    st.warning(f"⚠ TLS depth differs by {ratio:.1f}x - using measured")
                    depth_final = depth_measured
            else:
                st.warning(f"⚠ TLS depth unrealistic - using measured")
                depth_final = depth_measured
        else:
            depth_final = depth_measured
        
        st.write(f"**Final depth:** {depth_final*1e6:.1f} ppm")
        
        # Calculate significance - IMPROVED FOR LIMITED DATA
        if best_n_points > 0:
            # Normal SNR calculation with measured points
            significance = (depth_final * baseline) / (noise / np.sqrt(best_n_points))
        elif self.sde > 0:
            # FALLBACK: Use SDE-based SNR estimate when no in-transit points
            # SDE and SNR are related: SDE ≈ SNR for strong signals
            # Conservative estimate: SNR ≈ SDE / 1.5
            significance = self.sde / 1.5
            st.info(f"⚠️ Using SDE-based SNR estimate (no in-transit points measured)")
        else:
            significance = 0.0
        
        st.write(f"**Significance:** {significance:.2f}σ")
        
        # Duration
        if use_tls_depth and hasattr(self.tls_results, 'duration'):
            duration_h = float(self.tls_results.duration * 24)
            st.write(f"**Duration:** {duration_h:.2f}h (from TLS)")
        else:
            if depth_final > 3 * noise and best_n_points >= 5:
                threshold = baseline - 0.3 * (depth_final * baseline)
                below_mask = flux < threshold
                if np.sum(below_mask) >= 2:
                    phase_range = phase[below_mask].max() - phase[below_mask].min()
                    duration_h = phase_range * self.best_period * 24.0
                else:
                    duration_h = expected_h
            else:
                duration_h = expected_h
            st.write(f"**Duration:** {duration_h:.2f}h (measured)")
        
        duration_h = min(duration_h, 0.3 * self.best_period * 24.0)
        
        # Number of transits
        span = self.transit_features.get("data_span_days", 90.0)
        num_tr = max(1, int(span / max(self.best_period, 1e-6)))
        
        # Update features
        self.transit_features.update({
            'depth': depth_final,
            'depth_ppm': depth_final * 1e6,
            'depth_flux': depth_final * baseline,
            'depth_measured': depth_measured,
            'depth_tls': tls_depth_raw,
            'used_tls_depth': use_tls_depth,
            'duration_hours': duration_h,
            'significance': significance,
            'snr': significance,
            'num_transits': num_tr,
            'baseline_flux': baseline,
            'noise_level': noise,
            'expected_duration': expected_h,
            'n_in_transit_points': best_n_points,
            'transit_width_phase': best_width,
            'period_log': np.log10(max(self.best_period, 0.1)),
            'duration_ratio': duration_h / (self.best_period * 24) if self.best_period > 0 else 0,
            'snr_ratio': significance / max(noise, 1e-10),
        })
        
        st.success(f"✅ Features extracted - {num_tr} transits detected")

    def _expected_duration(self, period_days):
        """Calculate expected transit duration"""
        r_star = self.stellar_params.get("radius", 1.0)
        dur = 13.0 * (max(period_days, 1e-6) / 365.25) ** (1/3) * max(r_star, 0.1)
        return float(np.clip(dur, 0.5, 36.0))

    # ========== VETTING ==========

    def _run_vetting_checks(self):
        """Basic vetting"""
        
        f = self.transit_features
        sig = f.get("significance", 0.0)
        depth = f.get("depth", 0.0)
        period = f.get("period", 0.0)
        dur_cons = abs(f.get("duration_hours", 0) - f.get("expected_duration", 1)) / max(f.get("expected_duration", 1), 1)

        score_sig = np.clip(sig / 7.1, 0, 1)
        score_depth = 1.0 if 1e-5 <= depth <= 0.03 else 0.6
        score_period = 1.0 if 0.5 <= period <= 500 else 0.5
        score_dur = np.clip(1.0 - min(dur_cons, 2.0)/2.0, 0, 1)
        
        if self.sde > 0:
            score_sde = min(1.0, self.sde / 7.0)
        else:
            score_sde = 0.5

        overall = float(np.mean([score_sig, score_depth, score_period, score_dur, score_sde]))
        
        self.vetting_results = {
            'overall_score': overall,
            'significance_score': score_sig,
            'depth_score': score_depth,
            'period_score': score_period,
            'duration_score': score_dur,
            'sde_score': score_sde,
            'passes_vetting': overall >= 0.5,
        }

        st.markdown("### 🔬 Vetting Assessment")
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.write("**Signal**")
            st.write(f"SNR: {sig:.2f}σ")
            if self.sde > 0:
                st.write(f"SDE: {self.sde:.1f}")
        
        with c2:
            st.write("**Depth**")
            st.write(f"{score_depth:.0%}")
        
        with c3:
            st.write("**Duration**")
            st.write(f"{score_dur:.0%}")
        
        with c4:
            st.write("**Overall**")
            st.metric("Score", f"{overall:.0%}")

    def _run_advanced_vetting_tests(self):
        """Advanced vetting"""
        
        st.markdown("### 🔬 Advanced Vetting Tests")
        
        oe = self._check_odd_even_robust()
        sec = self._check_secondary()
        
        self.vetting_results['odd_even_test'] = oe
        self.vetting_results['secondary_eclipse_test'] = sec
        
        if oe['status'] == 'FAIL' or sec['status'] == 'FAIL':
            oe_ratio = oe.get('depth_ratio', 1.0)
            sec_ratio = abs(sec.get('secondary_ratio', 0.0))
            
            if oe_ratio < 0.5 or sec_ratio > 0.2:
                self.vetting_results['overall_score'] *= 0.3
                self.vetting_results['passes_vetting'] = False
                st.error("❌ Clear eclipsing binary signature")
            else:
                self.vetting_results['overall_score'] *= 0.7
                st.warning("⚠ Marginal vetting failure")

    def _check_odd_even_robust(self):
        """Improved odd-even test"""
        try:
            if self.folded_lc is None:
                return {'status': 'SKIP', 'reason': 'No data'}
            
            total_sig = self.transit_features.get('significance', 0)
            if total_sig < 5:
                st.info("ℹ️ Odd-Even: Low total SNR, skipping")
                return {'status': 'SKIP', 'reason': 'Low SNR'}
            
            t = self.original_time
            f = self.original_flux
            t0 = self.folded_lc['t0']
            P = self.best_period
            
            width = self.transit_features.get('transit_width_phase', 0.05)
            
            phase = ((t - t0) / P) % 1.0
            phase = np.where(phase > 0.5, phase - 1.0, phase)
            
            cycle = np.floor((t - t0) / P)
            
            odd_mask = (cycle % 2 == 0) & (np.abs(phase) < width)
            even_mask = (cycle % 2 == 1) & (np.abs(phase) < width)
            oot_mask = np.abs(phase) > 0.25
            
            n_odd = np.sum(odd_mask)
            n_even = np.sum(even_mask)
            
            st.write(f"**Odd-Even Test:**")
            st.write(f"  • Odd transit points: {n_odd}")
            st.write(f"  • Even transit points: {n_even}")
            
            if n_odd < 5 or n_even < 5:
                st.info("ℹ️ Odd-Even: Insufficient points per transit")
                return {'status': 'SKIP', 'reason': 'Insufficient points'}
            
            base = np.median(f[oot_mask])
            noise = np.std(f[oot_mask])
            
            odd_flux = np.median(f[odd_mask])
            even_flux = np.median(f[even_mask])
            
            d_odd = base - odd_flux
            d_even = base - even_flux
            
            st.write(f"  • Odd depth: {d_odd*1e6:.1f} ppm")
            st.write(f"  • Even depth: {d_even*1e6:.1f} ppm")
            
            sig_odd = d_odd / (noise / np.sqrt(n_odd))
            sig_even = d_even / (noise / np.sqrt(n_even))
            
            st.write(f"  • Odd significance: {sig_odd:.1f}σ")
            st.write(f"  • Even significance: {sig_even:.1f}σ")
            
            if sig_odd < 2 and sig_even < 2:
                st.info("ℹ️ Odd-Even: Both transits below 2σ")
                return {'status': 'SKIP', 'reason': 'Both weak'}
            
            if d_odd > 0 and d_even > 0:
                ratio = min(d_odd, d_even) / max(d_odd, d_even)
            elif d_odd > 0 or d_even > 0:
                ratio = 0.0
            else:
                ratio = 1.0
            
            st.write(f"  • Depth ratio: {ratio:.3f}")
            
            if ratio > 0.5:
                st.success(f"✅ Odd-Even: PASS (ratio={ratio:.2f})")
                status = 'PASS'
            elif ratio > 0.3:
                st.warning(f"⚠ Odd-Even: MARGINAL (ratio={ratio:.2f})")
                status = 'MARGINAL'
            else:
                st.error(f"❌ Odd-Even: FAIL (ratio={ratio:.2f})")
                status = 'FAIL'
            
            return {
                'status': status,
                'depth_ratio': ratio,
                'depth_odd': d_odd,
                'depth_even': d_even,
                'sig_odd': sig_odd,
                'sig_even': sig_even
            }
            
        except Exception as e:
            st.warning(f"Odd-even test error: {e}")
            import traceback
            st.code(traceback.format_exc())
            return {'status': 'ERROR', 'error': str(e)}

    def _check_secondary(self):
        """Secondary eclipse test"""
        try:
            if self.folded_lc is None:
                return {'status': 'SKIP', 'reason': 'No data'}
            
            if self.transit_features.get('significance', 0) < 3:
                st.info("ℹ️ Secondary: Low SNR, skipping")
                return {'status': 'SKIP', 'reason': 'Low SNR'}
            
            phase = self.folded_lc['phase']
            flux = self.folded_lc['flux']
            
            sec_mask = (np.abs(phase - 0.5) < 0.05) | (np.abs(phase + 0.5) < 0.05)
            base_mask = (np.abs(phase) > 0.2) & (np.abs(phase - 0.5) > 0.2)
            
            if np.sum(sec_mask) < 5:
                st.info("ℹ️ Secondary: Insufficient coverage")
                return {'status': 'SKIP', 'reason': 'Insufficient'}
            
            base = np.median(flux[base_mask])
            sec_flux = np.median(flux[sec_mask])
            
            sec_depth = base - sec_flux
            pri_depth = self.transit_features.get('depth_flux', 0.001)
            
            ratio = sec_depth / pri_depth if pri_depth > 0 else 0.0
            
            st.write(f"**Secondary Eclipse:**")
            st.write(f"  • Secondary depth: {sec_depth*1e6:.1f} ppm")
            st.write(f"  • Ratio to primary: {ratio:.3f}")
            
            if abs(ratio) < 0.15:
                st.success(f"✅ Secondary: PASS")
                status = 'PASS'
            else:
                st.error(f"❌ Secondary: FAIL")
                status = 'FAIL'
            
            return {
                'status': status,
                'secondary_ratio': ratio,
                'secondary_depth': sec_depth
            }
            
        except Exception as e:
            st.warning(f"Secondary test error: {e}")
            return {'status': 'ERROR', 'error': str(e)}

    # ========== CLASSIFICATION (CANDIDATE-BASED) ==========

    def classify_transit_signal(self):
        """
        Candidate-based classification
        
        We identify CANDIDATES for professional follow-up.
        Only professional astronomers can confirm planets.
        """
        try:
            if not self.transit_features:
                return {"classification": "ERROR", "confidence": 0.0}

            f = self.transit_features
            period = f.get("period", 0.0)
            depth = f.get("depth", 0.0)
            sig = f.get("significance", 0.0)
            dur = f.get("duration_hours", 0.0)
            power = f.get("bls_power", 0.0)
            ntr = f.get("num_transits", 0)
            vet = self.vetting_results.get("overall_score", 0.0)
            
            # CRITICAL FIX: Use TLS depth if measured depth is 0 or unrealistic
            tls_depth = f.get("tls_depth_raw", 0.0)
            if depth <= 0 and tls_depth > 0:
                st.info(f"⚠️ Using TLS-reported depth ({tls_depth*1e6:.0f} ppm) - measured depth was {depth*1e6:.1f} ppm")
                depth = tls_depth
            elif depth > 0.5:  # Unrealistically deep
                if 0 < tls_depth < 0.5:
                    st.info(f"⚠️ Using TLS depth ({tls_depth*1e6:.0f} ppm) - measured depth was unrealistic")
                    depth = tls_depth

            oe_status = self.vetting_results.get('odd_even_test', {}).get('status', 'UNKNOWN')
            sec_status = self.vetting_results.get('secondary_eclipse_test', {}).get('status', 'UNKNOWN')
            
            oe_pass = oe_status != 'FAIL'
            sec_pass = sec_status != 'FAIL'

            st.markdown("### 🎯 Transit Signal Classification")
            st.info("ℹ️ We identify **candidates** for professional follow-up. Only astronomers with follow-up observations can confirm planets.")
            st.write(f"Period: {period:.5f}d | Depth: {depth*1e6:.0f}ppm | SNR: {sig:.2f}σ")
            st.write(f"SDE: {self.sde:.1f} | Transits: {ntr} | Vetting: {vet:.0%}")

            # ========== NASA-BASED CLASSIFICATION CRITERIA ==========
            # Based on Kepler/TESS dispositions and literature:
            # - SDE ≥ 7.1: High-confidence detection (Kepler threshold)
            # - Period: 0.3-500 days (physical range)
            # - Depth: 20 ppm - 5% (Earth to Jupiter-size)
            # - Duration: 1-12 hours (typical)
            
            # Check for eclipsing binary (too deep or secondary eclipse)
            is_binary = False
            if depth > 0.05:  # >5% depth
                is_binary = True
                reasons = [f"Extremely deep transit: {depth*100:.1f}% → Likely eclipsing binary"]
            elif not sec_pass:
                is_binary = True  
                reasons = ["Secondary eclipse detected → Eclipsing binary star system"]
            elif not oe_pass and sig > 5:
                is_binary = True
                reasons = ["Failed odd-even consistency → Likely eclipsing binary"]
            
            # ========== STRONG PLANET CANDIDATE ==========
            # NASA Kepler PC threshold: SDE ≥ 7.1
            if (not is_binary and
                self.sde >= 7.1 and
                20e-6 <= depth <= 0.05 and
                0.3 <= period <= 500 and
                1.0 <= dur <= 12.0 and
                ntr >= 2):
                
                label = "STRONG PLANET CANDIDATE"
                conf = min(0.95, 0.7 + (self.sde - 7.1)/50.0)
                
                st.success("### 🟢 STRONG PLANET CANDIDATE")
                st.write("**High-confidence detection - meets NASA Kepler PC threshold (SDE ≥ 7.1)**")
                st.info("✨ This signal meets professional detection criteria")
                
                st.write("**Strong indicators:**")
                st.write(f"  ✅ SDE: {self.sde:.1f} (NASA threshold: ≥7.1)")
                st.write(f"  ✅ Signal-to-noise: {sig:.1f}σ")
                st.write(f"  ✅ Physical depth: {depth*1e6:.0f} ppm ({depth*1e6/84:.1f}× Earth)")
                st.write(f"  ✅ {ntr} transits detected")
                st.write(f"  ✅ Duration: {dur:.2f} hours (physical range)")
                
                st.warning("⚠️ **Requires professional confirmation** via spectroscopy and radial velocity")
            
            # ========== PLANET CANDIDATE ==========
            # Marginal detection: SDE 5.0-7.1
            elif (not is_binary and
                  self.sde >= 5.0 and
                  20e-6 <= depth <= 0.05 and
                  0.3 <= period <= 500 and
                  0.5 <= dur <= 15.0 and
                  ntr >= 2):
                
                label = "PLANET CANDIDATE"
                conf = 0.4 + (self.sde - 5.0) / 10.0
                
                st.warning("### 🟡 PLANET CANDIDATE")
                st.write("**Marginal detection - below NASA PC threshold but detectable**")
                
                st.write("**Positive indicators:**")
                st.write(f"  📊 SDE: {self.sde:.1f} (threshold: ≥7.1)")
                st.write(f"  📊 Depth: {depth*1e6:.0f} ppm (physical)")
                st.write(f"  📊 {ntr} transits detected")
                st.write(f"  📊 Duration: {dur:.2f} hours")
                
                needs_improvement = []
                if self.sde < 7.1:
                    needs_improvement.append(f"SDE {self.sde:.1f} below NASA threshold (7.1)")
                if ntr < 3:
                    needs_improvement.append(f"Only {ntr} transits (more observations needed)")
                
                if needs_improvement:
                    st.write("**To strengthen this candidate:**")
                    for item in needs_improvement:
                        st.write(f"  � {item}")
                    st.info("💡 Longer observation baseline recommended")
                
                st.warning("⚠️ **Requires professional follow-up** to validate")
            
            # ========== FALSE POSITIVE ==========
            else:
                label = "FALSE POSITIVE"
                conf = 0.25
                
                st.error("### ❌ FALSE POSITIVE")
                st.write("**No convincing planetary signal detected**")
                
                reasons = []
                if is_binary:
                    # Already set reasons above
                    pass
                else:
                    if self.sde < 5.0:
                        reasons.append(f"Low SDE: {self.sde:.1f} (NASA minimum: 5.0 marginal, 7.1 confident)")
                    if depth < 20e-6:
                        reasons.append(f"Too shallow: {depth*1e6:.1f} ppm (minimum: 20 ppm for detection)")
                    if depth > 0.05 and depth <= 0.15:
                        reasons.append(f"Unusually deep: {depth*100:.2f}% (check for binaries)")
                    if period < 0.3:
                        reasons.append(f"Unrealistic period: {period:.3f} days (too short)")
                    if period > 500:
                        reasons.append(f"Unrealistic period: {period:.1f} days (beyond search range)")
                    if dur < 0.5:
                        reasons.append(f"Too short: {dur:.2f} hours (non-physical)")
                    if dur > 15:
                        reasons.append(f"Too long: {dur:.2f} hours (non-physical)")
                    if ntr < 2:
                        reasons.append(f"Insufficient transits: {ntr} (minimum: 2)")
                
                if reasons:
                    st.write("**Reasons for rejection:**")
                    for r in reasons:
                        st.write(f"  • {r}")
                
                st.info("ℹ️ This is typical - most stars don't have detectable transiting planets")

            # ML classification
            ml_class = None
            ml_conf = None
            ml_probs = None
            
            try:
                from ml_classifier import MLExoplanetClassifier
                ml_clf = MLExoplanetClassifier()
                if ml_clf.model_loaded:
                    ml_class, ml_conf, ml_probs = ml_clf.classify_with_ml(self.transit_features)
                    
                    # Remap to candidate system
                    if ml_class == "CONFIRMED EXOPLANET":
                        ml_class = "STRONG PLANET CANDIDATE"
                    elif ml_class == "CANDIDATE":
                        ml_class = "PLANET CANDIDATE"
                    
                    # CRITICAL: BIDIRECTIONAL ML MODEL OVERRIDE
                    # Trust the ML model (trained on 3,200 real transits) when it strongly disagrees
                    
                    if ml_probs:
                        fp_prob = ml_probs.get("False Positive", 0)
                        candidate_prob = ml_probs.get("Candidate", 0) + ml_probs.get("Strong Candidate", 0)
                        
                        # CASE 1: ML says FALSE POSITIVE but we said CANDIDATE
                        if (label in ["STRONG PLANET CANDIDATE", "PLANET CANDIDATE"]) and fp_prob > 0.40:
                            st.warning("### 🤖 ML Model Override - DOWNGRADE")
                            st.write(f"**ML Confidence:** {fp_prob*100:.1f}% FALSE POSITIVE")
                            st.write("**Rule-based metrics look good, but ML model trained on 3,200 real transits flags this as likely false positive**")
                            
                            label = "FALSE POSITIVE"
                            conf = fp_prob
                            st.error("⬇️ Classification downgraded to: **FALSE POSITIVE**")
                            st.info("💡 **Reason:** ML detects patterns inconsistent with real planetary transits")
                        
                        # CASE 2: ML says CANDIDATE but we said FALSE POSITIVE  
                        elif label == "FALSE POSITIVE" and candidate_prob > 0.50:
                            st.warning("### 🤖 ML Model Override - UPGRADE")
                            st.write(f"**ML Confidence:** {candidate_prob*100:.1f}% CANDIDATE")
                            st.write("**Rule-based vetting flagged issues, but ML model trained on 3,200 real transits sees planetary signature**")
                            
                            # Upgrade to candidate
                            if candidate_prob > 0.70:
                                label = "PLANET CANDIDATE"
                                conf = candidate_prob
                                st.success("⬆️ Classification upgraded to: **PLANET CANDIDATE**")
                            else:
                                label = "PLANET CANDIDATE"
                                conf = candidate_prob
                                st.warning("⬆️ Classification upgraded to: **PLANET CANDIDATE** (marginal)")
                            
                            st.info("💡 **Recommendation:** This deserves closer examination despite vetting flags")
                            
            except Exception as e:
                st.warning(f"ML classification error: {e}")
                pass

            probs = ml_probs if ml_probs else self._pseudo_probs(sig, depth, vet, oe_pass, sec_pass)

            result = {
                'classification': label,
                'confidence': conf,
                'period': period,
                'depth': depth,
                'depth_ppm': depth * 1e6,
                'significance': sig,
                'duration_hours': dur,
                'bls_power': power,
                'sde': self.sde,
                'num_transits': ntr,
                'vetting_score': vet,
                'estimated_planet_radius': self._estimate_radius(depth),
                'baseline_flux': f.get("baseline_flux", 1.0),
                'noise_level': f.get("noise_level", 0.001),
                'all_features': self.transit_features,
                'vetting_details': self.vetting_results,
                'ml_probabilities': probs,
                'ml_prediction': ml_class if ml_class else label,
                'ml_confidence': ml_conf if ml_conf else conf,
            }
            
            self.classification_result = result
            return result

        except Exception as e:
            st.error(f"❌ Classification failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            return {"classification": "ERROR", "confidence": 0.0}

    # ========== PLOTTING ==========

    def _plot_folded_lightcurve(self):
        """Plot folded light curve"""
        try:
            if not self.folded_lc:
                return

            phase = self.folded_lc["phase"]
            flux = self.folded_lc["flux"]

            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.scatter(phase, flux, s=1, alpha=0.3, color='gray', label='Data', zorder=1)
            
            bins = np.linspace(-0.5, 0.5, 200)
            idx = np.digitize(phase, bins)
            med = [np.median(flux[idx == i]) if np.any(idx == i) else np.nan for i in range(len(bins))]
            ax.plot(bins, med, lw=2.5, color='blue', label="Binned", zorder=2)
            
            # TLS model
            if self.tls_results is not None and hasattr(self.tls_results, 'model_lightcurve_time'):
                try:
                    model_phase = ((self.tls_results.model_lightcurve_time - self.folded_lc['t0']) / self.best_period) % 1.0
                    model_phase = np.where(model_phase > 0.5, model_phase - 1.0, model_phase)
                    model_flux = self.tls_results.model_lightcurve_model
                    
                    sort_idx = np.argsort(model_phase)
                    ax.plot(model_phase[sort_idx], model_flux[sort_idx], 
                           lw=3, color='red', alpha=0.7, label='TLS Model', zorder=3)
                except:
                    pass
            
            ax.set_xlabel("Phase", fontsize=12, weight='bold')
            ax.set_ylabel("Normalized Flux", fontsize=12, weight='bold')
            ax.set_title(f"Folded Light Curve (P={self.best_period:.5f}d)", fontsize=13, weight='bold')
            ax.set_xlim(-0.5, 0.5)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.warning(f"Plot failed: {e}")

    def display_classification_results(self):
        """Display classification results - REMOVED, ML prediction is shown instead"""
        # Classification Summary removed - ML model prediction is the final classification
        # This function is kept for backward compatibility but does nothing
        pass

    def _estimate_radius(self, depth):
        """Estimate planet radius"""
        try:
            r_star = self.stellar_params.get("radius", 1.0)
            corrected_depth = depth * 1.1
            rp = np.sqrt(max(corrected_depth, 1e-6)) * r_star * 109.2
            return float(np.clip(rp, 0.1, 50.0))
        except:
            return 1.0

    def _pseudo_probs(self, sig, depth, vet, oe, sec):
        """Pseudo probabilities"""
        fp, conf, cand = 0.8, 0.1, 0.1
        
        if sig >= 7.1:
            conf *= 10; fp *= 0.15
        elif sig >= 4:
            cand *= 5; fp *= 0.6
        
        if 1e-5 <= depth <= 0.03:
            conf *= 2; fp *= 0.5
        
        if vet > 0.5:
            conf *= 2
        
        if not oe or not sec:
            fp *= 5; conf *= 0.05
        
        tot = fp + conf + cand
        return {
            "False Positive": fp/tot,
            "Confirmed Exoplanet": conf/tot,
            "Candidate": cand/tot,
        }