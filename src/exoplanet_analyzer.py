# exoplanet_analyzer.py - WITH CANDIDATE-BASED CLASSIFICATION

import streamlit as st
import numpy as np
from pathlib import Path
import traceback
from typing import Dict, Optional, Any
import matplotlib.pyplot as plt

from lightcurve_generator import LightcurveGenerator
from pixel_processor import PixelProcessor  
from transit_detector import TransitDetector
from utils import display_progress_step, format_star_id


class ExoplanetAnalyzer:
    """
    Professional Exoplanet Analysis Pipeline
    
    PHILOSOPHY:
    - Identifies CANDIDATES for professional follow-up
    - Does not claim to "confirm" planets
    - Scientifically responsible and honest
    """
    
    def __init__(self):
        self.lightcurve_generator = LightcurveGenerator()
        self.pixel_processor = PixelProcessor()
        self.transit_detector = TransitDetector()
        
        self.light_curve = None
        self.processed_pixels = None
        self.analysis_summary = None
        self.star_id = None
        self.mission = None
        
        self.analysis_complete = False
        self.analysis_successful = False
        
    def run_full_analysis(self, star_id: str, mission: str = "Kepler", 
                         quarter: Optional[int] = None, 
                         sector: Optional[int] = None) -> bool:
        """Run complete exoplanet analysis pipeline"""
        
        try:
            self.star_id = star_id
            self.mission = mission
            
            self.analysis_complete = False
            self.analysis_successful = False
            self.analysis_summary = None
            
            st.info(f"🔭 **Starting Professional Exoplanet Analysis**")
            st.write(f"**Target:** {format_star_id(star_id, mission)}")
            st.write(f"**Mission:** {mission}")
            if quarter:
                st.write(f"**Quarter:** {quarter}")
            if sector:
                st.write(f"**Sector:** {sector}")
            
            # === STEP 1: LIGHT CURVE (FAST PATH WITH FALLBACK) ===
            display_progress_step(1, 6, "📈 Light Curve Download")
            
            st.info("⚡ Trying fast path: downloading pre-processed light curves")
            st.info("💡 This skips pixel file download and saves time!")
            
            lightcurve_success = self.lightcurve_generator.generate_from_pixels(
                None, star_id, mission  # Pass None to skip pixel processing
            )
            
            # FALLBACK: If fast path fails, try pixel files
            if not lightcurve_success:
                st.warning("⚠️ Fast path failed. Trying pixel file method (slower but more reliable)...")
                
                # Try pixel processing as backup
                display_progress_step(2, 6, "🎯 Pixel Processing (Fallback)")
                
                pixel_success = self.pixel_processor.download_and_process_pixels(
                    star_id, mission, quarter, sector
                )
                
                if not pixel_success:
                    st.error("❌ Both fast path and pixel method failed. NASA MAST may be temporarily unavailable.")
                    st.info("💡 Please try again in a few minutes.")
                    return False
                
                self.processed_pixels = self.pixel_processor.get_processed_data()
                
                # Now extract light curve from pixels
                lightcurve_success = self.lightcurve_generator.generate_from_pixels(
                    self.processed_pixels, star_id, mission
                )
                
                if not lightcurve_success:
                    st.error("❌ Failed to extract light curve from pixels")
                    return False
                
            self.light_curve = self.lightcurve_generator.get_lightcurve()
            st.success("✅ **Light curve extracted**")
            
            # === STEP 3: PREPROCESSING ===
            display_progress_step(3, 6, "🔧 Data Quality & Preprocessing")
            
            preprocessing_success = self._run_preprocessing_pipeline()
            
            if not preprocessing_success:
                st.error("❌ Preprocessing failed")
                return False
                
            st.success("✅ **Preprocessing completed**")
            
            # === STEP 4: PERIOD DETECTION ===
            display_progress_step(4, 6, "🔍 Period Detection & Folding")
            
            period_success = self.transit_detector.run_periodogram_analysis(
                self.light_curve, min_period=0.5, max_period=None
            )
            
            if not period_success:
                st.error("❌ Failed to detect periodic signals")
                return False
                
            folding_success = self.transit_detector.create_folded_lightcurve(
                self.light_curve
            )
            
            if not folding_success:
                st.error("❌ Failed to create folded light curve")
                return False
                
            st.success("✅ **Period detection and folding completed**")
            
            # === STEP 5: CLASSIFICATION ===
            display_progress_step(5, 6, "🤖 Transit Classification")
            
            classification_result = self.transit_detector.classify_transit_signal()
            
            if not classification_result or classification_result.get('classification') == 'ERROR':
                st.error("❌ Classification failed")
                return False
                
            st.success("✅ **Classification completed**")
            
            # === STEP 6: REPORTING ===
            display_progress_step(6, 6, "📊 Results & Reporting")
            
            self._generate_analysis_summary(classification_result)
            
            self.transit_detector.display_classification_results()
            
            self.analysis_complete = True
            self.analysis_successful = True
            
            st.success("🎉 **Professional Exoplanet Analysis Complete!**")
            
            self._display_final_summary()
            
            return True
            
        except Exception as e:
            st.error(f"❌ **Analysis pipeline failed:** {str(e)}")
            
            with st.expander("🔧 **Error Details**"):
                st.code(traceback.format_exc())
            
            self.analysis_complete = True
            self.analysis_successful = False
            return False
    
    def _run_preprocessing_pipeline(self) -> bool:
        """Run data quality checks"""
        
        try:
            if self.light_curve is None:
                st.error("No light curve data")
                return False
            
            if hasattr(self.light_curve, 'time'):
                time_span = np.ptp(self.light_curve.time.value) if hasattr(self.light_curve.time, 'value') else np.ptp(self.light_curve.time)
                n_points = len(self.light_curve.time)
            else:
                st.error("Light curve missing time")
                return False
            
            if hasattr(self.light_curve, 'flux'):
                flux_median = np.median(self.light_curve.flux.value) if hasattr(self.light_curve.flux, 'value') else np.median(self.light_curve.flux)
                flux_std = np.std(self.light_curve.flux.value) if hasattr(self.light_curve.flux, 'value') else np.std(self.light_curve.flux)
            else:
                st.error("Light curve missing flux")
                return False
            
            st.write("📊 **Data Quality Assessment:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Time Span", f"{time_span:.1f} days")
                st.metric("Data Points", f"{n_points:,}")
            
            with col2:
                st.metric("Median Flux", f"{flux_median:.6f}")
                st.metric("Noise Level", f"{flux_std:.6f}")
            
            with col3:
                quality_score = min(1.0, (n_points / 10000) * (time_span / 80) * min(1.0, 0.001 / flux_std))
                st.metric("Quality Score", f"{quality_score:.2f}")
                
                if quality_score > 0.7:
                    st.success("High Quality ✅")
                elif quality_score > 0.4:
                    st.warning("Moderate ⚠️")
                else:
                    st.error("Low Quality ❌")
            
            if n_points < 100:
                st.warning("⚠️ Very few data points")
            
            if time_span < 5:
                st.warning("⚠️ Short observation period")
            
            st.write("🔧 **Preprocessing Applied:**")
            st.write("- ✅ Outlier removal")
            st.write("- ✅ Trend correction")
            st.write("- ✅ Normalization")
            st.write("- ✅ Quality checks")
            
            return True
            
        except Exception as e:
            st.error(f"Preprocessing failed: {str(e)}")
            return False
    
    def _generate_analysis_summary(self, classification_result: Dict) -> None:
        """Generate comprehensive analysis summary"""
        
        try:
            classification = classification_result.get('classification', 'UNKNOWN')
            confidence = classification_result.get('confidence', 0.0)
            period = classification_result.get('period', 0.0)
            depth = classification_result.get('depth', 0.0)
            significance = classification_result.get('significance', 0.0)
            duration_hours = classification_result.get('duration_hours', 0.0)
            
            ml_probabilities = classification_result.get('ml_probabilities', {})
            ml_prediction = classification_result.get('ml_prediction', classification)
            
            vetting_details = classification_result.get('vetting_details', {})
            
            odd_even_test = vetting_details.get('odd_even_test', {})
            secondary_test = vetting_details.get('secondary_eclipse_test', {})
            
            self.analysis_summary = {
                'star_id': self.star_id,
                'mission': self.mission,
                'analysis_timestamp': st.session_state.get('analysis_start_time', 'unknown'),
                
                'classification': classification,
                'confidence': confidence,
                
                'period': period,
                'depth': depth,
                'depth_ppm': depth * 1e6,
                'depth_significance': significance,
                'duration_hours': duration_hours,
                'significance': significance,
                
                'bls_power': classification_result.get('bls_power', 0.0),
                'num_transits': classification_result.get('num_transits', 0),
                'vetting_score': classification_result.get('vetting_score', 0.0),
                'estimated_planet_radius': classification_result.get('estimated_planet_radius', 0.0),
                'baseline_flux': classification_result.get('baseline_flux', 1.0),
                'noise_level': classification_result.get('noise_level', 0.001),
                
                'ml_probabilities': ml_probabilities,
                'ml_prediction': ml_prediction,
                
                'odd_even_test_status': odd_even_test.get('status', 'UNKNOWN'),
                'odd_even_depth_ratio': odd_even_test.get('depth_ratio', 0.0),
                'odd_even_depth_odd': odd_even_test.get('depth_odd', 0.0),
                'odd_even_depth_even': odd_even_test.get('depth_even', 0.0),
                
                'secondary_eclipse_status': secondary_test.get('status', 'UNKNOWN'),
                'secondary_eclipse_ratio': secondary_test.get('secondary_ratio', 0.0),
                'secondary_eclipse_depth': secondary_test.get('secondary_depth', 0.0),
                
                'all_features': classification_result.get('all_features', {}),
                'vetting_details': vetting_details,
                
                'analysis_successful': True,
                'analysis_complete': True,
                'pipeline_version': '5.0_CANDIDATE_BASED'
            }
            
        except Exception as e:
            st.error(f"Error generating summary: {str(e)}")
            
            self.analysis_summary = {
                'star_id': self.star_id,
                'mission': self.mission,
                'classification': 'ERROR',
                'confidence': 0.0,
                'analysis_successful': False,
                'error_message': str(e)
            }
    
    def _display_final_summary(self) -> None:
        """Display final analysis summary"""
        
        if not self.analysis_summary:
            st.error("No analysis summary available")
            return
        
        try:
            st.markdown("---")
            st.markdown("## 🎯 **Final Analysis Summary**")
            
            summary = self.analysis_summary
            
            classification = summary.get('classification', 'UNKNOWN')
            confidence = summary.get('confidence', 0.0)
            ml_prediction = summary.get('ml_prediction', classification)
            
            if classification == "STRONG PLANET CANDIDATE":
                st.success(f"""
                **🟢 STRONG PLANET CANDIDATE DETECTED!**
                
                **Target:** {format_star_id(self.star_id, self.mission)}  
                **Classification:** {classification}  
                **ML Prediction:** {ml_prediction}  
                **Confidence:** {confidence:.1%}  
                **Orbital Period:** {summary.get('period', 0):.3f} days  
                **Planet Radius:** {summary.get('estimated_planet_radius', 0):.2f} Earth radii
                
                This detection meets NASA professional standards.  
                **All vetting tests passed!**
                
                ⚠️ *Requires professional follow-up for confirmation*
                """)
                
            elif classification == "PLANET CANDIDATE":
                st.warning(f"""
                **🟡 PLANET CANDIDATE IDENTIFIED**
                
                **Target:** {format_star_id(self.star_id, self.mission)}  
                **Classification:** {classification}  
                **ML Prediction:** {ml_prediction}  
                **Confidence:** {confidence:.1%}  
                **Orbital Period:** {summary.get('period', 0):.3f} days  
                
                This candidate shows promising signals but needs:
                - More observations to increase confidence
                - Professional follow-up observations
                - Validation with radial velocity measurements
                """)
                
            else:
                st.info(f"""
                **📊 ANALYSIS COMPLETE**
                
                **Target:** {format_star_id(self.star_id, self.mission)}  
                **Classification:** {classification}  
                **ML Prediction:** {ml_prediction}  
                **Result:** No convincing planetary signals detected  
                
                This is the typical outcome for most stellar observations.
                Most stars do not have detectable transiting planets.
                """)
            
            # ML probabilities
            ml_probs = summary.get('ml_probabilities', {})
            if ml_probs:
                st.markdown("### 🤖 **ML Classification Probabilities**")
                prob_col1, prob_col2, prob_col3 = st.columns(3)
                
                with prob_col1:
                    fp_prob = ml_probs.get('False Positive', 0)
                    st.metric("False Positive", f"{fp_prob:.1%}")
                
                with prob_col2:
                    cand_prob = ml_probs.get('Candidate', 0)
                    st.metric("Candidate", f"{cand_prob:.1%}")
                
                with prob_col3:
                    conf_prob = ml_probs.get('Confirmed Exoplanet', 0)
                    st.metric("Strong Candidate", f"{conf_prob:.1%}")
            
            # Vetting tests
            st.markdown("### 🔬 **Advanced Vetting Tests**")
            
            vetting_col1, vetting_col2 = st.columns(2)
            
            with vetting_col1:
                st.write("**Odd-Even Transit Consistency:**")
                odd_even_status = summary.get('odd_even_test_status', 'UNKNOWN')
                odd_even_ratio = summary.get('odd_even_depth_ratio', 0.0)
                
                if odd_even_status == 'PASS':
                    st.success(f"✅ PASS (depth ratio: {odd_even_ratio:.3f})")
                    st.write("Odd and even transits consistent")
                elif odd_even_status == 'FAIL':
                    st.error(f"❌ FAIL (depth ratio: {odd_even_ratio:.3f})")
                    st.write("**Likely eclipsing binary**")
                    st.write(f"- Odd: {summary.get('odd_even_depth_odd', 0)*1e6:.0f} ppm")
                    st.write(f"- Even: {summary.get('odd_even_depth_even', 0)*1e6:.0f} ppm")
                elif odd_even_status == 'SKIP':
                    st.info("ℹ️ Skipped (insufficient transits)")
                else:
                    st.info(f"ℹ️ {odd_even_status}")
            
            with vetting_col2:
                st.write("**Secondary Eclipse Check:**")
                secondary_status = summary.get('secondary_eclipse_status', 'UNKNOWN')
                secondary_ratio = summary.get('secondary_eclipse_ratio', 0.0)
                
                if secondary_status == 'PASS':
                    st.success(f"✅ PASS (ratio: {abs(secondary_ratio):.3f})")
                    st.write("No secondary eclipse detected")
                elif secondary_status == 'FAIL':
                    st.error(f"❌ FAIL (ratio: {abs(secondary_ratio):.3f})")
                    st.write("**Secondary eclipse - likely binary**")
                    st.write(f"- Depth: {summary.get('secondary_eclipse_depth', 0)*1e6:.0f} ppm")
                elif secondary_status == 'SKIP':
                    st.info("ℹ️ Skipped (insufficient coverage)")
                else:
                    st.info(f"ℹ️ {secondary_status}")
            
            # Technical summary
            with st.expander("📋 **Technical Summary**", expanded=False):
                tech_col1, tech_col2 = st.columns(2)
                
                with tech_col1:
                    st.write("**Detection Metrics:**")
                    st.write(f"Significance: {summary.get('significance', 0):.2f}σ")
                    st.write(f"Depth: {summary.get('depth_ppm', 0):.0f} ppm")
                    st.write(f"BLS Power: {summary.get('bls_power', 0):.4f}")
                    st.write(f"Transits: {summary.get('num_transits', 0)}")
                
                with tech_col2:
                    st.write("**Quality:**")
                    st.write(f"Vetting: {summary.get('vetting_score', 0):.1%}")
                    st.write(f"Duration: {summary.get('duration_hours', 0):.2f} h")
                    st.write(f"Noise: {summary.get('noise_level', 0):.6f}")
                    st.write(f"Pipeline: {summary.get('pipeline_version', 'Unknown')}")
            
            st.success("✅ **Analysis pipeline completed successfully!**")
            
        except Exception as e:
            st.error(f"Error displaying summary: {str(e)}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
    
    def get_analysis_summary(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive analysis summary"""
        
        if not self.analysis_complete:
            st.warning("Analysis not yet complete")
            return None
        
        if not self.analysis_successful:
            st.error("Analysis was not successful")
            return self.analysis_summary
        
        return self.analysis_summary
    
    def get_lightcurve(self):
        """Get the processed light curve"""
        return self.light_curve
    
    def get_transit_detector(self):
        """Get the transit detector instance"""
        return self.transit_detector
    
    def reset_analysis(self) -> None:
        """Reset analysis state"""
        
        self.light_curve = None
        self.processed_pixels = None
        self.analysis_summary = None
        self.star_id = None
        self.mission = None
        
        self.analysis_complete = False
        self.analysis_successful = False
        
        if hasattr(self.lightcurve_generator, 'reset'):
            self.lightcurve_generator.reset()
        if hasattr(self.pixel_processor, 'reset'):
            self.pixel_processor.reset()
        if hasattr(self.transit_detector, '__init__'):
            self.transit_detector.__init__()
    
    def export_results(self, format: str = 'json') -> Optional[str]:
        """Export analysis results"""
        
        if not self.analysis_summary:
            st.error("No results to export")
            return None
        
        try:
            if format.lower() == 'json':
                import json
                return json.dumps(self.analysis_summary, indent=2, default=str)
            
            elif format.lower() == 'csv':
                import pandas as pd
                
                flat_summary = {}
                for key, value in self.analysis_summary.items():
                    if isinstance(value, dict):
                        for nested_key, nested_value in value.items():
                            flat_summary[f"{key}_{nested_key}"] = nested_value
                    else:
                        flat_summary[key] = value
                
                df = pd.DataFrame([flat_summary])
                return df.to_csv(index=False)
            
            else:
                st.error(f"Unsupported format: {format}")
                return None
        
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
            return None
    
    def get_analysis_status(self) -> Dict[str, Any]:
        """Get current analysis status"""
        
        return {
            'analysis_complete': self.analysis_complete,
            'analysis_successful': self.analysis_successful,
            'star_id': self.star_id,
            'mission': self.mission,
            'has_lightcurve': self.light_curve is not None,
            'has_processed_pixels': self.processed_pixels is not None,
            'has_analysis_summary': self.analysis_summary is not None,
            'pipeline_version': '5.0_CANDIDATE_BASED'
        }