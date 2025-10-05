# ml_classifier.py - HYBRID MODEL ONLY (CORRECTED)

import joblib
import numpy as np
import streamlit as st
from typing import Any, Dict, Tuple  # FIXED: Added Any import
import os


class MLExoplanetClassifier:
    """ML-powered exoplanet classifier using hybrid trained model."""
    
    def __init__(self, model_path=None):
        self.model_package = None
        self.model_loaded = False
        
        # Auto-detect model path if not provided
        if model_path is None:
            possible_paths = [
                "src/models/exoplanet_ml_model_hybrid.pkl",  # From root
                "models/exoplanet_ml_model_hybrid.pkl",       # From src
                os.path.join(os.path.dirname(__file__), "models", "exoplanet_ml_model_hybrid.pkl")  # Relative to this file
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            if model_path is None:
                model_path = possible_paths[0]  # Use first as default
        
        self.model_path = model_path
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the hybrid trained ML model"""
        try:
            if not os.path.exists(model_path):
                st.error(f"❌ Hybrid ML model not found at: {model_path}")
                st.error("*Required:* Hybrid model trained with real NASA data + synthetic data")
                st.info("*To create the model, run:*")
                st.code("python train_hybrid_model.py")
                st.warning("⚠ Analysis will continue with rule-based classification only")
                self.model_loaded = False
                return False
            
            st.info(f"📦 Loading hybrid model from: {model_path}")
            self.model_package = joblib.load(model_path)
            
            required_keys = ['model', 'scaler', 'feature_names']
            missing_keys = [k for k in required_keys if k not in self.model_package]
            
            if missing_keys:
                st.error(f"❌ Model file is corrupted or incomplete")
                st.error(f"Missing components: {missing_keys}")
                st.info("*Solution:* Re-train the model:")
                st.code("python train_hybrid_model.py")
                self.model_loaded = False
                return False
            
            self.model_loaded = True
            
            st.success("✅ Hybrid ML Model loaded successfully!")
            
            model_info_col1, model_info_col2 = st.columns(2)
            
            with model_info_col1:
                st.write(f"*Model Version:* {self.model_package.get('version', 'Unknown')}")
                st.write(f"*Trained On:* {self.model_package.get('trained_on', 'Unknown')}")
                # Accuracy removed - not shown to user
            
            with model_info_col2:
                st.write(f"*Real NASA Samples:* {self.model_package.get('real_data_samples', 0):,}")
                st.write(f"*Synthetic Samples:* {self.model_package.get('synthetic_samples', 0):,}")
            
            description = self.model_package.get('description', '')
            if description:
                st.info(f"ℹ {description}")
            
            with st.expander("🔍 Model Feature Information"):
                expected_features = self.model_package.get('feature_names', [])
                st.write("*Expected features (11 total):*")
                for i, feat in enumerate(expected_features, 1):
                    st.write(f"  {i}. {feat}")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading hybrid model: {e}")
            
            with st.expander("🔧 Error Details"):
                import traceback
                st.code(traceback.format_exc())
                
                st.write("*Troubleshooting:*")
                st.write("1. Ensure model was created:")
                st.code("python train_hybrid_model.py")
                st.write("2. Check model file exists:")
                st.code(f"ls -lh {model_path}")
                st.write("3. Verify file is not corrupted (should be 200-500 KB)")
            
            self.model_loaded = False
            return False
    
    def extract_ml_features(self, transit_features: Dict) -> np.ndarray:
        """Extract features for ML model from transit_detector features"""
        
        period = float(transit_features.get('period', 0.0))
        depth = float(transit_features.get('depth', 0.0))
        duration_hours = float(transit_features.get('duration_hours', 0.0))
        depth_significance = float(transit_features.get('significance', 0.0))
        baseline_flux = float(transit_features.get('baseline_flux', 1.0))
        noise_level = float(transit_features.get('noise_level', 0.001))
        bls_power = float(transit_features.get('bls_power', 0.0))
        
        period_log = float(transit_features.get('period_log', np.log10(max(period, 0.1))))
        depth_ppm = float(transit_features.get('depth_ppm', depth * 1e6))
        duration_ratio = float(transit_features.get('duration_ratio', 
                                                    duration_hours / (period * 24) if period > 0 else 0))
        snr_ratio = float(transit_features.get('snr_ratio', 
                                               depth_significance / max(noise_level, 1e-10)))
        
        features = np.array([
            period,
            depth,
            duration_hours,
            depth_significance,
            baseline_flux,
            noise_level,
            bls_power,
            period_log,
            depth_ppm,
            duration_ratio,
            snr_ratio
        ], dtype=np.float64).reshape(1, -1)
        
        if not np.all(np.isfinite(features)):
            st.warning("⚠ Some features are not finite, replacing with defaults")
            features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
        
        return features
    
    def classify_with_ml(self, transit_features: Dict) -> Tuple[str, float, Dict]:
        """Classify exoplanet using hybrid ML model"""
        
        if not self.model_loaded:
            st.warning("⚠ ML model not available - using rule-based classification only")
            return self._rule_based_fallback(transit_features)
        
        try:
            features = self.extract_ml_features(transit_features)
            
            expected_count = len(self.model_package['feature_names'])
            actual_count = features.shape[1]
            
            if actual_count != expected_count:
                st.error(f"❌ Feature count mismatch!")
                st.error(f"   Expected: {expected_count} features")
                st.error(f"   Got: {actual_count} features")
                return self._rule_based_fallback(transit_features)
            
            model = self.model_package['model']
            scaler = self.model_package['scaler']
            
            features_scaled = scaler.transform(features)
            
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            
            labels = {
                0: "FALSE POSITIVE",
                1: "PLANET CANDIDATE",
                2: "CONFIRMED EXOPLANET"
            }
            classification = labels[prediction]
            confidence = float(probabilities[prediction])
            
            prob_details = {
                'False Positive': float(probabilities[0]),
                'Candidate': float(probabilities[1]),
                'Confirmed Exoplanet': float(probabilities[2])
            }
            
            st.markdown("### 🤖 Hybrid ML Model Prediction")
            
            if classification == "CONFIRMED EXOPLANET":
                st.success(f"*Classification:* {classification}")
            elif classification == "PLANET CANDIDATE":
                st.warning(f"*Classification:* {classification}")
            else:
                st.info(f"*Classification:* {classification}")
            
            st.write(f"*ML Confidence:* {confidence:.1%}")
            
            st.write("*Probability Breakdown:*")
            prob_col1, prob_col2, prob_col3 = st.columns(3)
            
            with prob_col1:
                fp_prob = prob_details['False Positive']
                st.metric("False Positive", f"{fp_prob:.1%}")
            
            with prob_col2:
                cand_prob = prob_details['Candidate']
                st.metric("Candidate", f"{cand_prob:.1%}")
            
            with prob_col3:
                conf_prob = prob_details['Confirmed Exoplanet']
                st.metric("Confirmed", f"{conf_prob:.1%}")
            
            with st.expander("🔍 Features Used for ML Prediction"):
                feature_names = self.model_package['feature_names']
                st.write("*Input features (scaled):*")
                
                feature_df_data = []
                for name, value in zip(feature_names, features[0]):
                    feature_df_data.append({
                        'Feature': name,
                        'Value': f"{value:.6f}"
                    })
                
                import pandas as pd
                st.dataframe(pd.DataFrame(feature_df_data), use_container_width=True)
            
            return classification, confidence, prob_details
            
        except Exception as e:
            st.error(f"❌ ML classification error: {e}")
            
            with st.expander("🔧 Error Details"):
                import traceback
                st.code(traceback.format_exc())
            
            return self._rule_based_fallback(transit_features)
    
    def _rule_based_fallback(self, transit_features: Dict) -> Tuple[str, float, Dict]:
        """Rule-based classification when ML model unavailable"""
        
        st.info("ℹ Using rule-based classification (NASA criteria)")
        
        significance = transit_features.get('significance', 0)
        depth = transit_features.get('depth', 0)
        period = transit_features.get('period', 0)
        bls_power = transit_features.get('bls_power', 0)
        vetting_score = transit_features.get('vetting_score', 0.5)
        
        if (significance >= 7.1 and 
            0.00001 <= depth <= 0.05 and 
            0.5 <= period <= 500 and 
            bls_power > 0.05):
            
            classification = "CONFIRMED EXOPLANET"
            confidence = min(0.9, 0.6 + significance/30.0)
            probs = {
                'False Positive': 0.1,
                'Candidate': 0.2,
                'Confirmed Exoplanet': 0.7
            }
            
        elif (significance >= 4.0 and 
              0.00001 <= depth <= 0.1 and 
              0.5 <= period <= 500 and 
              bls_power > 0.02):
            
            classification = "PLANET CANDIDATE"
            confidence = min(0.7, 0.4 + significance/20.0)
            probs = {
                'False Positive': 0.3,
                'Candidate': 0.6,
                'Confirmed Exoplanet': 0.1
            }
            
        else:
            classification = "FALSE POSITIVE"
            confidence = 0.8
            probs = {
                'False Positive': 0.8,
                'Candidate': 0.15,
                'Confirmed Exoplanet': 0.05
            }
        
        st.write(f"*Rule-based result:* {classification} ({confidence:.1%} confidence)")
        
        return classification, confidence, probs
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model"""
        
        if not self.model_loaded or self.model_package is None:
            return {
                'loaded': False,
                'error': 'Model not loaded'
            }
        
        return {
            'loaded': True,
            'path': self.model_path,
            'version': self.model_package.get('version', 'Unknown'),
            'trained_on': self.model_package.get('trained_on', 'Unknown'),
            'description': self.model_package.get('description', ''),
            'train_accuracy': self.model_package.get('train_accuracy', 0),
            'test_accuracy': self.model_package.get('test_accuracy', 0),
            'real_data_samples': self.model_package.get('real_data_samples', 0),
            'synthetic_samples': self.model_package.get('synthetic_samples', 0),
            'total_samples': self.model_package.get('total_samples', 0),
            'feature_names': self.model_package.get('feature_names', [])
        }
    
    def is_model_loaded(self) -> bool:
        """Check if ML model is loaded and ready"""
        return self.model_loaded