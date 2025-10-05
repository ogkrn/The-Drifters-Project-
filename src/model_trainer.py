# model_trainer.py - SCIENTIFICALLY CORRECTED VERSION

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # FIXED: Use RandomForest, not XGBoost
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os


class ExoplanetMLTrainer:
    """
    Train ML model on scientifically accurate exoplanet features.
    
    FIXES APPLIED:
    - Uses RandomForest (consistent with create_model.py)
    - Significance calculated from physics
    - Realistic BLS power values
    - Duration follows Kepler's third law
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names = [
            'period', 'depth', 'duration_hours', 'depth_significance',
            'baseline_flux', 'noise_level', 'bls_power',
            'period_log', 'depth_ppm', 'duration_ratio', 'snr_ratio'
        ]
    
    def create_realistic_training_data(self):
        """
        Create training data based on REAL exoplanet physics.
        
        SCIENTIFICALLY CORRECT:
        - Duration ∝ P^(1/3)
        - Significance = (depth/noise) × sqrt(N_transits × N_points)
        - BLS power ~ log-normal distribution
        """
        
        st.write("🔄 Creating scientifically accurate training dataset...")
        
        data = []
        
        # ========== CONFIRMED EXOPLANETS ==========
        
        confirmed_examples = [
            {
                'name': 'Hot Jupiters',
                'period_range': (0.5, 10),
                'depth_range': (0.005, 0.03),
                'noise_ppm_range': (50, 150),
                'count': 150
            },
            {
                'name': 'Super-Earths',
                'period_range': (10, 50),
                'depth_range': (0.0005, 0.005),
                'noise_ppm_range': (60, 180),
                'count': 150
            },
            {
                'name': 'Earth-like',
                'period_range': (50, 400),
                'depth_range': (0.0001, 0.001),
                'noise_ppm_range': (50, 150),
                'count': 150
            },
        ]
        
        for example in confirmed_examples:
            for _ in range(example['count']):
                period = np.random.uniform(*example['period_range'])
                depth = np.random.uniform(*example['depth_range'])
                
                # FIXED: Physics-based duration
                stellar_radius = np.random.uniform(0.8, 1.2)
                base_duration = 13.0 * (period / 365.0) ** (1.0/3.0) * stellar_radius
                duration = base_duration * np.random.uniform(0.85, 1.15)
                
                # FIXED: Realistic noise
                noise_ppm = np.random.uniform(*example['noise_ppm_range'])
                noise = noise_ppm / 1e6
                
                # FIXED: Calculate significance from physics
                data_span = 90
                n_transits = max(3, int(data_span / period))
                points_per_transit = max(5, int(duration / 0.5))
                
                significance = (depth / noise) * np.sqrt(n_transits * points_per_transit)
                significance *= np.random.uniform(0.85, 1.15)
                
                # FIXED: Realistic BLS power
                bls_power = np.random.lognormal(np.log(0.08), 0.4)
                bls_power = np.clip(bls_power, 0.03, 0.4)
                
                baseline_flux = np.random.normal(1.0, 0.005)
                
                data.append({
                    'period': period,
                    'depth': depth,
                    'duration_hours': duration,
                    'depth_significance': significance,
                    'baseline_flux': baseline_flux,
                    'noise_level': noise,
                    'bls_power': bls_power,
                    'classification': 2
                })
        
        # ========== PLANET CANDIDATES ==========
        
        for _ in range(300):
            period = np.random.uniform(1, 150)
            depth = np.random.uniform(0.0002, 0.008)
            
            stellar_radius = np.random.uniform(0.7, 1.3)
            base_duration = 13.0 * (period / 365.0) ** (1.0/3.0) * stellar_radius
            duration = base_duration * np.random.uniform(0.8, 1.2)
            
            noise_ppm = np.random.uniform(80, 250)
            noise = noise_ppm / 1e6
            
            data_span = 90
            n_transits = max(2, int(data_span / period))
            points_per_transit = max(5, int(duration / 0.5))
            
            significance = (depth / noise) * np.sqrt(n_transits * points_per_transit)
            significance *= np.random.uniform(0.8, 1.2)
            
            bls_power = np.random.lognormal(np.log(0.03), 0.5)
            bls_power = np.clip(bls_power, 0.015, 0.15)
            
            baseline_flux = np.random.normal(1.0, 0.01)
            
            data.append({
                'period': period,
                'depth': depth,
                'duration_hours': duration,
                'depth_significance': significance,
                'baseline_flux': baseline_flux,
                'noise_level': noise,
                'bls_power': bls_power,
                'classification': 1
            })
        
        # ========== FALSE POSITIVES ==========
        
        false_positive_types = [
            {
                'period_range': (0.2, 5),
                'depth_range': (0.01, 0.15),
                'duration_factor': (1.5, 3.0),
                'noise_range': (50, 150),
                'bls_mean': 0.02,
                'count': 150
            },
            {
                'period_range': (5, 100),
                'depth_range': (0.00001, 0.0008),
                'duration_factor': (0.3, 3.0),
                'noise_range': (100, 400),
                'bls_mean': 0.01,
                'count': 150
            },
            {
                'period_range': (0.1, 200),
                'depth_range': (0.00001, 0.1),
                'duration_factor': (0.1, 5.0),
                'noise_range': (50, 500),
                'bls_mean': 0.005,
                'count': 150
            },
        ]
        
        for fp_type in false_positive_types:
            for _ in range(fp_type['count']):
                period = np.random.uniform(*fp_type['period_range'])
                depth = np.random.uniform(*fp_type['depth_range'])
                
                stellar_radius = np.random.uniform(0.5, 2.0)
                base_duration = 13.0 * (period / 365.0) ** (1.0/3.0) * stellar_radius
                duration = base_duration * np.random.uniform(*fp_type['duration_factor'])
                
                noise_ppm = np.random.uniform(*fp_type['noise_range'])
                noise = noise_ppm / 1e6
                
                data_span = 90
                n_transits = max(1, int(data_span / period))
                points_per_transit = max(3, int(duration / 0.5))
                
                significance = (depth / noise) * np.sqrt(n_transits * points_per_transit)
                significance *= np.random.uniform(0.5, 1.5)
                
                bls_power = np.random.lognormal(np.log(fp_type['bls_mean']), 0.7)
                bls_power = np.clip(bls_power, 0.001, 0.08)
                
                baseline_flux = np.random.normal(1.0, 0.02)
                
                data.append({
                    'period': period,
                    'depth': depth,
                    'duration_hours': duration,
                    'depth_significance': significance,
                    'baseline_flux': baseline_flux,
                    'noise_level': noise,
                    'bls_power': bls_power,
                    'classification': 0
                })
        
        df = pd.DataFrame(data)
        
        st.write(f"**Training Data Generated:** {len(df)} samples")
        st.write(f"- Confirmed: {len(df[df['classification']==2])}")
        st.write(f"- Candidates: {len(df[df['classification']==1])}")
        st.write(f"- False Positives: {len(df[df['classification']==0])}")
        
        return df
    
    def train_and_evaluate(self):
        """Train the model and show comprehensive evaluation"""
        
        # Create training data
        df = self.create_realistic_training_data()
        
        # Feature engineering
        df['period_log'] = np.log10(df['period'])
        df['depth_ppm'] = df['depth'] * 1e6
        df['duration_ratio'] = df['duration_hours'] / (df['period'] * 24)
        df['snr_ratio'] = df['depth_significance'] / np.maximum(df['noise_level'], 1e-10)
        
        # Features and target
        X = df[self.feature_names]
        y = df['classification']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        st.write("🤖 Training Random Forest model...")
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        st.success(f"**Training Accuracy:** {train_score:.3f}")
        st.success(f"**Test Accuracy:** {test_score:.3f}")
        
        # Detailed evaluation
        y_pred = self.model.predict(X_test_scaled)
        
        # Classification report
        report = classification_report(
            y_test, y_pred,
            target_names=['False Positive', 'Candidate', 'Confirmed'],
            output_dict=True
        )
        
        st.write("### 📊 Model Performance")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("False Positive Recall", f"{report['False Positive']['recall']:.3f}")
            st.metric("False Positive Precision", f"{report['False Positive']['precision']:.3f}")
        with col2:
            st.metric("Candidate Recall", f"{report['Candidate']['recall']:.3f}")
            st.metric("Candidate Precision", f"{report['Candidate']['precision']:.3f}")
        with col3:
            st.metric("Confirmed Recall", f"{report['Confirmed']['recall']:.3f}")
            st.metric("Confirmed Precision", f"{report['Confirmed']['precision']:.3f}")
        
        # Confusion Matrix
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['False Pos', 'Candidate', 'Confirmed'],
                   yticklabels=['False Pos', 'Candidate', 'Confirmed'],
                   ax=ax)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        st.pyplot(fig)
        plt.close()
        
        # Feature importance
        st.write("### Feature Importance")
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='importance', y='feature', ax=ax)
        plt.title('Feature Importance for Exoplanet Classification')
        st.pyplot(fig)
        plt.close()
        
        # Display feature importance table
        st.dataframe(feature_importance.style.format({'importance': '{:.4f}'}))
        
        return True
    
    def save_model(self, filepath="models/exoplanet_ml_model.pkl"):
        """Save the trained model"""
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'version': '3.0_SCIENTIFIC',
            'trained_on': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'description': 'Scientifically accurate exoplanet classifier'
        }
        
        joblib.dump(model_package, filepath)
        st.success(f"✅ Model saved to {filepath}")
        
        return filepath


if __name__ == "__main__":
    st.title("Exoplanet ML Model Trainer")
    
    trainer = ExoplanetMLTrainer()
    
    if st.button("Train Model"):
        if trainer.train_and_evaluate():
            trainer.save_model()