# create_model.py - SCIENTIFICALLY CORRECTED VERSION

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os


def create_pretrained_model():
    """
    Create a scientifically accurate pre-trained exoplanet classification model.
    
    FIXES APPLIED:
    - Significance calculated from physics (not random)
    - Realistic BLS power values (log-normal distribution)
    - Duration follows Kepler's third law
    - Proper noise models for Kepler/TESS
    """
    
    print("=" * 60)
    print("Creating Scientifically Accurate Exoplanet Classification Model")
    print("=" * 60)
    
    data = []
    
    # ========== CONFIRMED EXOPLANETS ==========
    # Based on actual Kepler/TESS discoveries
    
    confirmed_examples = [
        # Hot Jupiters (short period, deep transit, strong signal)
        {
            'name': 'Hot Jupiters',
            'period_range': (0.5, 10),
            'depth_range': (0.005, 0.03),  # 0.5% to 3%
            'noise_ppm_range': (50, 150),
            'count': 200
        },
        # Super-Earths (medium period, moderate transit)
        {
            'name': 'Super-Earths',
            'period_range': (10, 50),
            'depth_range': (0.0005, 0.005),  # 500 to 5000 ppm
            'noise_ppm_range': (60, 180),
            'count': 200
        },
        # Earth-like (longer period, shallow transit)
        {
            'name': 'Earth-like',
            'period_range': (50, 400),
            'depth_range': (0.0001, 0.001),  # 100 to 1000 ppm
            'noise_ppm_range': (50, 150),
            'count': 200
        },
    ]
    
    print("\nGenerating CONFIRMED EXOPLANET training data...")
    
    for example in confirmed_examples:
        print(f"  - {example['name']}: {example['count']} samples")
        
        for _ in range(example['count']):
            period = np.random.uniform(*example['period_range'])
            depth = np.random.uniform(*example['depth_range'])
            
            # FIXED: Duration from Kepler's third law
            # T_dur ≈ 13h × (P/365d)^(1/3) × (R*/R_sun)
            stellar_radius = np.random.uniform(0.8, 1.2)  # Solar radii
            base_duration = 13.0 * (period / 365.0) ** (1.0/3.0) * stellar_radius
            duration = base_duration * np.random.uniform(0.85, 1.15)  # ±15% scatter
            
            # FIXED: Realistic noise for Kepler
            noise_ppm = np.random.uniform(*example['noise_ppm_range'])
            noise = noise_ppm / 1e6
            
            # FIXED: Calculate significance from PHYSICS
            data_span_days = 90  # Typical Kepler quarter
            n_transits = max(3, int(data_span_days / period))
            
            # Kepler cadence: 30 minutes = 0.5 hours
            cadence_hours = 0.5
            points_per_transit = int((duration / cadence_hours))
            points_per_transit = max(5, points_per_transit)
            
            # Signal-to-noise ratio calculation
            # SNR = (depth / noise) × sqrt(N_transits × N_points_per_transit)
            significance = (depth / noise) * np.sqrt(n_transits * points_per_transit)
            
            # Add realistic scatter (±15%)
            significance *= np.random.uniform(0.85, 1.15)
            
            # FIXED: Realistic BLS power (log-normal distribution)
            # Confirmed planets: mean ~0.08, range 0.03-0.25
            bls_power = np.random.lognormal(np.log(0.08), 0.4)
            bls_power = np.clip(bls_power, 0.03, 0.4)
            
            # Baseline flux near 1.0 (normalized), stable
            baseline_flux = np.random.normal(1.0, 0.005)
            
            data.append({
                'period': period,
                'depth': depth,
                'duration_hours': duration,
                'depth_significance': significance,
                'baseline_flux': baseline_flux,
                'noise_level': noise,
                'bls_power': bls_power,
                'classification': 2  # Confirmed
            })
    
    print(f"Total confirmed exoplanets: {len(data)}")
    
    # ========== PLANET CANDIDATES ==========
    # Weaker signals, need follow-up
    
    print("\nGenerating PLANET CANDIDATE training data...")
    
    candidate_count = 400
    print(f"  - Planet Candidates: {candidate_count} samples")
    
    for _ in range(candidate_count):
        period = np.random.uniform(1, 150)
        depth = np.random.uniform(0.0002, 0.008)
        
        # Duration from Kepler's law
        stellar_radius = np.random.uniform(0.7, 1.3)
        base_duration = 13.0 * (period / 365.0) ** (1.0/3.0) * stellar_radius
        duration = base_duration * np.random.uniform(0.8, 1.2)
        
        # Higher noise for candidates
        noise_ppm = np.random.uniform(80, 250)
        noise = noise_ppm / 1e6
        
        # Calculate significance (weaker than confirmed)
        data_span_days = 90
        n_transits = max(2, int(data_span_days / period))
        points_per_transit = max(5, int(duration / 0.5))
        
        significance = (depth / noise) * np.sqrt(n_transits * points_per_transit)
        significance *= np.random.uniform(0.8, 1.2)
        
        # Lower BLS power for candidates
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
            'classification': 1  # Candidate
        })
    
    print(f"Total planet candidates: {candidate_count}")
    
    # ========== FALSE POSITIVES ==========
    # Various types of astrophysical false positives
    
    print("\nGenerating FALSE POSITIVE training data...")
    
    false_positive_types = [
        # Eclipsing binaries (very deep, wrong duration)
        {
            'name': 'Eclipsing Binaries',
            'period_range': (0.2, 5),
            'depth_range': (0.01, 0.15),  # Much deeper than planets
            'duration_factor': (1.5, 3.0),  # Longer than expected
            'noise_range': (50, 150),
            'bls_mean': 0.02,
            'count': 200
        },
        # Stellar variability (irregular, shallow)
        {
            'name': 'Stellar Variability',
            'period_range': (5, 100),
            'depth_range': (0.00001, 0.0008),  # Very shallow
            'duration_factor': (0.3, 3.0),  # Wrong duration
            'noise_range': (100, 400),  # High noise
            'bls_mean': 0.01,
            'count': 200
        },
        # Instrumental artifacts
        {
            'name': 'Instrumental Artifacts',
            'period_range': (0.1, 200),
            'depth_range': (0.00001, 0.1),
            'duration_factor': (0.1, 5.0),  # Completely wrong
            'noise_range': (50, 500),
            'bls_mean': 0.005,
            'count': 200
        },
    ]
    
    for fp_type in false_positive_types:
        print(f"  - {fp_type['name']}: {fp_type['count']} samples")
        
        for _ in range(fp_type['count']):
            period = np.random.uniform(*fp_type['period_range'])
            depth = np.random.uniform(*fp_type['depth_range'])
            
            # WRONG duration (doesn't follow Kepler's law properly)
            stellar_radius = np.random.uniform(0.5, 2.0)
            base_duration = 13.0 * (period / 365.0) ** (1.0/3.0) * stellar_radius
            duration = base_duration * np.random.uniform(*fp_type['duration_factor'])
            
            # Variable noise
            noise_ppm = np.random.uniform(*fp_type['noise_range'])
            noise = noise_ppm / 1e6
            
            # Calculate significance (often low)
            data_span_days = 90
            n_transits = max(1, int(data_span_days / period))
            points_per_transit = max(3, int(duration / 0.5))
            
            significance = (depth / noise) * np.sqrt(n_transits * points_per_transit)
            significance *= np.random.uniform(0.5, 1.5)  # More variable
            
            # Low BLS power
            bls_power = np.random.lognormal(np.log(fp_type['bls_mean']), 0.7)
            bls_power = np.clip(bls_power, 0.001, 0.08)
            
            # More variable baseline
            baseline_flux = np.random.normal(1.0, 0.02)
            
            data.append({
                'period': period,
                'depth': depth,
                'duration_hours': duration,
                'depth_significance': significance,
                'baseline_flux': baseline_flux,
                'noise_level': noise,
                'bls_power': bls_power,
                'classification': 0  # False Positive
            })
    
    print(f"Total false positives: {sum(fp['count'] for fp in false_positive_types)}")
    
    # ========== CREATE DATAFRAME ==========
    
    df = pd.DataFrame(data)
    
    print(f"\n{'='*60}")
    print(f"Total training samples: {len(df)}")
    print(f"  - Confirmed Exoplanets: {len(df[df['classification']==2])} ({len(df[df['classification']==2])/len(df)*100:.1f}%)")
    print(f"  - Planet Candidates: {len(df[df['classification']==1])} ({len(df[df['classification']==1])/len(df)*100:.1f}%)")
    print(f"  - False Positives: {len(df[df['classification']==0])} ({len(df[df['classification']==0])/len(df)*100:.1f}%)")
    print(f"{'='*60}")
    
    # ========== FEATURE ENGINEERING ==========
    
    print("\nEngineering features...")
    
    df['period_log'] = np.log10(df['period'])
    df['depth_ppm'] = df['depth'] * 1e6
    df['duration_ratio'] = df['duration_hours'] / (df['period'] * 24)
    df['snr_ratio'] = df['depth_significance'] / np.maximum(df['noise_level'], 1e-10)
    
    # Feature names (must match transit_detector.py output)
    feature_names = [
        'period', 'depth', 'duration_hours', 'depth_significance',
        'baseline_flux', 'noise_level', 'bls_power',
        'period_log', 'depth_ppm', 'duration_ratio', 'snr_ratio'
    ]
    
    X = df[feature_names]
    y = df['classification']
    
    # ========== TRAIN MODEL ==========
    
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train RandomForest
    print("\nTraining Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    
    model.fit(X_train_scaled, y_train)
    
    # ========== EVALUATE ==========
    
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"\nAccuracy:")
    print(f"  Training: {train_score:.3f} ({train_score*100:.1f}%)")
    print(f"  Test:     {test_score:.3f} ({test_score*100:.1f}%)")
    
    # Detailed classification report
    y_pred = model.predict(X_test_scaled)
    
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['False Positive', 'Candidate', 'Confirmed'],
        digits=3
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print("                  Predicted")
    print("                  FP   Cand  Conf")
    print(f"Actual FP       {cm[0,0]:4d} {cm[0,1]:4d} {cm[0,2]:4d}")
    print(f"       Cand     {cm[1,0]:4d} {cm[1,1]:4d} {cm[1,2]:4d}")
    print(f"       Conf     {cm[2,0]:4d} {cm[2,1]:4d} {cm[2,2]:4d}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    for idx, row in importance.iterrows():
        print(f"  {row['feature']:20s}: {row['importance']:.4f}")
    
    # ========== SAVE MODEL ==========
    
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    # Create models directory
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_file = os.path.join(models_dir, "exoplanet_ml_model.pkl")
    
    # Package everything
    model_package = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'version': '3.0_SCIENTIFIC',
        'trained_on': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'Scientifically accurate exoplanet classifier with physics-based features',
        'train_accuracy': train_score,
        'test_accuracy': test_score,
        'feature_importance': importance.to_dict('records')
    }
    
    joblib.dump(model_package, model_file)
    
    print(f"\n✅ Model saved to: {model_file}")
    print(f"   Version: {model_package['version']}")
    print(f"   Test Accuracy: {test_score:.3f}")
    print(f"   File size: {os.path.getsize(model_file) / 1024:.1f} KB")
    
    print("\n" + "="*60)
    print("MODEL CREATION COMPLETE!")
    print("="*60)
    
    return True


if __name__ == "__main__":
    try:
        success = create_pretrained_model()
        if success:
            print("\n✅ SUCCESS! Model is ready for exoplanet detection.")
        else:
            print("\n❌ FAILED! Model creation unsuccessful.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()