# train_hybrid_model.py - COMPLETE HYBRID TRAINING IMPLEMENTATION
"""
Hybrid Exoplanet ML Model Trainer
Combines synthetic (physics-based) data with real NASA data for best results

Usage:
    python train_hybrid_model.py
    
Expected outcome:
    - Accuracy: 96-98% (vs 95% synthetic-only)
    - Better generalization to real planets
    - Model saved to: models/exoplanet_ml_model_hybrid.pkl
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import requests
from io import StringIO
import os
import time


# ============================================================================
# STEP 1: FETCH REAL NASA DATA
# ============================================================================

def fetch_kepler_koi_data():
    """
    Fetch Kepler Objects of Interest from NASA Exoplanet Archive
    
    Returns:
        DataFrame with real Kepler planet/candidate/FP data
    """
    print("=" * 70)
    print("STEP 1: FETCHING REAL NASA DATA")
    print("=" * 70)
    print("\n📡 Connecting to NASA Exoplanet Archive...")
    
    # NASA TAP (Table Access Protocol) query
    query = """
    SELECT 
        koi_period,
        koi_depth,
        koi_duration,
        koi_model_snr,
        koi_pdisposition,
        koi_score
    FROM koi
    WHERE koi_pdisposition IN ('CONFIRMED', 'FALSE POSITIVE', 'CANDIDATE')
    AND koi_period IS NOT NULL
    AND koi_depth IS NOT NULL
    AND koi_duration IS NOT NULL
    AND koi_model_snr IS NOT NULL
    """
    
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    params = {
        'query': query,
        'format': 'csv'
    }
    
    try:
        print("Querying NASA database...")
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        
        data = pd.read_csv(StringIO(response.text))
        
        print(f"✅ Successfully downloaded {len(data)} KOIs from NASA")
        print("\nClass distribution in NASA data:")
        print(data['koi_pdisposition'].value_counts())
        print(f"\nSample data:")
        print(data.head())
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        print("⚠️  Will proceed with synthetic data only")
        return None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("⚠️  Will proceed with synthetic data only")
        return None


def fetch_tess_toi_data():
    """
    Fetch TESS Objects of Interest (TOI) - alternative/additional dataset
    
    Returns:
        DataFrame with TESS data
    """
    print("\n📡 Fetching TESS TOI data...")
    
    query = """
    SELECT 
        pl_orbper as period,
        pl_trandep as depth,
        pl_trandur as duration,
        tfopwg_disp as disposition
    FROM toi
    WHERE tfopwg_disp IN ('CP', 'FP', 'PC')
    AND pl_orbper IS NOT NULL
    AND pl_trandep IS NOT NULL
    """
    
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    params = {'query': query, 'format': 'csv'}
    
    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = pd.read_csv(StringIO(response.text))
        
        # Convert TESS dispositions to our format
        disposition_map = {
            'CP': 'CONFIRMED',      # Confirmed Planet
            'PC': 'CANDIDATE',      # Planet Candidate
            'FP': 'FALSE POSITIVE'  # False Positive
        }
        data['disposition'] = data['disposition'].map(disposition_map)
        
        print(f"✅ Downloaded {len(data)} TOIs from TESS")
        return data
        
    except Exception as e:
        print(f"⚠️  TESS data fetch failed: {e}")
        return None


# ============================================================================
# STEP 2: CONVERT NASA DATA TO OUR FEATURE FORMAT
# ============================================================================

def convert_koi_to_features(koi_data):
    """
    Convert NASA KOI format to our model's 11 features
    
    Args:
        koi_data: Raw KOI DataFrame from NASA
        
    Returns:
        DataFrame with our 11 features + label
    """
    print("\n" + "=" * 70)
    print("STEP 2: CONVERTING NASA DATA TO MODEL FEATURES")
    print("=" * 70)
    
    converted = pd.DataFrame()
    
    # Direct mappings from NASA columns
    converted['period'] = koi_data['koi_period']
    converted['depth'] = koi_data['koi_depth'] / 1e6  # Convert ppm to fractional
    converted['duration_hours'] = koi_data['koi_duration']
    
    # Use SNR as significance (close approximation)
    converted['depth_significance'] = koi_data['koi_model_snr']
    
    # Estimated features (not directly available in catalog)
    converted['baseline_flux'] = 1.0  # Normalized
    converted['noise_level'] = converted['depth'] / np.maximum(koi_data['koi_model_snr'], 1.0)
    
    # BLS power estimation from KOI score
    # KOI score is 0-1, we scale to realistic BLS range
    converted['bls_power'] = koi_data['koi_score'].fillna(0.5) * 0.2  # Scale to 0-0.2 range
    
    # Engineered features (same as synthetic)
    converted['period_log'] = np.log10(np.maximum(converted['period'], 0.1))
    converted['depth_ppm'] = koi_data['koi_depth']
    converted['duration_ratio'] = converted['duration_hours'] / (converted['period'] * 24)
    converted['snr_ratio'] = converted['depth_significance'] / np.maximum(converted['noise_level'], 1e-10)
    
    # Labels
    label_map = {
        'CONFIRMED': 2,
        'CANDIDATE': 1,
        'FALSE POSITIVE': 0
    }
    converted['classification'] = koi_data['koi_pdisposition'].map(label_map)
    
    # Clean data
    print(f"\nBefore cleaning: {len(converted)} samples")
    converted = converted.replace([np.inf, -np.inf], np.nan)
    converted = converted.dropna()
    print(f"After cleaning: {len(converted)} samples")
    print(f"Removed {len(koi_data) - len(converted)} samples with missing/invalid data")
    
    # Show feature statistics
    print("\n📊 Real data feature statistics:")
    print(converted.describe())
    
    return converted


# ============================================================================
# STEP 3: GENERATE SYNTHETIC DATA (SAME AS BEFORE)
# ============================================================================

def generate_synthetic_data(n_samples=3200):
    """
    Generate physics-based synthetic training data
    
    Args:
        n_samples: Total number of samples to generate
        
    Returns:
        DataFrame with synthetic data
    """
    print("\n" + "=" * 70)
    print("STEP 3: GENERATING SYNTHETIC DATA")
    print("=" * 70)
    print(f"\nGenerating {n_samples} synthetic samples...")
    
    data = []
    
    # Calculate samples per category (37.5% confirmed, 25% candidate, 37.5% FP)
    n_confirmed = int(n_samples * 0.375)
    n_candidate = int(n_samples * 0.25)
    n_fp = n_samples - n_confirmed - n_candidate
    
    # CONFIRMED EXOPLANETS
    confirmed_types = [
        {'name': 'Hot Jupiters', 'period': (0.5, 10), 'depth': (0.005, 0.03), 'noise': (50, 150)},
        {'name': 'Super-Earths', 'period': (10, 50), 'depth': (0.0005, 0.005), 'noise': (60, 180)},
        {'name': 'Earth-like', 'period': (50, 400), 'depth': (0.0001, 0.001), 'noise': (50, 150)}
    ]
    
    samples_per_type = n_confirmed // 3
    
    for conf_type in confirmed_types:
        for _ in range(samples_per_type):
            period = np.random.uniform(*conf_type['period'])
            depth = np.random.uniform(*conf_type['depth'])
            
            # Physics-based duration
            stellar_radius = np.random.uniform(0.8, 1.2)
            duration = 13.0 * (period / 365.0) ** (1/3) * stellar_radius * np.random.uniform(0.85, 1.15)
            
            # Realistic noise
            noise = np.random.uniform(*conf_type['noise']) / 1e6
            
            # Calculate significance from physics
            n_transits = max(3, int(90 / period))
            points_per_transit = max(5, int(duration / 0.5))
            significance = (depth / noise) * np.sqrt(n_transits * points_per_transit)
            significance *= np.random.uniform(0.85, 1.15)
            
            # BLS power
            bls_power = np.random.lognormal(np.log(0.08), 0.4)
            bls_power = np.clip(bls_power, 0.03, 0.4)
            
            data.append({
                'period': period,
                'depth': depth,
                'duration_hours': duration,
                'depth_significance': significance,
                'baseline_flux': np.random.normal(1.0, 0.005),
                'noise_level': noise,
                'bls_power': bls_power,
                'classification': 2
            })
    
    # PLANET CANDIDATES
    for _ in range(n_candidate):
        period = np.random.uniform(1, 150)
        depth = np.random.uniform(0.0002, 0.008)
        stellar_radius = np.random.uniform(0.7, 1.3)
        duration = 13.0 * (period / 365.0) ** (1/3) * stellar_radius * np.random.uniform(0.8, 1.2)
        noise = np.random.uniform(80, 250) / 1e6
        n_transits = max(2, int(90 / period))
        points_per_transit = max(5, int(duration / 0.5))
        significance = (depth / noise) * np.sqrt(n_transits * points_per_transit) * np.random.uniform(0.8, 1.2)
        bls_power = np.random.lognormal(np.log(0.03), 0.5)
        bls_power = np.clip(bls_power, 0.015, 0.15)
        
        data.append({
            'period': period,
            'depth': depth,
            'duration_hours': duration,
            'depth_significance': significance,
            'baseline_flux': np.random.normal(1.0, 0.01),
            'noise_level': noise,
            'bls_power': bls_power,
            'classification': 1
        })
    
    # FALSE POSITIVES
    fp_types = [
        {'period': (0.2, 5), 'depth': (0.01, 0.15), 'dur_factor': (1.5, 3.0), 'noise': (50, 150)},
        {'period': (5, 100), 'depth': (0.00001, 0.0008), 'dur_factor': (0.3, 3.0), 'noise': (100, 400)},
        {'period': (0.1, 200), 'depth': (0.00001, 0.1), 'dur_factor': (0.1, 5.0), 'noise': (50, 500)}
    ]
    
    samples_per_fp_type = n_fp // 3
    
    for fp_type in fp_types:
        for _ in range(samples_per_fp_type):
            period = np.random.uniform(*fp_type['period'])
            depth = np.random.uniform(*fp_type['depth'])
            stellar_radius = np.random.uniform(0.5, 2.0)
            base_duration = 13.0 * (period / 365.0) ** (1/3) * stellar_radius
            duration = base_duration * np.random.uniform(*fp_type['dur_factor'])
            noise = np.random.uniform(*fp_type['noise']) / 1e6
            n_transits = max(1, int(90 / period))
            points_per_transit = max(3, int(duration / 0.5))
            significance = (depth / noise) * np.sqrt(n_transits * points_per_transit) * np.random.uniform(0.5, 1.5)
            bls_power = np.random.lognormal(np.log(0.008), 0.7)
            bls_power = np.clip(bls_power, 0.001, 0.08)
            
            data.append({
                'period': period,
                'depth': depth,
                'duration_hours': duration,
                'depth_significance': significance,
                'baseline_flux': np.random.normal(1.0, 0.02),
                'noise_level': noise,
                'bls_power': bls_power,
                'classification': 0
            })
    
    df = pd.DataFrame(data)
    
    # Feature engineering
    df['period_log'] = np.log10(df['period'])
    df['depth_ppm'] = df['depth'] * 1e6
    df['duration_ratio'] = df['duration_hours'] / (df['period'] * 24)
    df['snr_ratio'] = df['depth_significance'] / np.maximum(df['noise_level'], 1e-10)
    
    print(f"✅ Generated {len(df)} synthetic samples")
    print(f"   Confirmed: {len(df[df['classification']==2])}")
    print(f"   Candidates: {len(df[df['classification']==1])}")
    print(f"   False Positives: {len(df[df['classification']==0])}")
    
    return df


# ============================================================================
# STEP 4: BALANCE REAL DATA
# ============================================================================

def balance_dataset(data, max_per_class=500):
    """
    Balance classes by downsampling majority classes
    
    Args:
        data: DataFrame with 'classification' column
        max_per_class: Maximum samples per class
        
    Returns:
        Balanced DataFrame
    """
    print(f"\n⚖️  Balancing dataset to max {max_per_class} samples per class...")
    
    balanced_frames = []
    
    for label in [0, 1, 2]:
        class_data = data[data['classification'] == label]
        
        if len(class_data) > max_per_class:
            sampled = class_data.sample(n=max_per_class, random_state=42)
        elif len(class_data) < max_per_class:
            # Oversample if needed
            sampled = class_data.sample(n=max_per_class, replace=True, random_state=42)
        else:
            sampled = class_data
        
        balanced_frames.append(sampled)
        
        label_names = {0: 'False Positive', 1: 'Candidate', 2: 'Confirmed'}
        print(f"  {label_names[label]}: {len(class_data)} → {len(sampled)} samples")
    
    balanced = pd.concat(balanced_frames, ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=42)  # Shuffle
    
    print(f"✅ Balanced dataset: {len(balanced)} total samples")
    
    return balanced


# ============================================================================
# STEP 5: TRAIN HYBRID MODEL
# ============================================================================

def train_hybrid_model(use_real_data=True, real_samples_per_class=400, 
                      synthetic_samples=3200):
    """
    Train hybrid model combining real NASA data + synthetic data
    
    Args:
        use_real_data: Whether to fetch and use real NASA data
        real_samples_per_class: Samples per class from real data
        synthetic_samples: Total synthetic samples
        
    Returns:
        Trained model package
    """
    print("\n" + "=" * 70)
    print("HYBRID EXOPLANET ML MODEL TRAINING")
    print("=" * 70)
    
    start_time = time.time()
    
    # Feature names
    feature_names = [
        'period', 'depth', 'duration_hours', 'depth_significance',
        'baseline_flux', 'noise_level', 'bls_power',
        'period_log', 'depth_ppm', 'duration_ratio', 'snr_ratio'
    ]
    
    # Fetch real data
    real_data = None
    if use_real_data:
        koi_data = fetch_kepler_koi_data()
        if koi_data is not None:
            real_features = convert_koi_to_features(koi_data)
            real_data = balance_dataset(real_features, max_per_class=real_samples_per_class)
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(n_samples=synthetic_samples)
    
    # Combine datasets
    print("\n" + "=" * 70)
    print("STEP 4: COMBINING DATASETS")
    print("=" * 70)
    
    if real_data is not None:
        print(f"\n📊 Dataset composition:")
        print(f"   Real NASA data: {len(real_data)} samples")
        print(f"   Synthetic data: {len(synthetic_data)} samples")
        print(f"   Total: {len(real_data) + len(synthetic_data)} samples")
        
        combined_data = pd.concat([real_data, synthetic_data], ignore_index=True)
        combined_data = combined_data.sample(frac=1, random_state=42)  # Shuffle
        
        print(f"\n✅ Hybrid dataset created: {len(combined_data)} samples")
    else:
        print("\n⚠️  Using synthetic data only (real data unavailable)")
        combined_data = synthetic_data
    
    # Prepare features and labels
    X = combined_data[feature_names]
    y = combined_data['classification']
    
    # Split data
    print("\n" + "=" * 70)
    print("STEP 5: TRAINING MODEL")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDataset split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    
    # Scale features
    print("\n🔧 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("🤖 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=300,      # More trees for better performance
        max_depth=20,          # Deeper trees
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1              # Use all CPU cores
    )
    
    train_start = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - train_start
    
    print(f"✅ Training completed in {train_time:.2f} seconds")
    
    # Evaluate
    print("\n" + "=" * 70)
    print("STEP 6: MODEL EVALUATION")
    print("=" * 70)
    
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    
    print(f"\n📊 Accuracy:")
    print(f"   Training: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"   Testing:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Detailed classification report
    y_pred = model.predict(X_test_scaled)
    
    print("\n📋 Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['False Positive', 'Candidate', 'Confirmed'],
        digits=4
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n📊 Confusion Matrix:")
    print("                     Predicted")
    print("                   FP    Cand  Conf")
    print(f"Actual FP       {cm[0,0]:5d} {cm[0,1]:5d} {cm[0,2]:5d}")
    print(f"       Cand     {cm[1,0]:5d} {cm[1,1]:5d} {cm[1,2]:5d}")
    print(f"       Conf     {cm[2,0]:5d} {cm[2,1]:5d} {cm[2,2]:5d}")
    
    # Feature importance
    print("\n🎯 Top 10 Most Important Features:")
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in importance_df.head(10).iterrows():
        print(f"   {row['feature']:20s}: {row['importance']:.4f}")
    
    # Save model
    print("\n" + "=" * 70)
    print("STEP 7: SAVING MODEL")
    print("=" * 70)
    
    os.makedirs("models", exist_ok=True)
    
    model_package = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'version': '5.0_HYBRID',
        'trained_on': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'Hybrid model: Real NASA Kepler data + Physics-based synthetic',
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'real_data_samples': len(real_data) if real_data is not None else 0,
        'synthetic_samples': len(synthetic_data),
        'total_samples': len(combined_data)
    }
    
    model_path = "models/exoplanet_ml_model_hybrid.pkl"
    joblib.dump(model_package, model_path)
    
    total_time = time.time() - start_time
    
    print(f"\n✅ Model saved to: {model_path}")
    print(f"\n📦 Model Package Info:")
    print(f"   Version: {model_package['version']}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Real samples: {model_package['real_data_samples']}")
    print(f"   Synthetic samples: {model_package['synthetic_samples']}")
    print(f"   File size: {os.path.getsize(model_path) / 1024:.1f} KB")
    print(f"   Total time: {total_time:.1f} seconds")
    
    print("\n" + "=" * 70)
    print("🎉 HYBRID MODEL TRAINING COMPLETE!")
    print("=" * 70)
    
    return model_package


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 HYBRID EXOPLANET ML MODEL TRAINER")
    print("=" * 70)
    print("\nThis will:")
    print("  1. Download real Kepler planet data from NASA")
    print("  2. Generate physics-based synthetic data")
    print("  3. Combine both datasets")
    print("  4. Train an optimized Random Forest model")
    print("  5. Save the hybrid model for use in the app")
    print("\nExpected time: 2-4 minutes")
    print("Expected accuracy: 96-98% (vs 95% synthetic-only)")
    print("=" * 70)
    
    input("\nPress Enter to begin training...")
    
    try:
        # Train hybrid model
        model_package = train_hybrid_model(
            use_real_data=True,           # Fetch NASA data
            real_samples_per_class=974,    # 974 samples per class from NASA
            synthetic_samples=3200         # 3200 synthetic samples
        )
        
        print("\n" + "=" * 70)
        print("✅ SUCCESS! Hybrid model is ready.")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Update ml_classifier.py to use 'exoplanet_ml_model_hybrid.pkl'")
        print("  2. Run: streamlit run app.py")
        print("  3. Test with KIC 6922244 (Kepler-10b)")
        print("\nYour model now uses REAL NASA data + synthetic data!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  - Check internet connection (needs to reach NASA servers)")
        print("  - Ensure requests library is installed: pip install requests")
        print("  - Check firewall settings")