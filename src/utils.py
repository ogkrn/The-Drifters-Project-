# utils.py - COMPLETE FINAL VERSION

import streamlit as st
import numpy as np
from typing import Dict, Any


def display_progress_step(current_step: int, total_steps: int, step_name: str):
    """
    Display analysis progress step with progress bar
    
    Args:
        current_step: Current step number (1-indexed)
        total_steps: Total number of steps
        step_name: Name/description of current step
    """
    
    # Calculate progress percentage
    progress = current_step / total_steps
    
    # Display progress bar
    st.progress(progress)
    
    # Display step information
    st.write(f"**Step {current_step}/{total_steps}:** {step_name}")


def format_star_id(star_id: str, mission: str) -> str:
    """
    Format star ID with mission prefix
    
    Args:
        star_id: Raw star identifier
        mission: Mission name ("Kepler" or "TESS")
        
    Returns:
        Formatted star ID string
    """
    
    if mission.lower() == "kepler":
        return f"KIC {star_id}"
    elif mission.lower() == "tess":
        return f"TIC {star_id}"
    else:
        return star_id


def calculate_transit_metrics(flux: np.ndarray, baseline_flux: float) -> Dict[str, Any]:
    """
    Calculate basic transit metrics from flux data
    
    Args:
        flux: Flux array
        baseline_flux: Baseline (out-of-transit) flux level
        
    Returns:
        Dictionary of calculated metrics
    """
    
    try:
        # Basic statistics
        min_flux = float(np.min(flux))
        max_flux = float(np.max(flux))
        std_flux = float(np.std(flux))
        
        # Transit depth
        depth = max(0.0, (baseline_flux - min_flux) / baseline_flux)
        
        # Signal-to-noise estimate
        noise_level = std_flux
        significance = depth / noise_level if noise_level > 0 else 0.0
        
        return {
            'depth': depth,
            'min_flux': min_flux,
            'max_flux': max_flux,
            'noise_level': noise_level,
            'significance': significance,
            'baseline_flux': baseline_flux
        }
        
    except Exception as e:
        st.warning(f"Error calculating transit metrics: {e}")
        return {
            'depth': 0.001,
            'min_flux': baseline_flux,
            'max_flux': baseline_flux,
            'noise_level': 0.001,
            'significance': 1.0,
            'baseline_flux': baseline_flux
        }


def estimate_planet_radius(depth: float, stellar_radius: float = 1.0) -> float:
    """
    Estimate planet radius from transit depth with limb darkening correction.
    
    IMPROVED: Accounts for limb darkening effect (~10% correction)
    
    Physics:
    - Transit depth = (R_planet / R_star)^2
    - Limb darkening makes transits appear ~10% shallower than geometric
    - Correction factor from Mandel & Agol (2002), Seager & Mallen-Ornelas (2003)
    
    Args:
        depth: Transit depth (fractional, e.g., 0.01 = 1%)
        stellar_radius: Stellar radius in solar radii (default 1.0)
        
    Returns:
        Planet radius in Earth radii
    """
    
    try:
        # Limb darkening makes transits appear ~10% shallower than geometric
        # Correct for this effect to get true planet radius
        limb_darkening_correction = 1.1
        corrected_depth = depth * limb_darkening_correction
        
        # Rp/Rs = sqrt(depth)
        planet_radius_stellar = np.sqrt(max(0.0001, corrected_depth))
        
        # Convert R* (solar radii) to R_earth
        # 1 R_sun = 109.2 R_earth (exact value: 109.16)
        planet_radius_earth = planet_radius_stellar * stellar_radius * 109.2
        
        # Physical limits: 0.1 to 50 Earth radii
        # Below 0.1: asteroids/moonlets (not planets)
        # Above 50: brown dwarfs (not planets)
        return float(np.clip(planet_radius_earth, 0.1, 50.0))
        
    except Exception as e:
        # Fallback to 1 Earth radius on error
        return 1.0


def validate_lightcurve_quality(lc) -> Dict[str, Any]:
    """
    Assess light curve data quality
    
    Args:
        lc: Light curve object (from lightkurve)
        
    Returns:
        Dictionary with quality assessment metrics
    """
    
    try:
        # Extract data safely
        if hasattr(lc.time, 'value'):
            time_vals = lc.time.value
            flux_vals = lc.flux.value
        else:
            time_vals = np.array(lc.time)
            flux_vals = np.array(lc.flux)
        
        # Calculate quality metrics
        n_points = len(time_vals)
        time_span = float(np.ptp(time_vals))
        flux_std = float(np.std(flux_vals))
        flux_median = float(np.median(flux_vals))
        
        # Quality score calculation (0 to 1)
        # Based on: number of points, time coverage, noise level
        
        # Point score: more points = better
        point_score = min(1.0, n_points / 10000)
        
        # Time score: longer coverage = better
        time_score = min(1.0, time_span / 80)
        
        # Noise score: lower noise = better
        # Typical Kepler noise: 50-200 ppm (0.00005 - 0.0002)
        # Good quality: < 100 ppm
        noise_score = min(1.0, 0.001 / flux_std) if flux_std > 0 else 0.0
        
        # Overall quality (average of subscores)
        overall_quality = float((point_score + time_score + noise_score) / 3)
        
        return {
            'n_points': int(n_points),
            'time_span': time_span,
            'flux_std': flux_std,
            'flux_median': flux_median,
            'point_score': point_score,
            'time_score': time_score,
            'noise_score': noise_score,
            'overall_quality': overall_quality,
            'quality_assessment': _get_quality_label(overall_quality)
        }
        
    except Exception as e:
        st.warning(f"Quality assessment failed: {e}")
        return {
            'overall_quality': 0.5,
            'n_points': 0,
            'time_span': 0,
            'flux_std': 0.01,
            'quality_assessment': 'Unknown'
        }


def _get_quality_label(score: float) -> str:
    """Get quality assessment label from score"""
    if score > 0.7:
        return "High Quality"
    elif score > 0.5:
        return "Good Quality"
    elif score > 0.3:
        return "Moderate Quality"
    else:
        return "Low Quality"


def format_scientific_notation(value: float, precision: int = 3) -> str:
    """
    Format value in scientific notation for display
    
    Args:
        value: Numerical value
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    
    try:
        if value == 0:
            return "0"
        elif abs(value) >= 1000 or abs(value) < 0.001:
            return f"{value:.{precision}e}"
        else:
            return f"{value:.{precision}f}"
    except:
        return str(value)


def create_summary_dict(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a standardized summary dictionary from analysis results
    
    Args:
        analysis_results: Raw analysis results from analyzer
        
    Returns:
        Standardized summary dictionary
    """
    
    try:
        return {
            'star_id': analysis_results.get('star_id', 'Unknown'),
            'mission': analysis_results.get('mission', 'Unknown'),
            'classification': analysis_results.get('classification', 'Unknown'),
            'confidence': analysis_results.get('confidence', 0.0),
            'period': analysis_results.get('period', 0.0),
            'depth': analysis_results.get('depth', 0.0),
            'depth_ppm': analysis_results.get('depth_ppm', 0.0),
            'significance': analysis_results.get('significance', 0.0),
            'planet_radius': analysis_results.get('estimated_planet_radius', 0.0),
            'vetting_score': analysis_results.get('vetting_score', 0.0),
            'bls_power': analysis_results.get('bls_power', 0.0),
            'num_transits': analysis_results.get('num_transits', 0),
            'odd_even_status': analysis_results.get('odd_even_test_status', 'UNKNOWN'),
            'secondary_status': analysis_results.get('secondary_eclipse_status', 'UNKNOWN'),
            'analysis_timestamp': analysis_results.get('analysis_timestamp', 'Unknown')
        }
        
    except Exception as e:
        st.warning(f"Error creating summary: {e}")
        return {}


def calculate_habitable_zone(stellar_radius: float = 1.0, stellar_temp: float = 5778) -> Dict[str, float]:
    """
    Calculate habitable zone boundaries for a star
    
    Based on Kopparapu et al. (2013) formulation
    
    Args:
        stellar_radius: Stellar radius in solar radii
        stellar_temp: Stellar effective temperature in Kelvin
        
    Returns:
        Dictionary with inner and outer HZ boundaries in AU
    """
    
    try:
        # Sun's effective temperature
        T_sun = 5778  # Kelvin
        
        # Stellar luminosity from Stefan-Boltzmann law
        # L/L_sun = (R/R_sun)^2 * (T/T_sun)^4
        luminosity = (stellar_radius ** 2) * ((stellar_temp / T_sun) ** 4)
        
        # Habitable zone boundaries (simplified)
        # Conservative HZ: 0.95 to 1.37 AU for Sun-like stars
        inner_hz = 0.95 * np.sqrt(luminosity)
        outer_hz = 1.37 * np.sqrt(luminosity)
        
        # Optimistic HZ: 0.75 to 1.77 AU
        inner_hz_optimistic = 0.75 * np.sqrt(luminosity)
        outer_hz_optimistic = 1.77 * np.sqrt(luminosity)
        
        return {
            'inner_hz_conservative': float(inner_hz),
            'outer_hz_conservative': float(outer_hz),
            'inner_hz_optimistic': float(inner_hz_optimistic),
            'outer_hz_optimistic': float(outer_hz_optimistic),
            'stellar_luminosity': float(luminosity)
        }
        
    except Exception as e:
        # Return Sun's HZ as fallback
        return {
            'inner_hz_conservative': 0.95,
            'outer_hz_conservative': 1.37,
            'inner_hz_optimistic': 0.75,
            'outer_hz_optimistic': 1.77,
            'stellar_luminosity': 1.0
        }


def orbital_period_to_semimajor_axis(period_days: float, stellar_mass: float = 1.0) -> float:
    """
    Convert orbital period to semi-major axis using Kepler's Third Law
    
    Args:
        period_days: Orbital period in days
        stellar_mass: Stellar mass in solar masses
        
    Returns:
        Semi-major axis in AU
    """
    
    try:
        # Kepler's Third Law: P^2 = (4π^2 / G M) a^3
        # For solar units: a^3 = P^2 * M
        # where P is in years, M in solar masses, a in AU
        
        period_years = period_days / 365.25
        
        # a = (P^2 * M)^(1/3)
        semimajor_axis = (period_years ** 2 * stellar_mass) ** (1.0/3.0)
        
        return float(semimajor_axis)
        
    except Exception as e:
        return 0.0


def is_in_habitable_zone(period_days: float, stellar_radius: float = 1.0, 
                         stellar_temp: float = 5778, stellar_mass: float = 1.0) -> Dict[str, Any]:
    """
    Check if a planet with given period is in the habitable zone
    
    Args:
        period_days: Orbital period in days
        stellar_radius: Stellar radius in solar radii
        stellar_temp: Stellar temperature in Kelvin
        stellar_mass: Stellar mass in solar masses
        
    Returns:
        Dictionary with HZ assessment
    """
    
    try:
        # Calculate semi-major axis
        semimajor_axis = orbital_period_to_semimajor_axis(period_days, stellar_mass)
        
        # Calculate HZ boundaries
        hz = calculate_habitable_zone(stellar_radius, stellar_temp)
        
        # Check if in HZ
        in_conservative_hz = (hz['inner_hz_conservative'] <= semimajor_axis <= hz['outer_hz_conservative'])
        in_optimistic_hz = (hz['inner_hz_optimistic'] <= semimajor_axis <= hz['outer_hz_optimistic'])
        
        # Determine status
        if in_conservative_hz:
            status = "In Conservative Habitable Zone"
            emoji = "🌍"
        elif in_optimistic_hz:
            status = "In Optimistic Habitable Zone"
            emoji = "🌎"
        else:
            if semimajor_axis < hz['inner_hz_optimistic']:
                status = "Too Hot (Inside HZ)"
                emoji = "🔥"
            else:
                status = "Too Cold (Outside HZ)"
                emoji = "❄️"
        
        return {
            'status': status,
            'emoji': emoji,
            'in_conservative_hz': in_conservative_hz,
            'in_optimistic_hz': in_optimistic_hz,
            'semimajor_axis_au': semimajor_axis,
            'inner_hz_au': hz['inner_hz_conservative'],
            'outer_hz_au': hz['outer_hz_conservative']
        }
        
    except Exception as e:
        return {
            'status': 'Unknown',
            'emoji': '❓',
            'in_conservative_hz': False,
            'in_optimistic_hz': False,
            'semimajor_axis_au': 0.0
        }


def estimate_equilibrium_temperature(period_days: float, stellar_radius: float = 1.0,
                                     stellar_temp: float = 5778, stellar_mass: float = 1.0,
                                     albedo: float = 0.3) -> float:
    """
    Estimate planet's equilibrium temperature
    
    Args:
        period_days: Orbital period in days
        stellar_radius: Stellar radius in solar radii
        stellar_temp: Stellar temperature in Kelvin
        stellar_mass: Stellar mass in solar masses
        albedo: Bond albedo (0 to 1, default 0.3 like Earth)
        
    Returns:
        Equilibrium temperature in Kelvin
    """
    
    try:
        # Calculate semi-major axis
        a_au = orbital_period_to_semimajor_axis(period_days, stellar_mass)
        
        # Convert to meters
        AU_TO_METERS = 1.496e11
        a_meters = a_au * AU_TO_METERS
        
        # Stellar luminosity
        L_sun = 3.828e26  # Watts
        luminosity = (stellar_radius ** 2) * ((stellar_temp / 5778) ** 4) * L_sun
        
        # Equilibrium temperature (assuming rapid rotation, no greenhouse effect)
        # T_eq = T_star * sqrt(R_star / 2a) * (1 - albedo)^(1/4)
        R_sun = 6.96e8  # meters
        R_star = stellar_radius * R_sun
        
        T_eq = stellar_temp * np.sqrt(R_star / (2 * a_meters)) * ((1 - albedo) ** 0.25)
        
        return float(T_eq)
        
    except Exception as e:
        return 0.0


def classify_planet_type(radius_earth: float, period_days: float) -> Dict[str, str]:
    """
    Classify planet type based on radius and period
    
    Args:
        radius_earth: Planet radius in Earth radii
        period_days: Orbital period in days
        
    Returns:
        Dictionary with planet type classification
    """
    
    try:
        # Radius-based classification
        if radius_earth < 1.25:
            size_class = "Earth-like"
            emoji = "🌍"
        elif radius_earth < 2.0:
            size_class = "Super-Earth"
            emoji = "🌎"
        elif radius_earth < 4.0:
            size_class = "Neptune-like"
            emoji = "🔵"
        elif radius_earth < 10.0:
            size_class = "Sub-Jovian"
            emoji = "🪐"
        else:
            size_class = "Jovian"
            emoji = "🪐"
        
        # Period-based classification
        if period_days < 1.0:
            temp_class = "Ultra-Hot"
        elif period_days < 10.0:
            temp_class = "Hot"
        elif period_days < 100.0:
            temp_class = "Warm"
        else:
            temp_class = "Cool"
        
        # Combined classification
        full_class = f"{temp_class} {size_class}"
        
        # Special cases
        if period_days < 10 and radius_earth > 8:
            full_class = "Hot Jupiter"
            emoji = "🔥🪐"
        elif 0.8 <= radius_earth <= 1.25 and 200 <= period_days <= 500:
            full_class = "Earth Analog"
            emoji = "🌍✨"
        
        return {
            'classification': full_class,
            'size_class': size_class,
            'temperature_class': temp_class,
            'emoji': emoji
        }
        
    except Exception as e:
        return {
            'classification': 'Unknown',
            'size_class': 'Unknown',
            'temperature_class': 'Unknown',
            'emoji': '❓'
        }


def format_analysis_results_for_display(summary: Dict[str, Any]) -> str:
    """
    Format analysis results for clean display with detailed info
    
    Args:
        summary: Analysis summary dictionary
        
    Returns:
        Formatted string for display
    """
    
    if not summary:
        return "No analysis results available"
    
    try:
        classification = summary.get('classification', 'Unknown')
        confidence = summary.get('confidence', 0)
        period = summary.get('period', 0)
        radius = summary.get('planet_radius', 0)
        ml_prediction = summary.get('ml_prediction', classification)
        
        # Vetting info
        odd_even = summary.get('odd_even_status', 'UNKNOWN')
        secondary = summary.get('secondary_status', 'UNKNOWN')
        
        # Planet type
        planet_type = classify_planet_type(radius, period)
        
        # HZ check
        hz_info = is_in_habitable_zone(period)
        
        result_text = f"""
**Classification:** {classification} {planet_type['emoji']}
**ML Prediction:** {ml_prediction}  
**Confidence:** {confidence:.1%}

**Planet Properties:**
- Type: {planet_type['classification']}
- Radius: {radius:.2f} R⊕
- Orbital Period: {period:.3f} days
- Semi-major Axis: {hz_info['semimajor_axis_au']:.3f} AU
- Habitable Zone: {hz_info['status']} {hz_info['emoji']}

**Vetting Tests:**
- Odd-Even: {odd_even}
- Secondary Eclipse: {secondary}

**Analysis Status:** {'✅ Successful' if summary.get('analysis_successful') else '❌ Failed'}
        """
        
        return result_text.strip()
        
    except Exception as e:
        return f"Error formatting results: {str(e)}"


def save_results_to_file(summary: Dict[str, Any], filename: str = "exoplanet_results.txt") -> bool:
    """
    Save analysis results to a text file
    
    Args:
        summary: Analysis summary dictionary
        filename: Output filename
        
    Returns:
        Success status
    """
    
    try:
        formatted_results = format_analysis_results_for_display(summary)
        
        with open(filename, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("EXOPLANET DETECTION ANALYSIS RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(formatted_results)
            f.write("\n\n" + "=" * 60 + "\n")
            f.write(f"Generated by NASA-Standard Exoplanet Detection System\n")
            f.write("=" * 60 + "\n")
        
        return True
        
    except Exception as e:
        st.error(f"Failed to save results: {e}")
        return False


def validate_inputs(star_id: str, mission: str, quarter: int = None, sector: int = None) -> Dict[str, Any]:
    """
    Validate user inputs before analysis
    
    Args:
        star_id: Star identifier
        mission: Mission name
        quarter: Kepler quarter (optional)
        sector: TESS sector (optional)
        
    Returns:
        Dictionary with validation results
    """
    
    errors = []
    warnings = []
    
    # Validate star ID
    if not star_id or not star_id.strip():
        errors.append("Star ID cannot be empty")
    elif not star_id.strip().isdigit():
        errors.append("Star ID must be numeric")
    else:
        star_id_int = int(star_id.strip())
        
        if mission.lower() == "kepler":
            if not (100000 <= star_id_int <= 999999999):
                warnings.append("KIC numbers are typically 6-9 digits")
        elif mission.lower() == "tess":
            if not (10000000 <= star_id_int <= 9999999999):
                warnings.append("TIC numbers are typically 8-10 digits")
    
    # Validate mission
    if mission.lower() not in ["kepler", "tess"]:
        errors.append("Mission must be 'Kepler' or 'TESS'")
    
    # Validate quarter/sector
    if mission.lower() == "kepler" and quarter is not None:
        if not (0 <= quarter <= 17):
            errors.append("Kepler quarter must be 0-17")
    
    if mission.lower() == "tess" and sector is not None:
        if not (0 <= sector <= 100):
            warnings.append("TESS sector seems unusually high")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }