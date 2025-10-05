# app.py - TWO-MODE SYSTEM WITH HOMEPAGE

import streamlit as st
import sys, os
from pathlib import Path
import joblib

# Add src to path
src_path = Path(__file__).parent
sys.path.append(str(src_path))

from exoplanet_analyzer import ExoplanetAnalyzer
from known_planets_viewer import create_known_planets_interface
from utils import format_star_id

# Page config
st.set_page_config(
    page_title="Professional Exoplanet Discovery",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
st.markdown(
    """
<style>
.main-header {font-size:3rem;color:#1f77b4;text-align:center;margin-bottom:2rem;}
.mission-badge {display:inline-block;padding:0.25rem 0.75rem;margin:0.25rem;
                border-radius:1rem;font-weight:bold;}
.kepler-badge {background:#ff7f0e;color:#fff;}
.tess-badge   {background:#2ca02c;color:#fff;}
.mode-card {
    padding: 2rem;
    border-radius: 1rem;
    border: 2px solid #ddd;
    margin: 1rem 0;
    cursor: pointer;
    transition: all 0.3s;
}
.mode-card:hover {
    border-color: #1f77b4;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
</style>
""",
    unsafe_allow_html=True,
)

# Model check - works from both root and src directory
def get_model_path():
    """Get correct model path regardless of working directory"""
    # Try multiple possible paths
    possible_paths = [
        "src/models/exoplanet_ml_model_hybrid.pkl",  # From root
        "models/exoplanet_ml_model_hybrid.pkl",       # From src
        os.path.join(os.path.dirname(__file__), "models", "exoplanet_ml_model_hybrid.pkl")  # Relative to this file
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return possible_paths[0]  # Return first as default

HYBRID_MODEL_PATH = get_model_path()


def check_hybrid_model() -> bool:
    """Check if hybrid model exists"""
    if os.path.exists(HYBRID_MODEL_PATH):
        try:
            pkg = joblib.load(HYBRID_MODEL_PATH)
            return True
        except:
            return False
    return False


# ================================================================
# HOMEPAGE - MODE SELECTION
# ================================================================

def show_homepage():
    """Display homepage with mode selection"""
    
    st.markdown('<h1 class="main-header">🔭 Professional Exoplanet Discovery System</h1>', 
                unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center;margin-bottom:3rem;color:#666;font-size:1.2rem;">'
        "NASA-Standard Analysis Pipeline with Transit Least Squares (TLS)</div>",
        unsafe_allow_html=True,
    )
    
    st.markdown("## 🚀 Choose Your Mode")
    st.write("Select how you'd like to explore exoplanet data:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="mode-card">
        <h2 style='text-align:center;'>📚 Known Planets Database</h2>
        <h3 style='text-align:center;color:#666;'>Browse NASA's Archive</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **What you'll do:**
        - Search by star ID (KIC/TIC)
        - View confirmed planets & candidates
        - See official NASA classifications
        - Access verified measurements
        
        **Data from:**
        - Kepler Mission (10,000+ objects)
        - TESS Mission (7,000+ objects)
        
        **Best for:**
        - Learning about known planets
        - Verified scientific data
        - Educational purposes
        """)
        
        if st.button("🔍 Browse Known Planets", use_container_width=True, type="primary"):
            st.session_state.mode = "known_planets"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="mode-card">
        <h2 style='text-align:center;'>🤖 Classify New Planets</h2>
        <h3 style='text-align:center;color:#666;'>AI-Powered Discovery</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **What you'll do:**
        - Analyze any star's light curve
        - Run AI/ML classification
        - Detect new planet candidates
        - Professional vetting pipeline
        
        **Technology:**
        - Transit Least Squares (TLS)
        - Machine Learning classifier
        - NASA-standard vetting
        - Advanced signal processing
        
        **Best for:**
        - Finding new candidates
        - Research & discovery
        - Stars not in archive
        """)
        
        model_ok = check_hybrid_model()
        
        if model_ok:
            if st.button("🚀 Classify New Planets", use_container_width=True, type="primary"):
                st.session_state.mode = "classify_new"
                st.rerun()
        else:
            st.error("⚠️ ML model not found")
            st.info("Run: `python train_hybrid_model.py`")
            if st.button("🚀 Classify New Planets", use_container_width=True, disabled=True):
                pass
    
    # Information section
    st.markdown("---")
    st.markdown("## ℹ️ About This System")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.markdown("""
        ### 🔬 Technology
        
        - Transit Least Squares (TLS)
        - Superior to traditional BLS
        - Limb-darkened transit models
        - Hybrid ML classifier
        - Real NASA data training
        """)
    
    with info_col2:
        st.markdown("""
        ### 📊 Capabilities
        
        - Period detection: 0.5-500 days
        - Depth sensitivity: 10 ppm+
        - Binary star vetting
        - Multi-planet detection
        - Habitable zone identification
        """)
    
    with info_col3:
        st.markdown("""
        ### 🎓 Educational
        
        - Learn exoplanet science
        - Understand detection methods
        - Explore NASA missions
        - Professional-grade tools
        - Open-source codebase
        """)


# ================================================================
# MODE 1: KNOWN PLANETS DATABASE
# ================================================================

def run_known_planets_mode():
    """Run the known planets database interface"""
    
    # Back button
    if st.sidebar.button("⬅️ Back to Home"):
        st.session_state.mode = "home"
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Run the known planets viewer
    create_known_planets_interface()


# ================================================================
# MODE 2: CLASSIFY NEW PLANETS
# ================================================================

def run_classify_new_mode():
    """Run the ML classification pipeline"""
    
    # Back button
    if st.sidebar.button("⬅️ Back to Home"):
        st.session_state.mode = "home"
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Check model
    model_ok = check_hybrid_model()
    
    if model_ok:
        st.success("✅ Hybrid ML model loaded")
        try:
            pkg = joblib.load(HYBRID_MODEL_PATH)
            col1 = st.columns(1)
            with col1:
                st.write(f"**Version:** {pkg.get('version','?')}")
                st.write(f"**Trained:** {pkg.get('trained_on','?')}")
        except:
            pass
    else:
        st.error("❌ ML model not found")
        st.stop()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Classify New Planets</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center;margin-bottom:2rem;color:#666;">'
        "AI-Powered Exoplanet Detection Pipeline</div>",
        unsafe_allow_html=True,
    )
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛 Analysis Parameters")
        
        mission = st.selectbox("Select Mission", ["Kepler", "TESS"])
        badge_class = "kepler-badge" if mission == "Kepler" else "tess-badge"
        st.markdown(f'<span class="mission-badge {badge_class}">{mission} Mission</span>', 
                   unsafe_allow_html=True)
        
        default_star_id = "11904151" if mission == "Kepler" else "25155310"
        
        star_id = st.text_input(
            f"Enter {'KIC' if mission=='Kepler' else 'TIC'} Number",
            default_star_id,
        )
        
        if mission == "Kepler":
            quarter = st.number_input(
                "Quarter (0 = ALL)", 
                0, 17, 0,
                key="kepler_quarter",
                help="Set to 0 for all quarters"
            )
            sector = None
        else:
            sector = st.number_input(
                "Sector (0 = ALL)", 
                0, 100, 0,
                key="tess_sector",
                help="Set to 0 for all sectors"
            )
            quarter = None
        
        # Performance mode selection
        st.markdown("### ⚡ Performance Settings")
        use_fast_mode = st.checkbox(
            "Fast Mode (4 files max, faster analysis)", 
            value=True,
            help="Fast: ~30s, downloads 4 quarters. Full: ~2min, downloads ALL quarters (better for known planets)"
        )
        
        # Store in session state for analyzer
        st.session_state['use_fast_mode'] = use_fast_mode
        
        if use_fast_mode:
            st.info("⚡ Fast Mode: Downloads 4 quarters/sectors for quick analysis")
        else:
            st.warning("🐢 Full Mode: Downloads ALL data (slower but more accurate for known planets)")
        
        st.markdown("---")
        analyze_button = st.button("🚀 Start Analysis", use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### 📊 Example Targets")
        
        if mission == "Kepler":
            st.info("💡 Set Quarter to 0")
            examples = [
                ("11904151", "Kepler-10b"),
                ("10593626", "Kepler-11"),
                ("10666592", "Kepler-22b"),
                ("8462852", "Tabby's Star"),
                ("10024862", "Kepler-5b"),
            ]
        else:
            st.info("💡 Set Sector to 0")
            examples = [
                ("25155310", "TOI-4633"),
                ("279741679", "TOI-2109"),
                ("441462736", "TOI-849"),
                ("307210830", "Multi-planet"),
            ]
        
        for ex_id, desc in examples:
            if st.button(f"{ex_id}", key=f"classify_{ex_id}", help=desc, use_container_width=True):
                st.session_state.selected_star = ex_id
                st.rerun()
    
    if "selected_star" in st.session_state:
        star_id = st.session_state.pop("selected_star")
    
    # Ready state
    if not analyze_button:
        st.markdown("## 🎯 Ready for Analysis")
        st.info("💡 Set Quarter/Sector to **0** for maximum sensitivity!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🔬 Pipeline Steps**
            - Download pixel data
            - Extract light curve
            - Remove stellar trends
            - Quality filtering
            """)
        
        with col2:
            st.markdown("""
            **🔍 TLS Detection**
            - Transit Least Squares
            - Limb-darkened models
            - Period search
            - Signal optimization
            """)
        
        with col3:
            st.markdown("""
            **🤖 Classification**
            - ML Random Forest
            - NASA vetting tests
            - Odd-even analysis
            - Binary detection
            """)
        
        return
    
    # Validate star ID
    if not star_id.strip():
        st.error("Please enter a star ID")
        return
    
    try:
        star_id_num = int(star_id.strip())
    except ValueError:
        st.error("❌ Star ID must be a number")
        return
    
    # Create analyzer
    analyzer = ExoplanetAnalyzer()
    
    # Convert 0 to None
    if mission == "Kepler":
        quarter_val = None if quarter == 0 else quarter
        if quarter == 0:
            st.success("✅ Downloading ALL Kepler quarters")
        else:
            st.warning(f"⚠ Only Quarter {quarter}")
    else:
        quarter_val = None
    
    if mission == "TESS":
        sector_val = None if sector == 0 else sector
        if sector == 0:
            st.success("✅ Downloading ALL TESS sectors")
        else:
            st.warning(f"⚠ Only Sector {sector}")
    else:
        sector_val = None
    
    # Run analysis
    success = analyzer.run_full_analysis(
        star_id=star_id,
        mission=mission,
        quarter=quarter_val,
        sector=sector_val,
    )
    
    if not success:
        st.error("❌ Analysis failed")
        return
    
    st.success("✅ Analysis completed!")
    
    # Display results
    summary = analyzer.get_analysis_summary()
    
    if summary:
        st.markdown("---")
        st.markdown("### 📈 Quick Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            classification = summary.get("classification", "UNKNOWN")
            if "STRONG" in classification:
                st.success(classification)
            elif "CANDIDATE" in classification:
                st.warning(classification)
            else:
                st.error(classification)
        
        with col2:
            st.metric("Confidence", f"{summary.get('confidence', 0):.1%}")
        
        with col3:
            st.metric("Period", f"{summary.get('period', 0):.3f} d")
        
        with col4:
            st.metric("SNR", f"{summary.get('depth_significance', 0):.1f} σ")
        
        analyzer._display_final_summary()
        
        # Export
        st.markdown("---")
        st.markdown("### 💾 Export Results")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            try:
                json_data = analyzer.export_results(format='json')
                if json_data:
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name=f"analysis_{star_id}_{mission}.json",
                        mime="application/json",
                    )
            except:
                pass
        
        with export_col2:
            try:
                csv_data = analyzer.export_results(format='csv')
                if csv_data:
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"analysis_{star_id}_{mission}.csv",
                        mime="text/csv",
                    )
            except:
                pass


# ================================================================
# MAIN APP ROUTER
# ================================================================

def main():
    """Main app router"""
    
    # Initialize session state
    if 'mode' not in st.session_state:
        st.session_state.mode = 'home'
    
    # Route to appropriate mode
    if st.session_state.mode == 'home':
        show_homepage()
    elif st.session_state.mode == 'known_planets':
        run_known_planets_mode()
    elif st.session_state.mode == 'classify_new':
        run_classify_new_mode()
    else:
        # Fallback
        st.session_state.mode = 'home'
        st.rerun()


if __name__ == "__main__":
    main()