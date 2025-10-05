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

# --- VISUAL THEME: Dark Space Theme + Subtle Starfield & Nebula ---
st.markdown(
    """
<style>
/* Import a futuristic font (falls back gracefully) */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;500;700&display=swap');

:root{
  --bg:#03040a; /* deep space black */
  --panel: rgba(255,255,255,0.03);
  --accent-1: #6EE7F3; /* cyan */
  --accent-2: #9B7BFF; /* purple */
  --accent-3: #FFB86B; /* warm */
  --glass: rgba(255,255,255,0.02);
}

/* Page background: starfield using multiple layered radial-gradients for subtle depth */
html, body, .appview-container, .main {
  background: radial-gradient(ellipse at 10% 10%, rgba(120,94,255,0.06), transparent 8%),
              radial-gradient(ellipse at 90% 80%, rgba(14,165,233,0.03), transparent 8%),
              linear-gradient(180deg, rgba(2,6,23,1) 0%, rgba(6,12,30,1) 100%);
  color: #E6F0FF;
  font-family: 'Rajdhani', 'Orbitron', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
}

/* Starfield animation: generated with repeating radial-gradient and keyframes */
#starfield{
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  background-image:
    radial-gradient(rgba(255,255,255,0.8) 1px, transparent 1px),
    radial-gradient(rgba(255,255,255,0.6) 1px, transparent 1px),
    radial-gradient(rgba(255,255,255,0.4) 1px, transparent 1px);
  background-size: 1200px 1200px, 800px 800px, 400px 400px;
  opacity: 0.28;
  animation: drift 60s linear infinite;
  mix-blend-mode: screen;
}

@keyframes drift{
  from { transform: translateY(0px) translateX(0px); }
  to   { transform: translateY(-120px) translateX(60px); }
}

/* Nebula glow overlay */
#nebula{
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  background: radial-gradient(circle at 15% 20%, rgba(155,123,255,0.06), transparent 8%),
              radial-gradient(circle at 80% 80%, rgba(110,231,243,0.035), transparent 8%),
              radial-gradient(circle at 70% 30%, rgba(255,184,107,0.02), transparent 6%);
  mix-blend-mode: screen;
  opacity: 0.9;
}

/* Page container to keep controls above the background */
.streamlit-container, .block-container {
  position: relative;
  z-index: 2;
}

/* Headline styling */
.main-header {
  font-size: 2.4rem !important;
  color: white;
  text-align: center;
  margin-bottom: 0.4rem;
  letter-spacing: 0.6px;
  text-shadow: 0 6px 24px rgba(147,122,255,0.16), 0 1px 1px rgba(0,0,0,0.6);
}

.sub-header {
  color: rgba(230,240,255,0.85);
  text-align:center;
  margin-bottom: 1.25rem;
}

/* Glowing badges */
.mission-badge {
  display:inline-block;padding:0.25rem 0.75rem;margin:0.25rem;border-radius:999px;font-weight:700;font-size:0.9rem;
  box-shadow: 0 6px 18px rgba(155,123,255,0.08);
}
.kepler-badge {background: linear-gradient(90deg,#ff7f0e22,#ff7f0e44); color:#fff}
.tess-badge   {background: linear-gradient(90deg,#2ca02c22,#2ca02c44); color:#fff}

/* Mode card: glass panel with subtle glow */
.mode-card {
    padding: 2rem;
    border-radius: 1rem;
    border: 1px solid rgba(255,255,255,0.04);
    margin: 0.6rem 0;
    cursor: pointer;
    transition: all 0.28s ease-in-out;
    background: linear-gradient(180deg, rgba(255,255,255,0.015), rgba(255,255,255,0.01));
    box-shadow: 0 6px 30px rgba(2,6,23,0.6);
}
.mode-card:hover {
    transform: translateY(-6px);
    border-color: rgba(155,123,255,0.55);
    box-shadow: 0 16px 48px rgba(91,50,255,0.12);
}

/* Buttons — modern, rounded and slightly glowing */
button.stButton, div.stButton > button {
  border-radius: 12px;
  padding: 0.6rem 1rem;
  border: none;
  font-weight: 600;
  letter-spacing: 0.4px;
  box-shadow: 0 10px 24px rgba(6,12,30,0.6);
  transition: transform 0.18s ease, box-shadow 0.18s ease;
  background: linear-gradient(90deg,var(--accent-1), var(--accent-2));
  color: #021027;
}

button.stButton:hover, div.stButton > button:hover{
  transform: translateY(-3px);
  box-shadow: 0 18px 40px rgba(110,231,243,0.12);
}

/* Disabled button slightly muted */
button[disabled], button:disabled{ opacity: 0.5; filter: grayscale(0.15); }

/* Sidebar styling */
[data-testid='stSidebar'] { background: linear-gradient(180deg, rgba(3,6,15,0.9), rgba(6,10,20,0.85)); border-right: 1px solid rgba(255,255,255,0.02); }

/* Panels / cards */
.widget-box, .stMarkdown, .stExpander { background: transparent !important; }

/* Metrics: make them stand out */
.metric-container { background: linear-gradient(180deg, rgba(255,255,255,0.015), rgba(255,255,255,0.01)); padding: 0.8rem; border-radius: 10px; }

/* Info boxes */
.stInfo, .stWarning, .stError {
  border-radius: 0.75rem; padding: 0.65rem 0.9rem; margin-bottom:0.6rem;
}

/* Responsive tweaks */
@media (max-width: 800px){
  .main-header { font-size: 1.6rem !important; }
}

</style>

<!-- Background layers placed outside Streamlit flow -->
<div id="starfield"></div>
<div id="nebula"></div>

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
    # Animated / styled header block (keeps semantics intact)
    st.markdown(
        '<div style="text-align:center;position:relative;z-index:2;">'
        "<h1 class=\"main-header\">🔭 Professional Exoplanet Discovery System</h1>"
        "<div class=\"sub-header\">NASA-Standard Analysis Pipeline · Transit Least Squares (TLS)</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    
    st.markdown("## 🚀 Choose Your Mode")
    st.write("Select how you'd like to explore exoplanet data:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="mode-card">
        <h2 style='text-align:center;margin-bottom:4px;'>📚 Known Planets Database</h2>
        <h4 style='text-align:center;color:#9fb3d9;margin-top:2px;'>Browse NASA's Archive</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **What you'll do:**
        - Search by star ID (KIC/TIC)
        - View confirmed planets & candidates
        - See official NASA classifications
        - Access verified measurements
        """)
        
        st.markdown("""
        **Data from:**
        - Kepler Mission (10,000+ objects)
        - TESS Mission (7,000+ objects)
        """)
        
        st.markdown("""
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
        <h2 style='text-align:center;margin-bottom:4px;'>🤖 Classify New Planets</h2>
        <h4 style='text-align:center;color:#9fb3d9;margin-top:2px;'>AI-Powered Discovery</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **What you'll do:**
        - Analyze any star's light curve
        - Run AI/ML classification
        - Detect new planet candidates
        - Professional vetting pipeline
        """)
        
        st.markdown("""
        **Technology:**
        - Transit Least Squares (TLS)
        - Machine Learning classifier
        - NASA-standard vetting
        - Advanced signal processing
        """)
        
        st.markdown("""
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
        '<div style="text-align:center;margin-bottom:2rem;color:#9fb3d9;">'
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
