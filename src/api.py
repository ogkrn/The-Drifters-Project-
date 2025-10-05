# api.py - FastAPI Backend for React Frontend

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import sys
from pathlib import Path
import traceback

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import your existing modules
from exoplanet_analyzer import ExoplanetAnalyzer
from known_planets_viewer import KnownPlanetsViewer

# Initialize FastAPI
app = FastAPI(
    title="Exoplanet Discovery API",
    description="Professional exoplanet detection and classification API",
    version="1.0.0"
)

# CORS - Allow React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "https://the-drifters-project.onrender.com/"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
# REQUEST/RESPONSE MODELS
# ================================================================

class KnownPlanetRequest(BaseModel):
    star_id: str
    mission: str  # "Kepler" or "TESS"

class ClassifyPlanetRequest(BaseModel):
    star_id: str
    mission: str
    quarter: Optional[int] = None
    sector: Optional[int] = None

class AnalysisResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# ================================================================
# ENDPOINTS
# ================================================================

@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "online",
        "message": "Exoplanet Discovery API",
        "version": "1.0.0",
        "endpoints": {
            "known_planets": "/api/known-planets",
            "classify": "/api/classify",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Check if API is running"""
    return {"status": "healthy", "service": "exoplanet-api"}

# ----------------------------------------------------------------
# KNOWN PLANETS ENDPOINT (NASA Archive Lookup)
# ----------------------------------------------------------------

@app.post("/api/known-planets", response_model=AnalysisResponse)
async def get_known_planets(request: KnownPlanetRequest):
    """
    Search NASA archive for known planets/candidates
    
    Example request:
    {
        "star_id": "10593626",
        "mission": "Kepler"
    }
    """
    
    try:
        viewer = KnownPlanetsViewer()
        
        # Search NASA archive
        results = viewer.search_star_direct(request.star_id, request.mission)
        
        if results is None or len(results) == 0:
            return AnalysisResponse(
                success=False,
                error=f"No data found for {request.mission} ID {request.star_id}"
            )
        
        # Convert DataFrame to dict
        planets_list = results.to_dict('records')
        
        return AnalysisResponse(
            success=True,
            data={
                "star_id": request.star_id,
                "mission": request.mission,
                "count": len(planets_list),
                "planets": planets_list
            }
        )
        
    except Exception as e:
        return AnalysisResponse(
            success=False,
            error=f"Search failed: {str(e)}"
        )

# ----------------------------------------------------------------
# CLASSIFY NEW PLANETS ENDPOINT (ML Analysis)
# ----------------------------------------------------------------

@app.post("/api/classify", response_model=AnalysisResponse)
async def classify_planet(request: ClassifyPlanetRequest):
    """
    Run ML classification on a star
    
    Example request:
    {
        "star_id": "11904151",
        "mission": "Kepler",
        "quarter": 0,
        "sector": null
    }
    """
    
    try:
        analyzer = ExoplanetAnalyzer()
        
        # Convert 0 to None
        quarter = None if request.quarter == 0 else request.quarter
        sector = None if request.sector == 0 else request.sector
        
        # Run analysis (this takes time - consider async/background tasks for production)
        success = analyzer.run_full_analysis(
            star_id=request.star_id,
            mission=request.mission,
            quarter=quarter,
            sector=sector
        )
        
        if not success:
            return AnalysisResponse(
                success=False,
                error="Analysis failed - check star ID and data availability"
            )
        
        # Get results
        summary = analyzer.get_analysis_summary()
        
        if not summary:
            return AnalysisResponse(
                success=False,
                error="No analysis results generated"
            )
        
        # Return results
        return AnalysisResponse(
            success=True,
            data=summary
        )
        
    except Exception as e:
        return AnalysisResponse(
            success=False,
            error=f"Classification failed: {str(e)}\n{traceback.format_exc()}"
        )

# ----------------------------------------------------------------
# ADDITIONAL UTILITY ENDPOINTS
# ----------------------------------------------------------------

@app.get("/api/missions")
async def get_missions():
    """Get available missions"""
    return {
        "missions": [
            {
                "id": "kepler",
                "name": "Kepler",
                "description": "NASA Kepler Mission",
                "id_type": "KIC",
                "examples": ["10593626", "11904151", "10666592"]
            },
            {
                "id": "tess",
                "name": "TESS",
                "description": "Transiting Exoplanet Survey Satellite",
                "id_type": "TIC",
                "examples": ["279741379", "460205581", "307210830"]
            }
        ]
    }

@app.get("/api/examples/{mission}")
async def get_examples(mission: str):
    """Get example star IDs for a mission"""
    
    examples = {
        "kepler": [
            {"id": "10593626", "name": "Kepler-11 (6 planets)"},
            {"id": "11904151", "name": "Kepler-10b"},
            {"id": "10666592", "name": "Kepler-22b"},
        ],
        "tess": [
            {"id": "279741379", "name": "TOI-270"},
            {"id": "460205581", "name": "TOI-700"},
            {"id": "307210830", "name": "TOI-175"},
        ]
    }
    
    mission_lower = mission.lower()
    if mission_lower not in examples:
        raise HTTPException(status_code=404, detail="Mission not found")
    
    return {"mission": mission, "examples": examples[mission_lower]}

# ----------------------------------------------------------------
# RUN SERVER
# ----------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)