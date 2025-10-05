# 🚀 Exoplanet Discovery API - Integration Guide

## 📋 Overview

This FastAPI backend provides exoplanet detection and classification services for React frontend applications.

**Base URL (Development):** `http://localhost:8000`

---

## 🔧 Setup & Running

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Run API Server**
```bash
# Development mode (with auto-reload)
cd src
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn api:app --host 0.0.0.0 --port 8000
```

### **3. Access API Documentation**
- **Interactive Docs:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## 🔐 CORS Configuration

### **Fixed CORS Error!**

The API now properly handles CORS for React frontends:

**Development Mode (Default):**
- Allows: `localhost:3000`, `localhost:5173`, `localhost:5174`, `localhost:8080`
- Credentials: Enabled
- Methods: All standard HTTP methods

**Production Mode:**
Set environment variables:
```bash
export ENVIRONMENT=production
export FRONTEND_URL=https://your-frontend.vercel.app
```

---

## 📡 API Endpoints

### **1. Health Check**

#### `GET /`
Root endpoint - API status

**Response:**
```json
{
  "status": "online",
  "message": "Exoplanet Discovery API",
  "version": "1.0.0",
  "endpoints": {
    "known_planets": "/api/known-planets",
    "classify": "/api/classify",
    "health": "/health"
  }
}
```

---

#### `GET /health`
Simple health check

**Response:**
```json
{
  "status": "healthy",
  "service": "exoplanet-api"
}
```

---

### **2. Known Planets Search**

#### `POST /api/known-planets`
Search NASA Exoplanet Archive for confirmed planets/candidates

**Request Body:**
```json
{
  "star_id": "10593626",
  "mission": "Kepler"
}
```

**Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `star_id` | string | Yes | KIC (Kepler) or TIC (TESS) ID |
| `mission` | string | Yes | "Kepler" or "TESS" |

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "star_id": "10593626",
    "mission": "Kepler",
    "count": 6,
    "planets": [
      {
        "kepoi_name": "K00082.01",
        "koi_period": 10.30375,
        "koi_depth": 2691.0,
        "koi_duration": 3.23,
        "koi_prad": 4.19,
        "koi_pdisposition": "CONFIRMED",
        "koi_model_snr": 122.3,
        "koi_srad": 0.961,
        "koi_kepmag": 13.826
      }
      // ... more planets
    ]
  },
  "error": null
}
```

**Error Response (200):**
```json
{
  "success": false,
  "data": null,
  "error": "No data found for Kepler ID 99999999"
}
```

---

### **3. Planet Classification (ML Analysis)**

#### `POST /api/classify`
Run machine learning analysis on a star to detect new planets

**Request Body:**
```json
{
  "star_id": "11904151",
  "mission": "Kepler",
  "quarter": 0,
  "sector": null
}
```

**Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `star_id` | string | Yes | KIC (Kepler) or TIC (TESS) ID |
| `mission` | string | Yes | "Kepler" or "TESS" |
| `quarter` | int | No | Kepler quarter (0 = all data) |
| `sector` | int | No | TESS sector (null = all data) |

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "star_id": "11904151",
    "mission": "Kepler",
    "light_curve": {
      "data_points": 48739,
      "time_span_days": 1426.78,
      "cadence": "long",
      "quarters_used": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    },
    "detection": {
      "detected": true,
      "period_days": 0.837495,
      "depth_ppm": 5284.3,
      "duration_hours": 2.45,
      "snr": 145.2,
      "sde": 218.4,
      "transit_count": 1704,
      "odd_even_mismatch": 0.023
    },
    "classification": {
      "category": "CONFIRMED PLANET",
      "confidence": "very_high",
      "ml_prediction": "CONFIRMED",
      "ml_confidence": 0.97,
      "reasoning": [
        "✅ Very strong detection (SDE=218.4, well above threshold)",
        "✅ ML model confirms: CONFIRMED (97.2% confidence)",
        "✅ High SNR (145.2σ) indicates clear signal",
        "✅ 1704 transits observed with consistent depth"
      ]
    },
    "stellar_properties": {
      "radius": 1.056,
      "temperature": 5627,
      "magnitude": 10.96
    },
    "planet_properties": {
      "radius_earth": 1.47,
      "radius_jupiter": 0.131,
      "orbital_period": 0.837495,
      "semi_major_axis_au": 0.0168,
      "equilibrium_temp_k": 1873
    }
  },
  "error": null
}
```

**Error Response (200):**
```json
{
  "success": false,
  "data": null,
  "error": "Analysis failed - check star ID and data availability"
}
```

---

### **4. Get Available Missions**

#### `GET /api/missions`
Get list of supported space missions

**Response:**
```json
{
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
```

---

### **5. Get Example Star IDs**

#### `GET /api/examples/{mission}`
Get example star IDs for testing

**Parameters:**
- `mission` (path): "kepler" or "tess"

**Response (Kepler):**
```json
{
  "mission": "kepler",
  "examples": [
    {"id": "10593626", "name": "Kepler-11 (6 planets)"},
    {"id": "11904151", "name": "Kepler-10b"},
    {"id": "10666592", "name": "Kepler-22b"}
  ]
}
```

---

## 🎨 React Integration Examples

### **1. Basic Fetch Example**

```javascript
// Search for known planets
async function searchKnownPlanets(starId, mission) {
  try {
    const response = await fetch('http://localhost:8000/api/known-planets', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        star_id: starId,
        mission: mission
      })
    });
    
    const result = await response.json();
    
    if (result.success) {
      console.log(`Found ${result.data.count} planets!`);
      return result.data.planets;
    } else {
      console.error('Error:', result.error);
      return null;
    }
  } catch (error) {
    console.error('Request failed:', error);
    return null;
  }
}

// Usage
const planets = await searchKnownPlanets('10593626', 'Kepler');
```

---

### **2. Axios Example**

```javascript
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 120000, // 2 minutes (classification takes time)
});

// Classify a planet
export async function classifyPlanet(starId, mission, quarter = 0) {
  try {
    const response = await api.post('/api/classify', {
      star_id: starId,
      mission: mission,
      quarter: quarter,
      sector: null
    });
    
    return response.data;
  } catch (error) {
    console.error('Classification failed:', error);
    throw error;
  }
}

// Usage
const result = await classifyPlanet('11904151', 'Kepler');
if (result.success) {
  console.log('Classification:', result.data.classification.category);
}
```

---

### **3. React Hook Example**

```javascript
import { useState } from 'react';

function useExoplanetAPI() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const classifyPlanet = async (starId, mission, quarter = 0) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:8000/api/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          star_id: starId,
          mission: mission,
          quarter: quarter,
          sector: null
        })
      });
      
      const result = await response.json();
      
      if (!result.success) {
        throw new Error(result.error);
      }
      
      return result.data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };
  
  return { classifyPlanet, loading, error };
}

// Usage in component
function ExoplanetClassifier() {
  const { classifyPlanet, loading, error } = useExoplanetAPI();
  const [result, setResult] = useState(null);
  
  const handleClassify = async () => {
    const data = await classifyPlanet('11904151', 'Kepler');
    setResult(data);
  };
  
  return (
    <div>
      <button onClick={handleClassify} disabled={loading}>
        {loading ? 'Analyzing...' : 'Classify Planet'}
      </button>
      {error && <p>Error: {error}</p>}
      {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
    </div>
  );
}
```

---

## ⚠️ Important Notes

### **1. Processing Time**
- **Known Planets Search:** ~1-3 seconds
- **Planet Classification:** **30-120 seconds** (downloads light curve data, runs ML analysis)
- **Recommendation:** Show loading spinner and progress indicators

### **2. Error Handling**
All endpoints return status 200 with `success: false` on application errors. Always check the `success` field:

```javascript
if (result.success) {
  // Handle data
} else {
  // Handle error message in result.error
}
```

### **3. Rate Limiting**
- No rate limiting currently implemented
- NASA MAST API may throttle requests
- Recommendation: Implement client-side request queuing

### **4. CORS Issues?**
If you still get CORS errors:

1. **Check your frontend port** is in the allowed list (api.py lines 35-42)
2. **Add your port** if different:
   ```python
   origins = [
       "http://localhost:YOUR_PORT",
       # ... existing origins
   ]
   ```
3. **Restart the API server** after changes

### **5. Production Deployment**
Set environment variables:
```bash
ENVIRONMENT=production
FRONTEND_URL=https://your-deployed-frontend.com
```

---

## 🧪 Testing Endpoints

### **Using cURL:**

```bash
# Health check
curl http://localhost:8000/health

# Known planets search
curl -X POST http://localhost:8000/api/known-planets \
  -H "Content-Type: application/json" \
  -d '{"star_id": "10593626", "mission": "Kepler"}'

# Classify planet
curl -X POST http://localhost:8000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"star_id": "11904151", "mission": "Kepler", "quarter": 0}'
```

### **Using Postman/Insomnia:**
1. Import the OpenAPI spec from http://localhost:8000/openapi.json
2. Or manually create POST requests to the endpoints above

---

## 📊 Example Star IDs for Testing

### **Kepler (KIC IDs):**
- `10593626` - Kepler-11 (6 confirmed planets)
- `11904151` - Kepler-10b (confirmed hot super-Earth)
- `10666592` - Kepler-22b (habitable zone planet)
- `9579641` - K00115 (weak candidates - needs full mode)

### **TESS (TIC IDs):**
- `279741379` - TOI-270 (multi-planet system)
- `460205581` - TOI-700 (habitable zone planet)
- `307210830` - TOI-175 (confirmed planet)

---

## 🐛 Troubleshooting

### **Issue: "Connection refused"**
- **Solution:** Make sure API server is running (`uvicorn api:app --reload`)

### **Issue: "CORS error"**
- **Solution:** Check your frontend port is in the allowed origins list

### **Issue: "Analysis takes too long"**
- **Expected:** Classification can take 30-120 seconds
- **Solution:** Increase timeout in your HTTP client

### **Issue: "Module not found"**
- **Solution:** Run API from `src/` directory or adjust Python path

---

## 📞 Support

For issues or questions:
- Check the interactive API docs: http://localhost:8000/docs
- Review this documentation
- Check the backend logs for error details

---

**🎉 Happy Coding!** Your React frontend is now ready to discover exoplanets! 🪐
