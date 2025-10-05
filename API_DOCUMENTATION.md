# 🌟 Exoplanet Discovery API Documentation

**Version:** 1.0.0  
**Base URL:** `http://localhost:8000`  
**Protocol:** REST API with JSON  
**Framework:** FastAPI + Python 3.11

---

## 📋 Table of Contents

1. [Getting Started](#getting-started)
2. [API Endpoints](#api-endpoints)
3. [Data Models](#data-models)
4. [Example Requests](#example-requests)
5. [Error Handling](#error-handling)
6. [React Integration Guide](#react-integration-guide)

---

## 🚀 Getting Started

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server
cd src
python api.py

# Server will start at http://localhost:8000
```

### CORS Configuration

The API is pre-configured to accept requests from:
- `http://localhost:3000` (Create React App)
- `http://localhost:5173` (Vite)

To add more origins, edit `api.py`:

```python
allow_origins=[
    "http://localhost:3000",
    "http://localhost:5173",
    "https://your-production-domain.com"
]
```

---

## 🔌 API Endpoints

### 1. Health Check

**GET** `/`

Check if API is online.

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

### 2. Get Known Planets (NASA Archive Search)

**POST** `/api/known-planets`

Search NASA Exoplanet Archive for confirmed planets and candidates.

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
        "kepoi_name": "K00082.05",
        "koi_period": 10.30375,
        "koi_depth": 2560.0,
        "koi_duration": 2.46,
        "koi_prad": 3.15,
        "koi_pdisposition": "CONFIRMED",
        "koi_model_snr": 147.3,
        "koi_srad": 1.1,
        "koi_kepmag": 13.8
      }
      // ... more planets
    ]
  },
  "error": null
}
```

**Error Response (200 with success=false):**
```json
{
  "success": false,
  "data": null,
  "error": "No data found for Kepler ID 99999999"
}
```

---

### 3. Classify New Planet (ML Analysis)

**POST** `/api/classify`

Run machine learning analysis to detect and classify exoplanets.

⚠️ **Note:** This is a **long-running operation** (30-120 seconds). Consider implementing:
- Frontend loading states
- WebSocket for real-time updates (future enhancement)
- Background task processing (future enhancement)

**Request Body:**
```json
{
  "star_id": "11904151",
  "mission": "Kepler",
  "quarter": null,
  "sector": null
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `star_id` | string | Yes | KIC (Kepler) or TIC (TESS) ID |
| `mission` | string | Yes | "Kepler" or "TESS" |
| `quarter` | int/null | No | Specific Kepler quarter (null = all) |
| `sector` | int/null | No | Specific TESS sector (null = all) |

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "star_id": "11904151",
    "mission": "Kepler",
    "analysis_timestamp": "2025-10-05 14:30:22",
    
    "classification": "CONFIRMED EXOPLANET",
    "confidence": 0.95,
    
    "period": 0.837495,
    "depth": 0.0064,
    "depth_ppm": 6400.0,
    "depth_significance": 142.5,
    "duration_hours": 1.82,
    "significance": 142.5,
    
    "bls_power": 0.245,
    "num_transits": 47,
    "vetting_score": 0.92,
    "estimated_planet_radius": 1.45,
    "baseline_flux": 1.0,
    "noise_level": 0.00012,
    
    "ml_probabilities": {
      "false_positive": 0.02,
      "candidate": 0.03,
      "confirmed": 0.95
    },
    "ml_prediction": "CONFIRMED",
    
    "odd_even_test_status": "PASS",
    "odd_even_depth_ratio": 1.03,
    "odd_even_depth_odd": 6420.0,
    "odd_even_depth_even": 6380.0,
    
    "secondary_eclipse_status": "NO_SECONDARY",
    "secondary_eclipse_ratio": 0.12,
    "secondary_eclipse_depth": 780.0,
    
    "all_features": {
      "period": 0.837495,
      "depth": 0.0064,
      "duration_hours": 1.82,
      "depth_significance": 142.5,
      "baseline_flux": 1.0,
      "noise_level": 0.00012,
      "bls_power": 0.245
    },
    
    "vetting_details": {
      "odd_even_test": {
        "status": "PASS",
        "depth_ratio": 1.03,
        "depth_odd": 6420.0,
        "depth_even": 6380.0
      },
      "secondary_eclipse_test": {
        "status": "NO_SECONDARY",
        "secondary_ratio": 0.12,
        "secondary_depth": 780.0
      }
    },
    
    "analysis_successful": true,
    "analysis_complete": true,
    "pipeline_version": "5.0_CANDIDATE_BASED"
  },
  "error": null
}
```

**Error Response (200 with success=false):**
```json
{
  "success": false,
  "data": null,
  "error": "Analysis failed - check star ID and data availability"
}
```

---

### 4. Get Available Missions

**GET** `/api/missions`

Get list of supported space missions.

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

### 5. Get Example Star IDs

**GET** `/api/examples/{mission}`

Get example star IDs for testing.

**Parameters:**
- `mission` (path): "kepler" or "tess"

**Example:** `GET /api/examples/kepler`

**Response:**
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

## 📦 Data Models

### AnalysisResponse

```typescript
interface AnalysisResponse {
  success: boolean;
  data?: any;
  error?: string;
}
```

### Classification Result

```typescript
interface ClassificationResult {
  // Identification
  star_id: string;
  mission: string;
  analysis_timestamp: string;
  
  // Classification
  classification: "CONFIRMED EXOPLANET" | "PLANET CANDIDATE" | "FALSE POSITIVE" | "ERROR";
  confidence: number; // 0.0 to 1.0
  
  // Orbital Parameters
  period: number; // days
  depth: number; // fractional (0-1)
  depth_ppm: number; // parts per million
  depth_significance: number; // sigma
  duration_hours: number;
  significance: number; // sigma
  
  // Detection Metrics
  bls_power: number;
  num_transits: number;
  vetting_score: number;
  estimated_planet_radius: number; // Earth radii
  baseline_flux: number;
  noise_level: number;
  
  // ML Predictions
  ml_probabilities: {
    false_positive: number;
    candidate: number;
    confirmed: number;
  };
  ml_prediction: string;
  
  // Vetting Tests
  odd_even_test_status: "PASS" | "FAIL" | "UNKNOWN";
  odd_even_depth_ratio: number;
  odd_even_depth_odd: number;
  odd_even_depth_even: number;
  
  secondary_eclipse_status: "NO_SECONDARY" | "POSSIBLE_SECONDARY" | "UNKNOWN";
  secondary_eclipse_ratio: number;
  secondary_eclipse_depth: number;
  
  // Metadata
  analysis_successful: boolean;
  analysis_complete: boolean;
  pipeline_version: string;
}
```

### Known Planet

```typescript
interface KnownPlanet {
  kepoi_name: string; // or toi_id for TESS
  koi_period: number; // orbital period (days)
  koi_depth: number; // transit depth (ppm)
  koi_duration: number; // transit duration (hours)
  koi_prad: number; // planet radius (Earth radii)
  koi_pdisposition: "CONFIRMED" | "CANDIDATE" | "FALSE POSITIVE";
  koi_model_snr: number; // signal-to-noise ratio
  koi_srad: number; // stellar radius (solar radii)
  koi_kepmag: number; // stellar magnitude
}
```

---

## 💡 Example Requests

### React with Axios

```javascript
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

// 1. Search NASA Archive
export const searchKnownPlanets = async (starId, mission) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/api/known-planets`, {
      star_id: starId,
      mission: mission
    });
    return response.data;
  } catch (error) {
    console.error('Search failed:', error);
    throw error;
  }
};

// 2. Classify New Planet
export const classifyPlanet = async (starId, mission, quarter = null, sector = null) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/api/classify`, {
      star_id: starId,
      mission: mission,
      quarter: quarter,
      sector: sector
    });
    return response.data;
  } catch (error) {
    console.error('Classification failed:', error);
    throw error;
  }
};

// 3. Get Missions
export const getMissions = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/api/missions`);
    return response.data;
  } catch (error) {
    console.error('Failed to get missions:', error);
    throw error;
  }
};
```

### React Component Example

```jsx
import React, { useState } from 'react';
import { classifyPlanet } from './api';

function ExoplanetAnalyzer() {
  const [starId, setStarId] = useState('');
  const [mission, setMission] = useState('Kepler');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await classifyPlanet(starId, mission);
      
      if (response.success) {
        setResult(response.data);
      } else {
        setError(response.error);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1>Exoplanet Classifier</h1>
      
      <input
        type="text"
        value={starId}
        onChange={(e) => setStarId(e.target.value)}
        placeholder="Enter Star ID"
      />
      
      <select value={mission} onChange={(e) => setMission(e.target.value)}>
        <option value="Kepler">Kepler</option>
        <option value="TESS">TESS</option>
      </select>
      
      <button onClick={handleAnalyze} disabled={loading}>
        {loading ? 'Analyzing...' : 'Analyze'}
      </button>

      {error && <div className="error">{error}</div>}
      
      {result && (
        <div className="result">
          <h2>Classification: {result.classification}</h2>
          <p>Confidence: {(result.confidence * 100).toFixed(1)}%</p>
          <p>Period: {result.period.toFixed(3)} days</p>
          <p>Depth: {result.depth_ppm.toFixed(0)} ppm</p>
          <p>Significance: {result.depth_significance.toFixed(1)}σ</p>
        </div>
      )}
    </div>
  );
}

export default ExoplanetAnalyzer;
```

---

## ⚠️ Error Handling

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success (check `success` field in response) |
| 404 | Endpoint not found |
| 422 | Validation error (invalid request body) |
| 500 | Server error |

### Error Response Structure

All errors return this structure:

```json
{
  "success": false,
  "data": null,
  "error": "Error message here"
}
```

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| "No data found for Kepler ID XXX" | Star ID doesn't exist in NASA archive | Verify star ID is correct |
| "Analysis failed - check star ID and data availability" | No light curve data available | Try different quarter/sector or star ID |
| "Classification failed: [error]" | Internal processing error | Check logs, report issue |
| "Mission not found" | Invalid mission name | Use "Kepler" or "TESS" |

---

## 🔄 React Integration Guide

### Step 1: Install Dependencies

```bash
npm install axios
# or
yarn add axios
```

### Step 2: Create API Service

Create `src/services/exoplanetApi.js`:

```javascript
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 180000, // 3 minutes for long-running analysis
  headers: {
    'Content-Type': 'application/json',
  },
});

export const exoplanetApi = {
  // Search NASA archive
  searchKnownPlanets: async (starId, mission) => {
    const response = await api.post('/api/known-planets', {
      star_id: starId,
      mission: mission,
    });
    return response.data;
  },

  // Classify new planet (long-running)
  classifyPlanet: async (starId, mission, quarter = null, sector = null) => {
    const response = await api.post('/api/classify', {
      star_id: starId,
      mission: mission,
      quarter: quarter,
      sector: sector,
    });
    return response.data;
  },

  // Get missions
  getMissions: async () => {
    const response = await api.get('/api/missions');
    return response.data;
  },

  // Get examples
  getExamples: async (mission) => {
    const response = await api.get(`/api/examples/${mission.toLowerCase()}`);
    return response.data;
  },

  // Health check
  healthCheck: async () => {
    const response = await api.get('/health');
    return response.data;
  },
};

export default exoplanetApi;
```

### Step 3: Create Custom Hook

Create `src/hooks/useExoplanetAnalysis.js`:

```javascript
import { useState } from 'react';
import { exoplanetApi } from '../services/exoplanetApi';

export const useExoplanetAnalysis = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);

  const analyze = async (starId, mission, quarter = null, sector = null) => {
    setLoading(true);
    setError(null);
    setData(null);

    try {
      const result = await exoplanetApi.classifyPlanet(starId, mission, quarter, sector);
      
      if (result.success) {
        setData(result.data);
        return result.data;
      } else {
        setError(result.error);
        return null;
      }
    } catch (err) {
      const errorMessage = err.response?.data?.error || err.message || 'Analysis failed';
      setError(errorMessage);
      return null;
    } finally {
      setLoading(false);
    }
  };

  const searchKnown = async (starId, mission) => {
    setLoading(true);
    setError(null);
    setData(null);

    try {
      const result = await exoplanetApi.searchKnownPlanets(starId, mission);
      
      if (result.success) {
        setData(result.data);
        return result.data;
      } else {
        setError(result.error);
        return null;
      }
    } catch (err) {
      const errorMessage = err.response?.data?.error || err.message || 'Search failed';
      setError(errorMessage);
      return null;
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setLoading(false);
    setError(null);
    setData(null);
  };

  return {
    loading,
    error,
    data,
    analyze,
    searchKnown,
    reset,
  };
};
```

### Step 4: Use in Component

```jsx
import React, { useState } from 'react';
import { useExoplanetAnalysis } from './hooks/useExoplanetAnalysis';

function App() {
  const [starId, setStarId] = useState('11904151');
  const [mission, setMission] = useState('Kepler');
  const { loading, error, data, analyze } = useExoplanetAnalysis();

  const handleSubmit = async (e) => {
    e.preventDefault();
    await analyze(starId, mission);
  };

  return (
    <div className="App">
      <h1>Exoplanet Discovery</h1>
      
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={starId}
          onChange={(e) => setStarId(e.target.value)}
          placeholder="Star ID"
          required
        />
        
        <select value={mission} onChange={(e) => setMission(e.target.value)}>
          <option value="Kepler">Kepler</option>
          <option value="TESS">TESS</option>
        </select>
        
        <button type="submit" disabled={loading}>
          {loading ? 'Analyzing... (this may take 1-2 minutes)' : 'Analyze'}
        </button>
      </form>

      {error && (
        <div className="error-box">
          <h3>Error</h3>
          <p>{error}</p>
        </div>
      )}

      {data && (
        <div className="results">
          <h2>Results for {data.star_id}</h2>
          
          <div className="classification">
            <h3>{data.classification}</h3>
            <p>Confidence: {(data.confidence * 100).toFixed(1)}%</p>
          </div>

          <div className="details">
            <h4>Orbital Parameters</h4>
            <ul>
              <li>Period: {data.period.toFixed(3)} days</li>
              <li>Depth: {data.depth_ppm.toFixed(0)} ppm</li>
              <li>Duration: {data.duration_hours.toFixed(2)} hours</li>
              <li>Significance: {data.depth_significance.toFixed(1)}σ</li>
            </ul>
          </div>

          <div className="ml-predictions">
            <h4>ML Probabilities</h4>
            <ul>
              <li>Confirmed: {(data.ml_probabilities.confirmed * 100).toFixed(1)}%</li>
              <li>Candidate: {(data.ml_probabilities.candidate * 100).toFixed(1)}%</li>
              <li>False Positive: {(data.ml_probabilities.false_positive * 100).toFixed(1)}%</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
```

---

## 🎨 UI/UX Recommendations

### Loading States

Since `/api/classify` takes 30-120 seconds:

1. **Progress Indicator:** Show a progress bar or spinner
2. **Status Messages:** Display intermediate steps:
   - "Downloading light curve data..."
   - "Processing pixels..."
   - "Running transit detection..."
   - "Applying ML classification..."
3. **Disable Submit:** Prevent multiple simultaneous requests
4. **Timeout Warning:** Warn users about long processing time

### Error Display

```jsx
{error && (
  <div className="alert alert-error">
    <strong>Analysis Failed:</strong>
    <p>{error}</p>
    <button onClick={reset}>Try Again</button>
  </div>
)}
```

### Success Visualization

- Use color coding: Green (Confirmed), Yellow (Candidate), Red (False Positive)
- Display charts/graphs (use Chart.js or D3.js)
- Show confidence meter/gauge
- Display vetting test results with icons (✓ Pass, ✗ Fail)

---

## 🔒 Security Considerations

1. **API Key:** Add authentication for production
2. **Rate Limiting:** Prevent abuse (use FastAPI dependencies)
3. **Input Validation:** Already handled by Pydantic models
4. **CORS:** Update origins for production domains

---

## 📊 Performance Tips

1. **Caching:** Cache known planet results (they don't change)
2. **Debouncing:** Debounce search inputs
3. **Pagination:** Paginate known planet results if >50 planets
4. **Background Jobs:** Use Celery/Redis for long-running tasks (future)
5. **WebSockets:** Real-time progress updates (future)

---

## 🐛 Debugging

### Enable Debug Mode

```bash
# Run with debug logging
python api.py --log-level debug
```

### Check API Status

```bash
curl http://localhost:8000/health
```

### Test Endpoints

```bash
# Known planets
curl -X POST http://localhost:8000/api/known-planets \
  -H "Content-Type: application/json" \
  -d '{"star_id":"10593626","mission":"Kepler"}'

# Classify
curl -X POST http://localhost:8000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"star_id":"11904151","mission":"Kepler","quarter":null,"sector":null}'
```

---

## 📚 Additional Resources

- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **React Axios:** https://axios-http.com/docs/intro
- **NASA Exoplanet Archive:** https://exoplanetarchive.ipac.caltech.edu/
- **Lightkurve Documentation:** https://docs.lightkurve.org/

---

## 🆘 Support

For issues or questions:
1. Check this documentation
2. Review API logs
3. Test with known example IDs
4. Report issues with full error messages and request details

---

**Happy Planet Hunting! 🌍🔭**
