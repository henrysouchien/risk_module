# LOGGING DEPLOYMENT PLAN - COMPREHENSIVE EDITION

## 🚨 **CRITICAL CONSTRAINTS**

### **HARD CONSTRAINTS - ABSOLUTELY NO EXCEPTIONS:**
- ✅ **PURELY ADDITIVE CHANGES ONLY** - No modifications to existing code whatsoever
- ✅ **SCOPE LIMITED TO LOGGING** - Only add logging calls, nothing else
- ✅ **PRESERVE ALL EXISTING FUNCTIONALITY** - Zero business logic changes
- ✅ **ONLY INSERT NEW LINES** - Never modify existing lines

### **IMPLEMENTATION APPROACH:**
1. **Read existing code** - Understand structure but treat as completely untouchable
2. **Only add new lines** - Insert logging calls as new lines before/after existing code blocks
3. **Use search_replace tool** - Add single lines at specific locations
4. **No function signature changes** - Never modify parameters or return values

---

## **TOOL USAGE GUIDE FOR IMPLEMENTER**

### **Step-by-Step Process:**
1. **Read the target file** - Use `read_file` to understand current structure
2. **Find exact insertion points** - Look for specific function signatures or patterns
3. **Use search_replace tool** - Insert logging lines with specific old_string/new_string pairs
4. **Test functionality** - Run existing code to ensure no regressions
5. **Verify logging works** - Check that logging appears in terminal

### **Example search_replace Usage:**
```python
# To insert logging at start of a function:
old_string = '''def analyze_portfolio(filepath: str):
    """Core portfolio analysis function"""
    config = load_portfolio_config(filepath)'''

new_string = '''def analyze_portfolio(filepath: str):
    """Core portfolio analysis function"""
    from utils.logging import log_portfolio_operation
    import time
    start_time = time.time()
    log_portfolio_operation("portfolio_analysis_start", {"filepath": filepath})
    config = load_portfolio_config(filepath)'''
```

### **Import Statement Placement:**
- **Always add imports inside functions** - Never at module level
- **Place imports immediately after function definition** - Before any business logic
- **Use consistent import pattern**: `from utils.logging import log_*` and `import time`

---

## **DEPLOYMENT TARGETS & INSERTION POINTS**

### **PHASE 1: CRITICAL USER FLOWS**

#### **1. BACKEND API ENDPOINTS** (`routes/api.py`) ✅ **COMMENTS ADDED**

**Target Functions:**
- `analyze()` endpoint (around line 40)
- Error handling blocks
- Service calls within endpoints

**Specific Additive Changes:**

**A. At start of analyze() function:**
```python
# SEARCH FOR THIS PATTERN:
def analyze():
    """Portfolio analysis endpoint"""
    
# INSERT LOGGING AFTER FUNCTION DEFINITION:
def analyze():
    """Portfolio analysis endpoint"""
    from utils.logging import log_api_request, log_performance_metric
    import time
    start_time = time.time()
    log_api_request("analyze", "POST", user_id=request.args.get("key"))
```

**B. Before each service call:**
```python
# SEARCH FOR SERVICE CALL PATTERNS LIKE:
result = portfolio_service.analyze_portfolio(...)

# INSERT TIMING BEFORE:
operation_start = time.time()
result = portfolio_service.analyze_portfolio(...)
log_performance_metric("portfolio_service.analyze", time.time() - operation_start)
```

**C. In error handling blocks:**
```python
# SEARCH FOR EXCEPT BLOCKS:
except Exception as e:
    return jsonify({"error": str(e)}), 500

# INSERT ERROR LOGGING:
except Exception as e:
    log_error_json("api_analyze", {"endpoint": "analyze", "user_key": user_key}, e)
    return jsonify({"error": str(e)}), 500
```

**D. At end of successful requests:**
```python
# SEARCH FOR RETURN STATEMENTS:
return jsonify(response_data), 200

# INSERT SUCCESS LOGGING:
log_api_request("analyze", "POST", user_id=request.args.get("key"), execution_time=time.time() - start_time, status_code=200)
return jsonify(response_data), 200
```

---

#### **2. AUTHENTICATION & SECURITY LOGGING** (`routes/auth.py`) ✅ **COMMENTS ADDED**

**Target Functions:**
- `auth_status()` endpoint
- `google_auth()` endpoint  
- `logout()` endpoint

**Specific Additive Changes:**

**A. At start of auth_status():**
```python
# SEARCH FOR FUNCTION DEFINITION:
def auth_status():
    """Check authentication status."""
    try:
        # Get session from cookie
        session_id = request.cookies.get('session_id')

# INSERT LOGGING:
def auth_status():
    """Check authentication status."""
    from utils.logging import log_api_request, log_performance_metric
    import time
    start_time = time.time()
    log_api_request("auth_status", "GET", user_id=None)
    try:
        # Get session from cookie
        session_id = request.cookies.get('session_id')
```

**B. At start of google_auth():**
```python
# SEARCH FOR FUNCTION DEFINITION:
def google_auth():
    """Handle Google OAuth authentication."""
    try:
        data = request.json

# INSERT LOGGING:
def google_auth():
    """Handle Google OAuth authentication."""
    from utils.logging import log_api_request, log_performance_metric
    import time
    start_time = time.time()
    log_api_request("google_auth", "POST", user_id=None)
    try:
        data = request.json
```

**C. After Google token verification:**
```python
# SEARCH FOR TOKEN VERIFICATION:
user_info, error = auth_service.verify_google_token(token)
if error:
    return jsonify({"error": f"Token verification failed: {error}"}), 401

# INSERT VERIFICATION LOGGING:
user_info, error = auth_service.verify_google_token(token)
log_performance_metric("google_token_verification", time.time() - start_time, {"success": error is None})
if error:
    return jsonify({"error": f"Token verification failed: {error}"}), 401
```

**D. At start of logout():**
```python
# SEARCH FOR FUNCTION DEFINITION:
def logout():
    """Logout user."""
    try:
        session_id = request.cookies.get('session_id')

# INSERT LOGGING:
def logout():
    """Logout user."""
    from utils.logging import log_api_request, log_performance_metric
    import time
    start_time = time.time()
    log_api_request("logout", "POST", user_id=None)
    try:
        session_id = request.cookies.get('session_id')
```

---

#### **3. PLAID INTEGRATION LOGGING** (`routes/plaid.py` & `plaid_loader.py`) ✅ **COMMENTS ADDED**

**routes/plaid.py:**

**A. At start of get_plaid_connections():**
```python
# SEARCH FOR FUNCTION DEFINITION:
def get_plaid_connections():
    """Get user's Plaid connections."""
    user = get_current_user()

# INSERT LOGGING:
def get_plaid_connections():
    """Get user's Plaid connections."""
    from utils.logging import log_api_request, log_performance_metric
    import time
    start_time = time.time()
    log_api_request("plaid_connections", "GET", user_id=None)
    user = get_current_user()
```

**B. At start of create_link_token():**
```python
# SEARCH FOR FUNCTION DEFINITION:
def create_link_token():
    """Create Plaid Link token for user."""
    user = get_current_user()

# INSERT LOGGING:
def create_link_token():
    """Create Plaid Link token for user."""
    from utils.logging import log_api_request, log_performance_metric
    import time
    start_time = time.time()
    log_api_request("plaid_create_link_token", "POST", user_id=None)
    user = get_current_user()
```

**plaid_loader.py:**

**A. At start of create_hosted_link_token():**
```python
# SEARCH FOR FUNCTION DEFINITION:
def create_hosted_link_token(...):
    """Create a hosted Plaid Link token..."""
    
# INSERT LOGGING:
def create_hosted_link_token(...):
    """Create a hosted Plaid Link token..."""
    from utils.logging import log_performance_metric
    import time
    start_time = time.time()
    log_performance_metric("plaid_link_token_creation_start", 0)
```

**B. At start of fetch_plaid_holdings():**
```python
# SEARCH FOR FUNCTION DEFINITION:
def fetch_plaid_holdings(access_token: str, client) -> Dict:
    """Fetch investment holdings from Plaid..."""
    request = InvestmentsHoldingsGetRequest(access_token=access_token)

# INSERT LOGGING:
def fetch_plaid_holdings(access_token: str, client) -> Dict:
    """Fetch investment holdings from Plaid..."""
    from utils.logging import log_performance_metric
    import time
    start_time = time.time()
    log_performance_metric("plaid_holdings_fetch_start", 0)
    request = InvestmentsHoldingsGetRequest(access_token=access_token)
```

---

#### **4. EXTERNAL API DEPENDENCIES** (`data_loader.py`) ✅ **COMMENTS ADDED**

**Target Functions:**
- `fetch_monthly_close()` - Financial Modeling Prep API
- `cache_read()` - Cache operations

**Specific Additive Changes:**

**A. At start of fetch_monthly_close():**
```python
# SEARCH FOR FUNCTION DEFINITION:
def fetch_monthly_close(ticker: str, start_date: Optional[Union[str, datetime]] = None, end_date: Optional[Union[str, datetime]] = None) -> pd.Series:
    """Fetch month-end closing prices for a given ticker from FMP."""
    
# INSERT LOGGING:
def fetch_monthly_close(ticker: str, start_date: Optional[Union[str, datetime]] = None, end_date: Optional[Union[str, datetime]] = None) -> pd.Series:
    """Fetch month-end closing prices for a given ticker from FMP."""
    from utils.logging import log_performance_metric
    import time
    start_time = time.time()
    log_performance_metric("fmp_api_fetch_start", 0, {"ticker": ticker})
```

**B. Before FMP API call:**
```python
# SEARCH FOR API CALL:
resp = requests.get(f"{BASE_URL}/historical-price-eod/full", params=params, timeout=30)

# INSERT API LOGGING:
api_start = time.time()
resp = requests.get(f"{BASE_URL}/historical-price-eod/full", params=params, timeout=30)
log_performance_metric("fmp_api_call", time.time() - api_start, {"ticker": ticker, "endpoint": "historical-price-eod/full"})
```

**C. At start of cache_read():**
```python
# SEARCH FOR FUNCTION DEFINITION:
def cache_read(*, key: Iterable[str | int | float], loader: Callable[[], Union[pd.Series, pd.DataFrame]], cache_dir: Union[str, Path] = "cache", prefix: Optional[str] = None) -> Union[pd.Series, pd.DataFrame]:
    """Returns cached object if present, else computes via loader() and caches."""
    
# INSERT LOGGING:
def cache_read(*, key: Iterable[str | int | float], loader: Callable[[], Union[pd.Series, pd.DataFrame]], cache_dir: Union[str, Path] = "cache", prefix: Optional[str] = None) -> Union[pd.Series, pd.DataFrame]:
    """Returns cached object if present, else computes via loader() and caches."""
    from utils.logging import log_performance_metric
    import time
    start_time = time.time()
    log_performance_metric("cache_operation_start", 0, {"key": str(key)})
```

**D. Cache hit logging:**
```python
# SEARCH FOR CACHE HIT:
if df is not None:
    return df.iloc[:, 0] if df.shape[1] == 1 else df

# INSERT CACHE HIT LOGGING:
if df is not None:
    log_performance_metric("cache_hit", time.time() - start_time, {"key": str(key)})
    return df.iloc[:, 0] if df.shape[1] == 1 else df
```

**E. Cache miss logging:**
```python
# SEARCH FOR CACHE MISS:
obj = loader()  # cache miss → compute

# INSERT CACHE MISS LOGGING:
log_performance_metric("cache_miss", time.time() - start_time, {"key": str(key)})
obj = loader()  # cache miss → compute
```

---

#### **5. MAIN APPLICATION** (`app.py`) ✅ **COMMENTS ADDED**

**Target Functions:**
- `send_key_to_kartra()` - Kartra API integration
- Request handling middleware
- CORS and session tracking

**Specific Additive Changes:**

**A. At start of send_key_to_kartra():**
```python
# SEARCH FOR FUNCTION DEFINITION:
def send_key_to_kartra(email, api_key_value):
    payload = [
        ('app_id', KARTRA_APP_ID),
        ('api_key', KARTRA_API_KEY),
        ('api_password', KARTRA_API_PASSWORD),
        ('lead[email]', email),
        ('lead[custom_fields][0][field_identifier]', 'api_key'),
        ('lead[custom_fields][0][field_value]', api_key_value),

# INSERT LOGGING:
def send_key_to_kartra(email, api_key_value):
    from utils.logging import log_performance_metric
    import time
    start_time = time.time()
    log_performance_metric("kartra_api_call_start", 0, {"email": email})
    payload = [
        ('app_id', KARTRA_APP_ID),
        ('api_key', KARTRA_API_KEY),
        ('api_password', KARTRA_API_PASSWORD),
        ('lead[email]', email),
        ('lead[custom_fields][0][field_identifier]', 'api_key'),
        ('lead[custom_fields][0][field_value]', api_key_value),
```

---

### **PHASE 2: CORE ANALYSIS PIPELINE**

#### **6. CLAUDE INTEGRATION** (`routes/claude.py` & `gpt_helpers.py`) ✅ **COMMENTS ADDED**

**routes/claude.py:**

**Target Function:** `claude_chat()` endpoint

**Specific Additive Changes:**

**A. At start of claude_chat():**
```python
# SEARCH FOR FUNCTION DEFINITION:
def claude_chat():
    """Enhanced Claude chat endpoint with function calling capability"""
    user_key = request.args.get("key", public_key)

# INSERT LOGGING:
def claude_chat():
    """Enhanced Claude chat endpoint with function calling capability"""
    from utils.logging import log_claude_integration
    import time
    start_time = time.time()
    user_key = request.args.get("key", public_key)
    log_claude_integration("chat_request", user_id=user_key)
```

**B. After successful Claude call:**
```python
# SEARCH FOR RESULT PROCESSING:
result = claude_service.process_chat(...)

# INSERT AFTER:
result = claude_service.process_chat(...)
log_claude_integration("chat_response", tokens_used=result.get('tokens'), execution_time=time.time() - start_time, status="success")
```

**C. In error handling:**
```python
# SEARCH FOR EXCEPT BLOCKS:
except Exception as e:
    return jsonify({"error": str(e)}), 500

# INSERT ERROR LOGGING:
except Exception as e:
    log_claude_integration("chat_error", user_id=user_key, execution_time=time.time() - start_time, status="error")
    return jsonify({"error": str(e)}), 500
```

**gpt_helpers.py:**

**Target Function:** `interpret_portfolio_risk()`

**Specific Additive Changes:**

**A. At start of function:**
```python
# SEARCH FOR FUNCTION DEFINITION:
def interpret_portfolio_risk(diagnostics_text: str) -> str:
    """Sends raw printed diagnostics to GPT for layman interpretation."""
    user_prompt = (

# INSERT LOGGING:
def interpret_portfolio_risk(diagnostics_text: str) -> str:
    """Sends raw printed diagnostics to GPT for layman interpretation."""
    from utils.logging import log_claude_integration
    import time
    start_time = time.time()
    log_claude_integration("openai_interpretation", user_id=None)
    user_prompt = (
```

**B. After OpenAI call:**
```python
# SEARCH FOR RESPONSE PROCESSING:
response = client.chat.completions.create(...)

return response.choices[0].message.content.strip()

# INSERT LOGGING:
response = client.chat.completions.create(...)
log_claude_integration("openai_response", tokens_used=response.usage.total_tokens if response.usage else None, execution_time=time.time() - start_time, status="success")
return response.choices[0].message.content.strip()
```

---

#### **7. CORE ANALYSIS - PORTFOLIO RISK** (`portfolio_risk.py`) ✅ **COMMENTS ADDED**

**Target Function:** `build_portfolio_view()` and `_build_portfolio_view_computation()`

**Specific Additive Changes:**

**A. At start of build_portfolio_view():**
```python
# SEARCH FOR FUNCTION DEFINITION:
def build_portfolio_view(weights: Dict[str, float], start_date: str, end_date: str, ...):
    """Build comprehensive portfolio view with LRU caching."""
    
# INSERT LOGGING:
def build_portfolio_view(weights: Dict[str, float], start_date: str, end_date: str, ...):
    """Build comprehensive portfolio view with LRU caching."""
    from utils.logging import log_portfolio_operation, log_performance_metric
    import time
    start_time = time.time()
    log_portfolio_operation("portfolio_view_start", {"num_positions": len(weights), "start_date": start_date, "end_date": end_date})
```

**B. At start of _build_portfolio_view_computation():**
```python
# SEARCH FOR FUNCTION DEFINITION:
def _build_portfolio_view_computation(weights: Dict[str, float], start_date: str, end_date: str, ...):
    """Builds a complete portfolio risk profile using historical returns..."""
    
# INSERT LOGGING:
def _build_portfolio_view_computation(weights: Dict[str, float], start_date: str, end_date: str, ...):
    """Builds a complete portfolio risk profile using historical returns..."""
    from utils.logging import log_portfolio_operation, log_performance_metric
    import time
    start_time = time.time()
    log_portfolio_operation("portfolio_computation_start", {"num_positions": len(weights)})
```

**C. After major computation steps:**
```python
# SEARCH FOR COVARIANCE COMPUTATION:
covariance_matrix = returns_df.cov()

# INSERT PERFORMANCE LOGGING:
covariance_matrix = returns_df.cov()
log_performance_metric("covariance_computation", time.time() - start_time, {"num_assets": len(returns_df.columns)})
```

---

#### **8. CORE ANALYSIS - SCENARIO ANALYSIS** (`core/scenario_analysis.py`) ✅ **COMMENTS ADDED**

**Target Function:** `analyze_scenario()`

**Specific Additive Changes:**

**A. At start of function:**
```python
# SEARCH FOR FUNCTION DEFINITION:
def analyze_scenario(filepath: str, scenario_yaml: Optional[str] = None, delta: Optional[str] = None) -> Dict[str, Any]:
    """Core scenario analysis business logic."""
    
# INSERT LOGGING:
def analyze_scenario(filepath: str, scenario_yaml: Optional[str] = None, delta: Optional[str] = None) -> Dict[str, Any]:
    """Core scenario analysis business logic."""
    from utils.logging import log_portfolio_operation, log_performance_metric
    import time
    start_time = time.time()
    log_portfolio_operation("scenario_analysis_start", {"filepath": filepath, "scenario_yaml": scenario_yaml, "delta": delta})
```

**B. After major processing steps:**
```python
# SEARCH FOR CONFIG LOADING COMPLETION:
config = load_portfolio_config(filepath)
with open("risk_limits.yaml", "r") as f:
    risk_config = yaml.safe_load(f)

# INSERT PERFORMANCE LOGGING:
config = load_portfolio_config(filepath)
log_performance_metric("scenario_config_load", time.time() - start_time)
with open("risk_limits.yaml", "r") as f:
    risk_config = yaml.safe_load(f)
```

**C. At end of function:**
```python
# SEARCH FOR RETURN STATEMENT:
return {
    "scenario_summary": scenario_summary,
    "risk_analysis": risk_analysis,
    "beta_analysis": beta_analysis,
    "comparison_analysis": comparison_analysis
}

# INSERT COMPLETION LOGGING:
log_portfolio_operation("scenario_analysis_complete", {"filepath": filepath, "execution_time": time.time() - start_time})
return {
    "scenario_summary": scenario_summary,
    "risk_analysis": risk_analysis,
    "beta_analysis": beta_analysis,
    "comparison_analysis": comparison_analysis
}
```

---

#### **9. CORE ANALYSIS - OPTIMIZATION** (`core/optimization.py`) ✅ **COMMENTS ADDED**

**Target Functions:** `optimize_min_variance()`, `optimize_max_return()` (if it exists)

**Specific Additive Changes:**

**A. At start of optimize_min_variance():**
```python
# SEARCH FOR FUNCTION DEFINITION:
def optimize_min_variance(filepath: str) -> Dict[str, Any]:
    """Core minimum variance optimization business logic."""
    
# INSERT LOGGING:
def optimize_min_variance(filepath: str) -> Dict[str, Any]:
    """Core minimum variance optimization business logic."""
    from utils.logging import log_portfolio_operation, log_performance_metric
    import time
    start_time = time.time()
    log_portfolio_operation("min_variance_optimization_start", {"filepath": filepath})
```

**B. After optimization steps:**
```python
# SEARCH FOR OPTIMIZATION CALL:
opt_result = run_min_var(...)

# INSERT PERFORMANCE LOGGING:
opt_result = run_min_var(...)
log_performance_metric("min_variance_calculation", time.time() - start_time)
```

**C. At end of function:**
```python
# SEARCH FOR RETURN STATEMENT:
return {
    "optimized_weights": optimized_weights,
    "risk_analysis": risk_analysis,
    "beta_analysis": beta_analysis,
    "optimization_metadata": optimization_metadata
}

# INSERT COMPLETION LOGGING:
log_portfolio_operation("min_variance_optimization_complete", {"filepath": filepath, "execution_time": time.time() - start_time})
return {
    "optimized_weights": optimized_weights,
    "risk_analysis": risk_analysis,
    "beta_analysis": beta_analysis,
    "optimization_metadata": optimization_metadata
}
```

---

#### **10. CORE ANALYSIS - STOCK ANALYSIS** (`core/stock_analysis.py`) ✅ **COMMENTS ADDED**

**Target Function:** `analyze_stock()`

**Specific Additive Changes:**

**A. At start of function:**
```python
# SEARCH FOR FUNCTION DEFINITION:
def analyze_stock(ticker: str, start: Optional[str] = None, end: Optional[str] = None, ...) -> Dict[str, Any]:
    """Core stock analysis business logic."""
    
# INSERT LOGGING:
def analyze_stock(ticker: str, start: Optional[str] = None, end: Optional[str] = None, ...) -> Dict[str, Any]:
    """Core stock analysis business logic."""
    from utils.logging import log_portfolio_operation, log_performance_metric
    import time
    start_time = time.time()
    log_portfolio_operation("stock_analysis_start", {"ticker": ticker, "start": start, "end": end})
```

**B. After major analysis steps:**
```python
# SEARCH FOR ANALYSIS CALLS:
detailed_profile = get_detailed_stock_factor_profile(...)
risk_profile = get_stock_risk_profile(...)

# INSERT PERFORMANCE LOGGING:
detailed_profile = get_detailed_stock_factor_profile(...)
log_performance_metric("stock_factor_analysis", time.time() - start_time)
risk_profile = get_stock_risk_profile(...)
log_performance_metric("stock_risk_analysis", time.time() - start_time)
```

**C. At end of function:**
```python
# SEARCH FOR RETURN STATEMENT:
return {
    "ticker": ticker,
    "analysis_results": analysis_results,
    "detailed_profile": detailed_profile,
    "risk_profile": risk_profile
}

# INSERT COMPLETION LOGGING:
log_portfolio_operation("stock_analysis_complete", {"ticker": ticker, "execution_time": time.time() - start_time})
return {
    "ticker": ticker,
    "analysis_results": analysis_results,
    "detailed_profile": detailed_profile,
    "risk_profile": risk_profile
}
```

---

#### **11. MAIN INTERFACE** (`run_risk.py`) ✅ **COMMENTS ADDED**

**Target Functions:** `run_portfolio()`, `run_stock()`, `run_what_if()`, `run_min_variance()`, `run_max_return()`

**Specific Additive Changes:**

**A. At start of run_portfolio():**
```python
# SEARCH FOR FUNCTION DEFINITION:
def run_portfolio(filepath: str, dual_mode: bool = False) -> Union[str, Dict[str, Any]]:
    """Main portfolio analysis function."""
    
# INSERT LOGGING:
def run_portfolio(filepath: str, dual_mode: bool = False) -> Union[str, Dict[str, Any]]:
    """Main portfolio analysis function."""
    from utils.logging import log_portfolio_operation, log_performance_metric
    import time
    start_time = time.time()
    log_portfolio_operation("portfolio_run_start", {"filepath": filepath, "mode": "CLI" if not dual_mode else "API"})
```

**B. After major steps:**
```python
# SEARCH FOR PORTFOLIO LOADING:
config = load_portfolio_config(filepath)
portfolio = standardize_portfolio_input(config)

# INSERT PERFORMANCE LOGGING:
config = load_portfolio_config(filepath)
log_performance_metric("portfolio_load", time.time() - start_time)
portfolio = standardize_portfolio_input(config)
```

**C. At end of function:**
```python
# SEARCH FOR RETURN STATEMENTS:
if dual_mode:
    return response_data
else:
    return formatted_output

# INSERT COMPLETION LOGGING:
log_portfolio_operation("portfolio_run_complete", {"filepath": filepath, "execution_time": time.time() - start_time})
if dual_mode:
    return response_data
else:
    return formatted_output
```

---

#### **12. PORTFOLIO OPTIMIZER** (`portfolio_optimizer.py`) ✅ **COMMENTS ADDED**

**Target Functions:**
- `run_min_var()` - Minimum variance optimization
- `run_max_return_portfolio()` - Maximum return optimization
- `run_what_if_scenario()` - Scenario optimization

**Specific Additive Changes:**

**A. At start of run_min_var():**
```python
# SEARCH FOR FUNCTION DEFINITION:
def run_min_var(base_weights, config, risk_config, proxies):
    """Run minimum variance optimization..."""
    
# INSERT LOGGING:
def run_min_var(base_weights, config, risk_config, proxies):
    """Run minimum variance optimization..."""
    from utils.logging import log_portfolio_operation, log_performance_metric
    import time
    start_time = time.time()
    log_portfolio_operation("optimization_min_var_start", {"num_positions": len(base_weights)})
```

---

#### **13. PORTFOLIO RISK SCORE** (`portfolio_risk_score.py`) ✅ **COMMENTS ADDED**

**Target Functions:**
- `run_risk_score_analysis()` - Risk score calculation
- Component score calculations

**Specific Additive Changes:**

**A. At start of run_risk_score_analysis():**
```python
# SEARCH FOR FUNCTION DEFINITION:
def run_risk_score_analysis(filepath: str) -> Dict[str, Any]:
    """Calculate comprehensive risk score..."""
    
# INSERT LOGGING:
def run_risk_score_analysis(filepath: str) -> Dict[str, Any]:
    """Calculate comprehensive risk score..."""
    from utils.logging import log_portfolio_operation, log_performance_metric
    import time
    start_time = time.time()
    log_portfolio_operation("risk_score_analysis_start", {"filepath": filepath})
```

---

#### **14. SERVICES LAYER** (`services/`) ✅ **COMMENTS ADDED**

**Target Files:**
- `services/portfolio_service.py`
- `services/auth_service.py`
- `services/claude/chat_service.py`

**Specific Additive Changes:**

**A. At start of service methods:**
```python
# SEARCH FOR SERVICE METHOD DEFINITIONS:
def analyze_portfolio(self, portfolio_data: PortfolioData) -> RiskAnalysisResult:
    """Analyze portfolio using core business logic."""
    
# INSERT LOGGING:
def analyze_portfolio(self, portfolio_data: PortfolioData) -> RiskAnalysisResult:
    """Analyze portfolio using core business logic."""
    from utils.logging import log_portfolio_operation, log_performance_metric
    import time
    start_time = time.time()
    log_portfolio_operation("service_portfolio_analysis_start", {"num_positions": len(portfolio_data.positions)})
```

---

### **PHASE 3: FRONTEND** ✅ **COMMENTS ADDED**

#### **15. FRONTEND COMMUNICATION** (`frontend/src/App.tsx`)

**Target Areas:**
- APIService class methods
- API call functions
- Error handling in frontend

**Specific Additive Changes:**

**A. In APIService methods:**
```typescript
// SEARCH FOR METHOD DEFINITIONS LIKE:
async makeRequest(endpoint: string, method: string = 'GET', data?: any): Promise<any> {
    const url = `${this.baseURL}${endpoint}`;

// INSERT LOGGING:
async makeRequest(endpoint: string, method: string = 'GET', data?: any): Promise<any> {
    console.log(`[API] ${method.toUpperCase()} ${endpoint} - Request started`);
    const startTime = performance.now();
    const url = `${this.baseURL}${endpoint}`;
```

**B. After API responses:**
```typescript
// SEARCH FOR FETCH RESPONSES:
const response = await fetch(url, config);

// INSERT AFTER:
const response = await fetch(url, config);
const endTime = performance.now();
console.log(`[API] ${method.toUpperCase()} ${endpoint} - Response: ${response.status} (${(endTime - startTime).toFixed(1)}ms)`);
```

**C. In error handling:**
```typescript
// SEARCH FOR CATCH BLOCKS:
} catch (error) {
    throw error;
}

// INSERT ERROR LOGGING:
} catch (error) {
    console.error(`[API] ${method.toUpperCase()} ${endpoint} - Error:`, error);
    throw error;
}
```

---

#### **16. FRONTEND CHASSIS SERVICES** ✅ **COMMENTS ADDED**

**Target Files:**
- `frontend/src/chassis/services/APIService.ts`
- `frontend/src/chassis/services/ClaudeService.ts`

**Specific Additive Changes:**

**A. In APIService request method:**
```typescript
// SEARCH FOR REQUEST METHOD:
private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
// INSERT LOGGING:
private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    console.log(`[API] Making request to: ${endpoint}`);
    const startTime = performance.now();
    const url = `${this.baseURL}${endpoint}`;
```

**B. In ClaudeService methods:**
```typescript
// SEARCH FOR CLAUDE METHODS:
async extractPortfolioData(fileContent: string): Promise<Portfolio> {
    const extractionPrompt = `...`;
    
// INSERT LOGGING:
async extractPortfolioData(fileContent: string): Promise<Portfolio> {
    console.log(`[Claude] Starting portfolio extraction`);
    const startTime = performance.now();
    const extractionPrompt = `...`;
```

---

#### **17. FRONTEND CHASSIS MANAGERS** ✅ **COMMENTS ADDED**

**Target Files:**
- `frontend/src/chassis/managers/AuthManager.ts`
- `frontend/src/chassis/managers/PlaidManager.ts`
- `frontend/src/chassis/managers/PortfolioManager.ts`
- `frontend/src/chassis/managers/ChatManager.ts`

**Specific Additive Changes:**

**A. In AuthManager methods:**
```typescript
// SEARCH FOR AUTH METHODS:
public async checkAuthStatus(): Promise<User | null> {
    try {
        const response = await this.apiService.checkAuthStatus();
        
// INSERT LOGGING:
public async checkAuthStatus(): Promise<User | null> {
    console.log(`[Auth] Checking authentication status`);
    const startTime = performance.now();
    try {
        const response = await this.apiService.checkAuthStatus();
```

**B. In PlaidManager methods:**
```typescript
// SEARCH FOR PLAID METHODS:
public async loadUserConnections(): Promise<{ connections: PlaidConnection[]; error: string | null }> {
    try {
        const response = await this.apiService.getConnections();
        
// INSERT LOGGING:
public async loadUserConnections(): Promise<{ connections: PlaidConnection[]; error: string | null }> {
    console.log(`[Plaid] Loading user connections`);
    const startTime = performance.now();
    try {
        const response = await this.apiService.getConnections();
```

**C. In PortfolioManager methods:**
```typescript
// SEARCH FOR PORTFOLIO METHODS:
public async extractPortfolioData(file: File): Promise<{ portfolio: Portfolio | null; error: string | null }> {
    try {
        const fileContent = await this.readFileContent(file);
        
// INSERT LOGGING:
public async extractPortfolioData(file: File): Promise<{ portfolio: Portfolio | null; error: string | null }> {
    console.log(`[Portfolio] Starting portfolio extraction from file: ${file.name}`);
    const startTime = performance.now();
    try {
        const fileContent = await this.readFileContent(file);
```

**D. In ChatManager methods:**
```typescript
// SEARCH FOR CHAT METHODS:
public async sendMessage(userMessage: string, currentMessages: ChatMessage[]): Promise<{ success: boolean; error: string | null }> {
    try {
        const userChatMessage: ChatMessage = {
            
// INSERT LOGGING:
public async sendMessage(userMessage: string, currentMessages: ChatMessage[]): Promise<{ success: boolean; error: string | null }> {
    console.log(`[Chat] Sending message to Claude`);
    const startTime = performance.now();
    try {
        const userChatMessage: ChatMessage = {
```

---

#### **18. FRONTEND COMPONENTS** ✅ **COMMENTS ADDED**

**Target Files:**
- `frontend/src/components/auth/GoogleSignInButton.tsx`
- `frontend/src/components/auth/LandingPage.tsx`
- `frontend/src/components/chat/RiskAnalysisChat.tsx`
- `frontend/src/components/portfolio/TabbedPortfolioAnalysis.tsx`
- `frontend/src/components/portfolio/RiskScoreDisplay.tsx`
- `frontend/src/components/plaid/PlaidLinkButton.tsx`
- `frontend/src/components/plaid/ConnectedAccounts.tsx`

**Specific Additive Changes:**

**A. In GoogleSignInButton:**
```typescript
// SEARCH FOR COMPONENT DEFINITION:
export const GoogleSignInButton: React.FC<GoogleSignInButtonProps> = ({ onSignIn }) => {
    useEffect(() => {
        const loadGoogleScript = () => {
            
// INSERT LOGGING:
export const GoogleSignInButton: React.FC<GoogleSignInButtonProps> = ({ onSignIn }) => {
    // Component-level logging for Google OAuth flow
    console.log('[GoogleSignIn] Component mounted');
    useEffect(() => {
        const loadGoogleScript = () => {
```

**B. In other components:**
```typescript
// SEARCH FOR COMPONENT DEFINITIONS AND INSERT SIMILAR LOGGING PATTERNS
// Focus on user interactions, API calls, and error states
```

---

### **PHASE 4: ADDITIONAL CRITICAL GAPS** ✅ **COMMENTS ADDED**

#### **19. FACTOR UTILITIES** (`factor_utils.py`)

**Target Functions:**
- `compute_factor_metrics()` - Factor calculations
- `calc_monthly_returns()` - Return calculations
- `fetch_excess_return()` - Excess return calculations

**Specific Additive Changes:**

**A. At start of compute_factor_metrics():**
```python
# SEARCH FOR FUNCTION DEFINITION:
def compute_factor_metrics(stock_returns: pd.Series, factor_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    """Runs independent single-factor regressions..."""
    
# INSERT LOGGING:
def compute_factor_metrics(stock_returns: pd.Series, factor_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    """Runs independent single-factor regressions..."""
    from utils.logging import log_performance_metric
    import time
    start_time = time.time()
    log_performance_metric("factor_metrics_calculation_start", 0, {"num_factors": len(factor_dict)})
```

---

#### **20. PROXY BUILDER** (`proxy_builder.py`)

**Target Functions:**
- `inject_all_proxies()` - Proxy injection
- Industry mapping functions
- Peer discovery functions

**Specific Additive Changes:**

**A. At start of inject_all_proxies():**
```python
# SEARCH FOR FUNCTION DEFINITION:
def inject_all_proxies(yaml_path: str = "portfolio.yaml", use_gpt_subindustry: bool = False) -> None:
    """Injects factor proxy mappings into a portfolio YAML file..."""
    
# INSERT LOGGING:
def inject_all_proxies(yaml_path: str = "portfolio.yaml", use_gpt_subindustry: bool = False) -> None:
    """Injects factor proxy mappings into a portfolio YAML file..."""
    from utils.logging import log_performance_metric
    import time
    start_time = time.time()
    log_performance_metric("proxy_injection_start", 0, {"yaml_path": yaml_path, "use_gpt": use_gpt_subindustry})
```

---

#### **21. HELPER MODULES** ✅ **COMMENTS ADDED**

**Target Files:**
- `helpers_input.py` - Input validation and transformation
- `helpers_display.py` - Display rendering and formatting
- `risk_helpers.py` - Risk calculation helpers
- `settings.py` - Configuration management

**Specific Additive Changes:**

**A. In helpers_input.py:**
```python
# SEARCH FOR PARSE FUNCTIONS:
def parse_delta(yaml_path: Optional[str] = None, literal_shift: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, float], Optional[Dict[str, float]]]:
    """Parse a what-if scenario..."""
    
# INSERT LOGGING:
def parse_delta(yaml_path: Optional[str] = None, literal_shift: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, float], Optional[Dict[str, float]]]:
    """Parse a what-if scenario..."""
    from utils.logging import log_performance_metric
    import time
    start_time = time.time()
    log_performance_metric("input_parsing_start", 0, {"yaml_path": yaml_path, "has_literal_shift": literal_shift is not None})
```

**B. In helpers_display.py:**
```python
# SEARCH FOR DISPLAY FUNCTIONS:
def _print_single_portfolio(risk_df, beta_df, title: str = "What-if") -> None:
    """Pretty-print risk-limit and factor-beta tables..."""
    
# INSERT LOGGING:
def _print_single_portfolio(risk_df, beta_df, title: str = "What-if") -> None:
    """Pretty-print risk-limit and factor-beta tables..."""
    from utils.logging import log_performance_metric
    import time
    start_time = time.time()
    log_performance_metric("display_rendering_start", 0, {"title": title, "has_risk_df": not risk_df.empty, "has_beta_df": not beta_df.empty})
```

**C. In risk_helpers.py:**
```python
# SEARCH FOR RISK CALCULATION FUNCTIONS:
def get_worst_monthly_factor_losses(stock_factor_proxies: Dict[str, Dict[str, Union[str, List[str]]]], start_date: str, end_date: str) -> Dict[str, float]:
    """For each unique factor proxy..."""
    
# INSERT LOGGING:
def get_worst_monthly_factor_losses(stock_factor_proxies: Dict[str, Dict[str, Union[str, List[str]]]], start_date: str, end_date: str) -> Dict[str, float]:
    """For each unique factor proxy..."""
    from utils.logging import log_performance_metric
    import time
    start_time = time.time()
    log_performance_metric("risk_calculation_start", 0, {"start_date": start_date, "end_date": end_date, "num_proxies": len(stock_factor_proxies)})
```

**D. In settings.py:**
```python
# SEARCH FOR CONFIGURATION CONSTANTS:
PORTFOLIO_DEFAULTS = {
    "start_date": "2019-01-31",
    "end_date":   "2025-06-27",
    "normalize_weights": False,
    "worst_case_lookback_years": 10
}

# INSERT LOGGING:
# Configuration loading and validation logging
from utils.logging import log_performance_metric
import time
start_time = time.time()
log_performance_metric("configuration_load_start", 0, {"config_type": "portfolio_defaults"})

PORTFOLIO_DEFAULTS = {
    "start_date": "2019-01-31",
    "end_date":   "2025-06-27",
    "normalize_weights": False,
    "worst_case_lookback_years": 10
}
```

---

#### **22. AI FUNCTION REGISTRY** (`ai_function_registry.py`)

**Target Functions:**
- Function registration and validation
- Function call routing and execution
- Result validation and error handling

**Specific Additive Changes:**

**A. At module level:**
```python
# SEARCH FOR MODULE HEADER:
"""
AI Function Registry - Centralized source of truth for all Claude function definitions.
...
"""

# INSERT LOGGING:
"""
AI Function Registry - Centralized source of truth for all Claude function definitions.
...
"""

from utils.logging import log_performance_metric
import time

# Function registry logging initialization
start_time = time.time()
log_performance_metric("ai_function_registry_load_start", 0, {"num_functions": len(AI_FUNCTIONS)})
```

---

## **IMPLEMENTATION SEQUENCE**

### **Phase 1: Critical User Flows**
1. ✅ `routes/api.py` - API endpoints
2. ✅ `routes/auth.py` - Authentication flow
3. ✅ `routes/plaid.py` - Plaid integration
4. ✅ `data_loader.py` - External APIs & cache
5. ✅ `app.py` - Main application endpoints

### **Phase 2: Core Analysis Pipeline**
6. ✅ `routes/claude.py` - Claude integration
7. ✅ `gpt_helpers.py` - OpenAI integration
8. ✅ `portfolio_risk.py` - Core risk calculations
9. ✅ `core/scenario_analysis.py` - Scenario analysis
10. ✅ `core/optimization.py` - Optimization functions
11. ✅ `core/stock_analysis.py` - Stock analysis
12. ✅ `run_risk.py` - Main interface
13. ✅ `portfolio_optimizer.py` - Optimization algorithms
14. ✅ `portfolio_risk_score.py` - Risk scoring
15. ✅ `services/portfolio_service.py` - Service layer

### **Phase 3: Frontend**
16. ✅ `frontend/src/App.tsx` - Main app API calls
17. ✅ `frontend/src/chassis/services/APIService.ts` - API service
18. ✅ `frontend/src/chassis/services/ClaudeService.ts` - Claude service
19. ✅ `frontend/src/chassis/managers/AuthManager.ts` - Auth manager
20. ✅ `frontend/src/chassis/managers/PlaidManager.ts` - Plaid manager
21. ✅ `frontend/src/chassis/managers/PortfolioManager.ts` - Portfolio manager
22. ✅ `frontend/src/chassis/managers/ChatManager.ts` - Chat manager
23. ✅ `frontend/src/components/auth/GoogleSignInButton.tsx` - Google auth
24. ✅ `frontend/src/components/auth/LandingPage.tsx` - Landing page
25. ✅ `frontend/src/components/chat/RiskAnalysisChat.tsx` - Chat component
26. ✅ `frontend/src/components/portfolio/TabbedPortfolioAnalysis.tsx` - Portfolio tabs
27. ✅ `frontend/src/components/portfolio/RiskScoreDisplay.tsx` - Risk display
28. ✅ `frontend/src/components/plaid/PlaidLinkButton.tsx` - Plaid button
29. ✅ `frontend/src/components/plaid/ConnectedAccounts.tsx` - Connected accounts

### **Phase 4: Additional Critical Gaps**
30. ✅ `factor_utils.py` - Factor calculations
31. ✅ `proxy_builder.py` - Proxy building
32. ✅ `helpers_input.py` - Input validation
33. ✅ `helpers_display.py` - Display rendering
34. ✅ `risk_helpers.py` - Risk calculations
35. ✅ `settings.py` - Configuration
36. ✅ `ai_function_registry.py` - Function registry
37. ✅ `plaid_loader.py` - Plaid API operations

---

## **TESTING APPROACH**

### **After Each Phase:**
1. **Run existing functionality** - Ensure no regressions
2. **Check log output** - Verify logging appears in terminal
3. **Test error scenarios** - Ensure error logging works
4. **Performance monitoring** - Check for >1s warnings

### **Final Integration Test:**
1. **Full portfolio analysis** - End-to-end logging
2. **API endpoint testing** - Frontend-backend communication
3. **Error injection testing** - Verify error logging
4. **External API testing** - FMP and Plaid integration logging
5. **Authentication flow testing** - Complete auth cycle logging

---

## **ROLLBACK PLAN**

If any issues arise:
1. **Identify problematic file** - Check which file caused issues
2. **Revert specific changes** - Remove only the additive logging lines
3. **Test functionality** - Ensure core business logic works
4. **Re-implement carefully** - Add logging more selectively

---

## **SUCCESS CRITERIA**

✅ **All existing functionality preserved**
✅ **Real-time logging visible in terminal**
✅ **Performance monitoring working (>1s warnings)**
✅ **Error tracking operational**
✅ **Frontend-backend communication visible**
✅ **External API calls tracked (FMP, Plaid)**
✅ **Authentication flow logged**
✅ **Cache performance** - Hit/miss ratios and timing
✅ **Core analysis pipeline visible**
✅ **No code modifications, only additions**
✅ **Comprehensive coverage** - All 37 target files have logging comments

---

## **NOTES FOR IMPLEMENTER**

### **Key Principles:**
1. **Always use search_replace tool** for precise line insertions
2. **Test after each file** to ensure no regressions
3. **Focus on frontend logging** for connection debugging
4. **Check imports** - Ensure logging imports work in each file
5. **Preserve exact formatting** - Match existing code style
6. **Document any issues** - Note any problems for future reference

### **Common Pitfalls to Avoid:**
- **Never modify existing lines** - Only add new lines
- **Don't change function signatures** - Keep parameters unchanged
- **Place imports inside functions** - Never at module level
- **Use consistent logging patterns** - Follow the examples exactly
- **Test immediately** - Check functionality after each change

### **Critical Focus Areas:**
- **Authentication logging** - Security events and session tracking
- **External API logging** - FMP and Plaid integration monitoring
- **Cache performance** - Hit/miss ratios and timing
- **Frontend connections** - API communication debugging
- **Core calculations** - Risk and optimization performance

### **🎯 COMPREHENSIVE COVERAGE ACHIEVED**

This plan now provides **complete logging coverage** for all 37 target files identified in the comprehensive analysis, with specific insertion points marked by comments in each file. The implementation will provide complete real-time visibility into the entire risk analysis system from authentication through data processing to final analysis delivery.

**FILES WITH LOGGING COMMENTS ADDED: 37/37 ✅**
- Backend: 22 files ✅
- Frontend: 15 files ✅
- Complete system coverage ✅

This comprehensive plan ensures that another Claude implementing the logging system will have exact guidance for every logging insertion point across the entire codebase. 