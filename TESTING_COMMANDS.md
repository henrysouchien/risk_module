# Risk Module Testing Commands

This document contains all the commands to test both CLI functions and API routes in your risk module system.

## 📋 Prerequisites

1. **Ensure you're in the risk_module directory:**
   ```bash
   cd /Users/henrychien/Documents/Jupyter/risk_module
   ```

2. **Verify configuration files exist:**
   ```bash
   ls -la portfolio.yaml risk_limits.yaml
   ```

3. **Test module loading:**
   ```bash
   python3 -c "from portfolio_risk_score import run_risk_score_analysis; print('✅ Modules loaded successfully')"
   ```

## 🖥️ CLI Testing Commands

### 1. Core Portfolio Analysis

**Full Portfolio Risk Analysis (Main Function)**
```bash
python3 run_risk.py --portfolio portfolio.yaml
```
*Expected Output: Risk decomposition, factor exposures, risk limits, beta checks*

**Portfolio Performance Analysis**
```bash
python3 run_risk.py --portfolio portfolio.yaml --performance
```
*Expected Output: Returns, Sharpe ratio, alpha, beta, maximum drawdown, benchmark comparison*

**Risk Score Analysis (0-100 Credit Score Style)**
```bash
python3 -c "from portfolio_risk_score import run_risk_score_analysis; run_risk_score_analysis('portfolio.yaml', 'risk_limits.yaml')"
```
*Expected Output: Overall risk score, component scores, risk factors, recommendations*

### 2. Portfolio Optimization

**Minimum Variance Optimization**
```bash
python3 run_risk.py --portfolio portfolio.yaml --minvar
```
*Expected Output: Optimized weights for minimum risk, risk/beta tables*

**Maximum Return Optimization**
```bash
python3 run_risk.py --portfolio portfolio.yaml --maxreturn
```
*Expected Output: Optimized weights for maximum return, risk/beta tables*

### 3. Scenario Analysis

**What-If Analysis with Inline Weight Changes**
```bash
python3 run_risk.py --portfolio portfolio.yaml --whatif --delta "AAPL:+500bp,SPY:-200bp"
```
*Expected Output: Before/after comparison, risk impact analysis*

**What-If Analysis with Scenario File**
```bash
python3 run_risk.py --portfolio portfolio.yaml --whatif --scenario what_if_portfolio.yaml
```
*Expected Output: Scenario comparison with detailed risk metrics*

### 4. Single Stock Analysis

**Basic Stock Analysis (No Factor Proxies)**
```bash
python3 run_risk.py --stock MSCI --start 2020-01-01 --end 2024-12-31
```
*Expected Output: Basic volatility metrics, market regression (beta, alpha, R²)*

**✅ Detailed Stock Analysis with YAML Factor Proxies (NEW CLI Support!)**
```bash
python3 run_risk.py --stock MSCI --start 2020-01-01 --end 2024-12-31 --yaml-path portfolio.yaml
```
*Expected Output: **Complete factor analysis** - market, momentum, value, industry, subindustry factor exposures*

**✅ Detailed Stock Analysis with JSON Factor Proxies (NEW CLI Support!)**
```bash
python3 run_risk.py --stock MSCI --start 2020-01-01 --end 2024-12-31 \
  --factor-proxies '{"market": "SPY", "momentum": "MTUM", "value": "IWD", "industry": "XLK"}'
```
*Expected Output: **Complete factor analysis** with specified proxies*

**Multiple Stock Detailed Analysis (Now with CLI!)**
```bash
o.yaml# MSFT with detailed factor analysis
python3 run_risk.py --stock MSCI --start 2022-01-01 --end 2024-12-31 --yaml-path portfoli

# SPY with detailed factor analysis  
python3 run_risk.py --stock SPY --start 2020-01-01 --end 2024-12-31 --yaml-path portfolio.yaml

# AAPL with custom factor proxies
python3 run_risk.py --stock AAPL --start 2020-01-01 --end 2024-12-31 \
  --factor-proxies '{"market": "SPY", "momentum": "MTUM", "value": "IWD"}'
```
*Expected Output: **Complete factor analysis** for each stock*

### 5. Setup and Proxy Management

**Set Up Factor Proxies (Required for New Portfolios)**
```bash
python3 run_risk.py --portfolio portfolio.yaml --inject_proxies
```
*Expected Output: Factor proxy setup confirmation*

**Set Up Factor Proxies with AI-Generated Peer Groups**
```bash
python3 run_risk.py --portfolio portfolio.yaml --inject_proxies --use_gpt
```
*Expected Output: AI-generated peer analysis & proxy setup only*

**Complete Setup + Analysis (Chained Commands)**
```bash
# Step 1: Inject proxies
python3 run_risk.py --portfolio portfolio.yaml --inject_proxies --use_gpt

# Step 2: Run analysis with injected proxies
python3 run_risk.py --portfolio portfolio.yaml
```
*Expected Output: **First** proxy setup, **then** complete portfolio analysis*

### 6. AI-Enhanced Analysis

**Get GPT Interpretation of Portfolio Analysis**
```bash
python3 run_risk.py --portfolio portfolio.yaml --gpt
```
*Expected Output: GPT interpretation of risk analysis in plain English*

## 🌐 API Testing Commands

### Setup: Start the Flask Server

**Terminal 1 - Start the Web Server:**
```bash
python3 app.py
```
*Expected Output: Server running on http://localhost:5001*

### Method 1: Quick API Output Viewer (Recommended)

**Terminal 2 - Test API Endpoints:**

**Complete Portfolio Analysis (Structured Data + Formatted Report)**
```bash
python3 show_api_output.py analyze
```
*Expected Output: Complete structured data + CLI-style formatted report*

**Portfolio Risk Score (Structured Data + Formatted Report)**
```bash
python3 show_api_output.py risk-score
```
*Expected Output: Complete structured data + CLI-style formatted report*

**Portfolio Performance (Structured Data + Formatted Report)**
```bash
python3 show_api_output.py performance
```
*Expected Output: Complete structured data + CLI-style formatted report*

**Portfolio Analysis with GPT Interpretation**
```bash
python3 show_api_output.py portfolio-analysis
```
*Expected Output: GPT interpretation of portfolio analysis*

**API Health Check**
```bash
python3 show_api_output.py health
```
*Expected Output: API status, version, timestamp*

**Help - Show Available Endpoints**
```bash
python3 show_api_output.py help
```
*Expected Output: List of all available API endpoints*

### Method 2: Comprehensive API Testing

**Run Full API Test Suite**
```bash
python3 test_api_endpoints.py
```
*Expected Output: Test results for all API endpoints with success/failure status*

### Method 3: Manual curl Commands (Alternative)

**Portfolio Risk Analysis**
```bash
curl -X POST http://localhost:5001/api/analyze?key=paid_key_789 \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Risk Score Analysis**
```bash
curl -X POST http://localhost:5001/api/risk-score?key=paid_key_789 \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Portfolio Performance**
```bash
curl -X POST http://localhost:5001/api/performance?key=paid_key_789 \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Health Check**
```bash
curl -X GET http://localhost:5001/api/health?key=paid_key_789
```

**Claude AI Chat**
```bash
curl -X POST http://localhost:5001/api/claude_chat?key=paid_key_789 \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Analyze my portfolio risk",
    "conversation_history": []
  }'
```

## 🚀 Recommended Testing Sequence

### Quick Validation (5 minutes)
```bash
# 1. Test basic portfolio analysis
python3 run_risk.py --portfolio portfolio.yaml

# 2. Test risk score
python3 -c "from portfolio_risk_score import run_risk_score_analysis; run_risk_score_analysis('portfolio.yaml', 'risk_limits.yaml')"

# 3. Test single stock analysis (detailed with factor proxies)
python3 run_risk.py --stock AAPL --start 2020-01-01 --end 2024-12-31 --yaml-path portfolio.yaml
```

### API Testing (10 minutes)
```bash
# Terminal 1: Start server
python3 app.py

# Terminal 2: Test APIs
python3 show_api_output.py health
python3 show_api_output.py analyze
python3 show_api_output.py risk-score
python3 show_api_output.py performance
```

### Full Testing (30 minutes)
```bash
# Run all CLI functions
python3 run_risk.py --portfolio portfolio.yaml
python3 run_risk.py --portfolio portfolio.yaml --performance
python3 run_risk.py --portfolio portfolio.yaml --minvar
python3 run_risk.py --portfolio portfolio.yaml --maxreturn
python3 run_risk.py --portfolio portfolio.yaml --whatif --delta "AAPL:+500bp,SPY:-200bp"

# Test single stock analysis (all variations)
python3 run_risk.py --stock AAPL --start 2020-01-01 --end 2024-12-31  # Basic
python3 run_risk.py --stock AAPL --start 2020-01-01 --end 2024-12-31 --yaml-path portfolio.yaml  # Detailed
python3 run_risk.py --stock MSFT --start 2022-01-01 --end 2024-12-31 --factor-proxies '{"market": "SPY", "momentum": "MTUM"}'  # Custom

python3 run_risk.py --portfolio portfolio.yaml --gpt

# Run comprehensive API tests
python3 test_api_endpoints.py
```

## 📊 What Each Command Shows You

### Single Stock Analysis - Output Comparison

**Basic Analysis (without proxies):**
```
=== MSCI Stock Analysis ===
Monthly Volatility: 15.2%
Annual Volatility: 52.6%
Market Beta: 1.23
Market Alpha: 0.04%
R-squared: 0.67
```

**Detailed Analysis (with proxies):**
```
=== MSCI Stock Analysis ===
Monthly Volatility: 15.2%
Annual Volatility: 52.6%

=== Factor Exposures ===
Market Beta: 1.23
Momentum Beta: 0.85
Value Beta: -0.12
Industry Beta: 1.45
Subindustry Beta: 0.92

=== Regression Summary ===
Market R-squared: 0.67
Factor R-squared: 0.84
Idiosyncratic Risk: 8.3%
```

**Key Difference:** The detailed analysis shows how the stock behaves relative to **5 different risk factors**, giving you a complete risk profile instead of just market exposure.

### CLI Outputs
- **`--portfolio`**: Risk decomposition, factor exposures, variance breakdown, risk limit checks
- **`--performance`**: Historical returns, Sharpe ratio, alpha, beta, maximum drawdown
- **`--minvar`**: Optimized portfolio weights for minimum risk with constraint checks
- **`--maxreturn`**: Optimized portfolio weights for maximum return with constraint checks
- **`--whatif`**: Before/after risk comparison for portfolio changes
- **`--stock`** (basic): Basic volatility metrics, market beta/alpha only
- **`--stock --yaml-path`**: **Complete factor analysis** - market, momentum, value, industry, subindustry exposures
- **`--stock --factor-proxies`**: **Complete factor analysis** with custom factor proxies
- **`--gpt`**: Plain English interpretation of portfolio analysis
- **Risk Score**: 0-100 credit-score style rating with component breakdown

### API Outputs
- **`analyze`**: Complete portfolio analysis in JSON format
- **`risk-score`**: Risk scoring with detailed component analysis
- **`performance`**: Performance metrics in structured JSON
- **`portfolio-analysis`**: GPT-interpreted portfolio analysis
- **`health`**: API server status and version info

## 🐛 Troubleshooting

### Common Issues

**Command not found: python**
- Use `python3` instead of `python`

**Module not found errors**
- Ensure you're in the correct directory: `/Users/henrychien/Documents/Jupyter/risk_module`
- Check if virtual environment is activated

**API connection errors**
- Ensure Flask server is running on port 5001
- Check if `app.py` started successfully

**Permission errors**
- Ensure you have read/write permissions in the directory
- Check if configuration files exist and are readable

### Debug Commands

**Check current directory:**
```bash
pwd
```

**List configuration files:**
```bash
ls -la *.yaml
```

**Test module imports:**
```bash
python3 -c "import portfolio_risk, portfolio_risk_score, run_risk; print('✅ All modules loaded')"
```

**Check Flask server status:**
```bash
curl -X GET http://localhost:5001/api/health
```

## 📝 Notes

- All commands assume you're in the `/Users/henrychien/Documents/Jupyter/risk_module` directory
- API testing requires the Flask server to be running on port 5001
- Some commands may take 30-60 seconds to complete due to data fetching
- Risk score analysis provides the most comprehensive overview
- GPT functions require API keys to be configured
- Use `show_api_output.py` for the cleanest API testing experience

## 🎉 CLI Interface Enhancement

**✅ FIXED**: The CLI interface now supports **complete factor analysis** for single stocks!

**Before (Limited):**
```bash
python3 run_risk.py --stock AAPL --start 2020-01-01 --end 2024-12-31
# Only basic market regression
```

**After (Complete):**
```bash
python3 run_risk.py --stock AAPL --start 2020-01-01 --end 2024-12-31 --yaml-path portfolio.yaml
# Full factor analysis with market, momentum, value, industry, subindustry factors
```

**New Arguments Added:**
- `--yaml-path`: Path to YAML file for factor proxy lookup
- `--factor-proxies`: JSON string of custom factor proxies

**✅ PROPER DESIGN**: The `--inject_proxies` command maintains separation of concerns - it only injects proxies and stops, allowing for flexible command chaining. 

This addresses the "Interface Coverage" gap mentioned in the architecture docs and eliminates the need for the awkward `-c` format for detailed stock analysis. 