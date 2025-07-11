#!/usr/bin/env python
"""
Test script to validate API endpoints with the new dual-mode functions.
This tests both existing service layer endpoints and new direct dual-mode endpoints.
"""

import requests
import json
import time
from datetime import datetime

# API Configuration
BASE_URL = "http://localhost:5001"
API_KEY = "test_key"  # Default test key

def test_endpoint(endpoint, method="POST", data=None, params=None):
    """Test an API endpoint and return the response"""
    url = f"{BASE_URL}/api/{endpoint}"
    
    if params is None:
        params = {}
    if 'key' not in params:
        params['key'] = API_KEY
    
    try:
        print(f"\n🔍 Testing {method} {url}")
        print(f"   Parameters: {params}")
        if data:
            print(f"   Data: {json.dumps(data, indent=2)}")
        
        if method == "GET":
            response = requests.get(url, params=params)
        elif method == "POST":
            response = requests.post(url, json=data, params=params)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ SUCCESS")
            print(f"   Response Keys: {list(result.keys())}")
            return result
        else:
            print(f"   ❌ FAILED")
            print(f"   Response: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"   ❌ CONNECTION ERROR - Is the Flask server running?")
        return None
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return None

def test_existing_endpoints():
    """Test the existing service layer API endpoints"""
    print("=" * 80)
    print("TESTING EXISTING SERVICE LAYER ENDPOINTS")
    print("=" * 80)
    
    # Test 1: Health check
    print("\n1. Testing health check endpoint")
    result = test_endpoint("health", method="GET")
    
    # Test 2: Portfolio analysis
    print("\n2. Testing portfolio analysis endpoint")
    result = test_endpoint("analyze", data={
        "portfolio_data": {}  # Uses default portfolio.yaml
    })
    
    if result:
        print(f"   Portfolio analysis completed successfully")
        print(f"   Keys in response: {list(result.keys())}")
        print(f"   Has formatted report: {'risk_results' in result and 'formatted_report' in result['risk_results']}")
        if 'risk_results' in result and 'formatted_report' in result['risk_results']:
            report_preview = result['risk_results']['formatted_report'][:100].replace('\n', ' ')
            print(f"   Formatted report preview: {report_preview}...")
    
    # Test 3: Risk score
    print("\n3. Testing risk score endpoint")
    result = test_endpoint("risk-score", data={})
    
    if result:
        print(f"   Risk score analysis completed successfully")
        print(f"   Risk score: {result.get('risk_score', 'N/A')}")
        print(f"   Has formatted report: {'formatted_report' in result}")
        if 'formatted_report' in result:
            report_preview = result['formatted_report'][:100].replace('\n', ' ')
            print(f"   Formatted report preview: {report_preview}...")
    
    # Test 4: Portfolio analysis with GPT
    print("\n4. Testing portfolio analysis with GPT endpoint")
    result = test_endpoint("portfolio-analysis", data={})
    
    if result:
        print(f"   Portfolio analysis with GPT completed successfully")
        print(f"   Has interpretation: {'interpretation' in result}")
        print(f"   Has full diagnostics: {'full_diagnostics' in result}")
        if 'interpretation' in result:
            interpretation = result['interpretation']
            if isinstance(interpretation, dict):
                print(f"   Interpretation length: {len(interpretation.get('ai_interpretation', ''))}")
                print(f"   Diagnostics length: {len(interpretation.get('full_diagnostics', ''))}")
            else:
                print(f"   Interpretation length: {len(interpretation)}")
                print(f"   Interpretation type: {type(interpretation)}")
    
    # Test 4b: Direct interpretation endpoint (if available)
    print("\n4b. Testing direct interpretation endpoint")
    result = test_endpoint("interpret", data={})
    
    if result:
        print(f"   Direct interpretation completed successfully")
        print(f"   Has AI interpretation: {'ai_interpretation' in result}")
        print(f"   Has full diagnostics: {'full_diagnostics' in result}")
        if 'ai_interpretation' in result:
            print(f"   AI interpretation length: {len(result['ai_interpretation'])}")
            print(f"   Full diagnostics length: {len(result['full_diagnostics'])}")
    
    # Test 5: Performance analysis
    print("\n5. Testing performance analysis endpoint")
    result = test_endpoint("performance", data={
        "benchmark_ticker": "SPY"
    })
    
    if result:
        print(f"   Performance analysis completed successfully")
        print(f"   Performance metrics keys: {list(result.get('performance_metrics', {}).keys())}")
        print(f"   Has formatted report: {'formatted_report' in result}")
        summary = result.get('summary', {})
        if summary:
            print(f"   Annualized return: {summary.get('annualized_return', 'N/A')}")
            print(f"   Sharpe ratio: {summary.get('sharpe_ratio', 'N/A')}")
            print(f"   Max drawdown: {summary.get('max_drawdown', 'N/A')}")
        if 'formatted_report' in result:
            report_preview = result['formatted_report'][:100].replace('\n', ' ')
            print(f"   Formatted report preview: {report_preview}...")

def test_dual_mode_functions():
    """Test the dual-mode functions directly (not through API yet)"""
    print("\n" + "=" * 80)
    print("TESTING DUAL-MODE FUNCTIONS DIRECTLY")
    print("=" * 80)
    
    from run_risk import run_portfolio, run_stock, run_what_if, run_min_variance, run_max_return
    
    # Test 1: run_portfolio in API mode
    print("\n1. Testing run_portfolio(return_data=True)")
    try:
        result = run_portfolio("portfolio.yaml", return_data=True)
        print(f"   ✅ SUCCESS - Keys: {list(result.keys())}")
        print(f"   Portfolio summary keys: {list(result['portfolio_summary'].keys())}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    # Test 2: run_stock in API mode
    print("\n2. Testing run_stock(return_data=True)")
    try:
        result = run_stock("AAPL", "2023-01-01", "2023-12-31", return_data=True)
        print(f"   ✅ SUCCESS - Keys: {list(result.keys())}")
        print(f"   Analysis type: {result.get('analysis_type')}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    # Test 3: run_what_if in API mode
    print("\n3. Testing run_what_if(return_data=True)")
    try:
        result = run_what_if("portfolio.yaml", delta="AAPL:-500bp,SGOV:+500bp", return_data=True)
        print(f"   ✅ SUCCESS - Keys: {list(result.keys())}")
        print(f"   Risk passes: {result.get('risk_analysis', {}).get('risk_passes')}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    # Test 4: run_min_variance in API mode
    print("\n4. Testing run_min_variance(return_data=True)")
    try:
        result = run_min_variance("portfolio.yaml", return_data=True)
        print(f"   ✅ SUCCESS - Keys: {list(result.keys())}")
        print(f"   Optimization type: {result.get('optimization_metadata', {}).get('optimization_type')}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    # Test 5: run_max_return in API mode
    print("\n5. Testing run_max_return(return_data=True)")
    try:
        result = run_max_return("portfolio.yaml", return_data=True)
        print(f"   ✅ SUCCESS - Keys: {list(result.keys())}")
        print(f"   Optimization type: {result.get('optimization_metadata', {}).get('optimization_type')}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    # Test 6: run_portfolio_performance in API mode
    print("\n6. Testing run_portfolio_performance(return_data=True)")
    try:
        from run_risk import run_portfolio_performance
        result = run_portfolio_performance("portfolio.yaml", return_data=True)
        print(f"   ✅ SUCCESS - Keys: {list(result.keys())}")
        print(f"   Has performance metrics: {'performance_metrics' in result}")
        if 'performance_metrics' in result:
            perf_metrics = result['performance_metrics']
            returns = perf_metrics.get('returns', {})
            print(f"   Annualized return: {returns.get('annualized_return', 'N/A')}")
            print(f"   Analysis successful: {result.get('analysis_metadata', {}).get('calculation_successful', 'N/A')}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")

def test_new_dual_mode_endpoints():
    """Test the new dual-mode API endpoints that directly use the dual-mode functions"""
    print("\n" + "=" * 80)
    print("TESTING NEW DUAL-MODE API ENDPOINTS")
    print("=" * 80)
    
    # Test 1: Direct portfolio analysis
    print("\n1. Testing direct portfolio analysis endpoint")
    result = test_endpoint("direct/portfolio", data={
        "portfolio_file": "portfolio.yaml"
    })
    
    if result:
        print(f"   Direct portfolio analysis completed successfully")
        print(f"   Data keys: {list(result.get('data', {}).keys())}")
    
    # Test 2: Direct stock analysis
    print("\n2. Testing direct stock analysis endpoint")
    result = test_endpoint("direct/stock", data={
        "ticker": "AAPL",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "yaml_path": "portfolio.yaml"
    })
    
    if result:
        print(f"   Direct stock analysis completed successfully")
        print(f"   Analysis type: {result.get('data', {}).get('analysis_type')}")
        print(f"   Ticker: {result.get('data', {}).get('ticker')}")
    
    # Test 3: Direct what-if analysis
    print("\n3. Testing direct what-if analysis endpoint")
    result = test_endpoint("direct/what-if", data={
        "portfolio_file": "portfolio.yaml",
        "delta": "AAPL:-500bp,SGOV:+500bp"
    })
    
    if result:
        print(f"   Direct what-if analysis completed successfully")
        print(f"   Data keys: {list(result.get('data', {}).keys())}")
        print(f"   Risk passes: {result.get('data', {}).get('risk_analysis', {}).get('risk_passes')}")
    
    # Test 4: Direct minimum variance optimization
    print("\n4. Testing direct minimum variance optimization endpoint")
    result = test_endpoint("direct/optimize/min-variance", data={
        "portfolio_file": "portfolio.yaml"
    })
    
    if result:
        print(f"   Direct min variance optimization completed successfully")
        print(f"   Optimization type: {result.get('data', {}).get('optimization_metadata', {}).get('optimization_type')}")
        print(f"   Risk passes: {result.get('data', {}).get('risk_analysis', {}).get('risk_passes')}")
    
    # Test 5: Direct maximum return optimization
    print("\n5. Testing direct maximum return optimization endpoint")
    result = test_endpoint("direct/optimize/max-return", data={
        "portfolio_file": "portfolio.yaml"
    })
    
    if result:
        print(f"   Direct max return optimization completed successfully")
        print(f"   Optimization type: {result.get('data', {}).get('optimization_metadata', {}).get('optimization_type')}")
        print(f"   Risk passes: {result.get('data', {}).get('risk_analysis', {}).get('risk_passes')}")
    
    # Test 6: Direct performance analysis
    print("\n6. Testing direct performance analysis endpoint")
    result = test_endpoint("direct/performance", data={
        "portfolio_file": "portfolio.yaml"
    })
    
    if result:
        print(f"   Direct performance analysis completed successfully")
        print(f"   Data keys: {list(result.get('data', {}).keys())}")
        if 'data' in result and 'performance_metrics' in result['data']:
            perf_metrics = result['data']['performance_metrics']
            returns = perf_metrics.get('returns', {})
            risk_metrics = perf_metrics.get('risk_metrics', {})
            print(f"   Annualized return: {returns.get('annualized_return', 'N/A')}")
            print(f"   Volatility: {risk_metrics.get('volatility', 'N/A')}")
            print(f"   Analysis successful: {result.get('data', {}).get('analysis_metadata', {}).get('calculation_successful', 'N/A')}")

def test_interpretation_comparison():
    """Test interpretation functionality by comparing service layer with backup functions."""
    print("\n" + "=" * 80)
    print("TESTING INTERPRETATION COMPARISON")
    print("=" * 80)
    
    try:
        # Test 1: Service layer interpretation
        print("\n1. Testing service layer interpretation via API")
        service_result = test_endpoint("portfolio-analysis", data={})
        
        if service_result and 'interpretation' in service_result:
            service_interpretation = service_result['interpretation']
            print(f"   ✅ Service layer interpretation successful")
            if isinstance(service_interpretation, dict):
                print(f"   • AI interpretation length: {len(service_interpretation.get('ai_interpretation', ''))}")
                print(f"   • Full diagnostics length: {len(service_interpretation.get('full_diagnostics', ''))}")
            else:
                print(f"   • Interpretation length: {len(service_interpretation)}")
                print(f"   • Interpretation type: {type(service_interpretation)}")
            
            # Show preview of service interpretation
            print(f"\n📝 Service AI Interpretation Preview:")
            if isinstance(service_interpretation, dict):
                ai_preview = service_interpretation.get('ai_interpretation', '')[:200]
                print(f"   {ai_preview}...")
                
                print(f"\n📝 Service Diagnostics Preview:")
                diag_preview = service_interpretation.get('full_diagnostics', '')[:200]
                print(f"   {diag_preview}...")
            else:
                # service_interpretation is the AI interpretation string directly
                ai_preview = service_interpretation[:200]
                print(f"   {ai_preview}...")
                print(f"\n📝 Service Diagnostics Preview:")
                print(f"   (No separate diagnostics - interpretation is a direct string)")
            
        else:
            print(f"   ❌ Service layer interpretation failed")
            return False
        
        # Test 2: Backup function interpretation
        print("\n2. Testing backup function interpretation")
        try:
            from backup_functions_for_testing import run_and_interpret
            from io import StringIO
            from contextlib import redirect_stdout
            
            # Capture backup function output
            backup_buf = StringIO()
            with redirect_stdout(backup_buf):
                run_and_interpret("portfolio.yaml")
            
            backup_output = backup_buf.getvalue()
            print(f"   ✅ Backup function interpretation successful")
            print(f"   • Backup output length: {len(backup_output)}")
            
            # Show preview of backup output
            print(f"\n📝 Backup Output Preview:")
            backup_preview = backup_output[:200]
            print(f"   {backup_preview}...")
            
        except Exception as e:
            print(f"   ❌ Backup function interpretation failed: {e}")
            return False
        
        # Test 3: Compare content structure
        print("\n3. Comparing content structure")
        
        if isinstance(service_interpretation, dict):
            service_diagnostics = service_interpretation.get('full_diagnostics', '')
            service_ai = service_interpretation.get('ai_interpretation', '')
        else:
            # service_interpretation is the AI interpretation string directly
            service_diagnostics = service_interpretation
            service_ai = service_interpretation
        
        # Check for key content in both outputs
        checks = [
            ("volatility", "volatility metrics"),
            ("factor", "factor analysis"),
            ("risk", "risk analysis"),
            ("beta", "beta exposures"),
            ("herfindahl", "concentration metrics")
        ]
        
        print(f"   Content comparison:")
        for keyword, description in checks:
            service_has = keyword.lower() in service_diagnostics.lower()
            backup_has = keyword.lower() in backup_output.lower()
            match = service_has == backup_has
            status = "✅" if match else "⚠️"
            print(f"   {status} {description}: Service={service_has}, Backup={backup_has}")
        
        print(f"\n✅ Interpretation comparison test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Interpretation comparison test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all API tests"""
    print("🚀 Starting API Endpoint Tests")
    print(f"Testing against: {BASE_URL}")
    print(f"Using API key: {API_KEY}")
    print(f"Test started at: {datetime.now()}")
    
    # Test existing endpoints
    test_existing_endpoints()
    
    # Test dual-mode functions directly
    test_dual_mode_functions()
    
    # Test new dual-mode endpoints
    test_new_dual_mode_endpoints()
    
    # Test interpretation comparison
    test_interpretation_comparison()
    
    print("\n" + "=" * 80)
    print("API TESTS COMPLETED")
    print("=" * 80)
    print(f"Test completed at: {datetime.now()}")

if __name__ == "__main__":
    main() 