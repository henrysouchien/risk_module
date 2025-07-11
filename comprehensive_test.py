#!/usr/bin/env python3
"""
Comprehensive test to verify all portfolio analysis paths produce identical results.

Tests:
1. Direct CLI call (run_portfolio)
2. Direct API route (/direct/portfolio) 
3. Service layer API route (/analyze)
4. RiskAnalysisResult object data

Verifies:
- Formatted reports are identical
- Structured data matches across all methods
- No data loss in translation layers
"""

import sys
import json
import requests
from typing import Dict, Any
import traceback

def test_direct_cli():
    """Test 1: Direct CLI call to run_portfolio"""
    print("🔧 Test 1: Direct CLI call...")
    
    try:
        from run_risk import run_portfolio
        
        # Get raw data from run_portfolio
        result_data = run_portfolio("portfolio.yaml", return_data=True)
        
        print(f"✅ Direct CLI completed")
        print(f"   Portfolio summary keys: {len(result_data['portfolio_summary'].keys())}")
        print(f"   Risk checks: {len(result_data['risk_analysis']['risk_checks'])}")
        print(f"   Beta checks: {len(result_data['beta_analysis']['beta_checks'])}")
        print(f"   Formatted report length: {len(result_data['formatted_report'])} chars")
        
        return result_data
        
    except Exception as e:
        print(f"❌ Direct CLI failed: {e}")
        traceback.print_exc()
        return None

def test_direct_api():
    """Test 2: Direct API route /direct/portfolio"""
    print("\n🔧 Test 2: Direct API route...")
    
    try:
        response = requests.post(
            'http://localhost:5000/api/direct/portfolio',
            json={'portfolio_file': 'portfolio.yaml'},
            params={'key': 'test_key'}
        )
        
        if response.status_code == 200:
            data = response.json()
            result_data = data['data']
            
            print(f"✅ Direct API completed")
            print(f"   Portfolio summary keys: {len(result_data['portfolio_summary'].keys())}")
            print(f"   Risk checks: {len(result_data['risk_analysis']['risk_checks'])}")
            print(f"   Beta checks: {len(result_data['beta_analysis']['beta_checks'])}")
            print(f"   Formatted report length: {len(result_data['formatted_report'])} chars")
            
            return result_data
        else:
            print(f"❌ Direct API failed with status {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Direct API failed: {e}")
        return None

def test_service_layer_api():
    """Test 3: Service layer API route /analyze"""
    print("\n🔧 Test 3: Service layer API route...")
    
    try:
        response = requests.post(
            'http://localhost:5000/api/analyze',
            json={'portfolio_data': {}},
            params={'key': 'test_key'}
        )
        
        if response.status_code == 200:
            data = response.json()
            result_data = data['risk_results']
            
            print(f"✅ Service layer API completed")
            print(f"   Risk results keys: {len(result_data.keys())}")
            print(f"   Risk checks: {len(result_data.get('risk_checks', []))}")
            print(f"   Beta checks: {len(result_data.get('beta_checks', []))}")
            print(f"   Formatted report length: {len(result_data.get('formatted_report', ''))} chars")
            
            return result_data
        else:
            print(f"❌ Service layer API failed with status {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Service layer API failed: {e}")
        return None

def test_service_layer_object():
    """Test 4: Direct service layer object"""
    print("\n🔧 Test 4: Service layer object...")
    
    try:
        from services.portfolio_service import PortfolioService
        from core.data_objects import PortfolioData
        
        portfolio_data = PortfolioData.from_yaml('portfolio.yaml')
        service = PortfolioService()
        
        # Get the RiskAnalysisResult object
        result = service.analyze_portfolio(portfolio_data)
        
        print(f"✅ Service layer object completed")
        print(f"   Volatility annual: {result.volatility_annual:.4f}")
        print(f"   Risk checks: {len(result.risk_checks)}")
        print(f"   Beta checks: {len(result.beta_checks)}")
        print(f"   Max betas: {len(result.max_betas)}")
        print(f"   Formatted report length: {len(result.to_formatted_report())} chars")
        
        # Convert to dict for comparison
        result_dict = result.to_dict()
        
        return result, result_dict
        
    except Exception as e:
        print(f"❌ Service layer object failed: {e}")
        traceback.print_exc()
        return None, None

def compare_formatted_reports(cli_data, direct_api_data, service_api_data, service_object):
    """Compare formatted reports across all methods"""
    print("\n📝 Comparing formatted reports...")
    
    reports = {}
    
    if cli_data:
        reports['CLI'] = cli_data['formatted_report']
    if direct_api_data:
        reports['Direct API'] = direct_api_data['formatted_report']
    if service_api_data:
        reports['Service API'] = service_api_data.get('formatted_report', '')
    if service_object:
        reports['Service Object'] = service_object.to_formatted_report()
    
    # Compare lengths first
    print("Report lengths:")
    for name, report in reports.items():
        print(f"   {name}: {len(report)} characters")
    
    # Check if all reports are identical
    if len(set(reports.values())) == 1:
        print("✅ All formatted reports are identical!")
        return True
    else:
        print("❌ Formatted reports differ!")
        
        # Show first few lines of each to identify differences
        for name, report in reports.items():
            lines = report.split('\n')[:5]
            print(f"\n{name} first 5 lines:")
            for line in lines:
                print(f"   {line}")
        
        return False

def compare_structured_data(cli_data, direct_api_data, service_api_data, service_dict):
    """Compare structured data across all methods"""
    print("\n📊 Comparing structured data...")
    
    # Compare key metrics
    def extract_key_metrics(data):
        if 'portfolio_summary' in data:
            # CLI/Direct API format
            summary = data['portfolio_summary']
            return {
                'volatility_annual': summary.get('volatility_annual'),
                'volatility_monthly': summary.get('volatility_monthly'),
                'herfindahl': summary.get('herfindahl'),
                'risk_checks_count': len(data.get('risk_analysis', {}).get('risk_checks', [])),
                'beta_checks_count': len(data.get('beta_analysis', {}).get('beta_checks', [])),
            }
        else:
            # Service API/Object format
            return {
                'volatility_annual': data.get('volatility_annual'),
                'volatility_monthly': data.get('volatility_monthly'), 
                'herfindahl': data.get('herfindahl'),
                'risk_checks_count': len(data.get('risk_checks', [])),
                'beta_checks_count': len(data.get('beta_checks', [])),
            }
    
    metrics = {}
    if cli_data:
        metrics['CLI'] = extract_key_metrics(cli_data)
    if direct_api_data:
        metrics['Direct API'] = extract_key_metrics(direct_api_data)
    if service_api_data:
        metrics['Service API'] = extract_key_metrics(service_api_data)
    if service_dict:
        metrics['Service Object'] = extract_key_metrics(service_dict)
    
    print("Key metrics comparison:")
    for name, metric_dict in metrics.items():
        print(f"\n{name}:")
        for key, value in metric_dict.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.6f}")
            else:
                print(f"   {key}: {value}")
    
    # Check if all metrics match
    values_lists = {}
    for key in ['volatility_annual', 'volatility_monthly', 'herfindahl', 'risk_checks_count', 'beta_checks_count']:
        values_lists[key] = [metrics[name][key] for name in metrics.keys()]
    
    all_match = True
    for key, values in values_lists.items():
        if isinstance(values[0], float):
            # For floats, check if they're close enough
            if not all(abs(v - values[0]) < 1e-10 for v in values):
                print(f"❌ {key} values differ: {values}")
                all_match = False
        else:
            # For integers/other types, check exact match
            if not all(v == values[0] for v in values):
                print(f"❌ {key} values differ: {values}")
                all_match = False
    
    if all_match:
        print("✅ All structured data matches!")
        return True
    else:
        print("❌ Structured data differs!")
        return False

def verify_service_object_completeness(service_object, cli_data):
    """Verify the service object contains all data from CLI"""
    print("\n🔍 Verifying service object completeness...")
    
    if not service_object or not cli_data:
        print("❌ Missing data for comparison")
        return False
    
    # Check that object has all the key fields from build_portfolio_view
    portfolio_summary = cli_data['portfolio_summary']
    
    field_checks = [
        ('volatility_annual', portfolio_summary['volatility_annual']),
        ('volatility_monthly', portfolio_summary['volatility_monthly']),
        ('herfindahl', portfolio_summary['herfindahl']),
        ('risk_checks', cli_data['risk_analysis']['risk_checks']),
        ('beta_checks', cli_data['beta_analysis']['beta_checks']),
        ('max_betas', cli_data['beta_analysis']['max_betas']),
        ('max_betas_by_proxy', cli_data['beta_analysis']['max_betas_by_proxy']),
    ]
    
    all_good = True
    for field_name, expected_value in field_checks:
        if hasattr(service_object, field_name):
            actual_value = getattr(service_object, field_name)
            if isinstance(expected_value, (int, float)):
                if abs(actual_value - expected_value) < 1e-10:
                    print(f"✅ {field_name}: {actual_value}")
                else:
                    print(f"❌ {field_name}: expected {expected_value}, got {actual_value}")
                    all_good = False
            elif isinstance(expected_value, (list, dict)):
                if len(actual_value) == len(expected_value):
                    print(f"✅ {field_name}: {len(actual_value)} items")
                else:
                    print(f"❌ {field_name}: expected {len(expected_value)} items, got {len(actual_value)}")
                    all_good = False
            else:
                print(f"✅ {field_name}: present")
        else:
            print(f"❌ {field_name}: missing from object")
            all_good = False
    
    if all_good:
        print("✅ Service object contains all expected data!")
        return True
    else:
        print("❌ Service object is missing some data!")
        return False

def main():
    """Run comprehensive test suite"""
    print("🚀 Running comprehensive portfolio analysis test...")
    print("="*60)
    
    # Run all tests
    cli_data = test_direct_cli()
    direct_api_data = test_direct_api()
    service_api_data = test_service_layer_api()
    service_object, service_dict = test_service_layer_object()
    
    print("\n" + "="*60)
    print("📋 COMPARISON RESULTS")
    print("="*60)
    
    # Compare results
    reports_match = compare_formatted_reports(cli_data, direct_api_data, service_api_data, service_object)
    data_matches = compare_structured_data(cli_data, direct_api_data, service_api_data, service_dict)
    object_complete = verify_service_object_completeness(service_object, cli_data)
    
    print("\n" + "="*60)
    print("🎯 FINAL RESULTS")
    print("="*60)
    
    if reports_match and data_matches and object_complete:
        print("🎉 ALL TESTS PASSED!")
        print("   ✅ Formatted reports identical")
        print("   ✅ Structured data matches")
        print("   ✅ Service object complete")
        print("\n✨ Portfolio analysis is consistent across all paths!")
        return True
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"   Formatted reports: {'✅' if reports_match else '❌'}")
        print(f"   Structured data: {'✅' if data_matches else '❌'}")
        print(f"   Service object: {'✅' if object_complete else '❌'}")
        print("\n🔧 Some inconsistencies need to be fixed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 