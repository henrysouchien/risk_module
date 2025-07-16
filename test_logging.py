#!/usr/bin/env python3
"""
Test script to demonstrate real-time logging in the risk module.

This script shows how the enhanced logging system provides real-time visibility
into portfolio analysis operations.
"""

import os
import time
import sys

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

from utils.logging import (
    log_portfolio_operation,
    log_api_request,
    log_performance_metric,
    log_data_transformation,
    log_error_json,
    portfolio_logger,
    api_logger,
    performance_logger
)

def test_logging_demo():
    """
    Demonstrate the real-time logging system with various operations.
    """
    print("🚀 Testing Real-Time Logging System")
    print("=" * 60)
    
    # Test 1: API Request Logging
    print("\n1. Testing API Request Logging...")
    start_time = time.time()
    
    log_api_request(
        endpoint="/api/analyze",
        method="POST",
        user_id=12345,
        execution_time=0.250,
        status_code=200
    )
    
    time.sleep(0.1)  # Brief pause to show real-time nature
    
    # Test 2: Portfolio Operation Logging
    print("\n2. Testing Portfolio Operation Logging...")
    
    portfolio_data = {
        "positions": 15,
        "total_value": 1250000.00,
        "file_path": "test_portfolio.yaml",
        "analysis_type": "risk_analysis"
    }
    
    log_portfolio_operation(
        operation="portfolio_upload",
        portfolio_data=portfolio_data,
        user_id=12345,
        execution_time=0.150
    )
    
    time.sleep(0.1)
    
    # Test 3: Performance Metric Logging
    print("\n3. Testing Performance Metric Logging...")
    
    # Fast operation
    log_performance_metric(
        operation="build_portfolio_view",
        execution_time=0.045,
        details={"positions": 15, "date_range": "2019-01-31 to 2025-06-27"}
    )
    
    time.sleep(0.1)
    
    # Slow operation (will show warning)
    log_performance_metric(
        operation="calc_max_factor_betas",
        execution_time=2.5,
        details={"lookback_years": 10, "num_factors": 6}
    )
    
    time.sleep(0.1)
    
    # Test 4: Data Transformation Logging
    print("\n4. Testing Data Transformation Logging...")
    
    input_data = {
        "type": "raw_portfolio_input",
        "size": 15,
        "file_path": "test_portfolio.yaml"
    }
    
    output_data = {
        "type": "standardized_weights",
        "size": 15,
        "total_weight": 1.0
    }
    
    log_data_transformation(
        transformation="standardize_portfolio_input",
        input_data=input_data,
        output_data=output_data,
        execution_time=0.025
    )
    
    time.sleep(0.1)
    
    # Test 5: Error Logging
    print("\n5. Testing Error Logging...")
    
    try:
        # Simulate an error
        raise ValueError("Test error for logging demonstration")
    except Exception as e:
        log_error_json(
            source="test_logging",
            context={
                "test_phase": "error_demonstration",
                "function": "test_logging_demo"
            },
            exc=e
        )
    
    time.sleep(0.1)
    
    # Test 6: Direct Logger Usage
    print("\n6. Testing Direct Logger Usage...")
    
    portfolio_logger.info("📁 Loading portfolio configuration...")
    portfolio_logger.info("📊 Calculating risk metrics...")
    portfolio_logger.warning("⚠️ High volatility detected: 35.2%")
    portfolio_logger.info("✅ Risk analysis completed successfully")
    
    time.sleep(0.1)
    
    api_logger.info("🌐 API server started on port 5000")
    api_logger.info("📋 Processing portfolio analysis request")
    api_logger.error("❌ Database connection failed, using fallback")
    
    time.sleep(0.1)
    
    performance_logger.info("⚡ Query executed in 12.5ms")
    performance_logger.warning("🐌 SLOW: Portfolio calculation took 1,250ms")
    
    total_time = time.time() - start_time
    
    print(f"\n✅ Logging demo completed in {total_time*1000:.1f}ms")
    print("\nCheck the error_logs/ directory for JSON files:")
    print("- portfolio.log")
    print("- api.log") 
    print("- performance.log")
    print("- portfolio_operations_YYYY-MM-DD.json")
    print("- api_requests_YYYY-MM-DD.json")
    print("- performance_metrics_YYYY-MM-DD.json")
    print("- data_transformations_YYYY-MM-DD.json")
    print("- error_YYYY-MM-DD.json")


def test_portfolio_analysis_with_logging():
    """
    Test actual portfolio analysis with logging (requires portfolio.yaml).
    """
    print("\n" + "=" * 60)
    print("🎯 Testing Portfolio Analysis with Real-Time Logging")
    print("=" * 60)
    
    # Check if portfolio.yaml exists
    if not os.path.exists("portfolio.yaml"):
        print("❌ portfolio.yaml not found - skipping portfolio analysis test")
        return
    
    try:
        # Import the portfolio analysis function
        from run_risk import run_portfolio
        
        print("\n📋 Running portfolio analysis with logging...")
        print("Watch the terminal for real-time log messages!")
        
        # Run portfolio analysis (this will trigger all our logging)
        result = run_portfolio("portfolio.yaml", return_data=True)
        
        print(f"\n✅ Portfolio analysis completed!")
        print(f"   - Positions: {len(result['portfolio_summary']['weights'])}")
        print(f"   - Execution time: {result['analysis_metadata']['execution_time_ms']:.1f}ms")
        print(f"   - Risk passes: {result['risk_analysis']['risk_passes']}")
        print(f"   - Beta passes: {result['beta_analysis']['beta_passes']}")
        
    except Exception as e:
        print(f"❌ Portfolio analysis test failed: {e}")
        # This error will also be logged by our error logging system
        

if __name__ == "__main__":
    print("🎯 Risk Module - Real-Time Logging System Test")
    print("=" * 60)
    print("This test demonstrates the enhanced logging system with real-time streaming.")
    print("You should see log messages appear in real-time in your terminal.")
    print("All logs are also saved to files in the error_logs/ directory.")
    
    # Run the demos
    test_logging_demo()
    test_portfolio_analysis_with_logging()
    
    print("\n🎉 All tests completed!")
    print("\nNext steps:")
    print("1. Run your normal portfolio analysis: python run_risk.py --portfolio portfolio.yaml")
    print("2. Watch the terminal for real-time log messages") 
    print("3. Check error_logs/ directory for persistent log files")
    print("4. The logging system is now active throughout your entire codebase!") 