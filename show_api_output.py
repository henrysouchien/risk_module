#!/usr/bin/env python3
import requests
import json

def show_api_output(endpoint="analyze", key="paid_key_789"):
    """Simple script to show API output without all the curl/jq nonsense"""
    
    url = f"http://localhost:5001/api/{endpoint}?key={key}"
    
    try:
        if endpoint == "health":
            response = requests.get(url)
        else:
            response = requests.post(url, json={})
        
        if response.status_code == 200:
            data = response.json()
            
            if endpoint == "analyze":
                # Show the complete structured data AND formatted report
                risk_results = data.get("risk_results", {})
                formatted_report = risk_results.get("formatted_report", "")
                
                print("=== STRUCTURED DATA ===")
                print(json.dumps(risk_results, indent=2))
                
                if formatted_report:
                    print("\n=== FORMATTED REPORT ===")
                    print(formatted_report)
                else:
                    print("\nNo formatted report found")
                    
            elif endpoint == "risk-score":
                # Show the complete structured data AND formatted report
                structured_data = {k: v for k, v in data.items() if k != 'formatted_report'}
                formatted_report = data.get("formatted_report", "")
                
                print("=== STRUCTURED DATA ===")
                print(json.dumps(structured_data, indent=2))
                
                if formatted_report:
                    print("\n=== FORMATTED REPORT ===")
                    print(formatted_report)
                else:
                    print("\nNo formatted report found")
                        

            elif endpoint == "performance":
                # Show the complete structured data AND formatted report
                performance_metrics = data.get("performance_metrics", {})
                formatted_report = data.get("formatted_report", "")
                
                print("=== STRUCTURED DATA ===")
                print(json.dumps(performance_metrics, indent=2))
                
                if formatted_report:
                    print("\n=== FORMATTED REPORT ===")
                    print(formatted_report)
                else:
                    print("\nNo formatted report found")
                
            elif endpoint == "portfolio-analysis":
                # Show GPT interpretation
                interpretation = data.get("interpretation", "")
                print("=== GPT PORTFOLIO INTERPRETATION ===")
                if isinstance(interpretation, str):
                    print(interpretation[:1000] + "..." if len(interpretation) > 1000 else interpretation)
                else:
                    ai_interp = interpretation.get("ai_interpretation", "")
                    print(ai_interp[:1000] + "..." if len(ai_interp) > 1000 else ai_interp)
                    
            elif endpoint == "interpret":
                # Show direct interpretation
                result = data.get("interpretation", "")
                print("=== DIRECT GPT INTERPRETATION ===")
                if isinstance(result, str):
                    print(result[:1000] + "..." if len(result) > 1000 else result)
                else:
                    ai_interp = result.get("ai_interpretation", "")
                    print(ai_interp[:1000] + "..." if len(ai_interp) > 1000 else ai_interp)
                    
            elif endpoint == "health":
                # Show health status
                print("=== API HEALTH CHECK ===")
                print(f"Status: {data.get('status', 'Unknown')}")
                print(f"Message: {data.get('message', 'No message')}")
                if 'timestamp' in data:
                    print(f"Timestamp: {data['timestamp']}")
                if 'version' in data:
                    print(f"Version: {data['version']}")
                    
            else:
                # Generic JSON output for unknown endpoints
                print(f"=== {endpoint.upper()} OUTPUT ===")
                print(json.dumps(data, indent=2)[:2000] + "..." if len(str(data)) > 2000 else json.dumps(data, indent=2))
                
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def show_help():
    """Show available commands"""
    print("=== AVAILABLE API ENDPOINTS ===")
    print("Usage: python3 show_api_output.py [endpoint]")
    print("")
    print("Available endpoints:")
    print("  analyze           - Complete portfolio analysis (structured data + formatted report)")
    print("  risk-score        - Portfolio risk score (structured data + formatted report)")
    print("  performance       - Portfolio performance (structured data + formatted report)")
    print("  portfolio-analysis - Portfolio analysis with GPT interpretation")
    print("  interpret         - Direct GPT interpretation")
    print("  health            - API health check")
    print("")
    print("Examples:")
    print("  python3 show_api_output.py")
    print("  python3 show_api_output.py risk-score")
    print("  python3 show_api_output.py performance")
    print("  python3 show_api_output.py help")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        endpoint = sys.argv[1]
        if endpoint == "help" or endpoint == "-h" or endpoint == "--help":
            show_help()
            sys.exit(0)
    else:
        endpoint = "analyze"
        
    show_api_output(endpoint) 