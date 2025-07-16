#!/usr/bin/env python3
"""
Reference Data Management Admin Script

Provides command-line interface for managing reference data mappings.
Supports adding, updating, and viewing cash, exchange, and industry mappings.

Usage:
    python admin/manage_reference_data.py cash add USD SGOV
    python admin/manage_reference_data.py cash-alias add "CUR:USD" USD
    python admin/manage_reference_data.py exchange add NASDAQ market SPY
    python admin/manage_reference_data.py industry add Technology XLK
    python admin/manage_reference_data.py cash list
    python admin/manage_reference_data.py exchange list
    python admin/manage_reference_data.py industry list
"""

import sys
import os
import argparse
from typing import Dict, Any

# Add parent directory to path so we can import the database client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def list_cash_mappings(db_client) -> None:
    """List all cash mappings"""
    try:
        mappings = db_client.get_cash_mappings()
        
        print("\n📊 Cash Proxy Mappings:")
        print("-" * 40)
        for currency, proxy_etf in mappings.get("proxy_by_currency", {}).items():
            print(f"  {currency:>3} → {proxy_etf}")
        
        print("\n🏦 Cash Alias Mappings:")
        print("-" * 40)
        for alias, currency in mappings.get("alias_to_currency", {}).items():
            print(f"  {alias:>12} → {currency}")
        
    except Exception as e:
        print(f"❌ Failed to list cash mappings: {e}")

def list_exchange_mappings(db_client) -> None:
    """List all exchange mappings"""
    try:
        mappings = db_client.get_exchange_mappings()
        
        print("\n📈 Exchange Factor Mappings:")
        print("-" * 50)
        for exchange, factors in mappings.items():
            print(f"  {exchange}:")
            for factor_type, proxy_etf in factors.items():
                print(f"    {factor_type:>10} → {proxy_etf}")
        
    except Exception as e:
        print(f"❌ Failed to list exchange mappings: {e}")

def list_industry_mappings(db_client) -> None:
    """List all industry mappings"""
    try:
        mappings = db_client.get_industry_mappings()
        
        print("\n🏭 Industry Proxy Mappings:")
        print("-" * 50)
        for industry, proxy_etf in mappings.items():
            print(f"  {industry:>25} → {proxy_etf}")
        
    except Exception as e:
        print(f"❌ Failed to list industry mappings: {e}")

def add_cash_proxy(db_client, currency: str, proxy_etf: str) -> None:
    """Add/update cash proxy mapping"""
    try:
        db_client.update_cash_proxy(currency, proxy_etf)
        print(f"✅ Updated cash proxy: {currency} → {proxy_etf}")
    except Exception as e:
        print(f"❌ Failed to update cash proxy: {e}")

def add_cash_alias(db_client, alias: str, currency: str) -> None:
    """Add/update cash alias mapping"""
    try:
        db_client.update_cash_alias(alias, currency)
        print(f"✅ Updated cash alias: {alias} → {currency}")
    except Exception as e:
        print(f"❌ Failed to update cash alias: {e}")

def add_exchange_proxy(db_client, exchange: str, factor_type: str, proxy_etf: str) -> None:
    """Add/update exchange proxy mapping"""
    try:
        db_client.update_exchange_proxy(exchange, factor_type, proxy_etf)
        print(f"✅ Updated exchange proxy: {exchange}/{factor_type} → {proxy_etf}")
    except Exception as e:
        print(f"❌ Failed to update exchange proxy: {e}")

def add_industry_proxy(db_client, industry: str, proxy_etf: str) -> None:
    """Add/update industry proxy mapping"""
    try:
        db_client.update_industry_proxy(industry, proxy_etf)
        print(f"✅ Updated industry proxy: {industry} → {proxy_etf}")
    except Exception as e:
        print(f"❌ Failed to update industry proxy: {e}")

def main():
    """Main admin function"""
    parser = argparse.ArgumentParser(description="Manage reference data mappings")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Cash proxy commands
    cash_parser = subparsers.add_parser("cash", help="Manage cash proxy mappings")
    cash_subparsers = cash_parser.add_subparsers(dest="cash_action")
    
    cash_add_parser = cash_subparsers.add_parser("add", help="Add cash proxy mapping")
    cash_add_parser.add_argument("currency", help="Currency code (e.g., USD)")
    cash_add_parser.add_argument("proxy_etf", help="Proxy ETF (e.g., SGOV)")
    
    cash_list_parser = cash_subparsers.add_parser("list", help="List cash mappings")
    
    # Cash alias commands
    alias_parser = subparsers.add_parser("cash-alias", help="Manage cash alias mappings")
    alias_subparsers = alias_parser.add_subparsers(dest="alias_action")
    
    alias_add_parser = alias_subparsers.add_parser("add", help="Add cash alias mapping")
    alias_add_parser.add_argument("alias", help="Broker alias (e.g., CUR:USD)")
    alias_add_parser.add_argument("currency", help="Currency code (e.g., USD)")
    
    alias_list_parser = alias_subparsers.add_parser("list", help="List cash alias mappings")
    
    # Exchange proxy commands
    exchange_parser = subparsers.add_parser("exchange", help="Manage exchange proxy mappings")
    exchange_subparsers = exchange_parser.add_subparsers(dest="exchange_action")
    
    exchange_add_parser = exchange_subparsers.add_parser("add", help="Add exchange proxy mapping")
    exchange_add_parser.add_argument("exchange", help="Exchange name (e.g., NASDAQ)")
    exchange_add_parser.add_argument("factor_type", help="Factor type (market, momentum, value)")
    exchange_add_parser.add_argument("proxy_etf", help="Proxy ETF (e.g., SPY)")
    
    exchange_list_parser = exchange_subparsers.add_parser("list", help="List exchange mappings")
    
    # Industry proxy commands
    industry_parser = subparsers.add_parser("industry", help="Manage industry proxy mappings")
    industry_subparsers = industry_parser.add_subparsers(dest="industry_action")
    
    industry_add_parser = industry_subparsers.add_parser("add", help="Add industry proxy mapping")
    industry_add_parser.add_argument("industry", help="Industry name (e.g., Technology)")
    industry_add_parser.add_argument("proxy_etf", help="Proxy ETF (e.g., XLK)")
    
    industry_list_parser = industry_subparsers.add_parser("list", help="List industry mappings")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize database client
    try:
        from inputs.database_client import DatabaseClient
        db_client = DatabaseClient()
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        sys.exit(1)
    
    # Handle commands
    if args.command == "cash":
        if args.cash_action == "add":
            add_cash_proxy(db_client, args.currency, args.proxy_etf)
        elif args.cash_action == "list":
            list_cash_mappings(db_client)
        else:
            cash_parser.print_help()
    
    elif args.command == "cash-alias":
        if args.alias_action == "add":
            add_cash_alias(db_client, args.alias, args.currency)
        elif args.alias_action == "list":
            list_cash_mappings(db_client)  # Shows both proxies and aliases
        else:
            alias_parser.print_help()
    
    elif args.command == "exchange":
        if args.exchange_action == "add":
            add_exchange_proxy(db_client, args.exchange, args.factor_type, args.proxy_etf)
        elif args.exchange_action == "list":
            list_exchange_mappings(db_client)
        else:
            exchange_parser.print_help()
    
    elif args.command == "industry":
        if args.industry_action == "add":
            add_industry_proxy(db_client, args.industry, args.proxy_etf)
        elif args.industry_action == "list":
            list_industry_mappings(db_client)
        else:
            industry_parser.print_help()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 