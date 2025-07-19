#!/usr/bin/env python3
"""
Reference Data Migration Script

Migrates existing YAML reference data files to the database.
This script populates the database with:
- Exchange mappings from exchange_etf_proxies.yaml
- Industry mappings from industry_to_etf.yaml
- Cash mappings from cash_map.yaml

Usage:
    python admin/migrate_reference_data.py
"""

import sys
import os
import yaml
from pathlib import Path

# Add parent directory to path so we can import the database client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_yaml_file(filepath: str) -> dict:
    """Load YAML file and return contents"""
    try:
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Failed to load {filepath}: {e}")
        return {}

def migrate_exchange_mappings(db_client):
    """Migrate exchange mappings from YAML to database"""
    print("\n📈 Migrating exchange mappings...")
    
    # Load exchange mappings from YAML
    exchange_data = load_yaml_file("../exchange_etf_proxies.yaml")
    
    if not exchange_data:
        print("⚠️  No exchange data found in exchange_etf_proxies.yaml")
        return
    
    migrated_count = 0
    
    for exchange, factors in exchange_data.items():
        print(f"  Processing exchange: {exchange}")
        
        for factor_type, proxy_etf in factors.items():
            try:
                db_client.update_exchange_proxy(exchange, factor_type, proxy_etf)
                print(f"    ✅ {exchange}/{factor_type} → {proxy_etf}")
                migrated_count += 1
            except Exception as e:
                print(f"    ❌ Failed to migrate {exchange}/{factor_type}: {e}")
    
    print(f"📈 Successfully migrated {migrated_count} exchange mappings")

def migrate_industry_mappings(db_client):
    """Migrate industry mappings from YAML to database"""
    print("\n🏭 Migrating industry mappings...")
    
    # Load industry mappings from YAML
    industry_data = load_yaml_file("../industry_to_etf.yaml")
    
    if not industry_data:
        print("⚠️  No industry data found in industry_to_etf.yaml")
        return
    
    migrated_count = 0
    
    for industry, proxy_etf in industry_data.items():
        try:
            db_client.update_industry_proxy(industry, proxy_etf)
            print(f"  ✅ {industry:>30} → {proxy_etf}")
            migrated_count += 1
        except Exception as e:
            print(f"  ❌ Failed to migrate {industry}: {e}")
    
    print(f"🏭 Successfully migrated {migrated_count} industry mappings")

def migrate_cash_mappings(db_client):
    """Migrate cash mappings from YAML to database"""
    print("\n💰 Migrating cash mappings...")
    
    # Load cash mappings from YAML
    cash_data = load_yaml_file("../cash_map.yaml")
    
    if not cash_data:
        print("⚠️  No cash data found in cash_map.yaml")
        return
    
    # Migrate cash proxies
    proxy_by_currency = cash_data.get("proxy_by_currency", {})
    for currency, proxy_etf in proxy_by_currency.items():
        try:
            db_client.update_cash_proxy(currency, proxy_etf)
            print(f"  ✅ Cash proxy: {currency} → {proxy_etf}")
        except Exception as e:
            print(f"  ❌ Failed to add cash proxy {currency}: {e}")
    
    # Migrate cash aliases
    alias_to_currency = cash_data.get("alias_to_currency", {})
    for alias, currency in alias_to_currency.items():
        try:
            db_client.update_cash_alias(alias, currency)
            print(f"  ✅ Cash alias: {alias} → {currency}")
        except Exception as e:
            print(f"  ❌ Failed to add cash alias {alias}: {e}")
    
    print(f"💰 Successfully migrated {len(proxy_by_currency)} cash proxies and {len(alias_to_currency)} cash aliases")

def verify_migration(db_client):
    """Verify the migration was successful"""
    print("\n🔍 Verifying migration...")
    
    try:
        # Check cash mappings
        cash_mappings = db_client.get_cash_mappings()
        cash_proxy_count = len(cash_mappings.get("proxy_by_currency", {}))
        cash_alias_count = len(cash_mappings.get("aliases", {}))
        print(f"  💰 Cash proxies: {cash_proxy_count} entries")
        print(f"  💰 Cash aliases: {cash_alias_count} entries")
        
        # Check exchange mappings
        exchange_mappings = db_client.get_exchange_mappings()
        exchange_count = sum(len(factors) for factors in exchange_mappings.values())
        print(f"  📈 Exchange mappings: {exchange_count} entries")
        
        # Check industry mappings
        industry_mappings = db_client.get_industry_mappings()
        industry_count = len(industry_mappings)
        print(f"  🏭 Industry mappings: {industry_count} entries")
        
        total_entries = cash_proxy_count + cash_alias_count + exchange_count + industry_count
        print(f"\n✅ Migration verification complete: {total_entries} total entries")
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")

def main():
    """Main migration function"""
    print("🚀 Starting Reference Data Migration")
    print("=" * 50)
    
    # Import and initialize database client
    try:
        from inputs.database_client import DatabaseClient
        db_client = DatabaseClient()
        print("✅ Database client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize database client: {e}")
        return
    
    # Migrate data
    migrate_exchange_mappings(db_client)
    migrate_industry_mappings(db_client)
    migrate_cash_mappings(db_client)
    
    # Verify migration
    verify_migration(db_client)
    
    print("\n🎉 Reference data migration completed successfully!")

if __name__ == "__main__":
    main() 