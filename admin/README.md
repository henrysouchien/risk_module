# Reference Data Management

This directory contains tools for managing reference data mappings used throughout the risk module.

## Overview

The risk module uses several types of reference data:
- **Cash Mappings**: Currency codes to ETF proxies (USD → SGOV)
- **Cash Aliases**: Broker-specific cash identifiers to currencies (CUR:USD → USD)
- **Exchange Mappings**: Exchange names to factor ETF proxies (NASDAQ → {market: SPY, momentum: MTUM, value: IWD})
- **Industry Mappings**: Industry names to representative ETFs (Technology → XLK)

## Architecture

The system uses a **database-first approach with YAML fallback**:

1. **Database**: Primary source of truth for all mappings
2. **YAML Files**: Fallback when database is unavailable
3. **Automatic Failover**: Code seamlessly switches to YAML if database fails

## Getting Started

### 1. Database Setup

Run the database schema to create the reference data tables:

```bash
psql -d risk_module_db -f db_schema.sql
```

### 2. Data Migration

Migrate existing YAML files to the database:

```bash
python migrate_reference_data.py
```

### 3. Managing Mappings

Use the admin script to manage mappings:

```bash
# List all mappings
python admin/manage_reference_data.py cash list
python admin/manage_reference_data.py exchange list
python admin/manage_reference_data.py industry list

# Add new mappings
python admin/manage_reference_data.py cash add EUR ESTR
python admin/manage_reference_data.py cash-alias add "CUR:EUR" EUR
python admin/manage_reference_data.py exchange add LSE market EFA
python admin/manage_reference_data.py industry add "Financial Services" XLF
```

## Database Schema

### Cash Mappings

```sql
-- Currency to ETF proxy mappings
CREATE TABLE cash_proxies (
    currency VARCHAR(3) PRIMARY KEY,
    proxy_etf VARCHAR(10) NOT NULL
);

-- Broker alias to currency mappings
CREATE TABLE cash_aliases (
    broker_alias VARCHAR(50) PRIMARY KEY,
    currency VARCHAR(3) NOT NULL
);
```

### Exchange Mappings

```sql
-- Exchange to factor proxy mappings
CREATE TABLE exchange_proxies (
    exchange VARCHAR(10) NOT NULL,
    factor_type VARCHAR(20) NOT NULL,
    proxy_etf VARCHAR(10) NOT NULL,
    PRIMARY KEY (exchange, factor_type)
);
```

### Industry Mappings

```sql
-- Industry to ETF proxy mappings
CREATE TABLE industry_proxies (
    industry VARCHAR(100) PRIMARY KEY,
    proxy_etf VARCHAR(10) NOT NULL
);
```

## Code Integration

The system automatically loads mappings from the database:

```python
# Cash mappings (in portfolio_manager.py)
cash_map = self.db_client.get_cash_mappings()

# Exchange mappings (in proxy_builder.py)
exchange_map = load_exchange_proxy_map()

# Industry mappings (in proxy_builder.py)
industry_map = load_industry_etf_map()
```

If the database is unavailable, the code automatically falls back to YAML files with appropriate logging.

## Benefits

1. **Operational Flexibility**: Add new brokerages without code deployment
2. **Reliability**: YAML fallback ensures system availability
3. **Auditability**: Database tracks all changes with timestamps
4. **Consistency**: Single source of truth for all reference data
5. **Scalability**: Database handles concurrent access and transactions

## Maintenance

### Adding New Brokerages

When connecting a new brokerage:

1. Add cash alias mappings for their cash identifiers
2. Add exchange mappings if they use different exchanges
3. Add industry mappings for new industry classifications

### Updating Mappings

Use the admin script to update existing mappings:

```bash
# Update cash proxy
python admin/manage_reference_data.py cash add USD SGOV

# Update exchange proxy
python admin/manage_reference_data.py exchange add NASDAQ market QQQ
```

### Backup and Recovery

The YAML files serve as backup configuration. To regenerate them from the database:

```bash
# Export current database mappings to YAML
python admin/export_reference_data.py
```

## Migration Notes

- Original YAML files are preserved as fallback
- Database seed data includes default mappings
- Migration is incremental and reversible
- All code changes are backward compatible 