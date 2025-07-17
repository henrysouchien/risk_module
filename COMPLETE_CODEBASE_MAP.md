# Complete Risk Module Codebase Map

## Overview
This document provides a comprehensive map of the entire risk_module codebase, including all directories (even those in .gitignore). Created on 2025-07-17.

## Directory Structure with Python File Counts

### Root Level Files (19 Python files)
- `ai_function_registry.py` - Registry for AI/Claude function definitions
- `app.py` - Main Flask application entry point
- `data_loader.py` - Data loading utilities
- `factor_utils.py` - Factor analysis utilities
- `gpt_helpers.py` - GPT/AI integration helpers
- `helpers_display.py` - Display formatting utilities
- `helpers_input.py` - Input processing utilities
- `plaid_loader.py` - Plaid API integration
- `portfolio_optimizer.py` - Portfolio optimization algorithms
- `portfolio_risk.py` - Portfolio risk calculations
- `portfolio_risk_score.py` - Risk scoring system
- `proxy_builder.py` - Proxy ETF builder
- `risk_helpers.py` - Risk calculation helpers
- `risk_summary.py` - Risk summary generation
- `run_portfolio_risk.py` - Portfolio risk runner script
- `run_risk.py` - Main risk calculation runner
- `settings.py` - Application settings
- `test_logging.py` - Logging test utilities

### Core Application Layers (CRITICAL - Gitignored)

#### `/services/` (16 Python files) - Business Logic Layer
Service layer implementing core business logic:
- `service_layer.py` - Main service orchestration
- `service_manager.py` - Service lifecycle management
- `portfolio_service.py` - Portfolio management service
- `stock_service.py` - Stock analysis service
- `optimization_service.py` - Portfolio optimization service
- `scenario_service.py` - Scenario analysis service
- `validation_service.py` - Data validation service
- `async_service.py` - Asynchronous service utilities
- `auth_service.py` - Authentication service
- `usage_examples.py` - Service usage examples
- **`/claude/`** subdirectory:
  - `chat_service.py` - Claude chat integration
  - `function_executor.py` - Function execution for Claude
- **`/portfolio/`** subdirectory:
  - `context_service.py` - Portfolio context management

#### `/routes/` (6 Python files) - API Endpoints Layer
Flask route definitions:
- `api.py` - Main API endpoints
- `auth.py` - Authentication routes
- `admin.py` - Admin panel routes
- `claude.py` - Claude AI integration routes
- `plaid.py` - Plaid integration routes

#### `/utils/` (5 Python files) - Utility Layer
Shared utilities:
- `auth.py` - Authentication utilities
- `config.py` - Configuration management
- `logging.py` - Logging infrastructure
- `serialization.py` - Data serialization utilities

#### `/inputs/` (7 Python files) - Data Access Layer
Data input and management:
- `database_client.py` - PostgreSQL database client
- `file_manager.py` - File system management
- `portfolio_manager.py` - Portfolio data management
- `returns_calculator.py` - Returns calculation
- `risk_config.py` - Risk configuration management
- `exceptions.py` - Custom exception definitions

#### `/core/` (10 Python files) - Core Business Objects
Core data structures and algorithms:
- `data_objects.py` - Core data object definitions
- `result_objects.py` - Result data structures
- `exceptions.py` - Core exception definitions
- `interpretation.py` - Result interpretation logic
- `optimization.py` - Optimization algorithms
- `performance_analysis.py` - Performance analytics
- `portfolio_analysis.py` - Portfolio analysis logic
- `scenario_analysis.py` - Scenario analysis logic
- `stock_analysis.py` - Stock analysis algorithms

### Frontend Application (React/TypeScript)

#### `/frontend/` (2 Python files + full React app)
Modern React frontend with TypeScript:
- **`/src/`** - Source code
  - **`/chassis/`** - Core frontend architecture
    - `/context/` - React context providers
    - `/hooks/` - Custom React hooks
    - `/managers/` - State management
    - `/services/` - API service layer
    - `/types/` - TypeScript type definitions
  - **`/components/`** - UI components
    - `/auth/` - Authentication components
    - `/chat/` - Chat/AI interface
    - `/layouts/` - Layout components
    - `/plaid/` - Plaid integration UI
    - `/portfolio/` - Portfolio management UI
    - `/shared/` - Shared components
- **`/tests/`** - Frontend tests
- **`/build/`** - Production build output

### Testing Infrastructure

#### `/tests/` (10 Python files)
Test suite:
- `test_api_endpoints.py` - API endpoint tests
- `test_auth_system.py` - Authentication tests
- `test_basic_functionality.py` - Basic functionality tests
- `test_cli.py` - CLI interface tests
- `test_final_status.py` - Integration tests
- `test_performance_benchmarks.py` - Performance tests
- `test_services.py` - Service layer tests
- `show_api_output.py` - API output debugging
- **`/api/`** subdirectory:
  - `test_full_workflow.py` - Full workflow tests
- **`/fixtures/`** - Test data files
- **`/cache_prices/`** - Test price data cache

### Development & Archive

#### `/archive/` (101 Python files)
Historical development files and backups

#### `/backup/` (201 Python files)
System backups including full copies of core files

#### `/prototype/` (5 Python files + notebooks)
Jupyter notebook prototypes converted to Python

#### `/tools/` (3 Python files)
Development tools:
- `check_dependencies.py` - Dependency checker
- `test_all_interfaces.py` - Interface testing
- `view_alignment.py` - Code alignment viewer

### Documentation

#### `/docs/`
- `API_REFERENCE.md` - API documentation
- `DATA_SCHEMAS.md` - Data schema documentation
- **`/interfaces/`** - Interface documentation
- **`/planning/`** - Planning documents

#### `/completed/`
Completed feature documentation and plans

#### `/admin/` (1 Python file)
- `manage_reference_data.py` - Reference data management tool

### Data & Cache

#### `/cache_prices/`
Price data cache (Parquet files)

#### `/error_logs/`
Application error logs

### Configuration Files
- Various YAML files for:
  - Portfolio configurations
  - Risk limits
  - Industry/exchange mappings
  - Cash mappings

### Security & Secrets

#### `/risk_module_secrets/` (163 Python files)
**CRITICAL**: Contains duplicated application code and secrets
- Appears to be a full backup/mirror of the application
- Contains sensitive configuration

### Critical Observations

1. **Service Layer Architecture**: The application uses a proper layered architecture:
   - Routes → Services → Core/Inputs → Database
   
2. **Gitignored Layers**: Critical application layers are gitignored:
   - `/services/` - Business logic
   - `/routes/` - API endpoints
   - `/utils/` - Utilities
   - `/inputs/` - Data access
   - `/frontend/` - UI application

3. **Database Integration**: 
   - PostgreSQL-based with connection pooling
   - Comprehensive logging decorators throughout
   - User isolation and portfolio management

4. **AI Integration**:
   - Claude AI chat integration
   - Function execution framework
   - AI-assisted portfolio analysis

5. **Security Concerns**:
   - `/risk_module_secrets/` contains sensitive data
   - Multiple backup directories with potential secrets

## File Count Summary
- Total Python files: ~560+
- Core application: ~50 files
- Tests: ~20 files
- Archive/Backup: ~300+ files
- Frontend: Full React application

## Key Integration Points
1. Database: PostgreSQL via `database_client.py`
2. External APIs: Plaid, Claude AI
3. Frontend: React with TypeScript
4. Authentication: Google OAuth integration
5. Real-time: WebSocket support for chat

This codebase represents a comprehensive financial risk analysis platform with:
- Modern web architecture
- AI-powered analysis
- External data integration
- Comprehensive testing
- Production-ready infrastructure