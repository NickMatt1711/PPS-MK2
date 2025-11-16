"""
Application configuration and settings.
"""

# UI Theme Configuration
THEME = {
    'primary': '#2E86DE',
    'secondary': '#54A0FF',
    'success': '#26DE81',
    'warning': '#FED330',
    'danger': '#FC5C65',
    'neutral': {
        'bg': '#F8F9FA',
        'border': '#DEE2E6',
        'text': '#495057',
        'text_light': '#6C757D'
    },
    'shadows': {
        'sm': '0 1px 3px rgba(0,0,0,0.08)',
        'md': '0 4px 6px rgba(0,0,0,0.1)',
        'lg': '0 10px 15px rgba(0,0,0,0.12)'
    }
}

# Default Optimization Parameters
DEFAULT_OPTIMIZATION_PARAMS = {
    'time_limit_minutes': 10,
    'buffer_days': 3,
    'stockout_penalty': 10,
    'transition_penalty': 10,
    'num_workers': 8,
    'random_seed': 42
}

# Solver Configuration
SOLVER_CONFIG = {
    'log_search_progress': True,
    'optimize_with_core': True,
    'cp_model_probing_level': 2,
}

# File Configuration
SAMPLE_TEMPLATE_FILENAME = 'polymer_production_template.xlsx'

# Required Excel Sheets
REQUIRED_SHEETS = ['Plant', 'Inventory', 'Demand']

# Validation Rules
VALIDATION_RULES = {
    'min_capacity': 1,
    'max_capacity': 100000,
    'min_inventory': 0,
    'max_inventory': 1000000,
}
