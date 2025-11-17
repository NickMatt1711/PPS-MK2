import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import streamlit as st
from ortools.sat.python import cp_model

# Import our modular components
from ui.styles import apply_custom_styles, create_header, create_info_box
from ui.components import (
    render_file_uploader,
    render_template_download,
    render_optimization_params,
    render_constraint_info,
    render_data_preview,
    render_results_metrics,
    render_inventory_stats,
    render_welcome_screen
)
from ui.visualizations import (
    get_grade_colors,
    create_gantt_chart,
    create_inventory_chart,
    create_schedule_table,
    create_production_summary
)
from data.loaders import ExcelDataLoader, DataValidator
from core.solver import ProductionScheduler
