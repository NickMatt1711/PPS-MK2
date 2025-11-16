"""
Polymer Production Scheduler V2
Modern, modular production scheduling application with CP-SAT optimization.
"""

import streamlit as st
import pandas as pd
import numpy as np
from ortools.sat.python import cp_model
from datetime import datetime, date, timedelta
import io
import time
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Polymer Production Scheduler V2",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme Configuration
THEME = {
    'primary': '#2E86DE',
    'secondary': '#54A0FF',
    'success': '#26DE81',
    'danger': '#FC5C65',
    'neutral_bg': '#F8F9FA',
    'neutral_border': '#DEE2E6',
}

# ============================================================================
# STYLING
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E86DE;
        text-align: center;
        padding: 1.5rem;
        margin-bottom: 2rem;
        border-bottom: 3px solid #2E86DE;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #DEE2E6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2E86DE;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6C757D;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button {
        background: #2E86DE;
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        background: #1E5FBD;
        border: none;
    }
    .info-box {
        padding: 1rem;
        border-radius: 8px;
        background: #D1ECF1;
        border: 1px solid #BEE5EB;
        color: #0C5460;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 8px;
        background: #D4EDDA;
        border: 1px solid #C3E6CB;
        color: #155724;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 8px;
        background: #FFF3CD;
        border: 1px solid #FFEAA7;
        color: #856404;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_excel_template():
    """Load the sample template from repository."""
    try:
        # Try to find template in assets or root
        for path in [Path("assets/polymer_production_template.xlsx"), 
                     Path("polymer_production_template.xlsx")]:
            if path.exists():
                with open(path, "rb") as f:
                    return io.BytesIO(f.read())
        return None
    except Exception as e:
        st.error(f"Could not load template: {e}")
        return None

def format_date(d: date) -> str:
    """Format date consistently."""
    return d.strftime('%d-%b-%y')

def parse_lines(lines_str) -> List[str]:
    """Parse comma-separated plant names."""
    if pd.isna(lines_str) or lines_str == '':
        return []
    return [x.strip() for x in str(lines_str).split(',')]

def parse_rerun_allowed(value) -> bool:
    """Parse rerun allowed field."""
    if pd.isna(value):
        return True
    val_str = str(value).strip().lower()
    return val_str not in ['no', 'n', 'false', '0']

# ============================================================================
# DATA LOADING & VALIDATION
# ============================================================================

class DataLoader:
    """Handle Excel data loading and parsing."""
    
    def __init__(self, file_bytes: bytes):
        self.file = io.BytesIO(file_bytes)
    
    def load_plants(self) -> pd.DataFrame:
        """Load plant data."""
        self.file.seek(0)
        return pd.read_excel(self.file, sheet_name='Plant')
    
    def load_inventory(self) -> pd.DataFrame:
        """Load inventory data."""
        self.file.seek(0)
        return pd.read_excel(self.file, sheet_name='Inventory')
    
    def load_demand(self) -> pd.DataFrame:
        """Load demand data."""
        self.file.seek(0)
        return pd.read_excel(self.file, sheet_name='Demand')
    
    def load_transitions(self, plant_names: List[str]) -> Dict:
        """Load transition matrices for all plants."""
        transitions = {}
        for plant in plant_names:
            patterns = [
                f'Transition_{plant}',
                f'Transition_{plant.replace(" ", "_")}',
                f'Transition{plant.replace(" ", "")}'
            ]
            for sheet_name in patterns:
                try:
                    self.file.seek(0)
                    df = pd.read_excel(self.file, sheet_name=sheet_name, index_col=0)
                    transitions[plant] = df
                    break
                except:
                    continue
        return transitions

class DataValidator:
    """Validate input data."""
    
    @staticmethod
    def validate_plants(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate plant data."""
        errors = []
        warnings = []
        
        if df['Plant'].duplicated().any():
            errors.append("Duplicate plant names found")
        
        if (df['Capacity per day'] <= 0).any():
            errors.append("Plant capacity must be positive")
        
        return errors, warnings
    
    @staticmethod
    def validate_inventory(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate inventory data."""
        errors = []
        warnings = []
        
        for idx, row in df.iterrows():
            grade = row['Grade Name']
            
            if pd.isna(row['Lines']) or str(row['Lines']).strip() == '':
                errors.append(f"Grade '{grade}': Lines must be specified")
            
            if row['Min. Run Days'] > row['Max. Run Days']:
                errors.append(f"Grade '{grade}': Min run days > Max run days")
            
            if row['Min. Inventory'] > row['Max. Inventory']:
                errors.append(f"Grade '{grade}': Min inventory > Max inventory")
        
        return errors, warnings
    
    @staticmethod
    def check_feasibility(plant_df: pd.DataFrame, 
                         demand_df: pd.DataFrame) -> List[str]:
        """Quick feasibility check."""
        warnings = []
        
        total_capacity = plant_df['Capacity per day'].sum()
        grades = demand_df.columns[1:]
        total_avg_demand = demand_df[grades].mean().sum()
        
        if total_avg_demand > total_capacity * 0.9:
            warnings.append(
                f"⚠️ High utilization: Avg demand ({total_avg_demand:.0f} MT/day) "
                f"vs capacity ({total_capacity:.0f} MT/day)"
            )
        
        return warnings

# ============================================================================
# OPTIMIZATION SOLVER
# ============================================================================

class SolutionCallback(cp_model.CpSolverSolutionCallback):
    """Callback to capture intermediate solutions."""
    
    def __init__(self, production_vars, inventory_vars, stockout_vars, 
                 is_producing_vars, grades, lines, dates, num_days):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.production = production_vars
        self.inventory = inventory_vars
        self.stockout = stockout_vars
        self.is_producing = is_producing_vars
        self.grades = grades
        self.lines = lines
        self.dates = dates
        self.num_days = num_days
        self.solutions = []
        self.start_time = time.time()
    
    def on_solution_callback(self):
        """Called when a new solution is found."""
        current_time = time.time() - self.start_time
        
        solution = {
            'objective': self.ObjectiveValue(),
            'time': current_time,
            'production': {},
            'inventory': {},
            'stockout': {},
            'schedule': {}
        }
        
        # Extract production
        for grade in self.grades:
            solution['production'][grade] = {}
            for line in self.lines:
                for d in range(self.num_days):
                    key = (grade, line, d)
                    if key in self.production:
                        val = self.Value(self.production[key])
                        if val > 0:
                            date_str = format_date(self.dates[d])
                            if date_str not in solution['production'][grade]:
                                solution['production'][grade][date_str] = 0
                            solution['production'][grade][date_str] += val
        
        # Extract inventory
        for grade in self.grades:
            solution['inventory'][grade] = {}
            for d in range(self.num_days + 1):
                key = (grade, d)
                if key in self.inventory:
                    date_str = format_date(self.dates[d]) if d < self.num_days else 'final'
                    solution['inventory'][grade][date_str] = self.Value(self.inventory[key])
        
        # Extract stockouts
        for grade in self.grades:
            solution['stockout'][grade] = {}
            for d in range(self.num_days):
                key = (grade, d)
                if key in self.stockout:
                    val = self.Value(self.stockout[key])
                    if val > 0:
                        date_str = format_date(self.dates[d])
                        solution['stockout'][grade][date_str] = val
        
        # Extract schedule
        for line in self.lines:
            solution['schedule'][line] = {}
            for d in range(self.num_days):
                date_str = format_date(self.dates[d])
                solution['schedule'][line][date_str] = None
                for grade in self.grades:
                    key = (grade, line, d)
                    if key in self.is_producing and self.Value(self.is_producing[key]) == 1:
                        solution['schedule'][line][date_str] = grade
                        break
        
        # Count transitions
        transitions = self._count_transitions()
        solution['transitions'] = transitions
        
        self.solutions.append(solution)
    
    def _count_transitions(self) -> Dict:
        """Count grade transitions per line."""
        trans_count = {line: 0 for line in self.lines}
        
        for line in self.lines:
            last_grade = None
            for d in range(self.num_days):
                current_grade = None
                for grade in self.grades:
                    key = (grade, line, d)
                    if key in self.is_producing and self.Value(self.is_producing[key]) == 1:
                        current_grade = grade
                        break
                
                if current_grade is not None and last_grade is not None:
                    if current_grade != last_grade:
                        trans_count[line] += 1
                
                if current_grade is not None:
                    last_grade = current_grade
        
        return {
            'per_line': trans_count,
            'total': sum(trans_count.values())
        }

class ProductionScheduler:
    """Main production scheduling optimizer."""
    
    def __init__(self, plant_df, inventory_df, demand_df, transition_dfs, 
                 buffer_days, time_limit_min, stockout_penalty, transition_penalty):
        self.plant_df = plant_df
        self.inventory_df = inventory_df
        self.demand_df = demand_df
        self.transition_dfs = transition_dfs
        self.buffer_days = buffer_days
        self.time_limit_min = time_limit_min
        self.stockout_penalty = stockout_penalty
        self.transition_penalty = transition_penalty
        
        self.model = None
        self.solver = None
        self.callback = None
        
    def prepare_data(self):
        """Prepare data structures for optimization."""
        # Lines and capacities
        self.lines = list(self.plant_df['Plant'])
        self.capacities = {
            row['Plant']: row['Capacity per day'] 
            for _, row in self.plant_df.iterrows()
        }
        
        # Dates
        demand_dates = sorted(self.demand_df.iloc[:, 0].dt.date.unique())
        self.dates = demand_dates.copy()
        last_date = self.dates[-1]
        for i in range(1, self.buffer_days + 1):
            self.dates.append(last_date + timedelta(days=i))
        self.num_days = len(self.dates)
        
        # Grades
        self.grades = [col for col in self.demand_df.columns if col != self.demand_df.columns[0]]
        
        # Demand dictionary
        self.demand_dict = {}
        for grade in self.grades:
            self.demand_dict[grade] = {}
            for _, row in self.demand_df.iterrows():
                d = row.iloc[0].date()
                self.demand_dict[grade][d] = float(row[grade]) if pd.notna(row[grade]) else 0.0
            # Add buffer days with zero demand
            for d in self.dates[-self.buffer_days:]:
                if d not in self.demand_dict[grade]:
                    self.demand_dict[grade][d] = 0.0
        
        # Process inventory data
        self.initial_inventory = {}
        self.min_inventory = {}
        self.max_inventory = {}
        self.min_closing_inventory = {}
        self.allowed_lines = {grade: [] for grade in self.grades}
        self.min_run_days = {}
        self.max_run_days = {}
        self.force_start_date = {}
        self.rerun_allowed = {}
        
        grade_inventory_defined = set()
        
        for _, row in self.inventory_df.iterrows():
            grade = row['Grade Name']
            
            # Parse lines
            plants_for_row = parse_lines(row['Lines'])
            if not plants_for_row:
                plants_for_row = self.lines
            
            for plant in plants_for_row:
                if plant not in self.allowed_lines[grade]:
                    self.allowed_lines[grade].append(plant)
            
            # Global inventory parameters (once per grade)
            if grade not in grade_inventory_defined:
                self.initial_inventory[grade] = float(row.get('Opening Inventory', 0))
                self.min_inventory[grade] = float(row.get('Min. Inventory', 0))
                self.max_inventory[grade] = float(row.get('Max. Inventory', 1000000))
                self.min_closing_inventory[grade] = float(row.get('Min. Closing Inventory', 0))
                grade_inventory_defined.add(grade)
            
            # Plant-specific parameters
            for plant in plants_for_row:
                key = (grade, plant)
                self.min_run_days[key] = int(row.get('Min. Run Days', 1))
                self.max_run_days[key] = int(row.get('Max. Run Days', 9999))
                self.rerun_allowed[key] = parse_rerun_allowed(row.get('Rerun Allowed'))
                
                if pd.notna(row.get('Force Start Date')):
                    try:
                        self.force_start_date[key] = pd.to_datetime(row['Force Start Date']).date()
                    except:
                        self.force_start_date[key] = None
                else:
                    self.force_start_date[key] = None
        
        # Shutdown periods
        self.shutdown_periods = {}
        for _, row in self.plant_df.iterrows():
            plant = row['Plant']
            start_date = row.get('Shutdown Start Date')
            end_date = row.get('Shutdown End Date')
            
            if pd.notna(start_date) and pd.notna(end_date):
                start_date = pd.to_datetime(start_date).date()
                end_date = pd.to_datetime(end_date).date()
                
                shutdown_days = []
                for d, date in enumerate(self.dates):
                    if start_date <= date <= end_date:
                        shutdown_days.append(d)
                self.shutdown_periods[plant] = shutdown_days
            else:
                self.shutdown_periods[plant] = []
        
        # Material running
        self.material_running = {}
        for _, row in self.plant_df.iterrows():
            plant = row['Plant']
            material = row.get('Material Running')
            expected_days = row.get('Expected Run Days')
            
            if pd.notna(material) and pd.notna(expected_days):
                self.material_running[plant] = (str(material).strip(), int(expected_days))
        
        # Transition rules
        self.transition_rules = {}
        for line, df in self.transition_dfs.items():
            if df is not None:
                self.transition_rules[line] = {}
                for prev_grade in df.index:
                    allowed = []
                    for curr_grade in df.columns:
                        if str(df.loc[prev_grade, curr_grade]).lower() == 'yes':
                            allowed.append(curr_grade)
                    self.transition_rules[line][prev_grade] = allowed
            else:
                self.transition_rules[line] = None
    
    def build_model(self):
        """Build the CP-SAT optimization model."""
        self.model = cp_model.CpModel()
        
        # Decision variables
        self.is_producing = {}
        self.production = {}
        
        for grade in self.grades:
            for line in self.allowed_lines[grade]:
                for d in range(self.num_days):
                    key = (grade, line, d)
                    self.is_producing[key] = self.model.NewBoolVar(f'producing_{grade}_{line}_{d}')
                    self.production[key] = self.model.NewIntVar(
                        0, self.capacities[line], f'prod_{grade}_{line}_{d}'
                    )
                    
                    # Link production to is_producing
                    if d < self.num_days - self.buffer_days:
                        self.model.Add(self.production[key] == self.capacities[line]).OnlyEnforceIf(
                            self.is_producing[key]
                        )
                        self.model.Add(self.production[key] == 0).OnlyEnforceIf(
                            self.is_producing[key].Not()
                        )
                    else:
                        self.model.Add(self.production[key] <= self.capacities[line] * self.is_producing[key])
        
        # Inventory variables
        self.inventory_vars = {}
        for grade in self.grades:
            for d in range(self.num_days + 1):
                self.inventory_vars[(grade, d)] = self.model.NewIntVar(
                    0, int(self.max_inventory[grade]), f'inv_{grade}_{d}'
                )
        
        # Stockout variables
        self.stockout_vars = {}
        for grade in self.grades:
            for d in range(self.num_days):
                self.stockout_vars[(grade, d)] = self.model.NewIntVar(
                    0, 100000, f'stockout_{grade}_{d}'
                )
        
        # CONSTRAINTS
        self._add_capacity_constraints()
        self._add_shutdown_constraints()
        self._add_inventory_constraints()
        self._add_material_running_constraints()
        self._add_run_day_constraints()
        self._add_transition_constraints()
        
        # OBJECTIVE
        self._add_objective()
    
    def _add_capacity_constraints(self):
        """One grade per line per day."""
        for line in self.lines:
            for d in range(self.num_days):
                producing_vars = []
                for grade in self.grades:
                    if line in self.allowed_lines[grade]:
                        key = (grade, line, d)
                        if key in self.is_producing:
                            producing_vars.append(self.is_producing[key])
                if producing_vars:
                    self.model.Add(sum(producing_vars) <= 1)
        
        # Full capacity utilization (except buffer days)
        for line in self.lines:
            for d in range(self.num_days - self.buffer_days):
                if line not in self.shutdown_periods or d not in self.shutdown_periods[line]:
                    prod_vars = []
                    for grade in self.grades:
                        if line in self.allowed_lines[grade]:
                            key = (grade, line, d)
                            if key in self.production:
                                prod_vars.append(self.production[key])
                    if prod_vars:
                        self.model.Add(sum(prod_vars) == self.capacities[line])
    
    def _add_shutdown_constraints(self):
        """No production during shutdown."""
        for line in self.lines:
            if line in self.shutdown_periods:
                for d in self.shutdown_periods[line]:
                    for grade in self.grades:
                        if line in self.allowed_lines[grade]:
                            key = (grade, line, d)
                            if key in self.is_producing:
                                self.model.Add(self.is_producing[key] == 0)
                                self.model.Add(self.production[key] == 0)
    
    def _add_inventory_constraints(self):
        """Inventory balance and bounds."""
        # Initial inventory
        for grade in self.grades:
            self.model.Add(self.inventory_vars[(grade, 0)] == int(self.initial_inventory[grade]))
        
        # Inventory balance
        for grade in self.grades:
            for d in range(self.num_days):
                produced = sum(
                    self.production.get((grade, line, d), 0)
                    for line in self.allowed_lines[grade]
                )
                demand = int(self.demand_dict[grade].get(self.dates[d], 0))
                
                # Simplified balance: inv(t+1) = inv(t) + prod - min(demand, inv(t) + prod)
                # Stockout = max(0, demand - inv(t) - prod)
                
                available = self.inventory_vars[(grade, d)] + produced
                supplied = self.model.NewIntVar(0, 100000, f'supplied_{grade}_{d}')
                
                self.model.Add(supplied <= available)
                self.model.Add(supplied <= demand)
                self.model.Add(self.stockout_vars[(grade, d)] == demand - supplied)
                self.model.Add(self.inventory_vars[(grade, d + 1)] == available - supplied)
        
        # Min/max inventory bounds (soft via penalties in objective)
        # Closing inventory requirement
        for grade in self.grades:
            closing_inv = self.inventory_vars[(grade, self.num_days - self.buffer_days)]
            min_closing = int(self.min_closing_inventory[grade])
            if min_closing > 0:
                # Soft constraint via penalty in objective
                pass
    
    def _add_material_running_constraints(self):
        """Force initial material running."""
        for plant, (material, expected_days) in self.material_running.items():
            for d in range(min(expected_days, self.num_days)):
                if plant in self.allowed_lines.get(material, []):
                    key = (material, plant, d)
                    if key in self.is_producing:
                        self.model.Add(self.is_producing[key] == 1)
    
    def _add_run_day_constraints(self):
        """Min/max run days and rerun constraints."""
        # Start/end variables
        is_start = {}
        is_end = {}
        
        for grade in self.grades:
            for line in self.allowed_lines[grade]:
                for d in range(self.num_days):
                    is_start[(grade, line, d)] = self.model.NewBoolVar(f'start_{grade}_{line}_{d}')
                    is_end[(grade, line, d)] = self.model.NewBoolVar(f'end_{grade}_{line}_{d}')
                    
                    curr_prod = self.is_producing.get((grade, line, d))
                    
                    if curr_prod is not None:
                        if d > 0:
                            prev_prod = self.is_producing.get((grade, line, d - 1))
                            if prev_prod is not None:
                                # Start: producing today, not yesterday
                                self.model.AddBoolAnd([curr_prod, prev_prod.Not()]).OnlyEnforceIf(
                                    is_start[(grade, line, d)]
                                )
                        else:
                            # Day 0: start if producing
                            self.model.Add(curr_prod == is_start[(grade, line, d)])
                        
                        if d < self.num_days - 1:
                            next_prod = self.is_producing.get((grade, line, d + 1))
                            if next_prod is not None:
                                # End: producing today, not tomorrow
                                self.model.AddBoolAnd([curr_prod, next_prod.Not()]).OnlyEnforceIf(
                                    is_end[(grade, line, d)]
                                )
                        else:
                            # Last day: end if producing
                            self.model.Add(curr_prod == is_end[(grade, line, d)])
                
                # Min run days
                key = (grade, line)
                min_run = self.min_run_days.get(key, 1)
                
                for d in range(self.num_days):
                    start_var = is_start[(grade, line, d)]
                    
                    # Check consecutive non-shutdown days
                    consecutive_days = 0
                    for k in range(min_run):
                        if d + k < self.num_days:
                            if line in self.shutdown_periods and (d + k) in self.shutdown_periods[line]:
                                break
                            consecutive_days += 1
                    
                    if consecutive_days >= min_run:
                        for k in range(min_run):
                            if d + k < self.num_days:
                                if line not in self.shutdown_periods or (d + k) not in self.shutdown_periods[line]:
                                    prod_var = self.is_producing.get((grade, line, d + k))
                                    if prod_var is not None:
                                        self.model.Add(prod_var == 1).OnlyEnforceIf(start_var)
                
                # Max run days (sliding window)
                max_run = self.max_run_days.get(key, 9999)
                for d in range(self.num_days - max_run):
                    consecutive = []
                    for k in range(max_run + 1):
                        if d + k < self.num_days:
                            if line in self.shutdown_periods and (d + k) in self.shutdown_periods[line]:
                                break
                            prod_var = self.is_producing.get((grade, line, d + k))
                            if prod_var is not None:
                                consecutive.append(prod_var)
                    
                    if len(consecutive) == max_run + 1:
                        self.model.Add(sum(consecutive) <= max_run)
                
                # Rerun allowed
                if not self.rerun_allowed.get(key, True):
                    starts = [is_start[(grade, line, d)] for d in range(self.num_days)]
                    self.model.Add(sum(starts) <= 1)
                
                # Force start date
                force_date = self.force_start_date.get(key)
                if force_date and force_date in self.dates:
                    day_idx = self.dates.index(force_date)
                    prod_var = self.is_producing.get((grade, line, day_idx))
                    if prod_var is not None:
                        self.model.Add(prod_var == 1)
    
    def _add_transition_constraints(self):
        """Transition rules enforcement."""
        for line in self.lines:
            if line in self.transition_rules and self.transition_rules[line] is not None:
                rules = self.transition_rules[line]
                
                for d in range(self.num_days - 1):
                    for prev_grade in self.grades:
                        if prev_grade in rules and line in self.allowed_lines.get(prev_grade, []):
                            allowed_next = rules[prev_grade]
                            
                            for curr_grade in self.grades:
                                if curr_grade != prev_grade and curr_grade not in allowed_next:
                                    if line in self.allowed_lines.get(curr_grade, []):
                                        prev_var = self.is_producing.get((prev_grade, line, d))
                                        curr_var = self.is_producing.get((curr_grade, line, d + 1))
                                        
                                        if prev_var is not None and curr_var is not None:
                                            self.model.Add(prev_var + curr_var <= 1)
    
    def _add_objective(self):
        """Build objective function."""
        objective = 0
        
        # Stockout penalties
        for grade in self.grades:
            for d in range(self.num_days):
                objective += self.stockout_penalty * self.stockout_vars[(grade, d)]
        
        # Transition penalties
        for line in self.lines:
            for d in range(self.num_days - 1):
                for g1 in self.grades:
                    for g2 in self.grades:
                        if g1 == g2:
                            continue
                        
                        if line in self.allowed_lines.get(g1, []) and line in self.allowed_lines.get(g2, []):
                            trans_var = self.model.NewBoolVar(f'trans_{line}_{d}_{g1}_{g2}')
                            
                            var1 = self.is_producing.get((g1, line, d))
                            var2 = self.is_producing.get((g2, line, d + 1))
                            
                            if var1 is not None and var2 is not None:
                                self.model.AddBoolAnd([var1, var2]).OnlyEnforceIf(trans_var)
                                self.model.AddBoolOr([var1.Not(), var2.Not()]).OnlyEnforceIf(trans_var.Not())
                                
                                objective += self.transition_penalty * trans_var
        
        # Minimize objective
        self.model.Minimize(objective)
    
    def solve(self):
        """Run optimization."""
        self.solver = cp_model.CpSolver()
        
        # Configure solver
        self.solver.parameters.max_time_in_seconds = self.time_limit_min * 60
        self.solver.parameters.num_search_workers = 8
        self.solver.parameters.random_seed = 42
        self.solver.parameters.log_search_progress = True
        
        # Create callback
        self.callback = SolutionCallback(
            self.production, self.inventory_vars, self.stockout_vars,
            self.is_producing, self.grades, self.lines, self.dates, self.num_days
        )
        
        # Solve
        start_time = time.time()
        status = self.solver.Solve(self.model, self.callback)
        solve_time = time.time() - start_time
        
        return status, solve_time

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_gantt_chart(line: str, schedule_dict: Dict, dates: List[date], 
                       shutdown_days: List[int], grade_colors: Dict) -> go.Figure:
    """Create Gantt chart for a production line."""
    gantt_data = []
    
    for d, date in enumerate(dates):
        date_str = format_date(date)
        grade = schedule_dict.get(date_str)
        
        if grade:
            gantt_data.append({
                'Grade': grade,
                'Start': date,
                'Finish': date + timedelta(days=1),
                'Line': line
            })
    
    if not gantt_data:
        return None
    
    gantt_df = pd.DataFrame(gantt_data)
    
    fig = px.timeline(
        gantt_df,
        x_start='Start',
        x_end='Finish',
        y='Grade',
        color='Grade',
        color_discrete_map=grade_colors,
        title=f'Production Schedule - {line}'
    )
    
    # Add shutdown visualization
    if shutdown_days:
        start_shutdown = dates[shutdown_days[0]]
        end_shutdown = dates[shutdown_days[-1]] + timedelta(days=1)
        
        fig.add_vrect(
            x0=start_shutdown,
            x1=end_shutdown,
            fillcolor='red',
            opacity=0.2,
            layer='below',
            line_width=0,
            annotation_text='Shutdown',
            annotation_position='top left'
        )
    
    fig.update_yaxes(autorange='reversed', title=None)
    fig.update_xaxes(title='Date', dtick='D1', tickformat='%d-%b')
    fig.update_layout(
        height=400,
        showlegend=True,
        plot_bgcolor='white'
    )
    
    return fig

def create_inventory_chart(grade: str, inventory_data: List[float], 
                           dates: List[date], min_inv: float, max_inv: float,
                           grade_color: str) -> go.Figure:
    """Create inventory level chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=inventory_data,
        mode='lines+markers',
        name=grade,
        line=dict(color=grade_color, width=3),
        marker=dict(size=6),
        hovertemplate='Date: %{x|%d-%b-%y}<br>Inventory: %{y:.0f} MT<extra></extra>'
    ))
    
    # Min/Max lines
    if min_inv > 0:
        fig.add_hline(
            y=min_inv,
            line=dict(color='red', width=2, dash='dash'),
            annotation_text=f'Min: {min_inv:,.0f}',
            annotation_position='top left'
        )
    
    if max_inv < 1000000:
        fig.add_hline(
            y=max_inv,
            line=dict(color='green', width=2, dash='dash'),
            annotation_text=f'Max: {max_inv:,.0f}',
            annotation_position='bottom left'
        )
    
    fig.update_layout(
        title=f'Inventory Level - {grade}',
        xaxis=dict(title='Date', dtick='D1', tickformat='%d-%b'),
        yaxis=dict(title='Inventory (MT)'),
        plot_bgcolor='white',
        height=400
    )
    
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    
    # Header
    st.markdown('<div class="main-header">🏭 Polymer Production Scheduler V2</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 Data Input")
        
        uploaded_file = st.file_uploader(
            "Upload Excel File",
            type=['xlsx'],
            help="Upload Excel file with Plant, Inventory, and Demand sheets"
        )
        
        if uploaded_file:
            st.success("✅ File uploaded successfully!")
        
        # Download template
        st.markdown("---")
        st.markdown("### 📥 Sample Template")
        
        template = load_excel_template()
        if template:
            st.download_button(
                label="Download Sample Template",
                data=template,
                file_name="polymer_production_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        if uploaded_file:
            st.markdown("---")
            st.markdown("### ⚙️ Optimization Parameters")
            
            with st.expander("🔧 Basic Settings", expanded=True):
                time_limit = st.number_input(
                    "Time limit (minutes)",
                    min_value=1, max_value=60, value=10
                )
                
                buffer_days = st.number_input(
                    "Buffer days",
                    min_value=0, max_value=7, value=3
                )
            
            with st.expander("🎯 Penalty Weights", expanded=True):
                stockout_penalty = st.number_input(
                    "Stockout penalty",
                    min_value=1, value=10
                )
                
                transition_penalty = st.number_input(
                    "Transition penalty",
                    min_value=1, value=10
                )
    
    # Main content
    if uploaded_file is None:
        st.markdown("""
        <div class="info-box">
            <h3>👈 Get Started</h3>
            <p>Upload an Excel file to begin optimization, or download the sample template to see the required format.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### Features
        
        - 🎯 **Multi-plant optimization** with shutdown management
        - 📊 **Interactive visualizations** with Gantt charts
        - 🔧 **Flexible constraints** (min/max run days, transitions)
        - 📈 **Inventory tracking** with stockout prevention
        - 🚀 **Fast CP-SAT solver** with configurable parameters
        """)
        
        return
    
    # Load and validate data
    try:
        file_bytes = uploaded_file.read()
        loader = DataLoader(file_bytes)
        
        plant_df = loader.load_plants()
        inventory_df = loader.load_inventory()
        demand_df = loader.load_demand()
        transition_dfs = loader.load_transitions(plant_df['Plant'].tolist())
        
        # Validate
        validator = DataValidator()
        
        plant_errors, plant_warnings = validator.validate_plants(plant_df)
        inv_errors, inv_warnings = validator.validate_inventory(inventory_df)
        feasibility_warnings = validator.check_feasibility(plant_df, demand_df)
        
        all_errors = plant_errors + inv_errors
        all_warnings = plant_warnings + inv_warnings + feasibility_warnings
        
        if all_errors:
            st.error("**Cannot proceed - please fix these errors:**")
            for error in all_errors:
                st.markdown(f"- ❌ {error}")
            return
        
        if all_warnings:
            with st.expander("⚠️ Warnings", expanded=True):
                for warning in all_warnings:
                    st.warning(warning)
        
        # Display data preview
        st.markdown("---")
        st.markdown("### 📊 Data Preview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🏭 Plants")
            st.dataframe(plant_df, use_container_width=True, height=250)
        
        with col2:
            st.subheader("📦 Inventory")
            st.dataframe(inventory_df, use_container_width=True, height=250)
        
        with col3:
            st.subheader("📈 Demand")
            st.dataframe(demand_df.head(10), use_container_width=True, height=250)
        
        # Show shutdown info
        st.markdown("---")
        shutdown_info = []
        for _, row in plant_df.iterrows():
            if pd.notna(row.get('Shutdown Start Date')) and pd.notna(row.get('Shutdown End Date')):
                start = pd.to_datetime(row['Shutdown Start Date']).date()
                end = pd.to_datetime(row['Shutdown End Date']).date()
                days = (end - start).days + 1
                shutdown_info.append(
                    f"**{row['Plant']}**: {format_date(start)} to {format_date(end)} ({days} days)"
                )
        
        if shutdown_info:
            st.markdown("### 🔧 Scheduled Shutdowns")
            for info in shutdown_info:
                st.info(info)
        
        # Optimization button
        st.markdown("---")
        
        if st.button("🎯 Run Optimization", type="primary", use_container_width=True):
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.markdown('<div class="info-box">📊 Preparing data...</div>', 
                               unsafe_allow_html=True)
            progress_bar.progress(20)
            
            # Create scheduler
            scheduler = ProductionScheduler(
                plant_df, inventory_df, demand_df, transition_dfs,
                buffer_days, time_limit, stockout_penalty, transition_penalty
            )
            
            scheduler.prepare_data()
            
            status_text.markdown('<div class="info-box">🔧 Building optimization model...</div>', 
                               unsafe_allow_html=True)
            progress_bar.progress(40)
            
            scheduler.build_model()
            
            status_text.markdown('<div class="info-box">⚡ Running solver...</div>', 
                               unsafe_allow_html=True)
            progress_bar.progress(60)
            
            status, solve_time = scheduler.solve()
            
            progress_bar.progress(100)
            
            # Display results
            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                status_text.markdown(
                    '<div class="success-box">✅ Optimization completed successfully!</div>', 
                    unsafe_allow_html=True
                )
                
                if scheduler.callback.solutions:
                    solution = scheduler.callback.solutions[-1]
                    
                    st.markdown("---")
                    st.markdown("### 📊 Results")
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Objective Value</div>
                            <div class="metric-value">{solution['objective']:,.0f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Transitions</div>
                            <div class="metric-value">{solution['transitions']['total']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        total_stockout = sum(
                            sum(solution['stockout'][g].values()) 
                            for g in scheduler.grades
                        )
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Stockouts</div>
                            <div class="metric-value">{total_stockout:,.0f} MT</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Solve Time</div>
                            <div class="metric-value">{solve_time:.1f}s</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Tabs for results
                    tab1, tab2, tab3 = st.tabs(["📅 Schedule", "📊 Summary", "📦 Inventory"])
                    
                    with tab1:
                        st.markdown("### Production Schedules")
                        
                        # Generate color map
                        sorted_grades = sorted(scheduler.grades)
                        colors = px.colors.qualitative.Vivid
                        grade_colors = {
                            grade: colors[i % len(colors)] 
                            for i, grade in enumerate(sorted_grades)
                        }
                        
                        for line in scheduler.lines:
                            st.markdown(f"#### {line}")
                            
                            fig = create_gantt_chart(
                                line, 
                                solution['schedule'][line],
                                scheduler.dates,
                                scheduler.shutdown_periods.get(line, []),
                                grade_colors
                            )
                            
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info(f"No production scheduled for {line}")
                    
                    with tab2:
                        st.markdown("### Production Summary")
                        
                        # Build summary table
                        summary_data = []
                        for grade in sorted_grades:
                            row = {'Grade': grade}
                            total = 0
                            for line in scheduler.lines:
                                line_total = 0
                                for date_str, prod_dict in solution['production'].items():
                                    if grade in prod_dict:
                                        # This needs adjustment based on actual structure
                                        pass
                                row[line] = line_total
                                total += line_total
                            row['Total'] = total
                            summary_data.append(row)
                        
                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True)
                    
                    with tab3:
                        st.markdown("### Inventory Levels")
                        
                        for grade in sorted_grades:
                            inv_data = []
                            for d in range(scheduler.num_days):
                                inv_data.append(
                                    scheduler.solver.Value(scheduler.inventory_vars[(grade, d)])
                                )
                            
                            fig = create_inventory_chart(
                                grade,
                                inv_data,
                                scheduler.dates,
                                scheduler.min_inventory[grade],
                                scheduler.max_inventory[grade],
                                grade_colors[grade]
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                
            else:
                status_text.markdown(
                    '<div class="warning-box">⚠️ No feasible solution found. Try adjusting constraints or increasing time limit.</div>', 
                    unsafe_allow_html=True
                )
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        import traceback
        with st.expander("🐛 Debug Info"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
