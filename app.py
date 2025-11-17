"""
Polymer Production Scheduler V2
Production scheduling with strict constraint enforcement.
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
# CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="Polymer Production Scheduler V2",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_excel_template():
    """Load the sample template from repository."""
    try:
        for path in [Path("assets/polymer_production_template.xlsx"), 
                     Path("polymer_production_template.xlsx")]:
            if path.exists():
                with open(path, "rb") as f:
                    return io.BytesIO(f.read())
        return None
    except Exception as e:
        return None

def format_date(d: date) -> str:
    """Format date consistently."""
    return d.strftime('%d-%b-%y')

# ============================================================================
# SOLUTION CALLBACK
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
        
        # Extract production by grade and date
        for grade in self.grades:
            solution['production'][grade] = {}
            for d in range(self.num_days):
                date_str = format_date(self.dates[d])
                daily_prod = 0
                for line in self.lines:
                    key = (grade, line, d)
                    if key in self.production:
                        daily_prod += self.Value(self.production[key])
                if daily_prod > 0:
                    solution['production'][grade][date_str] = daily_prod
        
        # Extract inventory
        for grade in self.grades:
            solution['inventory'][grade] = []
            for d in range(self.num_days + 1):
                key = (grade, d)
                if key in self.inventory:
                    solution['inventory'][grade].append(self.Value(self.inventory[key]))
        
        # Extract stockouts
        total_stockout = 0
        for grade in self.grades:
            for d in range(self.num_days):
                key = (grade, d)
                if key in self.stockout:
                    val = self.Value(self.stockout[key])
                    total_stockout += val
        solution['total_stockout'] = total_stockout
        
        # Extract schedule per line
        for line in self.lines:
            solution['schedule'][line] = []
            for d in range(self.num_days):
                grade_today = None
                for grade in self.grades:
                    key = (grade, line, d)
                    if key in self.is_producing and self.Value(self.is_producing[key]) == 1:
                        grade_today = grade
                        break
                solution['schedule'][line].append(grade_today)
        
        # Count transitions
        transitions = 0
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
                        transitions += 1
                
                if current_grade is not None:
                    last_grade = current_grade
        
        solution['transitions'] = transitions
        self.solutions.append(solution)

# ============================================================================
# MAIN SCHEDULER CLASS
# ============================================================================

class ProductionScheduler:
    """Production scheduler with strict constraint enforcement."""
    
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
        """Prepare all data structures."""
        # HARD CONSTRAINT 1: Lines and capacities
        self.lines = list(self.plant_df['Plant'])
        self.capacities = {row['Plant']: int(row['Capacity per day']) 
                          for _, row in self.plant_df.iterrows()}
        
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
                val = float(row[grade]) if pd.notna(row[grade]) else 0.0
                self.demand_dict[grade][d] = val
            # Buffer days have zero demand
            for d in self.dates[-self.buffer_days:]:
                if d not in self.demand_dict[grade]:
                    self.demand_dict[grade][d] = 0.0
        
        # Process inventory data - HARD CONSTRAINTS
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
            
            # HARD CONSTRAINT 8: Lines - parse allowed lines
            lines_str = row.get('Lines', '')
            if pd.isna(lines_str) or str(lines_str).strip() == '':
                st.warning(f"⚠️ Grade '{grade}': No lines specified, using all lines")
                plants_for_row = self.lines
            else:
                plants_for_row = [x.strip() for x in str(lines_str).split(',')]
            
            for plant in plants_for_row:
                if plant in self.lines and plant not in self.allowed_lines[grade]:
                    self.allowed_lines[grade].append(plant)
            
            # Global inventory parameters (once per grade)
            if grade not in grade_inventory_defined:
                # HARD CONSTRAINT 5: Opening Inventory
                self.initial_inventory[grade] = float(row.get('Opening Inventory', 0))
                # Soft constraints
                self.min_inventory[grade] = float(row.get('Min. Inventory', 0))
                self.max_inventory[grade] = float(row.get('Max. Inventory', 1000000))
                self.min_closing_inventory[grade] = float(row.get('Min. Closing Inventory', 0))
                grade_inventory_defined.add(grade)
            
            # Plant-specific parameters
            for plant in plants_for_row:
                if plant not in self.lines:
                    continue
                    
                key = (grade, plant)
                
                # HARD CONSTRAINT 6: Min/Max Run Days
                self.min_run_days[key] = int(row.get('Min. Run Days', 1))
                self.max_run_days[key] = int(row.get('Max. Run Days', 9999))
                
                # HARD CONSTRAINT 9: Rerun Allowed
                rerun_val = row.get('Rerun Allowed')
                if pd.notna(rerun_val):
                    val_str = str(rerun_val).strip().lower()
                    self.rerun_allowed[key] = val_str not in ['no', 'n', 'false', '0']
                else:
                    self.rerun_allowed[key] = True
                
                # HARD CONSTRAINT 7: Force Start Date
                if pd.notna(row.get('Force Start Date')):
                    try:
                        self.force_start_date[key] = pd.to_datetime(row['Force Start Date']).date()
                    except:
                        self.force_start_date[key] = None
                else:
                    self.force_start_date[key] = None
        
        # HARD CONSTRAINT 4: Shutdown periods
        self.shutdown_periods = {}
        for _, row in self.plant_df.iterrows():
            plant = row['Plant']
            start_date = row.get('Shutdown Start Date')
            end_date = row.get('Shutdown End Date')
            
            if pd.notna(start_date) and pd.notna(end_date):
                start_date = pd.to_datetime(start_date).date()
                end_date = pd.to_datetime(end_date).date()
                
                if start_date <= end_date:
                    shutdown_days = []
                    for d, date in enumerate(self.dates):
                        if start_date <= date <= end_date:
                            shutdown_days.append(d)
                    self.shutdown_periods[plant] = shutdown_days
                    st.info(f"🔧 {plant}: Shutdown {format_date(start_date)} to {format_date(end_date)} ({len(shutdown_days)} days)")
                else:
                    self.shutdown_periods[plant] = []
            else:
                self.shutdown_periods[plant] = []
        
        # HARD CONSTRAINT 2 & 3: Material running
        self.material_running = {}
        for _, row in self.plant_df.iterrows():
            plant = row['Plant']
            material = row.get('Material Running')
            expected_days = row.get('Expected Run Days')
            
            if pd.notna(material) and pd.notna(expected_days):
                material_str = str(material).strip()
                if material_str in self.grades:
                    self.material_running[plant] = (material_str, int(expected_days))
                    st.info(f"🔄 {plant}: Running {material_str} for {int(expected_days)} days")
        
        # HARD CONSTRAINT 10: Transition rules
        self.transition_rules = {}
        for line, df in self.transition_dfs.items():
            if df is not None and line in self.lines:
                self.transition_rules[line] = {}
                for prev_grade in df.index:
                    allowed = []
                    for curr_grade in df.columns:
                        if str(df.loc[prev_grade, curr_grade]).lower() == 'yes':
                            allowed.append(curr_grade)
                    self.transition_rules[line][str(prev_grade)] = allowed
                st.info(f"✓ Loaded transition rules for {line}")
            else:
                self.transition_rules[line] = None
    
    def build_model(self):
        """Build CP-SAT model with all constraints."""
        self.model = cp_model.CpModel()
        
        # Decision variables
        self.is_producing = {}
        self.production = {}
        
        # Create variables only for allowed grade-line combinations
        for grade in self.grades:
            for line in self.allowed_lines[grade]:
                for d in range(self.num_days):
                    key = (grade, line, d)
                    
                    # Binary: is this grade being produced on this line today?
                    self.is_producing[key] = self.model.NewBoolVar(f'prod_{grade}_{line}_{d}')
                    
                    # Production quantity
                    self.production[key] = self.model.NewIntVar(
                        0, self.capacities[line], f'qty_{grade}_{line}_{d}'
                    )
                    
                    # Link production to is_producing
                    # During non-buffer days: if producing, must be at full capacity
                    if d < self.num_days - self.buffer_days:
                        self.model.Add(
                            self.production[key] == self.capacities[line]
                        ).OnlyEnforceIf(self.is_producing[key])
                        self.model.Add(
                            self.production[key] == 0
                        ).OnlyEnforceIf(self.is_producing[key].Not())
                    else:
                        # Buffer days: can produce less than capacity
                        self.model.Add(
                            self.production[key] <= self.capacities[line] * self.is_producing[key]
                        )
        
        # Inventory and stockout variables
        self.inventory_vars = {}
        self.stockout_vars = {}
        
        for grade in self.grades:
            for d in range(self.num_days + 1):
                self.inventory_vars[(grade, d)] = self.model.NewIntVar(
                    0, int(self.max_inventory[grade] * 2), f'inv_{grade}_{d}'
                )
            
            for d in range(self.num_days):
                self.stockout_vars[(grade, d)] = self.model.NewIntVar(
                    0, 1000000, f'stockout_{grade}_{d}'
                )
        
        # Apply all constraints
        self._add_hard_constraints()
        self._add_soft_constraints()
        self._add_objective()
    
    def _add_hard_constraints(self):
        """Add all hard constraints."""
        
        # HARD CONSTRAINT 1 & 8: One grade per line per day, only allowed lines
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
        
        # HARD CONSTRAINT 1: Full capacity utilization (non-buffer, non-shutdown days)
        for line in self.lines:
            for d in range(self.num_days - self.buffer_days):
                # Skip shutdown days
                if d in self.shutdown_periods.get(line, []):
                    continue
                
                prod_vars = []
                for grade in self.grades:
                    if line in self.allowed_lines[grade]:
                        key = (grade, line, d)
                        if key in self.production:
                            prod_vars.append(self.production[key])
                
                if prod_vars:
                    self.model.Add(sum(prod_vars) == self.capacities[line])
        
        # HARD CONSTRAINT 4: No production during shutdown
        for line in self.lines:
            for d in self.shutdown_periods.get(line, []):
                for grade in self.grades:
                    if line in self.allowed_lines[grade]:
                        key = (grade, line, d)
                        if key in self.is_producing:
                            self.model.Add(self.is_producing[key] == 0)
                            self.model.Add(self.production[key] == 0)
        
        # HARD CONSTRAINT 2 & 3: Material running for expected days
        for plant, (material, expected_days) in self.material_running.items():
            if plant in self.lines and material in self.grades:
                for d in range(min(expected_days, self.num_days)):
                    if plant in self.allowed_lines[material]:
                        key = (material, plant, d)
                        if key in self.is_producing:
                            self.model.Add(self.is_producing[key] == 1)
                        
                        # Ensure no other grade runs
                        for other_grade in self.grades:
                            if other_grade != material and plant in self.allowed_lines[other_grade]:
                                other_key = (other_grade, plant, d)
                                if other_key in self.is_producing:
                                    self.model.Add(self.is_producing[other_key] == 0)
        
        # HARD CONSTRAINT 5: Initial inventory
        for grade in self.grades:
            self.model.Add(
                self.inventory_vars[(grade, 0)] == int(self.initial_inventory[grade])
            )
        
        # Inventory balance
        for grade in self.grades:
            for d in range(self.num_days):
                produced = sum(
                    self.production.get((grade, line, d), 0)
                    for line in self.allowed_lines[grade]
                )
                demand = int(self.demand_dict[grade].get(self.dates[d], 0))
                
                inv_before = self.inventory_vars[(grade, d)]
                inv_after = self.inventory_vars[(grade, d + 1)]
                stockout = self.stockout_vars[(grade, d)]
                
                # Balance: inv_after = inv_before + produced - demand + stockout
                # Where stockout represents unmet demand
                self.model.Add(inv_after == inv_before + produced - demand + stockout)
                
                # Stockout occurs when we can't meet demand
                # stockout >= demand - (inv_before + produced)
                self.model.Add(stockout >= demand - inv_before - produced)
                self.model.Add(stockout >= 0)
        
        # HARD CONSTRAINT 6 & 7 & 9: Min/max run days, force start, rerun
        self._add_run_constraints()
        
        # HARD CONSTRAINT 10: Transition rules
        self._add_transition_rules()
    
    def _add_run_constraints(self):
        """Add min/max run days, force start, and rerun constraints."""
        
        # Create start and end indicators
        is_start = {}
        is_end = {}
        
        for grade in self.grades:
            for line in self.allowed_lines[grade]:
                for d in range(self.num_days):
                    is_start[(grade, line, d)] = self.model.NewBoolVar(f'start_{grade}_{line}_{d}')
                    is_end[(grade, line, d)] = self.model.NewBoolVar(f'end_{grade}_{line}_{d}')
                    
                    curr = self.is_producing.get((grade, line, d))
                    if curr is None:
                        continue
                    
                    # Define start
                    if d == 0:
                        self.model.Add(is_start[(grade, line, d)] == curr)
                    else:
                        prev = self.is_producing.get((grade, line, d - 1))
                        if prev is not None:
                            # Start = producing today AND not producing yesterday
                            start_indicator = self.model.NewBoolVar(f'start_ind_{grade}_{line}_{d}')
                            self.model.AddBoolAnd([curr, prev.Not()]).OnlyEnforceIf(start_indicator)
                            self.model.AddBoolOr([curr.Not(), prev]).OnlyEnforceIf(start_indicator.Not())
                            self.model.Add(is_start[(grade, line, d)] == start_indicator)
                    
                    # Define end
                    if d == self.num_days - 1:
                        self.model.Add(is_end[(grade, line, d)] == curr)
                    else:
                        next_var = self.is_producing.get((grade, line, d + 1))
                        if next_var is not None:
                            # End = producing today AND not producing tomorrow
                            end_indicator = self.model.NewBoolVar(f'end_ind_{grade}_{line}_{d}')
                            self.model.AddBoolAnd([curr, next_var.Not()]).OnlyEnforceIf(end_indicator)
                            self.model.AddBoolOr([curr.Not(), next_var]).OnlyEnforceIf(end_indicator.Not())
                            self.model.Add(is_end[(grade, line, d)] == end_indicator)
                
                key = (grade, line)
                
                # HARD CONSTRAINT 6a: Minimum run days
                min_run = self.min_run_days.get(key, 1)
                for d in range(self.num_days - min_run + 1):
                    start_var = is_start.get((grade, line, d))
                    if start_var is None:
                        continue
                    
                    # If we start at day d, must produce for at least min_run days
                    for k in range(min_run):
                        if d + k < self.num_days:
                            # Skip if shutdown day
                            if (d + k) in self.shutdown_periods.get(line, []):
                                continue
                            
                            prod_var = self.is_producing.get((grade, line, d + k))
                            if prod_var is not None:
                                self.model.Add(prod_var == 1).OnlyEnforceIf(start_var)
                
                # HARD CONSTRAINT 6b: Maximum run days
                max_run = self.max_run_days.get(key, 9999)
                if max_run < 9999:
                    for d in range(self.num_days - max_run):
                        consecutive_vars = []
                        for k in range(max_run + 1):
                            if d + k < self.num_days:
                                # Skip shutdown days
                                if (d + k) in self.shutdown_periods.get(line, []):
                                    break
                                prod_var = self.is_producing.get((grade, line, d + k))
                                if prod_var is not None:
                                    consecutive_vars.append(prod_var)
                        
                        if len(consecutive_vars) == max_run + 1:
                            self.model.Add(sum(consecutive_vars) <= max_run)
                
                # HARD CONSTRAINT 9: Rerun allowed
                if not self.rerun_allowed.get(key, True):
                    # Can only start once
                    start_vars = [is_start.get((grade, line, d)) 
                                 for d in range(self.num_days) 
                                 if is_start.get((grade, line, d)) is not None]
                    if start_vars:
                        self.model.Add(sum(start_vars) <= 1)
                
                # HARD CONSTRAINT 7: Force start date
                force_date = self.force_start_date.get(key)
                if force_date and force_date in self.dates:
                    day_idx = self.dates.index(force_date)
                    prod_var = self.is_producing.get((grade, line, day_idx))
                    if prod_var is not None:
                        self.model.Add(prod_var == 1)
                        st.info(f"✓ Forcing {grade} on {line} at {format_date(force_date)}")
    
    def _add_transition_rules(self):
        """HARD CONSTRAINT 10: Enforce transition rules."""
        for line in self.lines:
            rules = self.transition_rules.get(line)
            if rules is None:
                continue
            
            for d in range(self.num_days - 1):
                for prev_grade in self.grades:
                    if prev_grade not in rules:
                        continue
                    
                    allowed_next = rules[prev_grade]
                    
                    # For each grade not in allowed list, forbid transition
                    for curr_grade in self.grades:
                        if curr_grade == prev_grade:
                            continue
                        
                        if curr_grade not in allowed_next:
                            # This transition is FORBIDDEN
                            prev_var = self.is_producing.get((prev_grade, line, d))
                            curr_var = self.is_producing.get((curr_grade, line, d + 1))
                            
                            if prev_var is not None and curr_var is not None:
                                # Cannot have both true
                                self.model.Add(prev_var + curr_var <= 1)
    
    def _add_soft_constraints(self):
        """Add soft constraints via penalties in objective."""
        # Soft constraints handled in objective (min inventory, closing inventory)
        pass
    
    def _add_objective(self):
        """Build objective function."""
        objective = 0
        
        # PRIMARY: Minimize stockouts (very high penalty)
        for grade in self.grades:
            for d in range(self.num_days):
                objective += self.stockout_penalty * self.stockout_vars[(grade, d)]
        
        # SECONDARY: Minimize transitions
        for line in self.lines:
            for d in range(self.num_days - 1):
                for g1 in self.grades:
                    if line not in self.allowed_lines[g1]:
                        continue
                    
                    for g2 in self.grades:
                        if g1 == g2 or line not in self.allowed_lines[g2]:
                            continue
                        
                        trans_var = self.model.NewBoolVar(f'trans_{line}_{d}_{g1}_{g2}')
                        
                        var1 = self.is_producing.get((g1, line, d))
                        var2 = self.is_producing.get((g2, line, d + 1))
                        
                        if var1 is not None and var2 is not None:
                            self.model.AddBoolAnd([var1, var2]).OnlyEnforceIf(trans_var)
                            self.model.AddBoolOr([var1.Not(), var2.Not()]).OnlyEnforceIf(trans_var.Not())
                            
                            objective += self.transition_penalty * trans_var
        
        # SOFT: Penalty for being below minimum inventory
        for grade in self.grades:
            min_inv = int(self.min_inventory[grade])
            if min_inv > 0:
                for d in range(1, self.num_days + 1):
                    deficit = self.model.NewIntVar(0, min_inv, f'deficit_{grade}_{d}')
                    self.model.Add(deficit >= min_inv - self.inventory_vars[(grade, d)])
                    self.model.Add(deficit >= 0)
                    objective += self.stockout_penalty * deficit // 2  # Lower penalty than stockout
        
        # SOFT: Penalty for not meeting min closing inventory
        for grade in self.grades:
            min_closing = int(self.min_closing_inventory[grade])
            if min_closing > 0:
                closing_day = self.num_days - self.buffer_days
                closing_deficit = self.model.NewIntVar(0, min_closing * 2, f'closing_deficit_{grade}')
                self.model.Add(closing_deficit >= min_closing - self.inventory_vars[(grade, closing_day)])
                self.model.Add(closing_deficit >= 0)
                objective += self.stockout_penalty * closing_deficit
        
        self.model.Minimize(objective)
    
    def solve(self):
        """Run optimization with configured solver."""
        self.solver = cp_model.CpSolver()
        
        # Configure solver
        self.solver.parameters.max_time_in_seconds = self.time_limit_min * 60
        self.solver.parameters.num_search_workers = 8
        self.solver.parameters.random_seed = 42
        self.solver.parameters.log_search_progress = True
        self.solver.parameters.optimize_with_core = True
        
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

def create_gantt_chart(line: str, schedule: List, dates: List[date], 
                       shutdown_days: List[int], grade_colors: Dict) -> go.Figure:
    """Create Gantt chart for production line."""
    gantt_data = []
    
    for d, (date, grade) in enumerate(zip(dates, schedule)):
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
            annotation_position='top left',
            annotation_font_color='red'
        )
    
    fig.update_yaxes(autorange='reversed', title=None)
    fig.update_xaxes(title='Date', dtick='D1', tickformat='%d-%b')
    fig.update_layout(
        height=400,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_inventory_chart(grade: str, inventory_data: List[float], 
                           dates: List[date], min_inv: float, max_inv: float,
                           demand_dict: Dict, grade_color: str) -> go.Figure:
    """Create inventory chart with demand overlay."""
    fig = go.Figure()
    
    # Inventory line
    fig.add_trace(go.Scatter(
        x=dates,
        y=inventory_data,
        mode='lines+markers',
        name='Inventory',
        line=dict(color=grade_color, width=3),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor=f'rgba{tuple(list(int(grade_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.1])}',
        hovertemplate='Date: %{x|%d-%b-%y}<br>Inventory: %{y:.0f} MT<extra></extra>'
    ))
    
    # Demand bars
    demand_values = [demand_dict.get(d, 0) for d in dates]
    fig.add_trace(go.Bar(
        x=dates,
        y=demand_values,
        name='Demand',
        marker=dict(color='rgba(255, 100, 100, 0.3)'),
        yaxis='y2',
        hovertemplate='Date: %{x|%d-%b-%y}<br>Demand: %{y:.0f} MT<extra></extra>'
    ))
    
    # Min/Max lines
    if min_inv > 0:
        fig.add_hline(
            y=min_inv,
            line=dict(color='red', width=2, dash='dash'),
            annotation_text=f'Min: {min_inv:,.0f}',
            annotation_position='top left',
            annotation_font_color='red'
        )
    
    if max_inv < 1000000:
        fig.add_hline(
            y=max_inv,
            line=dict(color='green', width=2, dash='dash'),
            annotation_text=f'Max: {max_inv:,.0f}',
            annotation_position='bottom left',
            annotation_font_color='green'
        )
    
    fig.update_layout(
        title=f'Inventory Level - {grade}',
        xaxis=dict(title='Date', dtick='D1', tickformat='%d-%b'),
        yaxis=dict(title='Inventory (MT)', side='left'),
        yaxis2=dict(title='Demand (MT)', side='right', overlaying='y'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=450,
        showlegend=True
    )
    
    return fig

def create_schedule_table(line: str, schedule: List, dates: List[date]) -> pd.DataFrame:
    """Create tabular schedule."""
    schedule_data = []
    current_grade = None
    start_day = None
    
    for d, (date, grade) in enumerate(zip(dates, schedule)):
        if grade != current_grade:
            if current_grade is not None:
                end_date = dates[d - 1]
                duration = (end_date - start_day).days + 1
                schedule_data.append({
                    'Grade': current_grade,
                    'Start Date': format_date(start_day),
                    'End Date': format_date(end_date),
                    'Days': duration
                })
            current_grade = grade
            start_day = date
    
    # Add last run
    if current_grade is not None:
        end_date = dates[-1]
        duration = (end_date - start_day).days + 1
        schedule_data.append({
            'Grade': current_grade,
            'Start Date': format_date(start_day),
            'End Date': format_date(end_date),
            'Days': duration
        })
    
    return pd.DataFrame(schedule_data)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    
    # Header
    st.markdown('<div class="main-header">🏭 Polymer Production Scheduler V2</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; color: #6C757D; margin-bottom: 2rem;">
        Production scheduling with <strong>strict constraint enforcement</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 Data Input")
        
        uploaded_file = st.file_uploader(
            "Upload Excel File",
            type=['xlsx'],
            help="Upload Excel file with Plant, Inventory, and Demand sheets"
        )
        
        if uploaded_file:
            st.success("✅ File uploaded!")
        
        # Download template
        st.markdown("---")
        st.markdown("### 📥 Sample Template")
        
        template = load_excel_template()
        if template:
            st.download_button(
                label="📥 Download Template",
                data=template,
                file_name="polymer_production_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        if uploaded_file:
            st.markdown("---")
            st.markdown("### ⚙️ Optimization Parameters")
            
            with st.expander("🔧 Solver Settings", expanded=True):
                time_limit = st.number_input(
                    "Time limit (minutes)",
                    min_value=1, max_value=60, value=10,
                    help="Maximum time for solver to run"
                )
                
                buffer_days = st.number_input(
                    "Buffer days",
                    min_value=0, max_value=7, value=3,
                    help="Additional planning days beyond demand horizon"
                )
            
            with st.expander("🎯 Penalty Weights", expanded=True):
                stockout_penalty = st.number_input(
                    "Stockout penalty",
                    min_value=1, max_value=1000, value=100,
                    help="Higher = prioritize meeting demand"
                )
                
                transition_penalty = st.number_input(
                    "Transition penalty",
                    min_value=1, max_value=100, value=10,
                    help="Higher = fewer grade changes"
                )
            
            st.markdown("---")
            st.markdown("""
            <div style="font-size: 0.85rem; color: #6C757D;">
            <strong>Hard Constraints (Strictly Enforced):</strong><br>
            ✓ Capacity utilization<br>
            ✓ Material running<br>
            ✓ Shutdown periods<br>
            ✓ Min/Max run days<br>
            ✓ Force start dates<br>
            ✓ Allowed lines<br>
            ✓ Rerun allowed<br>
            ✓ Transition rules
            </div>
            """, unsafe_allow_html=True)
    
    # Main content
    if uploaded_file is None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h3>👈 Get Started</h3>
                <p>Upload an Excel file to begin optimization, or download the sample template.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            ### ✨ Key Features
            
            - **🎯 Strict Constraint Enforcement**: All hard constraints are guaranteed
            - **📊 Interactive Visualizations**: Gantt charts and inventory graphs
            - **🔧 Flexible Configuration**: Adjust penalties and time limits
            - **📈 Real-time Tracking**: See solver progress and intermediate solutions
            - **🚀 Fast Optimization**: Powered by Google OR-Tools CP-SAT
            
            ### 📋 Hard Constraints (Always Enforced)
            
            1. **Full capacity utilization** (except buffer days and shutdowns)
            2. **Material running** for specified initial days
            3. **Expected run days** for current materials
            4. **Shutdown periods** - zero production during shutdowns
            5. **Opening inventory** - exact starting levels
            6. **Min & Max run days** - consecutive production limits
            7. **Force start dates** - mandatory production starts
            8. **Allowed lines** - grade-plant restrictions
            9. **Rerun allowed** - single vs. multiple runs
            10. **Transition rules** - forbidden grade changes
            
            ### 📊 Soft Constraints (Optimized)
            
            - Minimum inventory levels (penalized if violated)
            - Minimum closing inventory (penalized if not met)
            - Transition minimization (via penalty weights)
            - Stockout prevention (primary objective)
            """)
        
        with col2:
            st.markdown("""
            ### 📖 Quick Guide
            
            **1. Prepare Excel File**
            - Plant sheet: capacities, shutdowns
            - Inventory sheet: constraints per grade
            - Demand sheet: daily requirements
            - Transition sheets: allowed changes
            
            **2. Upload & Configure**
            - Upload your Excel file
            - Set time limit
            - Adjust penalty weights
            
            **3. Optimize**
            - Click "Run Optimization"
            - Monitor progress
            - Review results
            
            **4. Analyze**
            - View Gantt charts
            - Check inventory levels
            - Export schedules
            """)
        
        return
    
    # Load and validate data
    try:
        st.markdown("---")
        st.markdown("### 📊 Data Validation")
        
        file_bytes = uploaded_file.read()
        
        # Load data
        with st.spinner("Loading data..."):
            excel_file = io.BytesIO(file_bytes)
            
            plant_df = pd.read_excel(excel_file, sheet_name='Plant')
            
            excel_file.seek(0)
            inventory_df = pd.read_excel(excel_file, sheet_name='Inventory')
            
            excel_file.seek(0)
            demand_df = pd.read_excel(excel_file, sheet_name='Demand')
            
            # Load transitions
            transition_dfs = {}
            plant_names = plant_df['Plant'].tolist()
            for plant in plant_names:
                patterns = [
                    f'Transition_{plant}',
                    f'Transition_{plant.replace(" ", "_")}',
                    f'Transition{plant.replace(" ", "")}'
                ]
                for sheet_name in patterns:
                    try:
                        excel_file.seek(0)
                        df = pd.read_excel(excel_file, sheet_name=sheet_name, index_col=0)
                        transition_dfs[plant] = df
                        break
                    except:
                        continue
                if plant not in transition_dfs:
                    transition_dfs[plant] = None
        
        # Basic validation
        errors = []
        warnings = []
        
        # Check for required columns
        if 'Plant' not in plant_df.columns or 'Capacity per day' not in plant_df.columns:
            errors.append("❌ Plant sheet missing required columns")
        
        if 'Grade Name' not in inventory_df.columns:
            errors.append("❌ Inventory sheet missing 'Grade Name' column")
        
        # Check capacities
        if (plant_df['Capacity per day'] <= 0).any():
            errors.append("❌ All plant capacities must be positive")
        
        # Display validation results
        if errors:
            for error in errors:
                st.error(error)
            st.stop()
        
        st.success("✅ Data validation passed!")
        
        # Display data preview
        st.markdown("---")
        st.markdown("### 📋 Data Preview")
        
        tab1, tab2, tab3 = st.tabs(["🏭 Plants", "📦 Inventory", "📈 Demand"])
        
        with tab1:
            st.dataframe(plant_df, use_container_width=True, height=300)
        
        with tab2:
            st.dataframe(inventory_df, use_container_width=True, height=300)
        
        with tab3:
            st.dataframe(demand_df.head(15), use_container_width=True, height=300)
        
        # Optimization button
        st.markdown("---")
        
        if st.button("🎯 Run Optimization", type="primary", use_container_width=True):
            
            progress_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.markdown('<div class="info-box">📊 Initializing scheduler...</div>', 
                                   unsafe_allow_html=True)
                progress_bar.progress(10)
                
                # Create scheduler
                scheduler = ProductionScheduler(
                    plant_df, inventory_df, demand_df, transition_dfs,
                    buffer_days, time_limit, stockout_penalty, transition_penalty
                )
                
                status_text.markdown('<div class="info-box">🔧 Preparing data and constraints...</div>', 
                                   unsafe_allow_html=True)
                progress_bar.progress(20)
                
                scheduler.prepare_data()
                
                status_text.markdown('<div class="info-box">🏗️ Building optimization model...</div>', 
                                   unsafe_allow_html=True)
                progress_bar.progress(40)
                
                scheduler.build_model()
                
                status_text.markdown('<div class="info-box">⚡ Running solver (this may take a few minutes)...</div>', 
                                   unsafe_allow_html=True)
                progress_bar.progress(60)
                
                status, solve_time = scheduler.solve()
                
                progress_bar.progress(100)
            
            # Display results
            st.markdown("---")
            
            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                status_msg = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
                status_text.markdown(
                    f'<div class="success-box">✅ Optimization completed! Status: {status_msg}</div>', 
                    unsafe_allow_html=True
                )
                
                if scheduler.callback.solutions:
                    solution = scheduler.callback.solutions[-1]
                    
                    st.markdown("### 📊 Optimization Results")
                    
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
                            <div class="metric-value">{solution['transitions']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Total Stockout</div>
                            <div class="metric-value">{solution['total_stockout']:,.0f} MT</div>
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
                    
                    # Results tabs
                    tab1, tab2, tab3 = st.tabs(["📅 Production Schedule", "📊 Summary Tables", "📦 Inventory Tracking"])
                    
                    with tab1:
                        st.markdown("### 📅 Production Schedules by Line")
                        
                        # Generate colors
                        sorted_grades = sorted(scheduler.grades)
                        colors = px.colors.qualitative.Vivid + px.colors.qualitative.Safe
                        grade_colors = {
                            grade: colors[i % len(colors)] 
                            for i, grade in enumerate(sorted_grades)
                        }
                        
                        for line in scheduler.lines:
                            st.markdown(f"#### {line}")
                            
                            # Gantt chart
                            fig = create_gantt_chart(
                                line, 
                                solution['schedule'][line],
                                scheduler.dates,
                                scheduler.shutdown_periods.get(line, []),
                                grade_colors
                            )
                            
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Schedule table
                            schedule_df = create_schedule_table(
                                line,
                                solution['schedule'][line],
                                scheduler.dates
                            )
                            
                            if not schedule_df.empty:
                                st.dataframe(schedule_df, use_container_width=True)
                            else:
                                st.info(f"No production scheduled for {line}")
                    
                    with tab2:
                        st.markdown("### 📊 Production Summary")
                        
                        # Build summary by grade
                        summary_data = []
                        for grade in sorted_grades:
                            row = {'Grade': grade}
                            grade_total = 0
                            
                            for line in scheduler.lines:
                                line_total = sum(
                                    scheduler.solver.Value(scheduler.production.get((grade, line, d), 0))
                                    for d in range(scheduler.num_days)
                                )
                                row[line] = line_total
                                grade_total += line_total
                            
                            row['Total Production'] = grade_total
                            
                            # Total demand
                            total_demand = sum(
                                scheduler.demand_dict[grade].get(scheduler.dates[d], 0)
                                for d in range(scheduler.num_days - scheduler.buffer_days)
                            )
                            row['Total Demand'] = int(total_demand)
                            
                            # Service level
                            if total_demand > 0:
                                service_level = min(100, (grade_total / total_demand) * 100)
                                row['Service Level %'] = f"{service_level:.1f}%"
                            else:
                                row['Service Level %'] = "N/A"
                            
                            summary_data.append(row)
                        
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True)
                    
                    with tab3:
                        st.markdown("### 📦 Inventory Levels by Grade")
                        
                        for grade in sorted_grades:
                            inv_data = [
                                scheduler.solver.Value(scheduler.inventory_vars[(grade, d)])
                                for d in range(scheduler.num_days + 1)
                            ]
                            
                            fig = create_inventory_chart(
                                grade,
                                inv_data[:-1],  # Exclude final day for chart alignment
                                scheduler.dates,
                                scheduler.min_inventory[grade],
                                scheduler.max_inventory[grade],
                                scheduler.demand_dict[grade],
                                grade_colors[grade]
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Inventory stats
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Opening", f"{inv_data[0]:,.0f} MT")
                            with col2:
                                st.metric("Closing", f"{inv_data[scheduler.num_days - scheduler.buffer_days]:,.0f} MT")
                            with col3:
                                st.metric("Peak", f"{max(inv_data):,.0f} MT")
                            with col4:
                                st.metric("Minimum", f"{min(inv_data):,.0f} MT")
                
            else:
                status_text.markdown(
                    '<div class="warning-box">⚠️ No feasible solution found. The constraints may be too restrictive. Try:<br>'
                    '• Increasing time limit<br>'
                    '• Relaxing min/max run days<br>'
                    '• Adjusting shutdown periods<br>'
                    '• Increasing opening inventory</div>', 
                    unsafe_allow_html=True
                )
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        with st.expander("🐛 Debug Information"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
