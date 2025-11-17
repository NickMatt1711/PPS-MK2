"""
Production scheduling optimization engine.
Pure optimization logic with no UI dependencies.
"""

import pandas as pd
from ortools.sat.python import cp_model
from datetime import date, timedelta
import time
from typing import Dict, List, Tuple, Optional


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
            for d in range(self.num_days):
                daily_prod = 0
                for line in self.lines:
                    key = (grade, line, d)
                    if key in self.production:
                        daily_prod += self.Value(self.production[key])
                if daily_prod > 0:
                    solution['production'][grade][str(self.dates[d])] = daily_prod
        
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
                    total_stockout += self.Value(self.stockout[key])
        solution['total_stockout'] = total_stockout
        
        # Extract schedule
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
        
        # Prepared data structures
        self.lines = []
        self.grades = []
        self.dates = []
        self.num_days = 0
        self.capacities = {}
        self.demand_dict = {}
        self.initial_inventory = {}
        self.min_inventory = {}
        self.max_inventory = {}
        self.min_closing_inventory = {}
        self.allowed_lines = {}
        self.min_run_days = {}
        self.max_run_days = {}
        self.force_start_date = {}
        self.rerun_allowed = {}
        self.shutdown_periods = {}
        self.material_running = {}
        self.transition_rules = {}
        
        # Model variables
        self.is_producing = {}
        self.production = {}
        self.inventory_vars = {}
        self.stockout_vars = {}
    
    def prepare_data(self) -> List[str]:
        """
        Prepare all data structures.
        
        Returns:
            List of info messages about constraints
        """
        messages = []
        
        # Lines and capacities
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
            # Buffer days
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
            
            # Parse allowed lines
            lines_str = row.get('Lines', '')
            if pd.isna(lines_str) or str(lines_str).strip() == '':
                plants_for_row = self.lines
                messages.append(f"⚠️ Grade '{grade}': No lines specified, using all lines")
            else:
                plants_for_row = [x.strip() for x in str(lines_str).split(',')]
            
            for plant in plants_for_row:
                if plant in self.lines and plant not in self.allowed_lines[grade]:
                    self.allowed_lines[grade].append(plant)
            
            # Global inventory parameters
            if grade not in grade_inventory_defined:
                self.initial_inventory[grade] = float(row.get('Opening Inventory', 0))
                self.min_inventory[grade] = float(row.get('Min. Inventory', 0))
                self.max_inventory[grade] = float(row.get('Max. Inventory', 1000000))
                self.min_closing_inventory[grade] = float(row.get('Min. Closing Inventory', 0))
                grade_inventory_defined.add(grade)
            
            # Plant-specific parameters
            for plant in plants_for_row:
                if plant not in self.lines:
                    continue
                
                key = (grade, plant)
                
                self.min_run_days[key] = int(row.get('Min. Run Days', 1))
                self.max_run_days[key] = int(row.get('Max. Run Days', 9999))
                
                rerun_val = row.get('Rerun Allowed')
                if pd.notna(rerun_val):
                    val_str = str(rerun_val).strip().lower()
                    self.rerun_allowed[key] = val_str not in ['no', 'n', 'false', '0']
                else:
                    self.rerun_allowed[key] = True
                
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
                
                if start_date <= end_date:
                    shutdown_days = []
                    for d, dt in enumerate(self.dates):
                        if start_date <= dt <= end_date:
                            shutdown_days.append(d)
                    self.shutdown_periods[plant] = shutdown_days
                    messages.append(
                        f"🔧 {plant}: Shutdown {start_date.strftime('%d-%b-%y')} to "
                        f"{end_date.strftime('%d-%b-%y')} ({len(shutdown_days)} days)"
                    )
                else:
                    self.shutdown_periods[plant] = []
            else:
                self.shutdown_periods[plant] = []
        
        # Material running
        self.material_running = {}
        for _, row in self.plant_df.iterrows():
            plant = row['Plant']
            material = row.get('Material Running')
            expected_days = row.get('Expected Run Days')
            
            if pd.notna(material) and pd.notna(expected_days):
                material_str = str(material).strip()
                if material_str in self.grades:
                    self.material_running[plant] = (material_str, int(expected_days))
                    messages.append(f"🔄 {plant}: Running {material_str} for {int(expected_days)} days")
        
        # Transition rules
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
                messages.append(f"✓ Loaded transition rules for {line}")
            else:
                self.transition_rules[line] = None
        
        return messages
    
    def build_model(self):
        """Build CP-SAT model with all constraints."""
        self.model = cp_model.CpModel()
        
        # Create decision variables
        self._create_variables()
        
        # Add all constraints
        self._add_hard_constraints()
        self._add_soft_constraints()
        self._add_objective()
    
    def _create_variables(self):
        """Create all decision variables."""
        # Production variables
        for grade in self.grades:
            for line in self.allowed_lines[grade]:
                for d in range(self.num_days):
                    key = (grade, line, d)
                    
                    self.is_producing[key] = self.model.NewBoolVar(f'prod_{grade}_{line}_{d}')
                    self.production[key] = self.model.NewIntVar(
                        0, self.capacities[line], f'qty_{grade}_{line}_{d}'
                    )
                    
                    # Link production to is_producing
                    if d < self.num_days - self.buffer_days:
                        self.model.Add(
                            self.production[key] == self.capacities[line]
                        ).OnlyEnforceIf(self.is_producing[key])
                        self.model.Add(
                            self.production[key] == 0
                        ).OnlyEnforceIf(self.is_producing[key].Not())
                    else:
                        self.model.Add(
                            self.production[key] <= self.capacities[line] * self.is_producing[key]
                        )
        
        # Inventory and stockout variables
        for grade in self.grades:
            for d in range(self.num_days + 1):
                self.inventory_vars[(grade, d)] = self.model.NewIntVar(
                    0, int(self.max_inventory[grade] * 2), f'inv_{grade}_{d}'
                )
            
            for d in range(self.num_days):
                self.stockout_vars[(grade, d)] = self.model.NewIntVar(
                    0, 1000000, f'stockout_{grade}_{d}'
                )
    
    def _add_hard_constraints(self):
        """Add all hard constraints."""
        # One grade per line per day
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
        
        # Full capacity utilization
        for line in self.lines:
            for d in range(self.num_days - self.buffer_days):
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
        
        # No production during shutdown
        for line in self.lines:
            for d in self.shutdown_periods.get(line, []):
                for grade in self.grades:
                    if line in self.allowed_lines[grade]:
                        key = (grade, line, d)
                        if key in self.is_producing:
                            self.model.Add(self.is_producing[key] == 0)
                            self.model.Add(self.production[key] == 0)
        
        # Material running
        for plant, (material, expected_days) in self.material_running.items():
            if plant in self.lines and material in self.grades:
                for d in range(min(expected_days, self.num_days)):
                    if plant in self.allowed_lines[material]:
                        key = (material, plant, d)
                        if key in self.is_producing:
                            self.model.Add(self.is_producing[key] == 1)
                        
                        for other_grade in self.grades:
                            if other_grade != material and plant in self.allowed_lines[other_grade]:
                                other_key = (other_grade, plant, d)
                                if other_key in self.is_producing:
                                    self.model.Add(self.is_producing[other_key] == 0)
        
        # Initial inventory
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
                
                self.model.Add(inv_after == inv_before + produced - demand + stockout)
                self.model.Add(stockout >= demand - inv_before - produced)
                self.model.Add(stockout >= 0)
        
        # Run day constraints
        self._add_run_constraints()
        
        # Transition rules
        self._add_transition_rules()
    
    def _add_run_constraints(self):
        """Add min/max run days, force start, rerun constraints."""
        # Create start/end indicators
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
                            end_indicator = self.model.NewBoolVar(f'end_ind_{grade}_{line}_{d}')
                            self.model.AddBoolAnd([curr, next_var.Not()]).OnlyEnforceIf(end_indicator)
                            self.model.AddBoolOr([curr.Not(), next_var]).OnlyEnforceIf(end_indicator.Not())
                            self.model.Add(is_end[(grade, line, d)] == end_indicator)
                
                key = (grade, line)
                
                # Min run days
                min_run = self.min_run_days.get(key, 1)
                for d in range(self.num_days - min_run + 1):
                    start_var = is_start.get((grade, line, d))
                    if start_var is None:
                        continue
                    
                    for k in range(min_run):
                        if d + k < self.num_days:
                            if (d + k) in self.shutdown_periods.get(line, []):
                                continue
                            
                            prod_var = self.is_producing.get((grade, line, d + k))
                            if prod_var is not None:
                                self.model.Add(prod_var == 1).OnlyEnforceIf(start_var)
                
                # Max run days
                max_run = self.max_run_days.get(key, 9999)
                if max_run < 9999:
                    for d in range(self.num_days - max_run):
                        consecutive_vars = []
                        for k in range(max_run + 1):
                            if d + k < self.num_days:
                                if (d + k) in self.shutdown_periods.get(line, []):
                                    break
                                prod_var = self.is_producing.get((grade, line, d + k))
                                if prod_var is not None:
                                    consecutive_vars.append(prod_var)
                        
                        if len(consecutive_vars) == max_run + 1:
                            self.model.Add(sum(consecutive_vars) <= max_run)
                
                # Rerun allowed
                if not self.rerun_allowed.get(key, True):
                    start_vars = [is_start.get((grade, line, d)) 
                                 for d in range(self.num_days) 
                                 if is_start.get((grade, line, d)) is not None]
                    if start_vars:
                        self.model.Add(sum(start_vars) <= 1)
                
                # Force start date
                force_date = self.force_start_date.get(key)
                if force_date and force_date in self.dates:
                    day_idx = self.dates.index(force_date)
                    prod_var = self.is_producing.get((grade, line, day_idx))
                    if prod_var is not None:
                        self.model.Add(prod_var == 1)
    
    def _add_transition_rules(self):
        """Enforce transition rules."""
        for line in self.lines:
            rules = self.transition_rules.get(line)
            if rules is None:
                continue
            
            for d in range(self.num_days - 1):
                for prev_grade in self.grades:
                    if prev_grade not in rules:
                        continue
                    
                    allowed_next = rules[prev_grade]
                    
                    for curr_grade in self.grades:
                        if curr_grade == prev_grade:
                            continue
                        
                        if curr_grade not in allowed_next:
                            prev_var = self.is_producing.get((prev_grade, line, d))
                            curr_var = self.is_producing.get((curr_grade, line, d + 1))
                            
                            if prev_var is not None and curr_var is not None:
                                self.model.Add(prev_var + curr_var <= 1)
    
    def _add_soft_constraints(self):
        """Add soft constraints via penalties."""
        pass
    
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
        
        # Soft: Min inventory penalty
        for grade in self.grades:
            min_inv = int(self.min_inventory[grade])
            if min_inv > 0:
                for d in range(1, self.num_days + 1):
                    deficit = self.model.NewIntVar(0, min_inv, f'deficit_{grade}_{d}')
                    self.model.Add(deficit >= min_inv - self.inventory_vars[(grade, d)])
                    self.model.Add(deficit >= 0)
                    penalty_weight = max(1, self.stockout_penalty // 2)
                    objective += penalty_weight * deficit
        
        # Soft: Min closing inventory penalty
        for grade in self.grades:
            min_closing = int(self.min_closing_inventory[grade])
            if min_closing > 0:
                closing_day = self.num_days - self.buffer_days
                closing_deficit = self.model.NewIntVar(0, min_closing * 2, f'closing_deficit_{grade}')
                self.model.Add(closing_deficit >= min_closing - self.inventory_vars[(grade, closing_day)])
                self.model.Add(closing_deficit >= 0)
                objective += self.stockout_penalty * closing_deficit
        
        self.model.Minimize(objective)
    
    def solve(self) -> Tuple[int, float]:
        """
        Run optimization.
        
        Returns:
            Tuple of (status, solve_time)
        """
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
