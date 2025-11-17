"""
Data loading and validation.
Pure data handling with no UI or optimization logic.
"""

import pandas as pd
import io
from typing import Dict, List, Tuple


class ExcelDataLoader:
    """Load production scheduling data from Excel."""
    
    def __init__(self, file_bytes: bytes):
        """
        Initialize loader.
        
        Args:
            file_bytes: Excel file as bytes
        """
        self.file = io.BytesIO(file_bytes)
    
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Load all data from Excel.
        
        Returns:
            Tuple of (plant_df, inventory_df, demand_df, transition_dfs)
        """
        plant_df = self.load_plants()
        inventory_df = self.load_inventory()
        demand_df = self.load_demand()
        transition_dfs = self.load_transitions(plant_df['Plant'].tolist())
        
        return plant_df, inventory_df, demand_df, transition_dfs
    
    def load_plants(self) -> pd.DataFrame:
        """Load plant data from Plant sheet."""
        self.file.seek(0)
        return pd.read_excel(self.file, sheet_name='Plant')
    
    def load_inventory(self) -> pd.DataFrame:
        """Load inventory data from Inventory sheet."""
        self.file.seek(0)
        return pd.read_excel(self.file, sheet_name='Inventory')
    
    def load_demand(self) -> pd.DataFrame:
        """Load demand data from Demand sheet."""
        self.file.seek(0)
        return pd.read_excel(self.file, sheet_name='Demand')
    
    def load_transitions(self, plant_names: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load transition matrices for all plants.
        
        Args:
            plant_names: List of plant names
        
        Returns:
            Dictionary mapping plant name to transition DataFrame
        """
        transition_dfs = {}
        
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
                    transition_dfs[plant] = df
                    break
                except:
                    continue
            
            if plant not in transition_dfs:
                transition_dfs[plant] = None
        
        return transition_dfs


class DataValidator:
    """Validate production scheduling data."""
    
    @staticmethod
    def validate_all(plant_df: pd.DataFrame,
                     inventory_df: pd.DataFrame,
                     demand_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Validate all data.
        
        Args:
            plant_df: Plant data
            inventory_df: Inventory data
            demand_df: Demand data
        
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        
        # Validate plants
        plant_errors, plant_warnings = DataValidator.validate_plants(plant_df)
        errors.extend(plant_errors)
        warnings.extend(plant_warnings)
        
        # Validate inventory
        inv_errors, inv_warnings = DataValidator.validate_inventory(inventory_df)
        errors.extend(inv_errors)
        warnings.extend(inv_warnings)
        
        # Validate demand
        demand_errors, demand_warnings = DataValidator.validate_demand(demand_df)
        errors.extend(demand_errors)
        warnings.extend(demand_warnings)
        
        # Feasibility checks
        feasibility_warnings = DataValidator.check_feasibility(plant_df, demand_df)
        warnings.extend(feasibility_warnings)
        
        return errors, warnings
    
    @staticmethod
    def validate_plants(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate plant data."""
        errors = []
        warnings = []
        
        # Required columns
        if 'Plant' not in df.columns or 'Capacity per day' not in df.columns:
            errors.append("❌ Plant sheet missing required columns")
            return errors, warnings
        
        # Duplicate names
        if df['Plant'].duplicated().any():
            errors.append("❌ Duplicate plant names found")
        
        # Positive capacities
        if (df['Capacity per day'] <= 0).any():
            errors.append("❌ All plant capacities must be positive")
        
        # Very high capacities (warning)
        high_cap = df[df['Capacity per day'] > 10000]
        if not high_cap.empty:
            plants = ', '.join(high_cap['Plant'].tolist())
            warnings.append(f"⚠️ Very high capacities for: {plants}")
        
        return errors, warnings
    
    @staticmethod
    def validate_inventory(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate inventory data."""
        errors = []
        warnings = []
        
        if 'Grade Name' not in df.columns:
            errors.append("❌ Inventory sheet missing 'Grade Name' column")
            return errors, warnings
        
        for idx, row in df.iterrows():
            grade = row['Grade Name']
            
            # Lines specified
            if pd.isna(row.get('Lines')) or str(row.get('Lines', '')).strip() == '':
                warnings.append(f"⚠️ Grade '{grade}': No lines specified")
            
            # Run days
            min_run = row.get('Min. Run Days', 1)
            max_run = row.get('Max. Run Days', 9999)
            
            if pd.notna(min_run) and pd.notna(max_run) and min_run > max_run:
                errors.append(f"❌ Grade '{grade}': Min run days > Max run days")
            
            # Inventory bounds
            min_inv = row.get('Min. Inventory', 0)
            max_inv = row.get('Max. Inventory', 1000000)
            
            if pd.notna(min_inv) and pd.notna(max_inv) and min_inv > max_inv:
                errors.append(f"❌ Grade '{grade}': Min inventory > Max inventory")
        
        return errors, warnings
    
    @staticmethod
    def validate_demand(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate demand data."""
        errors = []
        warnings = []
        
        # Check date column
        if not pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
            try:
                pd.to_datetime(df.iloc[:, 0])
            except:
                errors.append("❌ First column must contain valid dates")
        
        # Check for negative demand
        numeric_cols = df.select_dtypes(include=['number']).columns
        if (df[numeric_cols] < 0).any().any():
            errors.append("❌ Negative demand values found")
        
        return errors, warnings
    
    @staticmethod
    def check_feasibility(plant_df: pd.DataFrame, 
                         demand_df: pd.DataFrame) -> List[str]:
        """Quick feasibility check."""
        warnings = []
        
        total_capacity = plant_df['Capacity per day'].sum()
        grades = demand_df.columns[1:]
        
        if len(grades) > 0:
            total_avg_demand = demand_df[grades].mean().sum()
            
            if total_avg_demand > total_capacity * 0.9:
                warnings.append(
                    f"⚠️ High utilization: Avg demand ({total_avg_demand:.0f} MT/day) "
                    f"vs capacity ({total_capacity:.0f} MT/day)"
                )
        
        return warnings
