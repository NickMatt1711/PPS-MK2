"""
Data models using Pydantic for type safety and validation.
"""
from datetime import date
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class PlantData(BaseModel):
    """Plant/Production line configuration."""
    plant: str = Field(min_length=1)
    capacity_per_day: int = Field(gt=0)
    material_running: Optional[str] = None
    expected_run_days: Optional[int] = Field(None, ge=0)
    shutdown_start_date: Optional[date] = None
    shutdown_end_date: Optional[date] = None
    
    @model_validator(mode='after')
    def validate_shutdown_dates(self):
        if self.shutdown_start_date and self.shutdown_end_date:
            if self.shutdown_end_date < self.shutdown_start_date:
                raise ValueError(f"Shutdown end before start for {self.plant}")
        return self


class InventoryData(BaseModel):
    """Inventory and production constraints."""
    grade_name: str
    lines: List[str] = Field(min_length=1)
    opening_inventory: float = Field(ge=0)
    min_inventory: float = Field(ge=0)
    max_inventory: float = Field(gt=0)
    min_closing_inventory: float = Field(ge=0)
    min_run_days: int = Field(ge=1)
    max_run_days: int = Field(ge=1)
    force_start_date: Optional[date] = None
    rerun_allowed: bool = True


class DemandData(BaseModel):
    """Demand forecast."""
    date: date
    grade: str
    quantity: float = Field(ge=0)


class OptimizationParams(BaseModel):
    """Optimization parameters."""
    time_limit_minutes: int = Field(10, ge=1, le=120)
    buffer_days: int = Field(3, ge=0, le=14)
    stockout_penalty: int = Field(10, ge=1)
    transition_penalty: int = Field(10, ge=1)
    num_workers: int = Field(8, ge=1, le=16)
    random_seed: int = Field(42, ge=0)
