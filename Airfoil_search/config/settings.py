import os
from dataclasses import dataclass
from typing import List
import json
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Paths:
    workdir: Path
    xfoil_cmd: str
    result_csv: str
    normalised_result_csv: str
    
    @classmethod
    def load(cls, config_data: dict) -> 'Paths':
        """Load from config dict"""
        raw = config_data['paths']
        workdir = Path(raw["workdir"])
        
        return cls(
            workdir=workdir,
            xfoil_cmd=raw["xfoil_cmd"],
            result_csv=os.path.join(workdir, raw["results_csv"]),
            normalised_result_csv=os.path.join(workdir, raw["normalised_results_csv"])
        )


@dataclass(frozen=True)
class Execution:
    max_workers: int
    purge: bool
    save_pol_dump: bool
    
    @classmethod
    def load(cls, config_data: dict) -> 'Execution':
        """Load from config dict"""
        raw = config_data['run_settings']
        
        return cls(
            max_workers=min(raw["workers"], (os.cpu_count() or 1)),
            purge=raw["purge"],
            save_pol_dump=raw["save_pol_dump"]
        )


@dataclass(frozen=True)
class NACAranges:
    L_vals: List[int]
    P_vals: List[int]
    S_vals: List[int]
    TT_vals: List[int]
    
    @classmethod
    def load(cls, config_data: dict) -> 'NACAranges':
        """Load from config dict"""
        raw = config_data['naca_ranges']
        
        return cls(
            L_vals=raw["L_vals"],
            P_vals=raw["P_vals"],
            S_vals=raw["S_vals"],
            TT_vals=raw["TT_vals"]
        )


@dataclass(frozen=True)
class Atmosphere:
    dynamic_viscosity: float
    density: float
    temperature: float
    pressure: float
    gamma: float

    @property
    def speed_of_sound(self) -> float:
        """Compute from fundamental properties"""
        return np.sqrt(self.gamma * self.pressure / self.density)
    
    @classmethod
    def load(cls, config_data: dict) -> 'Atmosphere':
        """Load from config dict"""
        raw = config_data['atmosphere']
        
        return cls(
            dynamic_viscosity=raw["dynamic_viscosity"],
            density=raw["density"],
            temperature=raw["temperature"],
            pressure=raw["pressure"],
            gamma=raw["gamma"]
        )


@dataclass(frozen=True)
class FlightParams:
    chord_length: float
    velocity: float
    aspect_ratio: float
    span_efficiency: float
    friction_coeff: float
    atmosphere: Atmosphere  # Injected dependency
    
    @property
    def mach(self) -> float:
        return self.velocity / self.atmosphere.speed_of_sound
    
    @property
    def reynolds(self) -> float:
        return (self.atmosphere.density * self.velocity * 
                self.chord_length / self.atmosphere.dynamic_viscosity)
    
    @classmethod
    def load(cls, flight_data: dict, atmosphere: Atmosphere) -> 'FlightParams':
        """Load from flight dict and inject atmosphere"""
        return cls(
            chord_length=flight_data["chord_length"],
            velocity=flight_data["velocity"],
            aspect_ratio=flight_data["aspect_ratio"],
            span_efficiency=flight_data["span_efficiency"],
            friction_coeff=flight_data["friction_coeff"],
            atmosphere=atmosphere
        )


@dataclass(frozen=True, slots=True)
class XFOILParams:
    panels: int
    aseq: str
    iter: int
    timeout: int
    flight_params: FlightParams  # Injected dependency
    
    @property
    def mach(self) -> float:
        return self.flight_params.mach
    
    @property
    def Re(self) -> float:
        return self.flight_params.reynolds


@dataclass(frozen=True, slots=True)
class Relaxed:
    aseq: str
    panels: int


@dataclass(frozen=True)
class MetricWeights:
    min_m: float
    max_ld: float
    max_cl: float
    aoa_at_max_cl: float
    max_drag: float
    thickness: float
    
    @property
    def array(self) -> np.ndarray:
        return np.array([
            self.min_m,
            self.max_ld,
            self.max_cl,
            self.aoa_at_max_cl,
            self.max_drag,
            self.thickness
        ])[:6]
    
    @classmethod
    def load(cls, weights_data: dict) -> 'MetricWeights':
        """Load from weights dict"""
        return cls(
            min_m=weights_data["min_m"],
            max_ld=weights_data["max_ld"],
            max_cl=weights_data["max_cl"],
            aoa_at_max_cl=weights_data["aoa_at_max_cl"],
            max_drag=weights_data["max_drag"],
            thickness=weights_data["thickness"]
        )


@dataclass(frozen=True)
class XFOILSolverParams:
    coarse: XFOILParams
    fine: XFOILParams
    relaxed: Relaxed
    max_retries: int
    
    @classmethod
    def load(cls, solver_data: dict, flight_params: FlightParams) -> 'XFOILSolverParams':
        """Load from solver dict"""
        return cls(
            coarse=XFOILParams(
                **solver_data['xfoil_coarse'],
                flight_params=flight_params
            ),
            fine=XFOILParams(
                **solver_data['xfoil_fine'],
                flight_params=flight_params
            ),
            relaxed=Relaxed(**solver_data['relaxed']),
            max_retries=solver_data['max_retries']
        )


@dataclass(frozen=True)
class Config:
    """Complete configuration bundle"""
    paths: Paths
    execution: Execution
    naca_ranges: NACAranges
    atmosphere: Atmosphere
    flight_params: FlightParams
    solver_params: XFOILSolverParams
    metric_weights: MetricWeights


def _load_config(
    config_path: str = "./config/config.json",
    flight_path: str = "./config/flight.json",
    solver_path: str = "./config/solver.json",
    weights_path: str = "./config/weights.json"
) -> Config:
    """
    Load all configuration from JSON files.
    Respects dependency ordering and returns an immutable Config instance.
    
    Args:
        config_path: Path to config.json
        flight_path: Path to flight.json
        solver_path: Path to solver.json
        weights_path: Path to weights.json
    
    Returns:
        Immutable Config instance with all loaded settings
        
    Raises:
        FileNotFoundError: If any config file is missing
        json.JSONDecodeError: If any config file has invalid JSON
        KeyError: If required config keys are missing
    """
    # Load all JSON files
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    with open(flight_path, 'r') as f:
        flight_data = json.load(f)
    
    with open(solver_path, 'r') as f:
        solver_data = json.load(f)
    
    with open(weights_path, 'r') as f:
        weights_data = json.load(f)
    
    # Build components respecting dependencies
    paths = Paths.load(config_data)
    execution = Execution.load(config_data)
    naca_ranges = NACAranges.load(config_data)
    atmosphere = Atmosphere.load(config_data)
    flight_params = FlightParams.load(flight_data, atmosphere)
    solver_params = XFOILSolverParams.load(solver_data, flight_params)
    metric_weights = MetricWeights.load(weights_data)
    
    return Config(
        paths=paths,
        execution=execution,
        naca_ranges=naca_ranges,
        atmosphere=atmosphere,
        flight_params=flight_params,
        solver_params=solver_params,
        metric_weights=metric_weights
    )

config = _load_config()

paths = config.paths
execution = config.execution
naca_ranges = config.naca_ranges
atmosphere = config.atmosphere
flight_params = config.flight_params
solver_params = config.solver_params
metric_weights = config.metric_weights