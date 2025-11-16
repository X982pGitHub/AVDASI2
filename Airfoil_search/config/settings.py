import os
from dataclasses import dataclass
from typing import List
import json
from pathlib import Path
import numpy as np

class Paths:
    workdir: Path
    xfoil_cmd: str
    result_csv: str
    normalised_result_csv: str
    
    @classmethod
    def load(cls, config_path: str) -> None:
        """Load from config.json"""
        with open(config_path, 'r') as f:
            raw = json.load(f)['paths']

        cls.workdir = Path(raw["workdir"])
        cls.xfoil_cmd = raw["xfoil_cmd"]
        cls.result_csv = os.path.join(cls.workdir, raw["results_csv"])
        cls.normalised_result_csv = os.path.join(cls.workdir, raw["normalised_results_csv"])

class Execution:
    max_workers: int
    purge: bool
    save_pol_dump: bool
    
    @classmethod
    def load(cls, config_path: str) -> None:
        """Load from config.json"""
        with open(config_path, 'r') as f:
            raw = json.load(f)['run_settings']

        cls.max_workers = min(raw["workers"], (os.cpu_count() or 1))
        cls.purge = raw["purge"]
        cls.save_pol_dump = raw["save_pol_dump"]


class NACAranges:
    L_vals: List[int]
    P_vals: List[int]
    S_vals: List[int]
    TT_vals: List[int]
    
    @classmethod
    def load(cls, config_path: str) -> None:
        """Load from config.json"""
        with open(config_path, 'r') as f:
            raw = json.load(f)['naca_ranges']
        
        cls.L_vals = raw["L_vals"]
        cls.P_vals = raw["P_vals"]
        cls.S_vals = raw["S_vals"]
        cls.TT_vals = raw["TT_vals"]


class Atmosphere:
    dynamic_viscosity: float
    density: float
    temperature: float
    pressure: float
    speed_of_sound: float
    
    @classmethod
    def load(cls, config_path: str) -> None:
        """Load from config.json"""
        with open(config_path, 'r') as f:
            raw = json.load(f)['atmosphere']

        cls.dynamic_viscosity = raw["dynamic_viscosity"]
        cls.density = raw["density"]
        cls.temperature = raw["temperature"]
        cls.pressure = raw["pressure"]
        cls.speed_of_sound = raw["speed_of_sound"]

class FlightParams:
    chord_length: float
    velocity: float
    aspect_ratio: float
    span_efficiency: float
    friction_coeff: float
    
    @property
    def mach(self) -> float:
        return self.velocity / Atmosphere.speed_of_sound
    
    @property
    def reynolds(self) -> float:
        return Atmosphere.density * self.velocity * self.chord_length / Atmosphere.dynamic_viscosity
    
    @classmethod
    def load(cls, flight_path: str) -> None:
        """Load from flight.json and inject atmosphere properties"""
        with open(flight_path, 'r') as f:
            raw = json.load(f)

        cls.chord_length = raw["chord_length"]
        cls.velocity = raw["velocity"]
        cls.aspect_ratio= raw["aspect_ratio"]
        cls.span_efficiency = raw["span_efficiency"]
        cls.friction_coeff = raw["friction_coeff"]

@dataclass(slots=True)
class XFOILParams:
    panels: int
    aseq: str
    iter: int
    timeout: int

    @property
    def mach(self):
        return FlightParams.mach
    
    @property
    def Re(self):
        return FlightParams.reynolds

@dataclass(slots=True)
class Relaxed:
    aseq: str
    panels: int

class MetricWeights:
    min_m: float
    max_ld: float
    max_cl: float
    aoa_at_max_cl: float
    max_drag: float
    thickness: float
    array: np.ndarray
    
    @classmethod
    def load(cls, weights_path: str) -> None:
        """Load from weights.json"""
        with open(weights_path, 'r') as f:
            raw = json.load(f)

        cls.min_m = raw["min_m"]
        cls.max_ld = raw["max_ld"]
        cls.max_cl = raw["max_cl"]
        cls.aoa_at_max_cl = raw["aoa_at_max_cl"]
        cls.max_drag = raw["max_drag"]
        cls.thickness = raw["thickness"]
        cls.array = np.array([cls.min_m,cls.max_ld,cls.max_cl,cls.aoa_at_max_cl,cls.max_drag,cls.thickness])[:6]




class XFOILSolverParams:
    coarse: XFOILParams
    fine: XFOILParams
    relaxed: Relaxed
    max_retries: int

    
    @classmethod
    def load(cls, solver_path: str) -> None:
        """Load from solver.json"""
        with open(solver_path, 'r') as f:
            data = json.load(f)
        
        cls.coarse = XFOILParams(**data['xfoil_coarse'])
        cls.fine = XFOILParams(**data['xfoil_fine'])
        cls.relaxed = Relaxed(**data['relaxed'])
        cls.max_retries = data['max_retries']



def _load_config(
    config_path: str = "./config/config.json",
    flight_path: str = "./config/flight.json",
    solver_path: str = "./config/solver.json",
    weights_path: str = "./config/weights.json"
) -> None:
    """
    Load all configuration from JSON files using individual dataclass loaders.
    Respects dependency ordering: Atmosphere must load before FlightParams.
    
    Args:
        config_path: Path to config.json
        flight_path: Path to flight.json
        solver_path: Path to solver.json
        weights_path: Path to weights.json

    """
    # Load with dependency ordering (Atmosphere before FlightParams)
    Paths.load(config_path)
    Execution.load(config_path)
    NACAranges.load(config_path)
    Atmosphere.load(config_path)
    FlightParams.load(flight_path)
    XFOILSolverParams.load(solver_path)
    MetricWeights.load(weights_path)
    
_load_config()