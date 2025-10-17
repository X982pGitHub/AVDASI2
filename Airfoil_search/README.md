# NACA sweep tool
Performs a full exhaustive sweep of all standard NACA 4 to 5-digit airfoils
using XFOIL’s internal geometry generation and aerodynamic analysis.

## Features:
  - Enumerates NACA combinations.
  - Runs XFOIL in parallel via subprocess.
    - Please download XFOIL6.99 from https://web.mit.edu/drela/Public/web/xfoil and add the directory to the executable to your device's PATH
  - Handles timeouts and retries with relaxed parameters.
  - Logs all outputs and results for later analysis.
  - Computes and ranked based on selection metric. (from MCDA matrix weighting)
      - 3D airfoil calculation. (Adapted from Natalie Segamwenge's)
      - Selection score should be calculated again [TODO] because parameters on each weights were not normalised
  - Saves to CSV.
