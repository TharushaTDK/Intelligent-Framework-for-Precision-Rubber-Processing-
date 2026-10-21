"""DRC from Metrolac reading and temperature, using the temperature corrected ready-reckoner chart.

Source: Kudaligama et al. (2014), Temperature corrected ready-reckoner chart for determination of
dry rubber content in Hevea latex using the Metrolac. Journal of the Rubber Research Institute of
Sri Lanka 94, 1-8 (Table 1, latex diluted 1:2 with water).
"""
import numpy as np

TEMPERATURES = np.arange(15, 36)
METROLAC = np.arange(50, 151, 10)
# Table 1: DRC as a fraction, one row per Metrolac reading, one column per temperature (15-35 C)
CHART = np.array([[float(v) for v in row.split()] for row in [
    "0.34 0.33 0.32 0.31 0.30 0.29 0.28 0.27 0.26 0.25 0.24 0.23 0.22 0.21 0.20 0.19 0.18 0.17 0.17 0.16 0.15",
    "0.36 0.35 0.34 0.33 0.32 0.31 0.30 0.29 0.28 0.27 0.26 0.25 0.24 0.23 0.22 0.21 0.20 0.19 0.19 0.18 0.17",
    "0.38 0.37 0.36 0.35 0.34 0.33 0.32 0.31 0.30 0.29 0.28 0.27 0.26 0.25 0.24 0.23 0.22 0.21 0.21 0.20 0.19",
    "0.40 0.39 0.38 0.37 0.36 0.35 0.34 0.33 0.32 0.31 0.30 0.29 0.28 0.27 0.26 0.25 0.24 0.23 0.23 0.22 0.21",
    "0.42 0.41 0.40 0.39 0.38 0.37 0.36 0.35 0.34 0.33 0.32 0.31 0.30 0.29 0.28 0.27 0.26 0.25 0.24 0.24 0.23",
    "0.44 0.43 0.42 0.41 0.40 0.39 0.38 0.37 0.36 0.35 0.34 0.33 0.32 0.31 0.30 0.29 0.28 0.27 0.26 0.25 0.24",
    "0.46 0.45 0.44 0.43 0.42 0.41 0.40 0.39 0.38 0.37 0.36 0.35 0.34 0.33 0.32 0.31 0.30 0.29 0.28 0.27 0.26",
    "0.48 0.47 0.46 0.45 0.44 0.43 0.42 0.41 0.40 0.39 0.38 0.37 0.36 0.35 0.34 0.33 0.32 0.31 0.30 0.29 0.28",
    "0.50 0.49 0.48 0.47 0.46 0.45 0.44 0.43 0.42 0.41 0.40 0.39 0.38 0.37 0.36 0.35 0.34 0.33 0.32 0.31 0.30",
    "0.52 0.51 0.50 0.49 0.48 0.47 0.46 0.45 0.44 0.43 0.42 0.41 0.40 0.39 0.38 0.37 0.36 0.35 0.34 0.33 0.32",
    "0.54 0.53 0.52 0.51 0.50 0.49 0.48 0.47 0.46 0.45 0.44 0.43 0.42 0.41 0.40 0.39 0.38 0.37 0.36 0.35 0.34",
]])


def drc_percent(metrolac_reading: float, temperature_c: float) -> float:
    """DRC % rounded to 1 decimal.

    The temperature is interpolated between chart columns. Along the Metrolac axis the chart is a
    straight line (+0.02 per 10 units), so readings above 150 follow the same line.
    """
    column = np.array([np.interp(temperature_c, TEMPERATURES, row) for row in CHART])
    slope, intercept = np.polyfit(METROLAC, column, 1)
    return round(100 * (slope * metrolac_reading + intercept), 1)
