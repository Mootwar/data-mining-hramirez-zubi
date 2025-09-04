# Simple Data gather for detected earthquakes 

## Requirements
```
- Python3
- Python3-venv
```

## Quick set up
```bash
# Set up environment
bash setup.sh

# Run with default parameters
bash run.sh
```

Then a folder called `data` is created with the requested information.

For help with an in-depth description on how to tailor information, run:
```bash
bash run.sh --help
```

## Examples
```bash
# Get earthquakes from the past year with magnitude 5+
bash run.sh --start 2024-09-04 --end 2025-09-04 --mag 5.0

# Get earthquakes near San Francisco
bash run.sh --loc sf

# Get earthquakes with custom location
bash run.sh --lat 37.7749 --lon -122.4194 --radius 100
```