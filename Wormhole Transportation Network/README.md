# Wormhole Transportation Network Solver

A genetic algorithm-based solver for the ESA SpOC2 Wormhole Transportation Network challenge.

## Overview

This solver implements a genetic algorithm to find optimal paths for 12 ships through a wormhole network, minimizing maximum variance while satisfying temporal synchronization constraints.

## Installation

### Requirements

- Python 3.10.8 or higher
- Required packages (see `requirements.txt`)

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure data files are in place:
- `data/database.npz` - Network data and problem parameters
- `data/example.npy` - Example solution (optional)

## Usage

### Basic Usage

Run the solver with default settings:
```bash
python solver/main.py
```

### Command-Line Options

```bash
python solver/main.py [OPTIONS]
```

**Data Options:**
- `--data PATH` - Path to database.npz file (default: `./data/database.npz`)

**Configuration Options:**
- `--config PATH` - Path to configuration JSON file
- `--population-size N` - Population size (overrides config)
- `--max-generations N` - Maximum generations (overrides config)
- `--crossover-rate RATE` - Crossover rate (overrides config)
- `--mutation-rate RATE` - Mutation rate (overrides config)

**Output Options:**
- `--output-dir DIR` - Output directory (default: `./results`)
- `--submission-file PATH` - Path to save submission JSON
- `--save-solution` - Save solution chromosome to .npy file

**Other Options:**
- `--log-level LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)
- `--validate-only` - Only validate data and configuration

### Example Commands

Run with custom parameters:
```bash
python solver/main.py \
    --population-size 200 \
    --max-generations 500 \
    --crossover-rate 0.8 \
    --mutation-rate 0.2 \
    --output-dir ./my_results
```

Validate data only:
```bash
python solver/main.py --validate-only
```

## Configuration

Configuration can be provided via JSON file. Example:

```json
{
  "population_size": 100,
  "max_generations": 1000,
  "crossover_rate": 0.8,
  "mutation_rate": 0.2,
  "selection_type": "tournament",
  "tournament_size": 3,
  "elitism_rate": 0.1,
  "replacement_strategy": "elitism",
  "initialization_type": "mixed",
  "heuristic_ratio": 0.3
}
```

See `solver/config/settings.py` for all available parameters.

## Output

The solver generates:

1. **Submission JSON** - Formatted for Optimise platform submission
2. **Solution Summary** - Complete results in `results/run_TIMESTAMP/`:
   - `solution.npy` - Best chromosome
   - `statistics.json` - Final statistics
   - `history.json` - Evolution history
   - `config.json` - Configuration used
   - `summary.txt` - Human-readable summary

## Algorithm

The solver uses a genetic algorithm with:

- **Population Initialization**: Mixed strategy (random + heuristic)
- **Selection**: Tournament selection
- **Crossover**: Single-point crossover at ship boundaries
- **Mutation**: Node replacement, path extension/truncation
- **Repair**: Constraint repair operators for origin, path, and window
- **Replacement**: Elitism strategy

## Project Structure

```
solver/
├── main.py              # Entry point
├── config/              # Configuration management
├── data/                # Data loading and validation
├── model/               # Network, chromosome, path models
├── evaluation/           # Fitness and constraint evaluation
├── genetic/              # GA components (population, operators, algorithm)
├── repair/               # Constraint repair operators
├── heuristics/           # Initialization and pathfinding
├── output/               # Solution formatting and saving
└── tests/                # Unit tests
```

## Development

### Running Tests

```bash
pytest solver/tests/
```

### Code Style

The code follows PEP 8 style guidelines with type hints.

## License

This project is part of the ESA SpOC2 competition.

## References

- [SpOC2 Challenge](https://optimize.esa.int/challenge/spoc-2-wormhole-transportation-network/About)
- [Problem Description](wormhole_description.md)
- [UDP Implementation](wormhole_udp.py)

