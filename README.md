# Gaussian processes on phylogenetic trees

### Instalation

Minimal installation is performed by typing
```
pip install -e .
```
in your terminal inside the `src` directory of the cloned repo.

To run tests, install the dependencies with
```
pip install -e ".[tests]"
```

### Usage

Use the following command to run the tests:
```
pytest .
```
Tests can be found in the `src/tests` directory.

To run benchmark, use the following command:
```
python src/scripts/benchmark.py --save_filepath=... --config_path=...
```
Note that `save_filepath` should be a `json` file. Examples of configs can be found in the `benchmarks` directory.

For example, the following command will run a simple benchmark and save its results to `benchmark_results.json`:
```
python src/scripts/benchmark.py --save_filepath="benchmark_results.json" --config_path="benchmarks/simple_config.json"
```