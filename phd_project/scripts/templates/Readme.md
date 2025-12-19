# Analysis Framework
In general each building / structural model has its own folder which contains the structural geometry in a json file and a structural_model.py file which converts the `.json`-file into an `openseespy` model.

For each type of analysis that will be run on the model there should be a "run_analysis.py" \(e.g. `run_nltha.py`\) and at least one "config_analysis.py" \(e.g. `config_nltha_record_120111.py`) file.

Each "run_analysis.py" file should contain a `run(config_data: str|Path|dict)` function that can be called to start the analysis

The "config_analysis.py" file should contain a variable `config: dict` which contains the parameters that need to be defined in the `run()` function in "run_analysis.py" 

Although not strictly required to use this framework in the general sense, the `standes` package is helpful because it contains some predefined modules for running individual and batches of different types of analyses. To run all the example templates here the `standes` package is required.

# Template files
Current template files are:
- run_nltha.py
- config_nltha.py
- run_snapback.py
- config_snapback.py
- structural_model.py
- run_batch_scripts.py

## structural_model.py
The file structural_model.py *must* contain the following variables
````python
design_json: str  # -> the name of the json file containing the data 
model_init:  Callable # -> a function that is called without arguments and creates the model in opensees. It should return two lists of recorders
````

It is assumed that `design_json.json` is located in the same folder as `structural_model.py

using `functools.partial` is a good way of creating the model_init function

The rest of the variables for model, recorder and damping config are not intended to be used outside of the model and are therefore not explicity required for structural_model.py to work with in the rest of the analysis script framework used in this study

