"""
The functions that get injected into the time history analysis are defined here.
valid keys for injection_functions are "pre_nltha","pre_analyse" and "post_analyse", "post_nltha",
which determines when the functions get called.
The values for each key should be a dictionary with the format {"function_name": function}.

The analysis config imports the structural model (by `model_file_name` /
`init_function`) and passes the model module into `make_injection_functions`, so this
file does not import `structural_model` itself and is model-name agnostic. Any extra
parameters are threaded through the config's `injection_function_params` dict (default {})
as keyword arguments.
"""

def make_injection_functions(model, **injection_function_params):
    damping_config = model.damping_config
    damping_model = model.damping_model

    def update_damping_model():
        # update the damping model in opensees using the model's damping_config and damping_model
        damping_model(damping_config)

    return {
        "post_analyse": {
            "update_damping_model": update_damping_model
            }
    }
