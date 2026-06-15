from structural_model import model_init, damping_config, damping_model # type: ignore

""" 
The functions that get injected into the time history analysis are defined here.
valid keys for injection_functions are "pre_nltha","pre_analyse" and "post_analyse", "post_nltha",
which determines when the functions get called. 
The values for each key should be a dictionary with the format {"function_name": function}.
"""

def update_damping_model():
    # update the damping model in opensees using the provided damping_config and damping_model function
    damping_model(damping_config)

injection_functions = {
    "post_analyse": {
        "update_damping_model": update_damping_model
        }
}