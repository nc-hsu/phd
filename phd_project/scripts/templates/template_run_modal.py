import argparse
import json
import pickle
import numpy as np
import numpy.typing as npt
from pathlib import Path
import openseespy.opensees as ops
from standes.utils import import_from_path
from standes.analysis.eigen import eigenvector_matrix, nodal_massmatrix

## run a modal analysis:
def run(config_data: str|Path|dict):
    # loading the configuration data from the provided file path
    if isinstance(config_data, str):
        config_data = Path(config_data)
        if not (config_data.exists() and config_data.is_file()):
            raise ValueError(f"Invalid path for config_data: {config_data}")

    if isinstance(config_data, Path):
        # import the configuration data from module path
        config_module = import_from_path(config_data)
        config: dict = config_module.config
    
    elif not isinstance(config_data, dict):
        raise ValueError
    
    else:
        config = config_data

    #print some stuff to screen
    print(f"Modal Analysis --> {config["n_modes"]} modes")

    # initialise the model
    _ = config["model_init"]()

    systemsize = ops.systemSize()

    # do the modal analysis
    ops.wipeAnalysis()
    ops.eigen(config["solver"], config["n_modes"])

    # extract the model results and save
    modal_props = ops.modalProperties("-return")

    # extract the mode shapes and masses
    M = nodal_massmatrix(systemsize)
    PHI = eigenvector_matrix(config["n_modes"], systemsize)
    
    # obtain the "r" maxtrix (columns are the influence vectors for each dof) for the each direction
    
    def _influence_vector(dof: int, system_size: int) -> npt.NDArray:
        # 1 if the dof interest at the node has mass, 0 otherwise
        # only works for nodal masses
        r = np.zeros(system_size)
        for node in ops.getNodeTags():
            for i, (dof_mass, eq_number) in enumerate(zip(ops.nodeMass(node), ops.nodeDOFs(node))):
                if i+1 == dof and dof_mass > 0:  
                    # we only want the influence vector for the specified dof if it has mass
                    # the dofs are alwasy 1, 2, 3, 4, 5, 6 (x, y, z, rx, ry, rz)
                    r[eq_number] = 1
        return r
    
    def _influence_matrix(system_size: int) -> npt.NDArray:
        return np.column_stack([_influence_vector(dof, system_size) for dof in range(1,7)])

    R = _influence_matrix(systemsize)

    # node tags, dofs, and equation numbers
    node_info = []
    for node in ops.getNodeTags():
        for i, dof in enumerate(ops.nodeDOFs(node)):
            node_info.append((node, i+1, dof))  # (node_tag, dof, eq_number)


    # save results
    output_folder: Path = config["result_dst"]
    output_folder.mkdir(parents=True, exist_ok=True)
    
    with open(config["result_dst"] / f"modal_properties.json", "w") as file:
        json.dump(modal_props, file, indent=4)

    np.savetxt(config["result_dst"] / f"mode_shapes.csv", PHI, delimiter=",")
    np.savetxt(config["result_dst"] / f"mass_matrix.csv", M, delimiter=",")
    np.savetxt(config["result_dst"] / f"influence_matrix.csv", R, delimiter=",")

    with open(config["result_dst"] / f"node_info.json", "w") as file:
        json.dump(node_info, file, indent=4)

    

if __name__ == "__main__":
    # launch from the terminal, e.g.
    #   python run_modal.py config_modal.py
    parser = argparse.ArgumentParser(description="Run a modal (eigenvalue) analysis.")
    parser.add_argument("config", nargs="?", default="config_modal.py",
                        help="config file (relative to this folder, or an absolute path)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path

    run(config_path)
