import json
import pickle
import numpy as np
from pathlib import Path

from standes.model import Model
from cbf_design_ULS_GQWEI import design_ULS_GQWIE
from standes.model import to_json

def  _gusset_end_i(beam: dict) -> bool:
    if max(beam["gusset_offset_i"], beam["column_offset_i"]) == beam["gusset_offset_i"]:
        return True
    return False

def  _gusset_end_j(beam: dict) -> bool:
    if max(beam["gusset_offset_j"], beam["column_offset_j"]) == beam["gusset_offset_j"]:
        return True
    return False

ROOT = Path("D:/case_studies_set1_dc2")

overwrite_log = False
S_alpha_RPs = [0.41]      # [g]  
storey_height = 3500        # mm
n_storeys = [3]
ductility_class = 2

g_floor = 0.0075            # N/mm²
g_roof = 0.0075
q_floor = 0.003             # N/mm²
q_roof = 0
q_floor_type = "QB"
q_roof_type = "QH"

# some file paths
template_filepath = ROOT / "template_design_parameters.json"
splice_template_filepath = ROOT / "template_beam_splices.json"
log_filepath = ROOT / "design_log.txt"

if overwrite_log:
    # create new log file before starting
    open(log_filepath, 'w').close()

ii = 1
n_buildings = len(S_alpha_RPs) * len(n_storeys)
for S_alpha_RP in S_alpha_RPs:
    for n_st in n_storeys:
        
        g_floors = list(g_floor * np.ones(n_st))
        g_floors[-1] = g_roof
        q_floors = list(q_floor * np.ones(n_st))
        q_floors[-1] = q_roof                            # no live load on roof
        q_floor_types = [q_floor_type for _ in range(n_st)]
        q_floor_types[-1] = q_roof_type 

        # basic input and output file names without extensions
        name = f"{n_st}s_cbf_dc{ductility_class}_{int(S_alpha_RP * 100)}"
        in_name_root = f"{name}_in"
        out_name_root = f"{name}_out"

        # set the folders
        building_folder = ROOT / f"{name}"
        building_folder.mkdir(parents=True, exist_ok=True)
        
        # write the input file
        with open(template_filepath, "r") as file:
            design_in_dict = json.load(file)

        # update the input parameters specific to this structure:
        design_in_dict["design_inputs"]["geometry"]["heights"] = int(n_st) * [storey_height]
        design_in_dict["design_inputs"]["load_parameters"]["eq_parameters"]["S_alpha_RP"] = S_alpha_RP
        design_in_dict["design_inputs"]["load_parameters"]["eq_parameters"]["ductility_class"] = ductility_class
        design_in_dict["design_inputs"]["load_parameters"]["g_floors"] = g_floors
        design_in_dict["design_inputs"]["load_parameters"]["q_floors"] = q_floors
        design_in_dict["design_inputs"]["load_parameters"]["q_floor_types"] = q_floor_types
        
        # save the input json file
        design_in_filepath = building_folder / f"{in_name_root}.json"
        with open(design_in_filepath, "w") as fid:
            json.dump(design_in_dict, fid, indent=4)

        # design the structure
        model = Model(design_in_filepath)
        model.set_ductility_class(ductility_class)
        model.set_behaviour_factors()
        model.set_allowable_column_sections(["HEB", "HEM"])
        model.set_allowable_beam_sections(["IPE", "HEA"])
        model.set_allowable_brace_sections(["CHS"])

        print(f"\nSTRUCTURE: {ii}/{n_buildings} --> {name}")

        try:
            model, checks, scs = design_ULS_GQWIE(model, max_iters=15)
        except ValueError as e:
            if "No suitable section available" in str(e):
                # log the failed design
                with open(log_filepath, "a") as file:
                    file.write(f"FAILED:: {name} -- no suitable sections\n")
                    print(f"FAILED:: {name} -- no suitable sections")

                break
            else:
                raise e
                # with open(log_filepath, "a") as file:
                #     file.write(f"FAILED:: {name} -- some other error")
                #     print(f"FAILED:: {name} -- some other error")

        with open(log_filepath, "a") as file:
            if scs:
                file.write(f"success:: {name}\n")
            else:
                file.write(f"FAILED:: {name} -- verifications failed --> check design\n")

        # output json file
        design_out_filepath = building_folder / f"{out_name_root}.json"
        to_json(model, design_out_filepath)

        # pickle model incase we want to look at it later
        model_save_filepath = building_folder / f"{out_name_root}.pickle"
        with open(model_save_filepath, "wb") as file:
            pickle.dump(model, file)

        # modify the out put json file by adding manually data for beamsplices and beamcolumnhinges
        with open(design_out_filepath, "r") as file:
            json_data = json.load(file)

        structure = json_data["structure"]
        beams = structure["beams"]
        columns = structure["columns"]

        with open(splice_template_filepath) as file:
            splice_data = json.load(file)

        for beam in beams:
            beam.pop("hinge_i", None)
            beam.pop("hinge_j", None)

            if _gusset_end_i(beam):     # then there is a beam splice
                key = "splice"
                beam["column_connection_i"] = None
                beam["beam_splice_i"] = splice_data[key]
            else:   # beam column connection
                beam["column_connection_i"] = splice_data[key]
                beam["beam_splice_i"] = None

            if _gusset_end_j(beam):     # then there is a beam splice
                beam["column_connection_j"] = None
                beam["beam_splice_j"] = splice_data[key]
            else:   # beam column connection
                beam["column_connection_j"] = splice_data[key]
                beam["beam_splice_j"] = None

        # save the data
        with open(design_out_filepath, "w") as file:
            json.dump(json_data, file, indent=4)

        ii+=1




            


