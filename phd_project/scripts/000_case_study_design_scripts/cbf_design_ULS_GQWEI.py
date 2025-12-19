""" 
Function for ULS design of concentrically braced steel frame structures

# TODO COMPLETE THIS

 - fixed base shear value


---------------
ASSUMPTIONS:
---------------


"""
from typing import List
import numpy as np
from standes.model import Model, to_dict
from standes.design.design import harmonise_design, harmonise_storey_braces 
from standes.design.columns import design_columns
from standes.design.beams import design_beams
from standes.design.braces import (design_braces, 
                           design_braces_drift_limitation,
                           design_braces_uniform_buckling_overstrength,
                           design_braces_regular_storey_stiffness)
from standes.design.gussets import design_gussets
from standes.design.loading import (get_storey_masses,
                            define_node_masses, 
                            calculate_seismic_base_shear,
                            define_seismic_loads,
                            define_wind_loads,
                            define_imperfection_loads,
                            define_gravity_loads,
                            define_snow_loads,
                            snow_pressure_single_pitch,
                            storey_forces_from_baseshear)
from standes.spectra import (calculate_spectrum_parameters,
                     calculate_R_q)
from standes.design.analyse import (static_analysis, 
                            modal_analysis)
from standes.opsmodels.lcbf_3d_01 import linear_cbf_3d_01
from standes.verification.structure import (verify_seismic_structure, 
                                            verify_nonseismic_structure,
                                            seismic_second_order_amplification,
                                            storey_drifts, VerificationSeismicCBF, VerificationNonseismicCBF)
from standes.get import behaviour_factors


GAMMA_G_ULS = 1.35
GAMMA_G_E = 1.0
GAMMA_Q_ULS = 1.5
GAMMA_Q_E = 1.0
GAMMA_I = 1.0
GAMMA_E = 1.0

PSI_G = 1.0
PSI_Q = 1.0         # when dominant live load case
PSI_0_Q = 0.7
PSI_2_Q = 0.3       # extraordinary loading (e.g. earthquake)
PSI_W = 1.0         # when dominant live load case
PSI_S = 1.0
PSI_0_W = 0.6
PSI_0_S = 0.5
PSI_I = 1.0
PSI_E = 1.0

DELTA = 1.0


def design_ULS_GQWIE(model: Model,
                           max_iters: int):

    # set Load case parameters
    tags_GQWSI = ["G", "Q", "W", "S", "I"]
    tags_GQEI = ["G", "Q", "E", "I"]
    
    gammas_GQWI = [GAMMA_G_ULS, GAMMA_Q_ULS, GAMMA_Q_ULS, GAMMA_Q_ULS, GAMMA_I]   # gamma values for ULS load cases
    gammas_GQEI = [GAMMA_G_E, GAMMA_Q_E, GAMMA_E, GAMMA_I]  # gamma values for EQ load case
    
    psis_GQWI_1 = [PSI_G, PSI_Q, PSI_0_W, PSI_0_S, PSI_I]    # psi values for first load combi - live
    psis_GQWI_2 = [PSI_G, PSI_0_Q, PSI_W, PSI_0_S, PSI_I]    # psi values for second load combi - wind
    psis_GQWI_3 = [PSI_G, PSI_0_Q, PSI_0_W, PSI_S, PSI_I]    # psi values for third load combi - snow
    psis_GQEI = [PSI_G, PSI_2_Q, PSI_E, GAMMA_I]    # psi values for EQ load case
    
    lc1 = (tags_GQWSI, gammas_GQWI, psis_GQWI_1, "GQWSI_LC1")  # load combination 1
    lc2 = (tags_GQWSI, gammas_GQWI, psis_GQWI_2, "GQWSI_LC2")  # load combination 2
    lc3 = (tags_GQWSI, gammas_GQWI, psis_GQWI_3, "GQWSI_LC3")  # load combination 2
    lc4 = (tags_GQEI, gammas_GQEI, psis_GQEI, "GQEI+")   # EQ load combination
    lc5 = (tags_GQEI, gammas_GQEI, psis_GQEI, "GQEI-")   # EQ load combination

    non_seismic_lcs = [lc1, lc2, lc3]
    all_lcs = [lc1, lc2, lc3, lc4, lc5]
    seismic_index = [3,4]
    
    # define some loads
    g_floors = model.load_parameters["g_floors"]
    q_floors = model.load_parameters["q_floors"]
    s_floors = [0]*model.g.nst
    s_floors[-1] = roof_snow_pressure(model)

    GQ_characteristic_floor_loads = [g_floors, q_floors]
    GQS_characteristic_floor_loads = [g_floors, q_floors, s_floors]

    # set dummy elements to allow model to be built
    model.set_dummy_elements(columns="IPE80", beams="IPE80",
                             braces="CHS33.7x2.5", gussets="Rect10x10")


    ######################################################################################
    print("\n[START] Gravity and Wind Design")
    
    # do the nonseismic design for the two ULS load cases with wind
    # build geometry
     
    
    
    build_model(model, seismic=False)
    for ii, lc in enumerate([lc1, lc2, lc3]):
        print(f"{'':3}--- Load combination {ii+1}")
        lc_tags, gammas, psis, verification_tag = lc
        
        non_seismic_design(model, lc_tags, gammas, psis, 
                           verification_tag, max_iters, GQS_characteristic_floor_loads)
  
    print("\n[END] Gravity and Wind Design")

    ######################################################################################

    # seismic design
    # sort out load case tags and factors etc for seismic design
    slc_tags, s_gammas, s_psis, verification_tag = lc4

    print("\n[CHECK] Seismic action index...")
    # Check the seismic action index to see if seismic design is required
    S_alpha_RP = model.load_parameters["eq_parameters"]["S_alpha_RP"]
    site_cat = model.load_parameters["eq_parameters"]["site_category"]
    model.set_spectrum_parameters(calculate_spectrum_parameters(S_alpha_RP, site_cat))

    if model.spectrum_parameters["seismic_action_class"] == "very low":
        # the non seismic design is sufficient -- no seismic design required
        # perform the final non seismic checks and exit the design
        
        print(("Seismic action class is 'very low' -" 
               "No seismic design required."))
        
        # save a few values relevant for seismic analysis

        period, seismic_mass, storey_masses, _ = fundamental_period(model, 
                                                                    [s_gammas[0], s_gammas[1]], 
                                                                    [s_psis[0], s_psis[1]],
                                                                    GQ_characteristic_floor_loads, 
                                                                    return_masses=True)
        seismic_design_outputs = {"seismic_mass": seismic_mass,
                                  "storey_masses": [s.item() for s in storey_masses],
                                  "ductility_class": model.ductility_class,
                                  "design_period": period,
                                 }
                
        model.set_seismic_design_outputs(seismic_design_outputs)
        model, lc_checks, scs = final_verifications(model, non_seismic_lcs, [], GQ_characteristic_floor_loads, GQS_characteristic_floor_loads)
        model.set_final_checks(lc_checks)
        model.add_design_parameter("design_success", scs)
        model.add_design_parameter("design_success", scs)
        return model, lc_checks, scs
    
    # if here: seismic design is required:
    period = fundamental_period(model, [s_gammas[0], s_gammas[1]], [s_psis[0], s_psis[1]],
                                GQ_characteristic_floor_loads)
    q_design, equivalent_base_shear = determine_q_design(model, slc_tags, s_gammas, s_psis, verification_tag,
                                                         period, "+x")   # checks elastic utilisation
    model.set_q_factor(q_design)

    print(f"\n[START] Seismic Design")
    print(f"  ** Design DC -> DC{model.ductility_class}")

    for ii, (lc, force_direction) in enumerate(zip([lc4, lc5], ["+x", "-x"])):
        
        lc_tags, gammas, psis, verification_tag = lc

        print(f"\n{'':3}--- Load combination {verification_tag}\n")
        seismic_design(model, slc_tags, s_gammas, s_psis, verification_tag,
                       force_direction, max_iters, GQ_characteristic_floor_loads,
                       equivalent_base_shear)
    
    print(f"\n[END] Seismic Design")

    ######################################################################################
    print("\n[CHECK] Final verification for all loadcases...")
    # perform final verifications
    period = fundamental_period(model, [s_gammas[0], s_gammas[1]], [s_psis[0], s_psis[1]],
                                GQ_characteristic_floor_loads)
    model, lc_checks, scs = final_verifications(model, all_lcs, seismic_index, GQ_characteristic_floor_loads, GQS_characteristic_floor_loads)
    
    model.set_final_checks(lc_checks)
    model.add_design_parameter("design_success", scs)
    return model, lc_checks, scs


def build_model_and_analyse(model: Model, lc_tags: List[str], 
                            gammas: List[float], psis: List[float],
                            seismic: bool,
                            update_wind_loads: bool=False):
    """ 
    # !!! the values in the lc_tags, gammas, and psis should follow the order:
    # !!! G, Q, W/E, [S], I
    # !!! the last values in lc_tags, gammas, psis should be reserved for 
    # !!! Imperfections
    """

    # build model geometry
    model_data = to_dict(model, seismic)

    if model.g.tension_only:
        linear_cbf_3d_01(model_data_dict=model_data, model_neg_braces=False)
    else:
        linear_cbf_3d_01(model_data_dict=model_data, model_neg_braces=True)

    gamma_G = gammas[0]
    gamma_Q = gammas[1]
    gamma_S = gammas[3]
    psi_Q = psis[1]
    psi_S = psis[3]

    if update_wind_loads:
        snow_pressure = snow_pressure_single_pitch(model.load_parameters["snow_parameters"]["s_k"],
                                                   model.load_parameters["snow_parameters"]["C_e"],
                                                   model.load_parameters["snow_parameters"]["C_t"],
                                                   model.load_parameters["snow_parameters"]["alpha_1"]) 
        define_wind_loads(model, gamma_G, gamma_Q, gamma_S, psi_Q, psi_S, snow_pressure)
    
    # perform static analysis on the new model
    static_analysis(model, lc_tags[:-1]) # analysis without imperfections
    design_actions = model.lc.combine_actions(lc_tags[:-1], gammas[:-1], 
                                              psis[:-1])
    
    # recalculate forces due to imperfections
    define_imperfection_loads(model, design_actions)
    static_analysis(model, lc_tags) # analysis with imperfections
    
    return


def build_model(model: Model, seismic: bool):
    # build model geometry
    model_data = to_dict(model, seismic)

    if model.g.tension_only:
        linear_cbf_3d_01(model_data_dict=model_data, model_neg_braces=False)
    else:
        linear_cbf_3d_01(model_data_dict=model_data, model_neg_braces=True)


def brace_utilisations(seismic_check: VerificationSeismicCBF):
    # returns the utilisation ratios for the braces in the models
    brace_checks: dict = seismic_check.checks[0].check_data  # brace checks
    etas = []

    for tag, veri_data in brace_checks.items():
        if not veri_data.dummy:
            sec_check = [chk for chk in veri_data.checks 
                           if ("section_N" in chk.check_tag) or ("buckling" in chk.check_tag)]
            
            if len(sec_check) == 0:
                raise ValueError(f"No axial capacity check for brace: {tag}")
            elif len(sec_check) > 1:
                raise ValueError(f"More than one 'section_N' check for brace: {tag}")
            else:
                etas.append(sec_check[0].eta)

    return etas


def non_seismic_design(model: Model, lc_tags, gammas, psis, verification_tag, max_iters,
                       GQS_characteristic_floor_loads):
    
    # calculate loads on the structure and design the sections
    define_gravity_loads(model)
    define_snow_loads(model)

    # build model geometry and analyse
    build_model_and_analyse(model, lc_tags, gammas, psis,
                            seismic=False, 
                            update_wind_loads=True)
    
    for ii in range(max_iters):  
        # design the beams, columns and braces for ULS gravity combination           
        design_columns(model, lc_tags, gammas, psis, verification_tag, 
                        seismic=False)
        design_beams(model, lc_tags, gammas, psis, verification_tag, 
                        seismic=False)
        design_braces(model, lc_tags, gammas, psis, verification_tag, 
                        seismic=False, selection_flag="lightest") 
        design_gussets(model, lc_tags, gammas, psis, verification_tag,
                       seismic=False)

        # harmonise the designs using simple rules
        harmonise_design(model)
        
        # compare how many changes need to be made to the design
        print(f"{'':5}-Model update: ii = {ii}")
        update_model(model, indent=6)
        
        # build model geometry and analyse
        snow_pressure = roof_snow_pressure(model)
        _, wind_base_shear, F_wind, F_lee = define_wind_loads(model, gammas[0], gammas[1], gammas[3], psis[1], psis[3], snow_pressure)
        build_model_and_analyse(model, lc_tags, gammas, psis, seismic=False)

        nonseismic_check = verify_nonseismic_structure(model, lc_tags, gammas, 
                                                       psis, verification_tag)
        
        if nonseismic_check.passes:
            set_nonseismic_design_output(model, lc_tags, gammas, psis, verification_tag,
                                         F_wind, F_lee, wind_base_shear, GQS_characteristic_floor_loads)
            print(f"Design successful\n")
            break # move on to the next load case
    
    if ii == max_iters-1:  
        print("Gravity Design NOT successful! --- Check model")

    
    
    return


def set_nonseismic_design_output(model, lc_tags, gammas, psis, verification_tag,
                                 F_wind, F_lee, wind_base_shear, GQS_characteristic_floor_loads) -> dict:
    period, _, storey_masses, _ = fundamental_period(model, 
                                                     [gammas[0], gammas[1], gammas[3]], 
                                                     [psis[0], psis[1], psis[3]],
                                                     GQS_characteristic_floor_loads, return_masses=True)
    x_drifts, *_ = storey_drifts(model, lc_tags, gammas, psis)
    output = {"storey_wind_forces_windward": [s.item() for s in F_wind],
              "storey_wind_forces_leeward": [s.item() for s in F_lee],
              "uls_wind_base_shear": wind_base_shear*gammas[2]*psis[2],
              "uls_storey_wind_forces_windward": [s.item()*gammas[2]*psis[2] for s in F_wind],
              "uls_storey_wind_forces_leeward": [s.item()*gammas[2]*psis[2] for s in F_lee],
              "storey_masses": [s.item() for s in storey_masses],
              "period": period,
              "drifts": [s.item() for s in x_drifts]}
    model.set_nonseismic_design_outputs(output, key=verification_tag)
    return output


def roof_snow_pressure(model) -> float:
    snow_pressure = snow_pressure_single_pitch(model.load_parameters["snow_parameters"]["s_k"],
                                               model.load_parameters["snow_parameters"]["C_e"],
                                               model.load_parameters["snow_parameters"]["C_t"],
                                               model.load_parameters["snow_parameters"]["alpha_1"])
    return snow_pressure


def seismic_design(model: Model, lc_tags, gammas, psis, verification_tag, 
                   force_direction, max_iters, GQ_characteristic_floor_loads, 
                   equivalent_minimum_base_shear):
    
    # sorting out variables
    gamma_G = gammas[0]
    gamma_Q = gammas[1]
    psi_Q = psis[1]

    # check required q factor
    period = fundamental_period(model, [gammas[0], gammas[1]], [psis[0], psis[1]],
                                GQ_characteristic_floor_loads)

    # iterate to find a suitable design and the period doesn't change
    periods = [period]

    if model.q_factor < model.q_max:
        update_q_factors(model, model.q_factor)

    # store the original q_factor incase irregularity changes it during design
    # q_start = model.q_factor
    # q_D_start = model.q_D
    # q_mod_flag = False

    # this is required to make it easier to achieve the DC3 requirement of uniform overstrength
    if model.ductility_class == 1 or model.ductility_class == 2:
        brace_selection = "lightest"
    else:
        brace_selection = "highest_utilisation"

    only_braces_change_counter = 0  # increments each time there is a design iteration where only the braces change
    max_iters_only_changing_braces = 3
    # when a maximum value is reached then there is most likely a "loop" in the design algorithm where the new braces change demands
    # causing another brace to be selected. when the maximum value of the counter is exceeded then the algorithm is tweaked slightly
    # to try and find a converging design. 
    
    for ii in range(max_iters):
        period = periods[-1]

        if ii > 0: # and q_start < model.q_max and model.q_factor < model.q_max: # the wind loads were originally larger than the elastic seismic loads
            # do a check on the base shear to see if the q factor needs to be updated
            R_q = calculate_R_q(period, 
                                model.spectrum_parameters["T_A"], 
                                model.spectrum_parameters["T_B"], 
                                1, 1, 1)        # elastic baseshear
            
            elastic_base_shear, S_r, lambda_, = calculate_seismic_base_shear(model, period, gamma_G, gamma_Q, 
                                                                             psi_Q, R_q, return_Sr=True)
            if elastic_base_shear >= equivalent_minimum_base_shear:     
                new_q = min(elastic_base_shear / equivalent_minimum_base_shear, model.q_max)     # reduce base shear down to as close to the equivalent minimum
                update_q_factors(model, new_q)
                if not seismic_check.is_regular:
                    new_q = update_q_factor_for_irregularity(model, new_q)
                model.set_q_factor(new_q)
                update_q_factors(model, new_q)

            else:
                new_q = 1
                update_q_factors(model, new_q)
                model.set_q_factor(new_q)
            
        # define seismic loads on the structure
        R_q = calculate_R_q(period, 
                            model.spectrum_parameters["T_A"], 
                            model.spectrum_parameters["T_B"], 
                            model.q_factor, model.q_R, model.q_S)
        
        base_shear, S_r, lambda_, = calculate_seismic_base_shear(model, period, gamma_G, gamma_Q, 
                                                                 psi_Q, R_q, return_Sr=True)
        
        define_seismic_loads(model, gamma_G, gamma_Q, psi_Q, base_shear, force_direction)

        # build model and analyse for all seismic load cases:
        build_model_and_analyse(model, lc_tags, gammas, psis, seismic=True)

        # Check if second order amplification is required for seismic load        
        amplification = seismic_second_order_amplification(model, lc_tags,
                                                           gammas, psis, 
                                                           period)        
        
        seismic_check = verify_seismic_structure(model, lc_tags, gammas, psis,
                                                 verification_tag, period)
        
        if not seismic_check.braces_pass:
            if only_braces_change_counter >= max_iters_only_changing_braces:
                if model.ductility_class == 1 or model.ductility_class == 2:
                    brace_selection = "highest_utilisation"
                else:
                    brace_selection = "second_highest_utilisation"
            design_braces(model, lc_tags, gammas, psis, verification_tag, True, 
                        amplification, brace_selection) 
            
        if not seismic_check.columns_pass:
            design_columns(model, lc_tags, gammas, psis, verification_tag, True,
                           amplification)

        if not seismic_check.beams_pass:        
            design_beams(model, lc_tags, gammas, psis, verification_tag, True, 
                         amplification)
        
        if not seismic_check.gussets_pass:
            design_gussets(model, lc_tags, gammas, psis, verification_tag, True)
        
        # harmonise the designs using simple rules
        harmonise_design(model)
        
        # compare how many changes need to be made to the design
        print(f"{'':5}-Model update: ii = {ii}")
        changes = update_model(model, indent=6, keep_new_braces=True)
        if changes[3] == changes[0]:
            only_braces_change_counter += 1

        # second analysis phase if changes were made
        if changes[0] > 0:   # calculate the new period and base shear

            new_period, *_ = fundamental_period(model, [gammas[0], gammas[1]], [psis[0], psis[1]],
                                                GQ_characteristic_floor_loads, return_masses=True)
            # define seismic loads on the structure
            R_q = calculate_R_q(period, 
                                model.spectrum_parameters["T_A"], 
                                model.spectrum_parameters["T_B"], 
                                model.q_factor, model.q_R, model.q_S)
            
            base_shear_2, *_ = calculate_seismic_base_shear(model, period, gamma_G, gamma_Q, 
                                                                    psi_Q, R_q, return_Sr=True)
            define_seismic_loads(model, gamma_G, gamma_Q, psi_Q, base_shear_2, force_direction)

        # build model and analyse:
        build_model_and_analyse(model, lc_tags, gammas, psis, seismic=True)
        
        seismic_check = verify_seismic_structure(model, lc_tags, gammas, psis,
                                                 verification_tag, period)

        # enter the next design phase if only the drift limit is not met
        if seismic_check.only_drifts_not_passing:
            
            print(f"{'':8}-Model update for drift limit")
            
            # modify the braces to improve drift performance
            update_levels = design_braces_drift_limitation(model, lc_tags, gammas, psis, verification_tag, 
                                                           seismic_check.drifts, seismic_check.drift_limits, True,
                                                           amplification)
            
            level_str = ", ".join([str(kk) for kk in update_levels])
            print(f"{'':10}-Updated braces at levels {level_str}.")
            harmonise_design(model, ["harmonise_storey"], ["braces"])
            update_model(model, indent=10, keep_new_braces=True)

        # enter the next design phase if overstrength ratios are not uniform --> DC3 requirement not regularity
        if seismic_check.only_dc3_overstrength_ratios_not_passing:
            print(f"{'':8}-Model update for uniform overstrength values")

            design_braces_uniform_buckling_overstrength(model, lc_tags,
                                                        gammas, psis, verification_tag,
                                                        seismic_check.checks[0].check_data,
                                                        amplification)
            harmonise_storey_braces(model)
            update_braces(model, indent=10, keep_new=True)

        # if the regularity is not passing then log this / print to screen
        if seismic_check.only_regularity_not_passing and model.enforce_regularity:
            if not seismic_check.checks[-3].passes:
                # regularity in storey stiffness not passing
                print(f"{'':8}-Model update for regular storey stiffnesses")

                design_braces_regular_storey_stiffness(model, lc_tags,
                                                    gammas, psis, verification_tag,
                                                    seismic_check.checks[-3].check_data["storey_stiffnesses"],
                                                    amplification)
                harmonise_design(model)
                update_model(model, indent=10, keep_new_braces=True)

            elif not seismic_check.checks[-2].passes:
                # regularity in mass not passing --> input parameters need changing
                raise ValueError("Regularity in mass not passing --> input parameters need changing")

            elif not seismic_check.checks[-1].passes:
                # regularity in overstrength not passing
                raise ValueError("Regularity in overstrength not passing --> input parameters need changing")

        new_period, seismic_mass, storey_masses, _ = fundamental_period(model, [gammas[0], gammas[1]], [psis[0], psis[1]],
                                                                        GQ_characteristic_floor_loads, return_masses=True)
        periods.append(new_period)
        T_diff = abs(period - new_period)
        
        if T_diff <= 0.0001:
            if ((seismic_check.passes or 
                (seismic_check.only_regularity_not_passing and not model.enforce_regularity)) 
               and ii <= max_iters-1):
                
                set_seismic_design_output(model, period, seismic_mass, storey_masses, 
                                          base_shear, S_r, lambda_, seismic_check.is_regular)
                print("Seismic Design successful!")
                return True

        if ii == max_iters-1:  
            print("Seismic Design NOT successful! --- Check model")
            return False


def set_seismic_design_output(model, period, seismic_mass, storey_masses, base_shear, S_r, lambda_, regularity
                              ):
    shape = model.load_parameters["eq_parameters"]["load_shape"]
    storey_forces = storey_forces_from_baseshear(model, base_shear, shape, storey_masses=storey_masses)
    seismic_design_outputs = {"gravity_frame_elastic_baseshear": model.design_params["elastic_baseshear_gravity_frame"],
                            "gravity_design_brace_etas_E": model.design_params["brace_etas_E"],
                            "gravity_design_brace_etas_G": model.design_params["brace_etas_G"],
                            "gravity_design_equivalent_seismic_baseshear": model.design_params["equivalent_base_shear"],
                            "seismic_mass": seismic_mass,
                            "storey_masses": [s.item() for s in storey_masses],
                            "ductility_class": model.ductility_class,
                            "vertical_regularity": regularity,
                            "q_design": model.q_factor,
                            "q_D": model.q_D,
                            "q_R": model.q_R,
                            "q_S": model.q_S,
                            "q_max": model.q_max,
                            "design_period": period,
                            "design_spectral_acceleration": S_r,
                            "design_baseshear": base_shear,
                            "lambda": lambda_,
                            "storey_forces": list(storey_forces),
                            "spectrum_parameters": model.spectrum_parameters,
                            }
                
    model.set_seismic_design_outputs(seismic_design_outputs)


def determine_q_design(model, lc_tags, gammas, psis, verification_tag, T_1,
                       force_direction, update_brace_etas=True):

    gamma_G = gammas[0]
    gamma_Q = gammas[1]
    psi_Q = psis[1]

    _, _, _, q_max = behaviour_factors(model.g.structure, model.ductility_class)

    # define elastic loads on the structure
    base_shear = calculate_seismic_base_shear(model, T_1, gamma_G, gamma_Q, 
                                              psi_Q, R_q=1)
    define_seismic_loads(model, gamma_G, gamma_Q, psi_Q, base_shear, force_direction)

    # build model and analyse for all seismic load cases:
    build_model_and_analyse(model, lc_tags, gammas, psis, seismic=True)
    
    # set the q_factor to 1 for this check - it gets reset once the real q is determined
    model.set_q_factor(1)

    # verify only the gravity load cases
    nonseismic_check = verify_nonseismic_structure(model, lc_tags, [gamma_G, gamma_Q, 0, 1], [1, psi_Q, 0, 1], "GQI")

    # verify only for the seismic case "E" as that is what can be reduced with q
    seismic_check = verify_seismic_structure(model, lc_tags, [0, 0, 1, 0], [0, 0, 1, 0], 
                                             "E", T_1=T_1) 

    brace_etas_E = np.array(brace_utilisations(seismic_check))
    brace_etas_G = np.array(brace_utilisations(nonseismic_check))

    q = max(brace_etas_E / (1 - brace_etas_G))

    equivalent_base_shear = base_shear / q    # equivalent seismic capacity of the gravity frame. (i.e. max_eta == 1.0 for the gravity design

    # add to the design parameters the whole list of etas
    if update_brace_etas:
        model.add_design_parameter("brace_etas_E", list(brace_etas_E))
        model.add_design_parameter("brace_etas_G", list(brace_etas_G))
        model.add_design_parameter("equivalent_base_shear", equivalent_base_shear)
        model.add_design_parameter("elastic_baseshear_gravity_frame", base_shear)
    
    # if the brace utilisation is less than 1.5 then q = 1.5 is used because 
    # we allow low levels of energy dissipation 
    return max(min(q, q_max), 1), equivalent_base_shear # value between 1.5 and q


def verify_load_combinations(model, lcs, seismic_index, GQ_characteristic_floor_loads,
                             GQS_characteristic_floor_loads):
    # perform final verifications 
    lc_verifications = []   
    for ii, lc in enumerate(lcs):
        lc_tags, gammas, psis, v_tag = lc

        if ii in seismic_index: # seismic check
            if "+" in v_tag:
                force_direction = "+x"
            elif "-" in v_tag:
                force_direction = "-x"
            else:
                raise ValueError
            
            period, seismic_mass, storey_masses, _ = fundamental_period(model, [gammas[0], gammas[1]], [psis[0], psis[1]],
                                                                        GQ_characteristic_floor_loads, return_masses=True)
            base_shear, S_r, lambda_ = _define_seismic_loads(model, gammas, psis, period, force_direction)
            build_model_and_analyse(model, lc_tags, gammas, psis, seismic=True)
            lc_verifications.append(verify_seismic_structure(model, lc_tags, gammas, 
                                                             psis, v_tag, period))
            regularity = lc_verifications[-1].is_regular
            set_seismic_design_output(model, period, seismic_mass, storey_masses, base_shear, S_r, lambda_, regularity)

        else:
            snow_pressure = roof_snow_pressure(model)
            _, wind_base_shear, F_wind, F_lee = define_wind_loads(model, gammas[0], gammas[1], gammas[3], psis[1], psis[3], snow_pressure)
            build_model_and_analyse(model, lc_tags, gammas, psis, seismic=False)
            lc_verifications.append(verify_nonseismic_structure(model, lc_tags, gammas, 
                                                                psis, v_tag))
            set_nonseismic_design_output(model, lc_tags, gammas, psis, v_tag,
                                         F_wind, F_lee, wind_base_shear, GQS_characteristic_floor_loads)

    verifications_passing = []
    
    for v in lc_verifications:
        if isinstance(v, VerificationSeismicCBF):
            if not model.enforce_regularity and not v.is_regular and v.only_regularity_not_passing:
                verifications_passing.append(True)
            else:
                verifications_passing.append(v.passes)
        else:
            verifications_passing.append(v.passes)

    return lc_verifications, verifications_passing


def final_verifications(model, lcs, seismic_index, GQ_characteristic_floor_loads, GQS_characteristic_floor_loads):
    lc_checks, checks_passing = verify_load_combinations(model, lcs, 
                                                         seismic_index, GQ_characteristic_floor_loads, GQS_characteristic_floor_loads)
    if all(checks_passing):
        print(f"\nAll load combinations pass!  -- Design successful\n")
        return model, lc_checks, True
    else:
        print(f"\nSome load combinations FAIL!!  -- Do another design round\n")
        failing_lcs = [lc for lc, passes in zip(lc_checks, checks_passing) if not passes]
        for lc in failing_lcs:
            if isinstance(lc, VerificationNonseismicCBF):
                non_seismic_design(model, lc.load_case_tags, lc.gamma_factors, lc.psi_factors, 
                           lc.verification_tag, 10, GQS_characteristic_floor_loads)

        #redo the verifications
        lc_checks, checks_passing = verify_load_combinations(model, lcs, 
                                                            seismic_index, GQ_characteristic_floor_loads, GQS_characteristic_floor_loads)

        if all(checks_passing):
            print(f"\nAll load combinations pass!  -- Design successful\n")
            return model, lc_checks, True
        else:
            print(f"\nSome load combinations FAIL!!  -- Check results\n")
            return model, lc_checks, False


def fundamental_period(model, gammas, psis, floor_loads, return_masses:bool=False):
    storey_masses = get_storey_masses(model, gammas, psis, floor_loads)
    node_masses = define_node_masses(model, storey_masses)
    total_mass = sum(storey_masses)
    modal_props = modal_analysis(model, 1, temp_masses=node_masses)
    T_1 = modal_props["eigenPeriod"][0]

    if return_masses:
        return T_1, total_mass, storey_masses, node_masses
    return T_1


def update_model(model, indent, keep_new_braces=False):

    changes = model.g.update_cross_sections(keep_new_braces)
    print((f"{' '*indent:} {changes[0]} changes made:\n"
           f"{' '*(indent+3)}-> columns: {changes[1]:>3}\n"
           f"{' '*(indent+3)}-> beams:   {changes[2]:>3}\n"
           f"{' '*(indent+3)}-> braces:  {changes[3]:>3}\n"
           f"{' '*(indent+3)}-> gussets: {changes[4]:>3}\n"))
    
    return changes


def update_braces(model, indent, keep_new: bool):
    changes = model.g.update_braces(keep_new=keep_new)
    print((f"{' '*indent:} {changes} changes made:\n"
           f"{' '*(indent+3)}-> braces:  {changes:>3}\n"))

    
    return changes


def _define_seismic_loads(model, gammas: list[float], psis: list[float], period, force_direction: str):
    # define the seismic loads acting on the structure for the final verifications
    gamma_G = gammas[0]
    gamma_Q = gammas[1]
    psi_Q = psis[1]

    R_q = calculate_R_q(period, 
                        model.spectrum_parameters["T_A"], 
                        model.spectrum_parameters["T_B"], 
                        model.q_factor, model.q_R, model.q_S)
    # define elastic loads on the structure
    base_shear, S_r, lambda_ = calculate_seismic_base_shear(model, period, gamma_G, gamma_Q, 
                                              psi_Q, R_q, return_Sr=True)
    define_seismic_loads(model, gamma_G, gamma_Q, psi_Q, base_shear, force_direction)
    return base_shear, S_r, lambda_


def update_q_factors(model, q_factor: float) -> None:
    if q_factor == 1:
        model.set_q_D(1.0)
        model.set_q_R(1.0)
        model.set_q_S(1.0)

    elif q_factor  <= model.q_S:
        model.set_q_D(1.0)
        model.set_q_R(1.0)
        model.set_q_S(q_factor)

    elif q_factor  < (model.q_S_max * model.q_R_max):
        model.set_q_D(1.0)
        model.set_q_R(model.q_factor / model.q_S_max)
        model.set_q_S(model.q_S_max)

    elif q_factor < model.q_max:
        model.set_q_D(model.q_factor/(model.q_R_max * model.q_S_max))
        model.set_q_R(model.q_R_max)
        model.set_q_S(model.q_S_max)


def update_q_factor_for_irregularity(model, q_factor: float) -> float:
    # a structure irregular in elevation according to §4.4.4.2(1) is allowed.
    # the value of q_d and therefore q should be reduced according to §5.3.2(2).
    if q_factor == 1:
        q_D_new = 1
    elif q_factor < model.q_max:
        q_D_eff = q_factor / model.q_R / model.q_S
        q_D_new = max(min(model.q_D_max * 0.8, q_D_eff), 1) # §5.3.2(2)
    else:
        q_D_new = max(model.q_D_max * 0.8, 1)
    
    q_new = max(min(q_factor, q_D_new * model.q_R * model.q_S), 1)
    return q_new
