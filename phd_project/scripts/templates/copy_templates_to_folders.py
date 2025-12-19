from pathlib import Path

def copy_and_edit_structural_model(
    src: str | Path,
    dst: str | Path,
    new_name: str|None = None,
    line_number: int=16
) -> None:
    """
    Copy a the structural model script to a new model, optionally replacing the name of the 
    design json 

    Args:
        src: Source file path
        dst: Destination file path
        replace: Replacement string
        line_number: If given, only modify this line (0-based). Otherwise modify all lines.
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8") as f_in, dst.open("w", encoding="utf-8") as f_out:
        for ii, line in enumerate(f_in):
            if new_name is not None and ii == line_number:
                line = f'design_json = "{new_name}.json"\n'
            f_out.write(line)

##############################


if __name__ == "__main__":
    src = "C:/Users/clemettn/OneDrive - Helmut-Schmidt-Universität/01_arbeit/14_PhD/scripts/templates/template_structural_model.py"
    dst = "D:/case_studies_set1_dc2/3s_cbf_dc2_10/structural_model.py"
    new_name = "3s_cbf_dc2_10_out"

    copy_and_edit_structural_model(src, dst, new_name)
