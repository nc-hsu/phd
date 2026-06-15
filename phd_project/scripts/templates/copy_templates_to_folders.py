from pathlib import Path
import re
import shutil
from pathlib import Path
from typing import Dict, Any

def copy_structural_model(
    src: str | Path,
    dst: str | Path,
    design_json: str | None = None,
    ops_updates: Dict[str, Any] | None = None,
    recorder_updates: Dict[str, Any] | None = None,
    damping_updates: Dict[str, Any] | None = None
) -> None:
    
    src_path = Path(src)
    dst_path = Path(dst)
    
    # Check if we are doing an in-place update
    is_in_place = src_path.resolve() == dst_path.resolve()
    
    if not is_in_place:
        # Create parent directories and copy only if paths are different
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
    
    content = dst_path.read_text(encoding='utf-8')
    
    # 1. Update design_json filename
    if design_json is not None:
        content = re.sub(
            r'(design_json\s*=\s*")[^"]+(")',
            f'\\g<1>{design_json}\\g<2>',
            content
        )
    
    # 2. Update dictionary parameters
    # Map update dicts to their variable names in the file
    updates = {
        "ops_model_config": ops_updates,
        "recorder_config": recorder_updates,
        "damping_config": damping_updates
    }
    
    for var_name, update_dict in updates.items():
        if update_dict:
            for key, value in update_dict.items():
                # Format the value for Python (handles strings, bools, etc)
                val_str = repr(value)
                
                # # Regex breakdown:
                # # {var_name}              -> Variable name
                # # (?::\s*(?:dict|Dict).*?)? -> Optional type hint (dict or Dict[...])
                # # \s*=\s*\{               -> Assignment and start of dict
                # # .*?["\']{key}["\']      -> Search for key inside this dict block
                # # \s*:\s* -> Colon after key
                # pattern = (
                #     rf'({var_name}(?::\s*(?:dict|Dict).*?)?\s*=\s*\{{.*?'
                #     rf'["\']{key}["\']\s*:\s*)[^,}}]+'
                # )
                
                # # We use re.DOTALL to ensure .* matches across multiple lines
                # content = re.sub(
                #     pattern,
                #     f'\\g<1>{val_str}',
                #     content,
                #     flags=re.DOTALL
                # )

                # This captures:
                # Group 1: Everything from 'var_name =' to the colon after 'key'
                # Group 2: The quote style used for the key (for matching)
                # Group 3: The comma and newline at the end of the line
                pattern = (
                    rf'({var_name}(?::\s*(?:dict|Dict).*?)?\s*=\s*\{{.*?'
                    rf'(["\']){key}\2\s*:\s*)'
                    rf'.*?(,\n)'
                )
                
                # Replacement: Group 1 (start + key) + New Value + Group 3 (,\n)
                content = re.sub(
                    pattern,
                    f'\\g<1>{val_str}\\g<3>',
                    content,
                    flags=re.DOTALL
                )
                
    dst_path.write_text(content, encoding='utf-8')


# def copy_pushover_config(
#     src: str | Path,
#     dst: str | Path,
#     ctrl_node: int | None = None,
#     u_max: tuple[str, float] | None = None,
#     du: float | None = None
# ) -> None:
    
#     src_path = Path(src)
#     dst_path = Path(dst)
    
#     dst_path.parent.mkdir(parents=True, exist_ok=True)
#     shutil.copy2(src_path, dst_path)
    
#     content = dst_path.read_text(encoding='utf-8')
    
#     if ctrl_node is not None:
#         content = re.sub(
#             r'("ctrl_node"\s*:\s*)[^,]+',
#             f'\\g<1>{ctrl_node}',
#             content
#         )
        
#     if u_max is not None:
#         # Matches "U_max" followed by any content inside [] or ()
#         # [\[\(] matches [ or (
#         # .*? matches content inside non-greedily
#         # [\]\)] matches ] or )
#         content = re.sub(
#             r'("U_max"\s*:\s*)[\[\(].*?[\]\)]',
#             f'\\g<1>{u_max}',
#             content
#         )
        
#     if du is not None:
#         content = re.sub(
#             r'("dU"\s*:\s*)[^,]+',
#             f'\\g<1>{du}',
#             content
#         )
        
#     dst_path.write_text(content, encoding='utf-8')


def copy_analysis_config(
        src: str | Path,
        dst: str | Path,
        results_folder_name: str | None = None,
        update_config: dict | None = None,
        **kwargs
) -> None:
    """
    Copies a config template and updates the results folder and config dict.
    
    Args:
        src: Source path of the template.
        dst: Destination path for the new config.
        results_folder_name: Optional string to rename the result destination folder.
        **updates: Key-value pairs to update in the config dictionary.
    """
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        raise FileNotFoundError(f"Source {src} not found.")

    content = src.read_text(encoding='utf-8')

    # 1. Handle kwargs: Replace existing or prepare for injection
    to_inject = {}
    for k, v in kwargs.items():
        fmt = f'"{v}"' if isinstance(v, str) else str(v)
        # Matches top-level 'var = val' but NOT '"var": val'
        # Group 1: start of line + var name + equals
        # Group 2: old value
        # Group 3: newline
        var_pattern = rf'(^{k}\s*=\s*)(.*?)(?=\n)'
        
        if re.search(var_pattern, content, re.MULTILINE):
            content = re.sub(
                var_pattern, 
                lambda m, val=fmt: f'{m.group(1)}{val}', 
                content, 
                flags=re.MULTILINE
            )
        else:
            to_inject[k] = fmt

    # 2. Update folder name and inject remaining kwargs at the anchor
    res_pattern = r'(results_folder_name\s*=\s*["\'])(.*?)(["\']\n)'
    
    def res_replacer(m):
        folder = results_folder_name if results_folder_name else m.group(2)
        line = f'{m.group(1)}{folder}{m.group(3)}'
        # Append only those that didn't exist in the file already
        for k, fmt_val in to_inject.items():
            line += f'{k} = {fmt_val}\n'
        return line

    content = re.sub(res_pattern, res_replacer, content)

    # 3. Update internal config dictionary
    if update_config:
        for key, value in update_config.items():
            # Group 1: Everything up to the colon and following space
            # Group 2: The newline and any indentation on the next line
            pattern = rf'((["\']){key}\2\s*:\s*)(?:.+?)(,\n)'
            
            if re.search(pattern, content):
                fmt = f'"{value}"' if isinstance(value, str) else str(value)
                content = re.sub(
                    pattern, 
                    lambda m, v=fmt: f'{m.group(1)}{v}{m.group(3)}', 
                    content
                )
            else:
                print(f"Warning: Key '{key}' not found in config.")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(content, encoding='utf-8')


def configure_batch_run_file(
    src: str | Path,
    dst: str | Path,
    scripts_configs_list: list[dict[str, str | list[str]]]
) -> None:
    
    src_path = Path(src)
    dst_path = Path(dst)
    
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    
    content = dst_path.read_text(encoding='utf-8')

    # Helper to convert a dictionary containing Path objects into a clean string
    def format_entry(entry: Dict[str, Any]) -> str:
        # Convert to string first to handle both Path objects and raw strings
        # Then replace any backslashes with forward slashes
        script_val = str(entry['script']).replace('\\', '/')
        
        # Apply the same logic to the list of config paths
        configs_list = [f"Path('{str(p).replace('\\', '/')}')" for p in entry['config']]
        configs_val = ", ".join(configs_list)
        
        return (
            f'{{\n'
            f'        "script": Path("{script_val}"),\n'
            f'        "config": [{configs_val}]\n'
            f'    }}'
        )

    # Build the full list string
    formatted_items = ",\n    ".join([format_entry(e) for e in scripts_configs_list])
    list_str = f"[\n    {formatted_items}\n]"

    # Use the balanced regex from our previous step
    pattern = r'(scripts_and_configs\s*=\s*)\[.*?\](?=\n+(?:def|if|$))'
    
    content = re.sub(
        pattern,
        f'\\g<1>{list_str}',
        content,
        flags=re.DOTALL
    )
    
    dst_path.write_text(content, encoding='utf-8')


def configure_optimisation_batch_run(
        src: str | Path,
        dst: str | Path,
        configs_list: list[dict[str, Any]]
    ) -> None:
    
    src_path = Path(src)
    dst_path = Path(dst)
    
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    
    def format_value(v: Any) -> str:
        """Recursively formats objects into valid Python code strings."""
        if isinstance(v, Path):
            clean_path = str(v).replace('\\', '/')
            return f"Path(r'{clean_path}')"
        elif isinstance(v, str):
            return f"'{v}'"
        elif isinstance(v, (list, tuple)):
            # Recursively format elements within containers
            items = [format_value(i) for i in v]
            if isinstance(v, tuple):
                # Handle single-element tuple syntax (val,)
                return f"({', '.join(items)})" if len(items) != 1 \
                    else f"({items[0]},)"
            return f"[{', '.join(items)}]"
        else:
            # Handles int, float, bool
            return str(v)

    def format_entry(entry: Dict[str, Any]) -> str:
        lines = []
        for key, value in entry.items():
            formatted_val = format_value(value)
            lines.append(f'        "{key}": {formatted_val},')
        
        inner_content = "\n".join(lines)
        return f"    {{\n{inner_content}\n    }}"

    formatted_items = ",\n".join([format_entry(e) for e in configs_list])
    list_str = f"[\n{formatted_items},\n]"

    content = dst_path.read_text(encoding='utf-8')

    # Regex targeting the 'configs' variable assignment
    pattern = r'(configs\s*=\s*)\[.*?\](?=\n+(?:def|if|class|$))'
    
    new_content = re.sub(
        pattern,
        f'\\g<1>{list_str}',
        content,
        flags=re.DOTALL
    )
    
    dst_path.write_text(new_content, encoding='utf-8')



def copy_file(src: str | Path, dst: str | Path) -> None:
    try:
        # copy2 preserves metadata (timestamps, etc.)
        _ = shutil.copy2(src, dst)
    except FileNotFoundError:
        print("Error: The source file was not found.")
    except PermissionError:
        print("Error: Permission denied.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


##############################
