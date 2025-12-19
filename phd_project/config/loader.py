import yaml
from pathlib import Path

def load_config(path=None, cfg_file:str|None=None):
    if path is None and cfg_file is None:
        path = Path(__file__).parent / "config.yaml"
    elif path is None and not cfg_file is None:
        path = Path(__file__).parent / cfg_file
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Convert all paths to absolute Path objects
    root = (path.parent / cfg["project_root"]).resolve()
    
    def fix(p):
        return (root / p).expanduser().resolve()

    for section in cfg.values():
        if isinstance(section, dict):
            for k, v in section.items():
                # if isinstance(v, str) and "/" in v:
                section[k] = fix(v)

    return cfg


if __name__ == "__main__":
    cfg = load_config()