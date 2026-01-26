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
        path_obj = Path(p)
        # if the path is already absolut (i.e. like C:/...) don't prefix with root
        if path_obj.is_absolute():
            return path_obj.resolve()
        return (root / p).expanduser().resolve()

    for section_name, section in cfg.items():
        if isinstance(section, dict):
            for k, v in section.items():
                if isinstance(v, str):
                    section[k] = fix(v)

    return cfg


def setup_project(cfg: dict):
    """
    Explicitly creates the directory structure. 
    """
    print("Initializing project directories...")
    for section in cfg.values():
        if isinstance(section, dict):
            for path in section.values():
                if isinstance(path, Path):
                    # If it's a file path, create the parent directory
                    # If it's a directory path, create it directly
                    if path.suffix:  # Simple check if it looks like a file
                        path.parent.mkdir(parents=True, exist_ok=True)
                    else:
                        path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # 1. Load configuration (Read-only)
    config = load_config()
    
    # 2. Setup environment (Side-effect - explicit)
    setup_project(config)
    
    # 3. Proceed with analysis
    print(f"Project root is set to: {config['project_root']}")