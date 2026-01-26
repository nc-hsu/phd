import os
os.environ['RPY2_CFFI_MODE'] = 'ABI'

from pymer4.models import lmer
import pandas as pd
import numpy as np
import subprocess
import sys

def check_r_health():
    print("--- R & Pymer4 Health Check ---")
    
    # 1. Check R_HOME
    r_home = os.environ.get('R_HOME')
    if r_home:
        print(f"[SUCCESS] R_HOME is set to: {r_home}")
    else:
        print("[ERROR] R_HOME environment variable is NOT set.")
    
    # 2. Check if R is in PATH
    try:
        r_version = subprocess.check_output(["R", "--version"], stderr=subprocess.STDOUT).decode()
        print(f"[SUCCESS] R is accessible via PATH.")
    except Exception:
        print("[ERROR] R is not found in your system PATH.")

    # 3. Test rpy2 and lme4 connection
    try:
        import rpy2.robjects.packages as rpackages
        from rpy2.robjects.vectors import StrVector
        
        utils = rpackages.importr('utils')
        lme4_installed = rpackages.isinstalled('lme4')
        
        if lme4_installed:
            print("[SUCCESS] R package 'lme4' is installed and detectable.")
        else:
            print("[ERROR] 'lme4' is NOT installed in R. Run install.packages('lme4') in R.")
            
    except ImportError:
        print("[ERROR] 'rpy2' is not installed in Python. Run 'pip install rpy2'.")
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")

    # 4. Final pymer4 test
    try:
        from pymer4.models import lmer
        print("[SUCCESS] pymer4 is installed and ready to use.")
    except ImportError:
        print("[ERROR] 'pymer4' is not installed. Run 'pip install pymer4'.")

if __name__ == "__main__":
    check_r_health()

# # Quick test data
# df = pd.DataFrame({'y': np.random.randn(100), 'group': np.repeat(range(10), 10)})
# model = Lmer('y ~ 1 + (1|group)', data=df)
# print(model.fit())