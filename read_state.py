import sys, argparse, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]   # adjust if needed
sys.path[:0] = [str(ROOT), str(ROOT / "peps_torch")]

from ipeps.integration_yastn import load_PepsAD
import yastn.yastn as yastn
from yastn.yastn.backend import backend_torch as backend
from yastn.yastn.sym import sym_U1


def read_state(args):
    # define the yastn config for U(1)-symmetric tensors
    yastn_config = yastn.make_config(
        backend=backend,
        sym=sym_U1,
        fermionic=True,
        default_device='cpu',
        default_dtype='complex128',
        tensordot_policy="no_fusion",
    )

    # Load the state in PepsAD format (see integration_yastn.py).
    if os.path.isfile(args.instate):
        stateAD = load_PepsAD(yastn_config, args.instate)
    else:
        raise FileNotFoundError(f"Input state file {args.instate} not found.")
    stateAD.normalize_()

    # To convert it to Peps format (used in YASTN), use state = stateAD.to_Peps()
    return stateAD.to_Peps()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instate", default="none", help="Input state JSON")

    state = read_state(parser.parse_args())
    print(state[(0,0)].get_shape())
