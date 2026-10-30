"""Apply the measured posterior root/strand transport to one independent copy."""
import runpy
from pathlib import Path

folder=Path(__file__).resolve().parent
api=runpy.run_path(str(folder/'root_surface_transport_r09.py'))
report=api['apply_and_save'](
    source_relative='work/reconstruction-r02/reference-forelock-flow-r09.blend',
    output_relative='work/reconstruction-r02/reference-root-flow-r09.blend',
    report_relative='work/reconstruction-r02/root-surface-transport-r09.json')
print('R09_ROOT_CANDIDATE_SAVED', flush=True)
