"""Read-only diagnostic of the adopted R09 groom substrate."""
import runpy
from pathlib import Path

api = runpy.run_path(str(Path(__file__).with_name('root_surface_diagnostic_r09.py')))
result = api['run'](
    source_relative='work/reconstruction-r02/reference-forelock-flow-r09.blend',
    report_relative='work/reconstruction-r02/root-surface-diagnostic-r09.json',
    export_arrays=True,
)
print('R09_ROOT_DIAGNOSTIC_COMPLETE', result['sourceSha256'], flush=True)
