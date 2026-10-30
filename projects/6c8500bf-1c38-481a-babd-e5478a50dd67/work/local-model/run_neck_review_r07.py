"""Serial, read-only four-pose review in the saved current production copy."""
import runpy
from pathlib import Path
api=runpy.run_path(str(Path(__file__).with_name('head_neck_pose_review_r07.py')))
api['run']()
