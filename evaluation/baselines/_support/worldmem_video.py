import sys,runpy
from pathlib import Path
# Delegate to the existing official dataset adapter, now with a strict prefix input mode.
sys.argv=[sys.argv[0],sys.argv[1],str(Path(sys.argv[2]).parent)]
runpy.run_path(str(Path(__file__).with_name('worldmem_infer.py')),run_name='__main__')
