"""Download the pinned official corn evaluation subset through process-local company proxy."""
import os
from pathlib import Path
for key in ('http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):
    os.environ[key]='http://100.68.170.171:3128'
os.environ['NO_PROXY']=os.environ['no_proxy']='localhost,127.0.0.1,::1,100.68.170.171'
os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import snapshot_download
root=Path('/mnt/vdc_5t/model/acwm/DreamDojo/official_validation')
root.mkdir(parents=True,exist_ok=True)
(root/'dataset_revision.txt').write_text('9f68429ad12535d30d1aae33b7945c79c830c8c4')
snapshot_download('nvidia/PhysicalAI-Robotics-GR00T-Teleop-GR1',repo_type='dataset',revision=(root/'dataset_revision.txt').read_text().strip(),allow_patterns=['In-lab_Eval/gr1_unified.pnp_corn_robot/**'],local_dir=root/'dataset',max_workers=4)
print('DOWNLOAD_COMPLETE',flush=True)
