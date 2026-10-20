"""2026-10-14 policy: inference and judging are restricted to physical GPUs 0-3."""
import subprocess

ELIGIBLE_GPUS = list(range(4))
POLICY_ID = 'checkpoint_handoff_0_3_v1'


def validate_policy(config):
    if config.get('gpus') != ELIGIBLE_GPUS or config.get('gpu_policy') != POLICY_ID:
        raise ValueError('Explicit checkpoint-handoff 0-3 GPU policy required')


def validate_active(gpus):
    if (not isinstance(gpus,list) or not gpus or len(gpus) != len(set(gpus))
            or any(type(gpu) is not int or gpu not in ELIGIBLE_GPUS for gpu in gpus)):
        raise ValueError('Active GPUs must be a nonempty unique subset of physical GPUs 0-3')
    return gpus


def idle_snapshot():
    """Zero utilization is not idle: memory AND process ownership must be free."""
    rows = subprocess.check_output(['nvidia-smi','-i','0,1,2,3,4,5,6,7',
        '--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    memory, identities = {}, {}
    for line in rows.splitlines():
        index,uuid,used = (part.strip() for part in line.split(','))
        gpu=int(index)
        if gpu in memory or uuid in identities: raise ValueError('Duplicate GPU inventory')
        memory[gpu]=int(used); identities[uuid]=gpu
    if set(memory) != set(range(8)): raise ValueError('GPU inventory differs')
    rows = subprocess.check_output(['nvidia-smi','-i','0,1,2,3,4,5,6,7',
        '--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    owners = {gpu:[] for gpu in range(8)}
    for line in rows.splitlines():
        uuid,pid = (part.strip() for part in line.split(','))
        if uuid not in identities or not pid.isdigit():
            raise ValueError('Cannot resolve GPU process ownership')
        owners[identities[uuid]].append(int(pid))
    free = [gpu for gpu in ELIGIBLE_GPUS if memory[gpu] <= 64 and not owners[gpu]]
    return free,{g:memory[g] for g in ELIGIBLE_GPUS},{g:owners[g] for g in ELIGIBLE_GPUS}
