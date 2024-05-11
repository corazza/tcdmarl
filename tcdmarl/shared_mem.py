from typing import Dict

# from tcdmarl.path_consts import WORK_DIR
from tcdmarl.tcrl.reward_machines.rm_common import ProbabilisticRewardMachine

PRM_TLCD_MAP: Dict[str, ProbabilisticRewardMachine] = dict()


# def record_prm_tlcd(path: Path, prm: ProbabilisticRewardMachine):

#     # extract filename from path: Path
#     filename: str = path.name
#     memoize_file_path = WORK_DIR / filename

STUCK_COUNTER: int = 0
