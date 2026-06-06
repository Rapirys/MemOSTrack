from lib.test.utils import TrackerParams
import os
import multiprocessing
from lib.test.evaluation.environment import env_settings
from lib.config.ostrack.config import cfg, update_config_from_file

_PRINTED_TEST_CONFIG = False


def _should_print_test_config_once():
    global _PRINTED_TEST_CONFIG
    if _PRINTED_TEST_CONFIG:
        return False

    proc_name = multiprocessing.current_process().name
    # In sequential mode: MainProcess prints once.
    # In parallel mode: allow only the first pool worker to print once.
    if proc_name != "MainProcess" and not proc_name.endswith("-1"):
        return False

    _PRINTED_TEST_CONFIG = True
    return True


def parameters(yaml_name: str, checkpoint_epoch=None):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/ostrack/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    if _should_print_test_config_once():
        print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    epoch = cfg.TEST.EPOCH if checkpoint_epoch is None else checkpoint_epoch
    params.checkpoint_epoch = epoch
    params.checkpoint = os.path.join(save_dir, "checkpoints/train/ostrack/%s/OSTrack_ep%04d.pth.tar" %
                                     (yaml_name, epoch))

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
