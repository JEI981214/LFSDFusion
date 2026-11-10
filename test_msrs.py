"""Test the supplied msrs checkpoints with the original inference model."""
from argparse import Namespace

from inference import main, run_test


def run_condition(condition, device, checkpoint, out_root, config):
    args = Namespace(condition=condition, device=str(device), checkpoint=checkpoint,
                     out_root=out_root, config=config, data_dir=None,
                     pretrained_root=None, captions=None, max_samples=None,
                     skip_precompute=False, save_stages=True)
    return run_test("msrs", args)


if __name__ == "__main__":
    main("msrs")
