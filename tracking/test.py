import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker
from lib.test.utils.checkpoint_epochs import parse_checkpoint_epochs


def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                num_gpus=8, checkpoint_epoch=None):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id, checkpoint_epoch=checkpoint_epoch)]
    checkpoint = trackers[0].get_parameters().checkpoint
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint))

    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--checkpoint_epoch', type=int, default=None,
                        help='Override TEST.EPOCH for one checkpoint, e.g. 70.')
    parser.add_argument('--checkpoint_epochs', type=str, default=None,
                        help='Run multiple checkpoints. Examples: "50,60,70" or "50-75:5". '
                             'Results are stored with run ids matching the epoch.')
    parser.add_argument('--skip_missing_checkpoints', type=int, default=0, choices=[0, 1],
                        help='When using checkpoint epochs, skip missing checkpoint files instead of failing.')

    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    checkpoint_epochs = parse_checkpoint_epochs(args.checkpoint_epochs)
    if args.checkpoint_epoch is not None and checkpoint_epochs:
        raise ValueError('Use either --checkpoint_epoch or --checkpoint_epochs, not both.')
    if args.runid is not None and checkpoint_epochs:
        raise ValueError('--runid cannot be combined with --checkpoint_epochs because epochs use run ids.')

    if checkpoint_epochs:
        skipped_epochs = []
        for epoch in checkpoint_epochs:
            print('Running checkpoint epoch {}'.format(epoch))
            try:
                run_tracker(args.tracker_name, args.tracker_param, epoch, args.dataset_name, seq_name, args.debug,
                            args.threads, num_gpus=args.num_gpus, checkpoint_epoch=epoch)
            except FileNotFoundError as e:
                if not args.skip_missing_checkpoints:
                    raise
                skipped_epochs.append(epoch)
                print('Skipping checkpoint epoch {}: {}'.format(epoch, e))
        if skipped_epochs:
            print('Skipped missing checkpoint epochs: {}'.format(
                ','.join(str(epoch) for epoch in skipped_epochs)))
    else:
        try:
            run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
                        args.threads, num_gpus=args.num_gpus, checkpoint_epoch=args.checkpoint_epoch)
        except FileNotFoundError as e:
            if not args.skip_missing_checkpoints:
                raise
            print('Skipping missing checkpoint: {}'.format(e))


if __name__ == '__main__':
    main()
