import argparse
import os

from lib.test.analysis.plot_results import print_results
from lib.test.evaluation import get_dataset, Tracker, trackerlist


def parse_checkpoint_epochs(epoch_spec):
    """Parse comma-separated epochs and ranges like '50,60-75:5'."""
    if epoch_spec is None:
        return []

    epochs = []
    for raw_part in epoch_spec.split(","):
        part = raw_part.strip()
        if not part:
            continue

        step = 1
        if ":" in part:
            part, raw_step = part.split(":", 1)
            step = int(raw_step)
            if step <= 0:
                raise ValueError("Epoch range step must be positive")

        if "-" in part:
            raw_start, raw_end = part.split("-", 1)
            start = int(raw_start)
            end = int(raw_end)
            if end < start:
                raise ValueError("Epoch range end must be >= start")
            epochs.extend(range(start, end + 1, step))
        else:
            epochs.append(int(part))

    return list(dict.fromkeys(epochs))


def epoch_result_name(base_name, epoch):
    return "{}_{:03d}".format(base_name, epoch)


def has_first_sequence_result(tracker, dataset):
    if not dataset:
        return False
    result_path = os.path.join(tracker.results_dir, "{}.txt".format(dataset[0].name))
    return os.path.isfile(result_path)


def main():
    parser = argparse.ArgumentParser(description="Print tracking metrics from saved test outputs.")
    parser.add_argument("--dataset_name", default="got10k_val")
    parser.add_argument("--tracker_name", default="ostrack")
    parser.add_argument("--tracker_param", default="vitb_256_mae_32x4_ep300/got10k")
    parser.add_argument("--display_name", default="OSTrack")
    parser.add_argument("--report_name", default=None)
    parser.add_argument("--plot_types", default="success,prec,norm_prec")
    parser.add_argument("--merge_results", type=int, default=1, choices=[0, 1])
    parser.add_argument("--force_evaluation", type=int, default=0, choices=[0, 1])
    parser.add_argument("--checkpoint_epochs", default=None,
                        help='Evaluate multiple checkpoint result folders. Examples: "50,60,70" or "50-75:5".')
    parser.add_argument("--got10k_subdir", type=int, default=None, choices=[0, 1],
                        help="Append /got10k to each epoch result name. Defaults to 1 for got10k datasets.")
    parser.add_argument("--skip_missing_results", type=int, default=0, choices=[0, 1],
                        help="When evaluating checkpoint epochs, skip epochs with missing result files.")
    args = parser.parse_args()

    plot_types = tuple(x.strip() for x in args.plot_types.split(",") if x.strip())
    report_name = args.report_name or args.dataset_name
    checkpoint_epochs = parse_checkpoint_epochs(args.checkpoint_epochs)
    dataset = get_dataset(args.dataset_name)

    if checkpoint_epochs:
        use_got10k_subdir = args.got10k_subdir
        if use_got10k_subdir is None:
            use_got10k_subdir = int(args.dataset_name.startswith("got10k"))
        trackers = []
        for epoch in checkpoint_epochs:
            tracker_param = epoch_result_name(args.tracker_param, epoch)
            if use_got10k_subdir:
                tracker_param = "{}/got10k".format(tracker_param)
            trackers.append(Tracker(
                args.tracker_name,
                tracker_param,
                args.dataset_name,
                display_name="{}_ep{:04d}".format(args.display_name, epoch),
            ))
        if args.skip_missing_results:
            kept_trackers = []
            skipped_epochs = []
            for tracker, epoch in zip(trackers, checkpoint_epochs):
                if has_first_sequence_result(tracker, dataset):
                    kept_trackers.append(tracker)
                else:
                    skipped_epochs.append(epoch)
            trackers = kept_trackers
            if skipped_epochs:
                print("Skipped missing result epochs: {}".format(
                    ",".join(str(epoch) for epoch in skipped_epochs)))
            if not trackers:
                raise ValueError("No checkpoint result folders found to evaluate.")
    else:
        trackers = trackerlist(
            args.tracker_name,
            args.tracker_param,
            args.dataset_name,
            run_ids=None,
            display_name=args.display_name,
        )

    print_results(
        trackers,
        dataset,
        report_name=report_name,
        merge_results=bool(args.merge_results),
        plot_types=plot_types,
        force_evaluation=bool(args.force_evaluation),
        skip_missing_seq=bool(args.skip_missing_results),
    )


if __name__ == "__main__":
    main()
