import argparse

from lib.test.analysis.plot_results import print_results
from lib.test.evaluation import get_dataset, Tracker, trackerlist
from lib.test.utils.checkpoint_epochs import epoch_result_name, parse_checkpoint_epochs


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
    args = parser.parse_args()

    plot_types = tuple(x.strip() for x in args.plot_types.split(",") if x.strip())
    report_name = args.report_name or args.dataset_name
    checkpoint_epochs = parse_checkpoint_epochs(args.checkpoint_epochs)

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
    else:
        trackers = trackerlist(
            args.tracker_name,
            args.tracker_param,
            args.dataset_name,
            run_ids=None,
            display_name=args.display_name,
        )
    dataset = get_dataset(args.dataset_name)

    print_results(
        trackers,
        dataset,
        report_name=report_name,
        merge_results=bool(args.merge_results),
        plot_types=plot_types,
        force_evaluation=bool(args.force_evaluation),
    )


if __name__ == "__main__":
    main()
