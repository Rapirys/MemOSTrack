import argparse

from lib.test.analysis.plot_results import print_results
from lib.test.evaluation import get_dataset, trackerlist


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
    args = parser.parse_args()

    plot_types = tuple(x.strip() for x in args.plot_types.split(",") if x.strip())
    report_name = args.report_name or args.dataset_name

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
