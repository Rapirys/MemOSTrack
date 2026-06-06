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
