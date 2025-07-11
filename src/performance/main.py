from performance.args import parse_args
from performance.service import run_performance


def main() -> None:
    args = parse_args()
    run_performance(args)


if __name__ == "__main__":
    main()
