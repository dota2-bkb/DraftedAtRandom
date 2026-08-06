"""Dispatch: python -m dota2ad.pipeline <command>"""

import sys

COMMANDS = {
    "collect": "dota2ad.pipeline.collect",
    "build-dataset": "dota2ad.pipeline.build_dataset",
    "build-stats": "dota2ad.pipeline.build_match_stats",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: python -m dota2ad.pipeline <{'|'.join(COMMANDS)}>")
        sys.exit(1)
    cmd = sys.argv[1]
    sys.argv = sys.argv[1:]
    mod = __import__(COMMANDS[cmd], fromlist=["main"])
    mod.main()


if __name__ == "__main__":
    main()
