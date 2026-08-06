"""Dispatch: python -m dota2ad.inference <command>"""

import sys

COMMANDS = {
    "serve": "dota2ad.inference.server",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: python -m dota2ad.inference <{'|'.join(COMMANDS)}>")
        sys.exit(1)
    cmd = sys.argv[1]
    sys.argv = sys.argv[1:]
    mod = __import__(COMMANDS[cmd], fromlist=["main"])
    mod.main()


if __name__ == "__main__":
    main()
