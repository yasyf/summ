import argparse
from pathlib import Path

from summ.summ import Summ


class CLI:
    @staticmethod
    def run():
        parser = argparse.ArgumentParser()
        parser.add_argument("command", help="Command to run")
        parser.add_argument("args", nargs="*", help="Arguments to command")
        args = parser.parse_args()
        if args.command == "populate":
            Summ().populate(Path(args.args[0]))
        elif args.command == "query":
            Summ().query(args.args[0])
        else:
            print("Unknown command")
