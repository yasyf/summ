import argparse
from pathlib import Path

from summ.pipeline import Pipeline
from summ.summ import Summ


class CLI:
    @staticmethod
    def run(summ: Summ, pipe: Pipeline):
        parser = argparse.ArgumentParser()
        parser.add_argument("command", help="Command to run")
        parser.add_argument("args", nargs="*", help="Arguments to command")
        args = parser.parse_args()
        if args.command == "populate":
            summ.populate(Path(args.args[0]), pipe=pipe)
        elif args.command == "query":
            summ.query(args.args[0])
        else:
            print("Unknown command")
