"""Run the Q1/Q2 example, Q3 continuation, or Q4 from-initial example."""
import sys
import argparse
sys.dont_write_bytecode=True
if __name__=="__main__":
    arguments=sys.argv[1:]
    selector=argparse.ArgumentParser(add_help=False)
    selector.add_argument('--question',choices=('1','2','3','4','all'),default='all')
    selected,remaining=selector.parse_known_args(arguments)
    if selected.question=='4':
        from minimal_q4 import main
        main(remaining)
    elif selected.question=='3':
        from minimal_q3 import main
        main(remaining)
    else:
        from minimal_solver import main
        main(arguments)
