from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
from run_all import run_question
if __name__=="__main__":run_question(1)
