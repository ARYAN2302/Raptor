#!/usr/bin/env python3
import sys
sys.path.insert(0,"src")
from src.evaluation.metrics import range_rmse, az_wrap_loss
import numpy as np
def main():
    print(f"range RMSE demo: {range_rmse([100,105],[100,100]):.2f}")
    print(f"az wrap: {az_wrap_loss([359,10],[1,355]):.2f}")
if __name__=="__main__": main()
