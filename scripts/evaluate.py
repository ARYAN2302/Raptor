#!/usr/bin/env python3
import sys
sys.path.insert(0, "/tmp/raptor_build/Raptor")
from src.evaluation.metrics import range_rmse, az_rmse
print("range RMSE", range_rmse([100,105],[100,100]))
print("az RMSE", az_rmse([359,10],[1,355]))
