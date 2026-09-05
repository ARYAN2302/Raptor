def expected_calibration_error(*a,**k):
    from .metrics import calibration_ece
    return calibration_ece(*a,**k)
