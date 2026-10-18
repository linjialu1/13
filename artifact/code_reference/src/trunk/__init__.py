"""S1 trunk model-interface package.

Module layout
-------------
adapters.py         raw-data adaptation for the four datasets (reuses the accepted preflight adapters)
descriptors.py      descriptor implementations (UCI-HAR / WISDM reused; generic 8-dim for UWave & CWRU)
acbm_interface.py    frozen input standardization -> CfC representation -> 56-dim joint -> sparse-head interface
check_forward.py     the zero-training forward-check entry point

No training, no optimizer step, no backward anywhere in this package.
"""
