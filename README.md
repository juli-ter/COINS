# COINS Python port

This is a Python rewrite of the core behavioural analysis pipeline from the MATLAB `coins_meg_analysis` repository.

## Included
- options/paths
- subject metadata
- robust behavioural CSV loading
- subject-level behavioural analysis
- tracking performance metrics
- block kernels (average and regression-based)
- sessionwise regression kernels
- post-jump adjustment extraction
- group-level adjustment and reaction-time summaries
- matplotlib plots in place of MATLAB `.fig` outputs

## Main entry points
- `python -m coins_py.scripts.analyse_subject 1 --main-dir /path/to/COINS`
- `python -m coins_py.scripts.run_behaviour_loop --main-dir /path/to/COINS`

## Notes
- Outputs are saved as `.pkl` and `.png` instead of `.mat` and `.fig`.
- The behavioural pipeline is the primary focus of this port.
- MEG/model helper code is included only as light placeholders/loaders.
- Some MATLAB plotting utilities (`notBoxPlot`, `shadedErrorBar`) were replaced with simpler matplotlib equivalents.
- Several MATLAB workspace-dependent bugs were cleaned up during the port.
