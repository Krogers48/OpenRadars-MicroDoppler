# Krogers mmWave Package


This repository contains our implementation of TI IWR6843 + DCA1000 radar data processing, centered around `main.py`.  
The script reads raw ADC captures and generates plots that demonstrate the importance of clutter removal and the extraction of gait signatures.  

The pipeline includes:
1. Reading raw ADC data and TI profile configuration.
2. Range and Doppler FFT processing with angle estimation.
3. A/B plots of Range–Doppler and Range–Time (before vs after clutter removal).
4. Micro-Doppler spectrogram of a walking subject.
5. Wall distance estimation using formulas from the thesis and filtering against the theoretical ghost curve.

## Documentation
- [openradar.readthedocs.io](https://openradar.readthedocs.io)

## Directory Structure
```bash
.
├── data                    # Small size sample data.
├── demo                    # Python implementations of TI demos.
├── docs                    # Documentation for mmwave package and hardware setup.
├── mmwave                  # mmwave package including all the DSP, tracking, etc algorithms.
├── PreSense Applied radar  # Jupyter notebook series explaining how apply radar concepts to real data
├── scripts                 # Various setup scripts for mmwavestudio, etc
├── Krogers/
│   └── MicroDoppler/
│       ├── main.py                  # Entry point for all processing and plots
│       ├── draft.py                 # Prototype / scratchpad script
│       └── PersonWalkingData/
│           ├── iqData_Raw_0.bin     # Raw ADC capture from DCA1000
│           ├── iqData_Cooked_0.bin  # preprocessed copy written by main.py
│           ├── iqData_Raw_LogFile.csv
│           ├── iqData_RecordingParameters.mat
│           └── xwr68xx_profile_2025_06_08.cfg.txt  # TI radar configuration file
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt        # Required dependencies to run this package.
└── setup.py                # Install mmwave package.
```

## Current Roadmap for this project
- [ ] Promote the tunable parameters in `main.py` (frame index, save options, thresholds) to command-line arguments.
- [ ] Add automatic wall-distance tracking over time with stability metrics.
- [ ] Package consistent plotting helpers for A/B comparisons (shared color scales, labeled metrics).
- [ ] Support live/online processing mode instead of only offline `.bin` playback.
- [ ] Integrate Wigait’s stable phase extraction algorithm for gait features.


## Future Plan
1. Convert the processing pipeline to a live processing script
2. Implement a stable walking phase algorithm


## Installation

### Pip installation
```
pip install OpenRadars-MicroDoppler
```

### Debug Mode
```
git clone https://github.com/Krogers48/OpenRadars-MicroDoppler
cd OpenRadars-MicroDoppler
pip install -r requirements.txt
python setup.py develop
```

## Uninstallation

```
pip uninstall OpenRadars-MicroDoppler
```

## Example Import and Usage

```python
import mmwave as mm
from mmwave.dataloader import DCA1000

dca = DCA1000()
adc_data = dca.read()
radar_cube = mm.dsp.range_processing(adc_data)
```

## Running main.py

Place your capture (`iqData_Raw_0.bin`) and matching TI profile (`xwr68xx_profile_2025_06_08.cfg.txt`) into:


Krogers/MicroDoppler/PersonWalkingData/

Run the script:

```bash
python Krogers/MicroDoppler/main.py
The script will produce:

Range–Doppler BEFORE vs AFTER plots with clutter metrics

Range–Time (RTI) BEFORE vs AFTER

Micro-Doppler spectrogram

Printed wall distance estimate

```
