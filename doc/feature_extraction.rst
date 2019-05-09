Feature extraction
======================
This is the core of the whole library. The algorithms for feature extraction are divided into 3 subgroups:

- Univariate
- Bivariate
text
- Event detection

All the algorithms accept raw or filtered data and provide pandas dataframes as their output.


Univariate feature extraction
*********************************



Bivariate feature extraction
*********************************
Bivariate feature extraction algorithms server for calculating relationships between two signals. They can be used for example to obtain connectivity between different areas of the brain.

Event detection
*********************************
This subsection provides algorithms for detection of events occurring in the signal. All algorithms provide event position or event start/stop and some of them provide additional features of detected events. Currently the library contains algorithms for detecting interictal epileptiform discharges (IEDs),i.e. epileptic spikes, and a number of algorithms for detection of high frequency oscillations (HFOs).
