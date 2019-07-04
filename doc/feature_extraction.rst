Feature extraction
======================
This is the core of the whole library. The algorithms for feature extraction are divided into 3 subgroups:

- Univariate
- Bivariate
- Event detection

All the algorithms accept raw or filtered data and provide pandas dataframes as their output.


Univariate feature extraction
*********************************



Bivariate feature extraction
*********************************
Bivariate feature extraction algorithms server for calculating relationships between two signals. They can be used for example to obtain connectivity between different areas of the brain.

- Linear correlation

  The linear correlation (LC) varies in interval <-1,1> and reflects shape similarities between two signals. LC=1 indicates perfect conformity between two signals, LC=-1 indicates opposite signals and LC=0 indicates two different signals. LC is calculated by Pearson’s correlation coefficient as: LCX,Y=[cov(Xt,Yt)/std(Xt)・std(Yt)], where Xt,Yt are the two evaluated signals, cov is the covariance and std is the standard deviation. The linear correlation between two signals can be calculated with a time-lag. Maximum time-lag should not exceed fmax/2. Lagged linear correlation (LLC) for each time-lag k was calculated by Pearson’s correlation coefficient as: LLCX(k),Y(k)=[cov(Xt(k),Yt(k))/std(Xt(k))・std(Yt(k))], where Xt,Yt are the two evaluated signals, cov is the covariance and std is the standard deviation. The maximum value of correlation is stored with its time-lag value.


Event detection
*********************************
This subsection provides algorithms for detection of events occurring in the signal. All algorithms provide event position or event start/stop and some of them provide additional features of detected events. Currently the library contains algorithms for detecting interictal epileptiform discharges (IEDs),i.e. epileptic spikes, and a number of algorithms for detection of high frequency oscillations (HFOs).
