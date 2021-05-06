# Thesis Project: Time-series Analysis of GNSS Processing Strategies
# Estimating Tidally Modulated Ice-flow motion of Priestley Glacier, East Antarctica

This Script is written as a BSc. thesis project collaborated with [Glaciology & Geophysics Research Group](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/geowissenschaften/arbeitsgruppen/mineralogie-geodynamik/forschungsbereich/glaciology-geophysics/glaciology-geophysics/) in Eberhard Karls University of Tübingen submitted to [University College Freiburg](https://www.ucf.uni-freiburg.de/).

The aim of this project is to assess three different GNSS processing software for estimating influences of ocean tides on glacier ice-flow in Antarctica. RTKLIB, CSRS, and MagicGNSS are being tested, in both differential GNSS (dGNSS) and Precise Point Positioning mode (PPP). Sinusoidal motion of ice-flow velocity is being extracted by singular spectrum analysis (SSA) to assess the preservation of tidal patterns by the software.

Successfully Assessed Software:
1. [CSRS-PPP](https://webapp.geod.nrcan.gc.ca/geod/tools-outils/ppp.php?locale=en)
2. [MagicGNSS](https://magicgnss.gmv.com/)
3. [RTKLIB](http://www.rtklib.com/)

# **Study Site**
The data collection was conducted in 2018 at Priestley Glacier, Antarctica, where five stations are set up, including one base station. GNSS receivers Trimble NetR9 with antenna Zephyr 3 are used.

## **Methods**
Positioning results are obtained from GNSS processing automation online and locally. Accuracies of the software are assessed from fast Fourier transform (FFT) of the signals and comparison to bootstrapping results. Ocean tidal signals are extracted from the positioning results using SSA for comparison.

## **Accuracy**
Results indicates that CSRS-PPP provides optimal performance in both accuracy and preservation of tidal patterns. The accuracy table is shown below:

![accuracy](https://github.com/pinkychow1010/GNSS_Project/blob/main/graphicsOutput/positioningAccuracy.JPG)

Positioning Results:

![boxplot](https://github.com/pinkychow1010/GNSS_Project/blob/main/graphicsOutput/positioningBoxplot.JPG)


## **FFT & Bootstrapping**
FFT periodogram is an estimate of the power spectral density (PSD) of signals which separates noise signals, whilst bootstrapping is useful for assigning measures of accuracy to GNSS sample estimates when true value is unknown. Both results show CSRS and PPP GNSS processing regime lead to superior performance.

![FFT](https://github.com/pinkychow1010/GNSS_Project/blob/main/graphicsOutput/FFT.JPG)

![bootstrap](https://github.com/pinkychow1010/GNSS_Project/blob/main/graphicsOutput/bootstrapping.JPG)

## **Singular Spectrum Analysis (SSA)**
SSA has been frequently used in geophysical domain, yet GNSS signals are conventionally modelled by least square method. This study proposes the use of SSA in the GNSS time series signals to separate tidal signals near the grounding zone. Signals in both diurnal and semi-diurnal frequencies are successfully extracted from vertical displacement, horizontal and vertical velocity components using SSA, revealing spatial variations depending on the relative position to the grounding line. Example result is shown below:

![bootstrap](https://github.com/pinkychow1010/GNSS_Project/blob/main/graphicsOutput/SSAshirase.JPG)

## **Dynamic Time Warping (DTW)**
DTW can be used to aid measurement in similarity between multiple time series. It is used to compare displacement sequences in different GNSS sites: results illustrate similarity in tidal patterns between sites at the same side of the grounding line. It also shows that CSRS-PPP offers the most stable results for DTW time series analysis.

![DTW](https://github.com/pinkychow1010/GNSS_Project/blob/main/graphicsOutput/DTW.JPG)

## Author
Ka Hei Chow (Graduated) BSc. Student in Environmental and Earth Sciences
Linkedin: https://www.linkedin.com/in/ka-hei-chow-231345188/) E-mail: ka-hei-pinky.chow@stud-mail.uni-wuerzburg.de
