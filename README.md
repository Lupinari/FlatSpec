# FlatSpec
FlatSpec is the Python-based control engine for an SLM-based spectral flattener in a Laser Frequency Comb. It uses a feedback loop to automatically and precisely flatten the intensity profile of the optical spectrum by controlling the amplitude of different frequency components via the SLM.

The code is written to be used with a Holoeye SLM and a Yokohama OSA. It is my intention to make a flexible version available in the future that allows using other equipment.

The following variables are inputs decided by the user:

-OSA
- WL_i=5e-7 #inicial wavelength (m)
- WL_f=5.7e-7 #final wavelength (m)
- sample_points=801 # OSA sample points

-For WL calibrations
- Col_w=15 # width of pixel collumn 
- Col_i=0 # starting collumn for calibration
- Col_sampled=15 # how many samples to use in the wl calibration

-For Gama calibrations
- Delta_GS=10 #step size with which Intensity vs. Grayscale Calibration will occur
- GS_max=256
- GS_min=0

-data name
- DATANAME=2708 #marker to identify data
- SERIES=3 #marker to identify subset of data
- LOOPS=200 #number of loops


The following variables are saved throughout the code run ('SERIES' and 'DATANAME' are markers, defined by the user, to identify and differentiate experimental data): 

- GamaSpecs_%d%d.npy %(SERIES,DATANAME)
  Array of arrays containing one spectrum for each GS level GamaSpecs[0][ ] contains wavelength (y axis) of the spectrum.

- CalibSpecs_%d%d.npy %(SERIES,DATANAME)
  Array of arrays containing one spectrum for each highlighted collumn, on the wavelength vs pixel calibration, and one more for the reference spec 'blank screen'[1][ ]. CalibSpecs[0][ ] contains wavelength (x axis) of the spectrum.

- fit_coef_%d%d.npy %(SERIES,DATANAME)
  Linear pixel vs wavelength fitting. The fitting is done considering number of pixel (x axis) VS wavelength (y axis).
    
- WLperColl_%d%d.npy %(SERIES,DATANAME)
  Array of arrays containing wl [0] and corresponding pixel collumn [1] for wl calibration.

- IntenseperColl_%d%d.npy %(SERIES,DATANAME)
  List of arrays, each array contains the intensities (varying with gray scale) in one wavelength sample point.
    
- Sin_param_%d%d.npy %(SERIES,DATANAME)
  Parameters for sine fitting on intensity vs grayscale plots, per wavelength.
  Sin_param[ i ]=[A, B, C, D]; where A * sin(B * x + C) + D; x is grayscale and i changes with wavelength.

- RulerRef_%d%d.npy %(SERIES,DATANAME)
  Target intensity value calculated.

- Initial_correction_%d%d.npy %(SERIES,DATANAME)
  Pixel collumn array containing the inital correction applied to the SLM.

- InLoopCorrPix_%d%d%d.npy %(SERIES,DATANAME,LoopCounter)
  Correction applied to the SLM on each loop.

- InLoopSpec_%d%d%d.npy %(SERIES,DATANAME,LoopCounter)
  Spectrum collected on each loop (OSA resolution).

- OSAinPix_%d%d%d.npy %(SERIES,DATANAME,LoopCounter)
  Spectrum collected on each loop (SLM resolution).
