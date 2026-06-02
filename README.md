# Causal inference for quantifying chemical–dynamical pathways controlling tropical middle stratospheric ozone variability

This repository is part of the manuscript published in Atmospheric Chemistry and Physics (ACP) and presents a code for causal inference used for quantifying dominant chemical–dynamical pathways that control ozone variability in the tropical middle stratosphere. 


> Galytska, E., Hassler, B., Arosio, C., Chipperfield. M. P., Dhomse S. S., Dube, K., Feng, W., Iglesias-Suarez, F., Runge, J. (2026). Causal inference for quantifying chemical–dynamical pathways controlling tropical middle stratospheric ozone variability. [Link to publication](https://doi.org/10.21203/rs.3.rs-6426983/v2).


Author: Evgenia Galytska, [egalytska@iup.physik.uni-bremen.de](mailto:egalytska@iup.physik.uni-bremen.de)

## Repository Content

- `.py scripts`: Python helper functions;
- Jupyter Notebooks:
  - `Data_preprocessing.ipynb`
  - `Motivation_Fig_1.ipynb`
  - `Toymodel_Fig_3.ipynb`
  - `Causal_inference_Fig_4_5_6_7.ipynb` 
- `environment.yml`:  environment needed to reproduce the results;
- `list_of_figures.txt`: list of Figures linked to a specific Jupyter Notebook.

# I. Preparation for data analysis

### Step 1. Download the repository. 
Includes the code needed to reproduce the published results. 

```
git clone https://github.com/EyringMLClimateGroup/galytska26acp_CausalInferenceForQuantOzoneVariability
cd galytska26acp_CausalInferenceForQuantOzoneVariability
```
### Step 2.Create the environment from the environment.yml file and activate it.

```
conda env create --name my_env --file environment.yml
conda activate my_env
```
In case the environment file environment.yml is not working (most likely because some of the dependencies are not available anymore), we suggest creating the environment with the following key dependencies:

```
conda create -n my_env python=3.9 numpy matplotlib scipy scikit-learn pandas seaborn xarray
```

Note, in that case for causal inference you need to install Tigramite package. Either keep your environment activated and proceed with Step 3, or create a separate environment for causal inference.

### Step 3.Install Tigramite.

To install Tigramite (https://doi.org/10.5281/zenodo.7747255) follow the [official GitHub repository](https://github.com/jakobrunge/tigramite) for the installation instructions. It is the User's responsibility to install the Tigramite package. Please, follow the official [Tigramite tutorials](https://github.com/jakobrunge/tigramite/tree/master/tutorials) to get acquainted with the application of Tigramite package.


# II. Data analysis

#### [Data_preprocessing.ipynb](Data_preprocessing.ipynb)

Use this notebook to prepare data from observations and the TOMCAT Chemistry-Transport Model (CTM) simulation. 

- Transformed Eulerian mean v0.1.1 *(Serva, 2022* and *Serva et al., 2024)* from the ERA5 reanalysis is available on Zenodo (https://zenodo.org/records/7081721). 
- MLS nitrous oxide (N<sub>2</sub>O) data v5.01 *(Lambert et al., 2020)* is publicly available (https://disc.gsfc.nasa.gov) upon registration. 
- OSIRIS ozone (O<sub>3</sub>) *(Bognar et al., 2022)* and nitrogen dioxide (NO<sub>2</sub>) *(Dubé et al., 2022)* data v7.3 are available at
https://research-groups.usask.ca/osiris/data-products.php#. 
- Quasi-Biennial Oscillation (QBO) equatorial winds are provided by the Institute of Meteorology and Climate Research at the Karlsruhe Institute of Technology (KIT, see *Kerzenmacher and Braesicke 2026*) and are publicly available on Zenodo (https://zenodo.org/records/18850668). 
- Data from the TOMCAT CTM simulation is available upon request from the authors.

It is users responsibility to download necessary data. 

 1. **Cell 1**: Insert the path to the [helper_functions](helper_functions.py).
 
 2. **Cell 2**: Insert the path to OSIRIS, Aura-MLS, and ERA5 data. Also insert the path where preprocessed data should be saved.

 3. **Cell 3**:  Insert the path to the TOMCAT CTM simulation. Also insert the path where preprocessed data should be saved.


#### [Motivation_Fig_1.ipynb](Motivation_Fig_1.ipynb)

Use this notebook to reproduce scatter plots of detrended monthly mean anomalies in the tropical middle stratosphere for 2004–2021, from observations and the TOMCAT CTM simulation shown in Fig. 1.

 1. **Cell 1**: Insert the path to the [helper_functions](helper_functions.py).
 2. **Cell 2**: Provide the path to `base_in`, where preprocessed data is stored.
 

#### [Toymodel_Fig_3.ipynb](Toymodel_Fig_3.ipynb)

This notebook guides the user through the process of causal justification and validation through a toy model shown in Fig. 3.

 1. **Cell 1**: Insert the path to the [helper_functions](helper_functions.py).


#### [Causal_inference_Fig_4_5_6_7.ipynb](Causal_inference_Fig_4_5_6_7.ipynb)
This notebooks provides step-by-step guide on calculation of direct path coefficients (Fig. 4), sensitivity tests for observations (Fig. 5), regime-oriented direct causal effects (Fig. 6), and total causal effects across different time lags (Fig. 7). 

 1. **Cell 1**: Insert the path to the [helper_functions](helper_functions.py).
 2. **Cell 2**: Select the Figure to be plotted.  
 3. **Cell 3**: Provide the path to `base_in`, where preprocessed data is stored.

# III. Results

Running Jupyter notebooks will reproduce the Figures published in the manuscript. Please see `list_of_figures.txt` to locate the notebook needed to reproduce each specific figure. 

## References
1. Serva, F.: Transformed Eulerian mean data from the ERA5 reanalysis (monthly means), https://doi.org/10.5281/zenodo.7081721, 2022
2. Serva, F., Christiansen, B., Davini, P., von Hardenberg, J., van den Oord, G., Reerink, T. J., Wyser, K., and Yang, S.: Changes in Stratospheric Dynamics Simulated by the EC-Earth Model From CMIP5 to CMIP6, Journal of Advances in Modeling Earth Systems, 16,
e2023MS003 756, https://doi.org/10.1029/2023MS003756, e2023MS003756 2023MS003756, 2024
3. Lambert, A., Livesey, N., Read, W., and Fuller, R.: MLS/Aura Level 3 Monthly Binned Nitrous Oxide (N2O) Mixing Ratio on Assorted Grids V005, https://disc.gsfc.nasa.gov/datasets/ML2N2O_003/summary, https://doi.org/10.5067/Aura/MLS/DATA/3545, accessed: [05.03.2025], 2020.
4.Bognar, K., Tegtmeier, S., Bourassa, A., Roth, C., Warnock, T., Zawada, D., and Degenstein, D.: Stratospheric ozone trends for 1984–2021 in the SAGE II–OSIRIS–SAGE III/ISS composite dataset, Atmospheric Chemistry and Physics, 22, 9553–9569, https://doi.org/10.5194/acp-
22-9553-2022, 2022.
5. Dubé, K., Zawada, D., Bourassa, A., Degenstein, D., Randel, W., Flittner, D., Sheese, P., and Walker, K.: An improved OSIRIS NO2 profile retrieval in the upper troposphere–lower stratosphere and intercomparison with ACE-FTS and SAGE III/ISS, Atmospheric Measurement
Techniques, 15, 6163–6180, https://doi.org/10.5194/amt-15-6163-2022, 2022.
6. Kerzenmacher, T. and Braesicke, P.: QBO: monthly zonal stratospheric winds from tropical radiosonde data (mainly Singapore), https://doi.org/10.5281/zenodo.18472673, 2026.

