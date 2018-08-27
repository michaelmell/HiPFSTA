# Description
HiPFSTA is a high-precision, high-throughput contour tracking algorithm for flicker spectroscopy with the following properties:
* It is written in Python and OpenCL and capable of levaraging the GPU for accelerated image processing (although it will run on CPU as well provided an OpenCL driver).
* It is capable of achieving nanometer localization accuracy in phase contrast light-microscopy images.

# References
* HiPFSTA was developed as part of Michael Mells PhD thesis and is published here: **[TODO: ADD REFERENCE, ONCE PUBLISHED]**
* If you use HiPFSTA in your research, please cite:  **[TODO: ADD REFERENCE, ONCE PUBLISHED]**

# Install OpenCL driver
* The GPU/CPU Driver Packages for modern Intel chips with Intel Graphics are include with Windows Graphics driver (see here: https://software.intel.com/en-us/articles/opencl-drivers#win64). If your CPU does not include integrated graphics you need install the CPU-only driver found here: https://software.intel.com/en-us/articles/opencl-drivers#latest_CPU_runtime
* For any other type of graphics card or CPU (e.g. AMD or NVidia) please see the instructions of the vendor.

# Git LFS
* This repository uses Git LFS to include image files for running the examples and code-tests.
* If you have problems checking out this repository with Git LFS, please refer to the documentation of your Git client.

# Setup python environment
* Install the newest version of Anaconda Python 5.2.X from ContinuumAnalytics for your respective OS from here: https://repo.continuum.io/archive/
* Start the anaconda prompt: open Windows Start-Menu -> type "anaconda prompt" -> hit Enter
* Install python 3.7 (see https://anaconda.org/anaconda/python):
    ```sh
    $ conda install -c anaconda python (installed 3.6.6 on 2018-08-24)
    $ conda install -c anaconda python=3.6.6
    ```
	 
* Install dependencies:
    ```sh
	$ conda install -c conda-forge pyopencl=2018.1.1 (installed pyopencl-2018.1.1 on 2018-08-24)
	$ conda install -c conda-forge ipdb=0.11 (installed ipdb-0.11 on 2018-08-24)
	$ conda install -c anaconda pillow=5.2.0 (installed pillow-5.2.0 on 2018-08-24 (and updated many other packages))
    ```

# Generating figures 10A and 10B of the publication
The repository includes sample data from POPC GUV and RBC measurements to generate figures 10A and 10B of the corresponding publication (using Git LFS, see above), which are generated by following the instructions below. Due to space-limitation, these are _reduced_ datasets, which is why the resulting spectra are _noisier_ than in the published figure.
In the following `$GITREPOPATH$` refers to the path were you cloned the Git repository.

* Run python contour tracking for GUV dataset. In the anaconda prompt enter:
    ```sh
	$ cd $GITREPOPATH$\examples\guv\tracking
	$ ipython
	$ %run run_tracking.py
    ```
	
* Run python contour tracking for RBC dataset. In the anaconda prompt enter:
    ```sh
	$ cd $GITREPOPATH$\examples\rbc\tracking
	$ ipython
	$ %run run_tracking.py
    ```
	
* Run Matlab contour tracking for GUV dataset. In Matlab prompt enter:
    ```sh
	> cd $GITREPOPATH$\examples\guv\tracking
	> run_matlab_tracking_002
    ```
	
* Run Matlab contour tracking for RBC dataset. In Matlab prompt enter:
    ```sh
	> cd $GITREPOPATH$\examples\rbc\tracking
	> run_matlab_tracking_002
    ```

* Generate comparison figure 10a for the GUV dataset. In Matlab prompt enter (you will have to wait for each script to finish before running the next one):
    ```sh
    > cd $GITREPOPATH$\examples\publication_figures\figure_10a_guv
    > calculate_transforms_fourier_for_algorithm_comparison
    > popcGuvSpectrumAlgorithmComparison_1_v000
    ```
		
* Generate comparison figure 10b for the RBC dataset. In Matlab prompt enter (you will have to wait for each script to finish before running the next one):
    ```sh
    > cd $GITREPOPATH$\examples\publication_figures\figure_10b_rbc
    > calculate_transforms_fourier_for_algorithm_comparison
    > popcGuvSpectrumAlgorithmComparison_1_v000
    ```
