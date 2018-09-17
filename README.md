# Description
HiPPFSTA is a **h**igh-**p**recision, high-**p**erformance **f**licker **s**pectroscopy contour **t**racking **a**lgorithm with the following properties:
* It is capable of achieving nanometer localization accuracy in phase contrast light-microscopy images given correct image material (see publication).
* It is written in Python and OpenCL and capable of levaraging the GPU for accelerated image processing (although it will run on CPU as well provided an OpenCL driver).

# References
* HiPPFSTA was developed by Michael Mell as part of his research work during his PhD thesis, which was conducted at the University Complutense of Madrid under the supervision of Prof. Francisco Monroy.
* If you use HiPPFSTA in your research, please cite:  **[TODO: ADD REFERENCE, ONCE PUBLISHED]**

# Getting the code and images
This repository uses Git LFS to manage the image files that are necessary for running the examples and code-tests. To obtain these you need to use the Git client to checkout the respository by following these instructions: 
Instructions:
* Download Git from gitscm.com by selecting the button "Download 2.XX for Windows"
* Install it with default configuration (unless you know what you are doing).
* Pull this repository:
    ```
    > git clone https://gitlab.com/michaelmell/cellcontourtracker.git
    ```
* Enter GitLab credentials  **[TODO: REMOVE THIS, WHEN RESPOSITORY IS PUBLIC]**
* Git will download the code along with the image necessary for running the examples
* If you have problems checking out this repository with Git LFS, please refer to the documentation of your Git client.

**Important**: Please note that the tar-ball/zip-file provided by GitLab.com for download on this page (above) does not include the Git LFS image files.

# OpenCL runtimes (installation instructions for Windows)
To run the HiPPFSTA you need a functioning OpenCL runtime supporting OpenCL 1.2. OpenCL runtimes are currently not included in major operating systems (except for macOS). There are several OpenCL runtimes provided from different hardware vendors. In particular, the runtimes from AMD, Intel and Nvidia have been tested with the HiPPFSTA.

### Intel runtime
Modern Intel CPUs include integrated GPUs capable of running OpenCL. The corresponding runtime is included with Graphics driver, which can be downloaded from Intel. Furthermore, there exists an OpenCL runtime for Intel CPUs, which is included with the Intel SDK.

* Drivers for the integrated GPUs can be download here: https://software.intel.com/en-us/articles/opencl-drivers#graph-win
* The CPU runtime can be found here (requires registration): https://software.intel.com/en-us/articles/opencl-drivers#cpu-section

### AMD runtime
* The AMD runtime can be found here: https://www.amd.com/en/support
* It is capable of running on AMD GPUs as well as AMD and Intel CPUs (_not_ Intel GPUs).

### NVIDIA  runtime
* NVIDIA GPUs support OpenCL as well: https://developer.nvidia.com/opencl
* OpenCL support is included in the NVIDIA GPU drivers: www.nvidia.com/drivers

# Setup python environment (instructions for Windows)
* Download and install the newest version of Anaconda3-5.X with Python3 support from ContinuumAnalytics: https://repo.continuum.io/archive/
* Start the anaconda prompt: open the Windows Start-Menu -> type "anaconda prompt" -> hit Enter
* Install required python version:
    ```
    > conda install -c anaconda python=3.6.6
    > conda install -c anaconda ipython=6.1
    ```
* Install PyOpenCL:
Download PyOpenCL for OpenCL 1.2  https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl
pyopencl-2018.1.1+cl12-cp36-cp36m-win_amd64.whl
    ```
	> conda install -c conda-forge pyopencl=2018.1.1
    ```
	 
* Install dependencies:
    ```
	> conda install -c conda-forge ipdb=0.11
	> conda install -c anaconda pillow=5.2.0
	> conda install -c anaconda icu=58.2 ipykernel=4.6.1 libtiff libpng jpeg qt
    ```

# Generating figures 10A and 10B of the corresponding publication
The repository includes sample data from POPC GUV and RBC measurements to generate figures 10A and 10B of the corresponding publication (see section **Getting the code and images**), which can be generated using the following the instructions. To limit data-usage, these are _reduced_ datasets, which is why the resulting spectra are _noisier_ than in the published figure.
The scripts are written in Matlab code and were tested with Matlab 2016b, but older versions >2014a should work too. The scripts were also tested with the free and open-source GNU Octave (https://www.gnu.org/software/octave/), but performance is worse than when using Matlab.
In the following `$GITREPOPATH$` refers to the path were you cloned the Git repository:

* Run python contour tracking for GUV dataset. In the anaconda prompt enter:
    ```
	> cd $GITREPOPATH$\examples\guv\tracking
	> ipython
	> %run run_tracking.py
    ```
**Note**: When running the program in interactive mode (default setting used in this repository), the program will display a window showing the first frame with its tracking (black line at the contour of the cell). Please close this window to continue the processing of the dataset.
	
* Run python contour tracking for RBC dataset. In the anaconda prompt enter:
    ```
	> cd $GITREPOPATH$\examples\rbc\tracking
	> ipython
	> %run run_tracking.py
    ```
	
* Run Matlab contour tracking for GUV dataset. In Matlab prompt enter:
    ```
	> cd $GITREPOPATH$\examples\guv\tracking
	> run_matlab_tracking_002
    ```
	
* Run Matlab contour tracking for RBC dataset. In Matlab prompt enter:
    ```
	> cd $GITREPOPATH$\examples\rbc\tracking
	> run_matlab_tracking_002
    ```

* Generate comparison figure 10a for the GUV dataset. In Matlab prompt enter (you will have to wait for each script to finish before running the next one):
    ```
    > cd $GITREPOPATH$\examples\publication_figures\figure_10a_guv
    > calculate_transforms_fourier_for_algorithm_comparison
    > popcGuvSpectrumAlgorithmComparison_1_v000
    ```
		
* Generate comparison figure 10b for the RBC dataset. In Matlab prompt enter (you will have to wait for each script to finish before running the next one):
    ```
    > cd $GITREPOPATH$\examples\publication_figures\figure_10b_rbc
    > calculate_transforms_fourier_for_algorithm_comparison
    > rbcHealthySpectrumAlgorithmComparison_1_v000
    ```
