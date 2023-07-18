This repository and the folders therein contain the code and data required to reproduce the figures that illustrate data and results in the manuscript by Petousakis et al., 2023.

The reproduction of the figures requires Anaconda Python to be installed on the system, and the repository files to be downloaded.
Anaconda Python can be found and downloaded from the following URL:  https://www.anaconda.com/products/distribution

Once Anaconda Python is installed on the system:

For Windows operating systems:
	Open an Anaconda Prompt and navigate to the folder containing the data and code.
	Execute the command "conda env create -f environment.yml", which will create a new environment with all requisite libraries installed.
	Activate the newly created environment by executing "conda activate petousakis2023"
	Run the scripts titled "01_..." to "19_..." to produce the figures from the manuscript. Each figure that is generated contains a title indicating the figure or panel being reproduced.

For Unix-based operating systems:
	Open a terminal window from within the folder containing the data and code.
	Execute the command "conda env create -f environment.yml", which will create a new environment with all requisite libraries installed.
	Activate the newly created environment by executing "conda activate petousakis2023"
	Run the scripts titled "01_..." to "19_..." to produce the figures from the manuscript. Each figure that is generated contains a title indicating the figure or panel being reproduced.

If you have any issues or questions, please contact me at kepetousakis[at]gmail.com.