For this course, I have created the following Python package: A527_package. 
Within this package, I will organize general functions for each homework assignments as to remain modular. 
Each homework will have it's own .py file in the main directory that makes use of A527_package. 
Figure output of each homework can then be found in the Figures folder. 

Homework One - JWST:
Potential, acceleration, and bisection functions found within A527_package in JWST.py. 
Output figures within Figures/homework_one
To run: git clone and open A527 then run in terminal: python homework_one.py  

Homework Two - Levenberg Marquardt Fitting
All functions for the Levenberg Marquardt algorithm are found within A527_package in levenberg_marquardt.py
Output figures within Figures/homework_two
To run: git clone and open A527 then run in terminal: python homework_two.py 
Note that in homework_two.py you must set in line 17 whether to do the fitting with a Gaussian or Lorentzian function, for Gaussian set func = "G" and for Lorentzian func = "L"
