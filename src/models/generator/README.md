All things core to the generator role of a BornMachine, called BornGenerator. 
'generator.py' contains the architecture and sequential contraction algorithms to obtain univariate marginals or conditionals. 
'differential_sampling.py' contains sampling algorithms from a univariate distribution with gradient signal which is crucial for GAN-style training schemes that need to differentiate through the observed samples. For BornMachines, one samples directly from the distribution, thus to get gradientflow to the model parameters the sampling algorithm needs to allow for this. 

