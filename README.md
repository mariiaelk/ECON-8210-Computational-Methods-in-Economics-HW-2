# ECON-8210-Computational-Methods-in-Economics-HW-2

This project contains solutions to HW 2 of ECON 8210 Computational Methods in Economics course:
1. Matlab and Dynare codes
2. Matlab files with solution obtained using generalized endogenous grid method and neural network
3. PDF file with explaination of the approach taken in each of the problems and results obtained

## Files
| File Name            | Description                              |
|----------------------|------------------------------------------|
| `EGM.m`              | Matlab code which solves stochastic neoclassical growth model using generalized endogenous grid method. |
| `Q1.m`               | Matlab code which solves stochastic neoclassical growth model using approximation with Chebychev polynomials (based on code from course materials). |
| `Q2.m`               | Matlab code which solves stochastic neoclassical growth model using approzimation with finite elements. |
| `Q3.m`               | Matlab code which solves stochastic neoclassical growth model using third order perturbation. |
| `Q4.m`               | Matlab code which solves stochastic neoclassical growth model using neural network. |
| `expLayer.m`         | Matlab code for exponential layer used to solve the model using neural network. |
| `rando.m`            | Matlab function which generates a random variable given a distribution vector (from course materials). |
| `tauchen.m`          | Matlab function which uses Tauchen method to discretize an AR(1) process (adapted version of analogous Julia code from QuantEcon). |
| `Q3.mod`             | Dynare code with stochastic neoclassical growth model. |
| `EGM_sol.mat`        | Matlab file with solution of stochastic neoclassical growth model obtained using generalized endogenous grid method. |
| `EGM_simul.mat`      | Matlab file with simulation of stochastic neoclassical growth model using solution obtained using generalized endogenous grid method. |
| `EGM_IRF.mat`        | Matlab file with impulse response functions for stochastic neoclassical growth model obtained using solution using generalized endogenous grid method. |
| `Q4_net.mat`         | Matlab file with solution of stochastic neoclassical growth model obtained using neural network. |
| `HW2_Elk_sol.pdf`    | PDF file which contains description of the approach taken in each problem and results obtained. |
