# Description

A software package for solving the problem of self-focusing of beams with different profiles, including vortex beams, as well as noise accounting. It is solved numerically a nonlinear wave equation for 3D-beam propagation:
<p align="center">
 <img src="https://latex.codecogs.com/gif.latex?2&space;i&space;k_0&space;\frac{\partial&space;A(\mathbf{r},z)}{\partial&space;z}&space;=&space;\Delta_\perp&space;A(\mathbf{r},z)&space;&plus;&space;\frac{2&space;i&space;k_0}{n_0}&space;n_2&space;I(\mathbf{r})&space;A(\mathbf{r},z)">
</p>
The program supports self-focusing calculation both in the axisymmetric approximation, and taking into account both spatial coordinates x and y. Depending on the approximation, the equation and the initial condition can have the following form:

|             | Full model taking into account both x and y | Axisymmetric approximation (x,y)->r |
|:-----------:|:-------------------------------------------:|:-----------------------------------:|
|Wave equation| someth                                      | smth2
