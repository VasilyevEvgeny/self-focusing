# Description

A software package for solving the problem of self-focusing of beams with different profiles, including vortex beams, as well as noise accounting. It is solved numerically a nonlinear wave equation for 3D-beam propagation:
<p align="center">
 <img src="https://latex.codecogs.com/gif.latex?2&space;i&space;k_0&space;\frac{\partial&space;A(\mathbf{r},z)}{\partial&space;z}&space;=&space;\Delta_\perp&space;A(\mathbf{r},z)&space;&plus;&space;\frac{2&space;i&space;k_0}{n_0}&space;n_2&space;I(\mathbf{r})&space;A(\mathbf{r},z)">
</p>
where 
The program supports self-focusing calculation both in the axisymmetric approximation, and taking into account both spatial coordinates ```x``` and ``y``. Depending on the approximation, the equation and the initial condition can have the following form:<br/>
 ```x```
 
  
   
    
     
|             | Full model taking into account both x and y | Axisymmetric approximation (x,y)->r |
|:-----------:|:-------------------------------------------:|:-----------------------------------:|
|Wave equation| <img src="https://latex.codecogs.com/gif.latex?2&space;i&space;k_0&space;\frac{\partial&space;A(x,y,z)}{\partial&space;z}&space;=&space;\biggl(\frac{\partial^2}{\partial&space;x^2}+\frac{\partial^2}{\partial&space;y^2}\biggr)&space;A(x,y,z)&space;&plus;&space;\frac{2&space;i&space;k_0}{n_0}&space;n_2&space;I(x,y)&space;A(x,y,z)"> | <img src="https://latex.codecogs.com/gif.latex?2&space;i&space;k_0&space;\frac{\partial&space;A(r,z)}{\partial&space;z}&space;=&space;\biggl(\frac{\partial^2}{\partial&space;r^2}+\frac1{r}\frac{\partial}{\partial&space;r}-\frac{m^2}{r^2}\biggr)&space;A(r,z)&space;&plus;&space;\frac{2&space;i&space;k_0}{n_0}&space;n_2&space;I(r)&space;A(r,z)">|
|Initial condition|<img src="https://latex.codecogs.com/gif.latex?A(x,y,z=0)=A_0\biggl(\frac{x^2}{x_0^2}+\frac{y^2}{y_0^2}\biggr)^{M/2}\exp\biggl\{-\frac1{2}\biggl(\frac{x^2}{x_0^2}+\frac{y^2}{y_0^2}\biggr)\biggr\}\exp\biggl\{i&space;m&space;\varphi\biggr\}">|<img src="https://latex.codecogs.com/gif.latex?A(r,z=0)=A_0\biggl(\frac{r}{r_0}\biggr)^M\exp\biggl\{-\frac{r^2}{2r_0^2}\biggr\}">|
