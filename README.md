# What is it?

Program for scientific research, which allows to simulate the phenomenon of [self-focusing](https://en.wikipedia.org/wiki/Self-focusing) of different laser beams (including [Gaussian](https://en.wikipedia.org/wiki/Gaussian_beam), ring and [vortex beams](https://en.wikipedia.org/wiki/Optical_vortex)) in condensed media in different approximations taking into account noise.

### [**>>> wiki <<<**](https://github.com/VasilyevEvgeny/self-focusing/wiki)
  
<p align="center">
<img src=resources/demonstration.gif>
</p>

# Requirements

* Python 3

![python](resources/python.jpg)

* pdflatex

![latex](resources/latex.png)

# Installation

* **Windows**:
```pwsh
virtualenv venv
cd venv/Scripts
activate
pip install -r <path_to_project>/requirements.txt
```

* **Linux**
```bash
virtualenv venv -p python3
cd venv
source activate
pip install -r <path_to_project>/requirements.txt
```

# [Mathematical model](math_model/math_model.pdf)

A mathematical model of beams self-focusing was obtained using the [approximation of slowly varying amplitude](https://en.wikipedia.org/wiki/Slowly_varying_envelope_approximation) and the terms responsible for [diffraction](https://en.wikipedia.org/wiki/Diffraction) and instantaneous [Kerr effect](https://en.wikipedia.org/wiki/Kerr_effect) are included. The model can be used to consider three-dimensional beams both in the axisymmetric approximation, and with both transverse spatial coordinates including ring beams with a phase singularity on the optical axis - the so-called [optical vortices](https://en.wikipedia.org/wiki/Optical_vortex). The possibility of considering ring beams without phase singularity, as well as [Gaussian beams](https://en.wikipedia.org/wiki/Gaussian_beam), is supported. Implemented accounting for complex noise in the initial condition. In addition, two-dimensional beams are also considered.
