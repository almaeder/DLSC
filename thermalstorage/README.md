## Problem Setting
This subproject contains multiple tasks about a thermal energy storage model. We model a device used for storing energy in a cyclic fashion with every cycle containing the following steps: Charging, idle, discharging and idle. The energy storage is happening through a interaction between a solid and a liquid phase.

## Mathematical Model
The main quantities to model are the solid and fluid temperatures. The system is described by the following reaction-convection-diffusion equations:

$$\varepsilon \rho_{f} C_{f} \frac{\partial T_{f}}{\partial t}+\varepsilon \rho_{f} C_{f} u_{f}(t) \frac{\partial T_{f}}{\partial x}=\lambda_{f} \frac{\partial^{2} T_{f}}{\partial x^{2}}-h_{v}\left(T_{f}-T_{s}\right) \quad x \in[0, L], t \in[0, T]$$

$$(1-\varepsilon) \rho_{s} C_{s} \frac{\partial T_{s}}{\partial t}=\lambda_{s} \frac{\partial^{2} T_{s}}{\partial x^{2}}+h_{v}\left(T_{f}-T_{s}\right) \quad x \in[0, L], t \in[0, T]$$

Where $\rho$ is the density of the phases, $C$ is the specific heat capacity, $\lambda$ is the diffusivity, $\epsilon$ is the solid porosity, $u_{f}$ is the fluid velocity, and $h_{v}$ is the heat exchange rate coefficient.

Further, suitable initial and boundary conditions are needed:

$$T_{f}(x, t=0)=T_{s}(x, t=0)=T_{0},\quad x \in[0, L] $$

$$\left.\frac{\partial T_{s}(x, t)}{\partial x}\right|_{x=0}=\left.\frac{\partial T_{s}(x, t)}{\partial x}\right|_{x=L}=0,\quad t \in[0, T]$$

- Charging State

$$T_{f}(0, t)=T_{h o t},\left.\quad \frac{\partial T_{f}(x, t)}{\partial x}\right|_{x=L}=0, \quad t \in[0, T]$$


- Discharging State:


$$\left.\frac{\partial T_{f}(x, t)}{\partial x}\right|_{x=0}=0, \quad T_{f}(L, t)=T_{\text {cold }}, \quad t \in[0, T]$$


- Idle Phase:


$$\left.\frac{\partial T_{f}(x, t)}{\partial x}\right|_{x=0}=0,\left.\quad \frac{\partial T_{f}(x, t)}{\partial x}\right|_{x=L}=0, \quad t \in[0, T]$$

## Task 1
The equations are normalized and only the system during charging is examined:

$$\frac{\partial \bar{T}_{f}}{\partial t}+U_{f} \frac{\partial \bar{T}_{f}}{\partial x}=\alpha_{f} \frac{\partial^{2} \bar{T}_{f}}{\partial x^{2}}-h_{f}\left(\bar{T}_{f}-\bar{T}_{s}\right) \quad x \in[0,1], t \in[0,1]$$

$$\frac{\partial \bar{T}_{s}}{\partial t}=\alpha_{s} \frac{\partial^{2} \bar{T}_{s}}{\partial x^{2}}+h_{s}\left(\bar{T}_{f}-\bar{T}_{s}\right) \quad x \in[0,1], t \in[0,1]$$

with

$$\bar{T}_{f}(x, t=0) =\bar{T}_{s}(x, t=0)=T_{0}, \quad x \in[0,1]$$

$$\left.\frac{\partial \bar{T}_{s}}{\partial x}\right|_{x=0} =\left.\frac{\partial \bar{T}_{s}}{\partial x}\right|_{x=1}=\left.\frac{\partial \bar{T}_{f}}{\partial x}\right|_{x=1}=0, \quad t \in[0,1]$$

$$\bar{T}_{f}(x=0, t) =\frac{T_{h o t}-T_{0}}{1+\exp (-200(t-0.25))}+T_{0}, \quad t \in[0,1]$$

and

$$\alpha_{f}=0.05 \quad h_{f}=5 \quad T_{h o t}=4 \quad U_{f}=1$$

$$\alpha_{s}=0.08 \quad h_{s}=6 \quad T_{0}=1$$

The task is to approximate the solution of the system of PDEs with a physics informed neural network.

### Results

## Task 2