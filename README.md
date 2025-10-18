# Delay-Robust Tube Model-Predictive Control for Six-DOF Spacecraft RVD

This repository contains the implementation and validation of a delay-aware Tube Model-Predictive Control (Tube-MPC) framework developed for a six-degree-of-freedom spacecraft during rendezvous and docking (RVD).  

## Contents
- `/report` — All technical reports (PDF)
- `/src` — All simulation and controller scripts  
- `/results` — Plots, figures, and comparison tables

## Description
The project implements a robust Tube-MPC formulation capable of handling actuator delays in spacecraft control loops. The approach guarantees bounded tracking within an invariant ellipsoid while maintaining constraint satisfaction.

## Highlights
- 6-DOF nonlinear spacecraft dynamics with attitude–translation coupling  
- Discrete-time linearization and delay modeling  
- Tube-MPC controller using Lyapunov-based invariant set computation  
- Quantitative comparison against nominal delay-afflicted MPC  

## How to Run
1. Clone this repository  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

