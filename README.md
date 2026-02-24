
# Model Inference Repository

## Overview
This repository provides Python scripts for running inference on rheological models, as described in *Martin Lardy, Sham Tlili, Simon Gsell, Inferring viscoplastic models from velocity fields: A physics-informed neural network approach, Journal of Non-Newtonian Fluid Mechanics*(https://www.sciencedirect.com/science/article/pii/S0377025725001302). 
In this github version it supports three models:
- **Carreau**
- **Herschel-Bulkley**
- **Papanastasiou**
- **Model selection between viscoplastic models**

But the PINNs is tuned to be adaptable to any viscoplastic model.
The dataset includes numerical simulations of flows (Herschel-Bulkley, Carreau, or Papanastasiou) across different geometries and parameter sets.

---

## Features
- **Easy-to-use scripts**: Just provide a data file (specified in each `.py` script).
- **Customizable sampling**: Configure the sampling area and point probability (lines 108–153 in the scripts).

---

## Updates
We refined the **model selection loss equations** to improve computational efficiency:
- Each viscosity is now weighted by **βᵢ** and included in the total loss (Equation 19).
- Updated Equations (19) and (20) from the paper:

  ![Updated Equation 19](https://github.com/martinLARD/Pinn/blob/main/eq18.png)
  and
  ![Updated Equation 20](https://github.com/martinLARD/Pinn/blob/main/eq20.png)

  where:
  ![Weighted Viscosity](https://github.com/martinLARD/Pinn/blob/main/eq3.png)

**Note**: These changes do not affect the results of the paper.

---

## Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/martinLARD/Pinn.git

## License
 
Distributed under the terms of the BSD 3 license.
