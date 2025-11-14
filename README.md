# Wind Turbine Power Prediction System


[![Streamlit App](https://img.shields.io/badge/Streamlit-App-blue.svg)](https://your-streamlit-app-url.streamlit.app/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)


A machine learning-based system to forecast the power output of wind turbines using SCADA data. This project trains regression models on historical wind speed, direction, and power readings to enable real-time predictions, supporting renewable energy planning and grid integration.


## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [Performance Metrics](#performance-metrics)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)


## Overview
Wind turbines generate variable power based on environmental factors like wind speed and direction. Accurate forecasting is essential for efficient energy management, maintenance scheduling, and reducing reliance on fossil fuels.


This project uses supervised machine learning to predict active power output (in kW) from input parameters. We process data from a real Turkish wind farm, engineer features for better model performance, and deploy an interactive web app for end-users.


Key goals:
- Achieve high prediction accuracy (R² > 0.95 on test data).
- Provide an intuitive interface for what-if scenarios.
- Demonstrate scalability for multi-turbine systems.


## Features
- **Input Handling**: Sliders for wind speed (0-25 m/s) and direction (0-360°).
- **Prediction Engine**: XGBoost regressor (outperforms baseline Linear Regression).
- **Visualizations**: Bar chart comparing predicted vs. theoretical power output.
- **Status Indicators**: Colour-coded alerts (Low <300 kW, Normal 300-1500 kW, High >1500 kW).
- **Model Insights**: Sidebar with feature lists and evaluation metrics.
- **Sample Data**: Built-in table of example predictions.


## Dataset
- **Source**: [Wind Turbine Scada Dataset](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset) (Kaggle, 2018-2019 data from a Turkish turbine).
- **Format**: CSV with 50,530 rows, 10-minute intervals.
- **Columns**:
  - `Date/Time`: Timestamp.
  - `Wind Speed (m/s)`: Primary input (range: 0-25 m/s).
  - `Wind Direction (°)`: Secondary input (range: 0-360°).
  - `LV ActivePower (kW)`: Target variable (range: 0-3618 kW).
  - `Theoretical_Power_Curve (KWh)`: Baseline for comparisons.
- **Preprocessing**: Time-based split (80/20 train/test), feature engineering (e.g., wind speed cubed for aerodynamic scaling).


## Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/wind-turbine-prediction.git
   cd wind-turbine-prediction
   ```
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   Core libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `streamlit`, `joblib`, `matplotlib`.


4. Download the dataset:
   - Place `WindTurbine_Data.csv` in the `data/` folder (or load via Kaggle API in the notebook).


## Usage
### Local Training
Run the Jupyter notebook for data exploration and model training:
```
jupyter notebook notebooks/training_notebook.ipynb
```
- Loads and cleans data.
- Engineers features (e.g., `Wind_Speed_Cubed`, `Dir_Sin`).
- Trains/evaluates models.
- Saves artifacts to `models/`.


### Local App Demo
Launch the Streamlit app:
```
streamlit run app/streamlit_app.py
```
- Enter wind parameters via sliders.
- Click "Predict Power Output" for results.
- View comparisons and metrics.


Example: At 10 m/s wind speed and 180° direction, expect ~2471 kW (HIGH status).


## Project Structure
```
wind-turbine-prediction/
├── data/
│   └── WindTurbine_Data.csv          # Raw dataset
├── notebooks/
│   └── training_notebook.ipynb       # EDA, training, evaluation
├── models/
│   ├── wind_power_model.pkl          # Trained XGBoost model
│   └── model_metadata.pkl            # Features and metrics
├── app/
│   └── streamlit_app.py              # Interactive UI
├── requirements.txt                  # Dependencies
├── README.md                         # This file
└── LICENSE                           # MIT License
```


## Model Training
- **Algorithms**: Linear Regression (baseline), XGBoost Regressor (primary).
- **Features**: `Wind Speed (m/s)`, `Wind_Speed_Cubed`, `Dir_Sin`, `Dir_Cos`, `Rolling_Speed`.
- **Evaluation**: Time-series split; metrics computed on holdout set.
- **Selection**: XGBoost chosen for lowest RMSE (~250 kW).


See `notebooks/training_notebook.ipynb` for full code and plots (e.g., actual vs. predicted scatter).


## Deployment
- **Platform**: Streamlit Cloud (free tier).
- **Live Demo**: [Try the App](https://your-streamlit-app-url.streamlit.app/).
- **Customization**: Edit `app/streamlit_app.py` for new features (e.g., batch predictions).
- **Scaling**: For production, integrate with Flask/Docker and host on AWS/Heroku.


## Performance Metrics
| Model            | RMSE (kW) | MAE (kW) | R² Score |
|------------------|-----------|----------|----------|
| Linear Regression| 867.08   | 724.23  | 0.583   |
| XGBoost         | 594.97   | 275.80  | 0.803   |


- Tested on 20% holdout (10,106 samples).
- XGBoost reduces error by ~31% over baseline.


## Contributing
Contributions welcome! Please:
1. Fork the repo and create a feature branch (`git checkout -b feature/amazing-feature`).
2. Commit changes (`git commit -m 'Add amazing feature'`).
3. Push to the branch (`git push origin feature/amazing-feature`).
4. Open a Pull Request.


For major changes, open an issue first to discuss. Use conventional commits (e.g., `feat: add new metric`).


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments
- Dataset credit: Berker İsen (Kaggle).
- Inspiration: UCI Energy Efficiency Dataset adaptations.
- Tools: Streamlit team for seamless deployment.


---


*Built with ❤️ for sustainable energy. Last updated: November 15, 2025.*
