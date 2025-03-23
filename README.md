# HealthVista-360

A predictive analytics system for chronic disease risk assessment.

## Setup
```bash
git clone https://github.com/yourusername/HealthVista-360.git
pip install -r requirements.txt\
```
## Directory Structure
```data/```: Contains raw and processed datasets

```notebooks/```: Jupyter notebooks for EDA and modeling

```src/```: Core Python scripts for data processing and modeling

```models/```: Saved model artifacts

Commit: "Add README.md with project documentation"

3. config/config.yaml:
```yaml
```
# config/config.yaml
data_paths:
  raw_medical: "data/raw/medical_records.csv"
  raw_lifestyle: "data/raw/lifestyle_surveys.json"
  external_pollution: "data/external/air_quality.geojson"

model_params:
  test_size: 0.2
  random_state: 42
  logistic_regression:
    C: 1.0
    max_iter: 1000

preprocessing:
  numeric_impute: "median"
  categorical_impute: "mode"
