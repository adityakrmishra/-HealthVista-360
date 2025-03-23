# System Architecture

```mermaid```
graph TD
    A[Raw Data Sources] --> B{Data Processing}
    B --> C[Clean Dataset]
    C --> D[Feature Engineering]
    D --> E[Model Training]
    E --> F[Risk Predictions]
    F --> G[SHAP Explanations]
