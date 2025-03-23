from evidently.report import Report
from evidently.metrics import DataDriftTable

def generate_data_drift_report(current_data, reference_data):
    data_drift_report = Report(metrics=[DataDriftTable()])
    data_drift_report.run(
        current_data=current_data,
        reference_data=reference_data
    )
    data_drift_report.save_html("results/drift_report.html")
