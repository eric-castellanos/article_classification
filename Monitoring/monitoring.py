import pandas as pd
import numpy as np 

from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset,NoTargetPerformanceTestPreset
from evidently.tests import *

class Monitoring():

    def __init__(self, reference_path, current_path):
        self.reference = pd.read_csv(reference_path).drop(columns=['Title', 'Description'], axis=1)
        self.current = pd.read_csv(current_path).drop(columns=['Title', 'Description'], axis=1) 

    def generate_report(self):
        report = Report(metrics=[
            DataDriftPreset(), 
        ])

        report.run(reference_data=self.reference, current_data=self.current)
        report.save_html("monitoring.html")

if __name__ == "__main__":
    monitoring = Monitoring(reference_path='../data/processed/train.csv', current_path='../data/processed/test.csv')
    monitoring.generate_report()

    