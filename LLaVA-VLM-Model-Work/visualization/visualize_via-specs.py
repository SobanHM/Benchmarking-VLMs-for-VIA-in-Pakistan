# evaluation/visualize_via_specs.py

import pandas as pd
import matplotlib.pyplot as plt

via_specs_report_file = r"C:\Users\soban\PycharmProjects\LLaVA\analysis\via_specs_results.csv"
df = pd.read_csv(via_specs_report_file)


plt.scatter(df["spatial_SPECS"], df["hazard_SPECS"])
plt.xlabel("Spatial SPECS")
plt.ylabel("Hazard SPECS")
plt.title("Navigation vs Safety")
plt.show()
