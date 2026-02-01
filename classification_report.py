import pandas as pd

# MLP classification report
mlp_report = """
astro-ph     0.9418    0.9394    0.9406      3531
cond-mat     0.8718    0.8868    0.8792      3435
cs           0.7925    0.8008    0.7966      1049
gr-qc        0.6908    0.7174    0.7038       651
hep-ex       0.8682    0.7957    0.8304       323
hep-lat      0.8492    0.7308    0.7855       208
hep-ph       0.8010    0.8522    0.8258      1441
hep-th       0.7928    0.7511    0.7714      1177
math         0.8928    0.9103    0.9015      4125
math-ph      0.3333    0.3388    0.3361       425
nlin         0.4802    0.3333    0.3935       255
nucl-ex      0.7500    0.5967    0.6646       181
nucl-th      0.7039    0.7384    0.7207       367
physics      0.5969    0.5988    0.5978      1296
q-bio        0.6435    0.5968    0.6192       248
q-fin        0.7473    0.7083    0.7273        96
quant-ph     0.7495    0.7474    0.7484      1049
stat         0.5439    0.4336    0.4825       143
accuracy     0.8196
macro avg    0.7250    0.6931    0.7069     20000
weighted avg 0.8178    0.8196    0.8182     20000
"""

# CNN classification report
cnn_report = """
astro-ph     0.9255    0.9422    0.9338      3531
cond-mat     0.8716    0.8771    0.8743      3435
cs           0.7799    0.7636    0.7717      1049
gr-qc        0.6396    0.6651    0.6521       651
hep-ex       0.8951    0.7399    0.8102       323
hep-lat      0.9396    0.6731    0.7843       208
hep-ph       0.8071    0.8307    0.8187      1441
hep-th       0.7577    0.7519    0.7548      1177
math         0.8942    0.9052    0.8997      4125
math-ph      0.3138    0.3529    0.3322       425
nlin         0.4737    0.3882    0.4267       255
nucl-ex      0.6960    0.4807    0.5686       181
nucl-th      0.6504    0.7248    0.6856       367
physics      0.5475    0.5826    0.5645      1296
q-bio        0.6125    0.5927    0.6025       248
q-fin        0.8000    0.6250    0.7018        96
quant-ph     0.7716    0.6797    0.7228      1049
stat         0.4074    0.4615    0.4328       143
accuracy     0.8056
macro avg    0.7102    0.6687    0.6854     20000
weighted avg 0.8074    0.8056    0.8056     20000
"""

def parse_report(report_str):
    data = []
    for line in report_str.strip().split("\n"):
        parts = line.split()
        if not parts:
            continue
        # accuracy
        if parts[0] == "accuracy":
            data.append(["accuracy", float(parts[1]), None, None, None])
        # macro avg / weighted avg
        elif "avg" in parts[0] or (len(parts) > 1 and "avg" in parts[1]):
            category = " ".join(parts[:2]) if "avg" in parts[1] else parts[0]
            precision = float(parts[-4])
            recall = float(parts[-3])
            f1 = float(parts[-2])
            support = int(parts[-1])
            data.append([category, precision, recall, f1, support])
        # الفئات العادية
        else:
            category = parts[0]
            precision = float(parts[1])
            recall = float(parts[2])
            f1 = float(parts[3])
            support = int(parts[4])
            data.append([category, precision, recall, f1, support])
    return pd.DataFrame(data, columns=["category", "precision", "recall", "f1", "support"])

# تحويل التقارير إلى DataFrame
mlp_df = parse_report(mlp_report)
cnn_df = parse_report(cnn_report)

# دمج البيانات
merged_df = mlp_df.merge(cnn_df, on="category", suffixes=("_mlp", "_cnn"))

# حفظ CSV
merged_df.to_csv("/rep/msalahat/COMP9323-ST-LLMs/COMP9319-DL/Assets/classification_reports_merged.csv", index=False)
print("✅ CSV تم إنشاؤه بنجاح: classification_reports_merged.csv")
