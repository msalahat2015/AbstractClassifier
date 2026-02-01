import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# بيانات الـ 20 الأعلى
data = {
    "category": [
        "cs.LG","hep-ph","hep-th","cs.CV","quant-ph","cs.AI","gr-qc","astro-ph",
        "cond-mat.mtrl-sci","cond-mat.mes-hall","cs.CL","math.MP","math-ph","cond-mat.str-el",
        "cond-mat.stat-mech","math.CO","astro-ph.CO","stat.ML","astro-ph.GA","math.AP"
    ],
    "num_papers": [
        242508,191810,177961,173824,170398,151883,117690,105380,104154,98338,
        97377,86880,86880,80258,79055,74707,74381,74330,73745,70446
    ]
}

df = pd.DataFrame(data)

plt.figure(figsize=(14,7))
bars = sns.barplot(x='category', y='num_papers', data=df, palette="tab20")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Number of Papers")
plt.xlabel("Category")
plt.title("Top 20 Categories Distribution of Research Papers")

# إضافة تسميات فوق كل الأعمدة
for i, row in df.iterrows():
    bars.text(i, row['num_papers'] + 2000, f"{row['num_papers']:,}", 
              color='black', ha='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig("/rep/msalahat/COMP9323-ST-LLMs/COMP9319-DL/Assets/matplotlib/top20_paper_distribution_all_labels.png")
plt.show()
