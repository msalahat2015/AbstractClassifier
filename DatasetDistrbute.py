import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# -------------------------------
# 1. Load JSON data
# -------------------------------
JSON_PATH = "/rep/msalahat/COMP9323-ST-LLMs/COMP9319-DL/Dataset/arxiv-metadata-oai-snapshot.json"

papers = []
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        paper = json.loads(line)
        if 'categories' in paper:
            categories = paper['categories']
            if isinstance(categories, list):
                categories = " ".join(categories)
            papers.append({'category': categories})

# تحويل إلى DataFrame
df = pd.DataFrame(papers)

# -------------------------------
# 2. Split multiple categories
# -------------------------------
df = df['category'].str.split(expand=True).stack().reset_index(drop=True).to_frame(name='category')

# -------------------------------
# 3. Count papers per category
# -------------------------------
category_counts = df['category'].value_counts().sort_values(ascending=False)

# -------------------------------
# 4. Save distribution to CSV
# -------------------------------
CSV_PATH = "/rep/msalahat/COMP9323-ST-LLMs/COMP9319-DL/Assets/category_distribution.csv"
category_counts_df = category_counts.reset_index()
category_counts_df.columns = ['category', 'num_papers']
category_counts_df.to_csv(CSV_PATH, index=False)
print(f"✅ Category distribution saved to CSV: {CSV_PATH}")

# -------------------------------
# 5. Plot distribution
# -------------------------------
plt.figure(figsize=(12,6))
sns.barplot(x=category_counts.index[:20], y=category_counts.values[:20], palette="tab20")
plt.xticks(rotation=45, ha='right')
plt.xlabel("Category")
plt.ylabel("Number of Papers")
plt.title("Top 20 Categories Distribution of Research Papers")
plt.tight_layout()
plt.savefig("/rep/msalahat/COMP9323-ST-LLMs/COMP9319-DL/Assets/matplotlib/top20_paper_distribution.png")
plt.show()
