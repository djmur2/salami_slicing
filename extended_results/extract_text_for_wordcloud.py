import os
import pandas as pd

# Read paper IDs
papers = pd.read_csv('papers_for_wordcloud.csv')
text_path = r'C:\\Users\\s14718\\Desktop\\salami_modularized\\data\\processed\\clean_text\\'

# Extract text for each paper
texts = []
for paper in papers['paper']:
    # Add w prefix for filename
    paper_str = f'w{int(paper):05d}'
    file_path = os.path.join(text_path, f'{paper_str}.txt')
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            texts.append({'paper': paper, 'text': text[:5000]})

# Save as CSV
df = pd.DataFrame(texts)
df.to_csv('abstracts_for_wordcloud.csv', index=False)
print(f'Extracted text for {len(texts)} papers')
