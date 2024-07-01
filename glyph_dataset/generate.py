from preprocess import draw_centers_with_text
import pandas as pd
import os
import numpy as np
from huggingface_hub import login
from huggingface_hub import HfApi
from tqdm import tqdm   

word_df = pd.read_csv('word/word.csv', usecols=['word'])
word_df = word_df[pd.notnull(word_df)]

# DataFrame 출력
word_list = word_df['word'].tolist()
word_list = word_list[:10000]
new_word_list = []
for word in word_list:
    new_word_list.append(word)
    new_word_list.append(word)
    new_word_list.append(word.upper())
    word = word[0].upper()+word[1:]
    new_word_list.append(word)
    
        
files = os.listdir("./font")
files = [file for file in files if (file.endswith('.ttf') or file.endswith('.otf'))]
#check glyph_dataset.npy exists
glyph_datasets = draw_centers_with_text(new_word_list, files)
if not os.path.exists('./glyph.dataset.npy'):
    np.save('glyph.dataset',glyph_datasets)
    with open('label.txt', 'w') as f:
        for word in new_word_list:
            for _ in range(5):
                f.write("%s\n" % word)
    login()
    api = HfApi()
    api.upload_file(
        path_or_fileobj="./glyph.dataset.npy",
        path_in_repo="glyph.dataset.npy",
        repo_id="ghlee420/glyph-word",
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj="./label.txt",
        path_in_repo="label.txt",
        repo_id="ghlee420/glyph-word",
        repo_type="dataset",
    )            
    

