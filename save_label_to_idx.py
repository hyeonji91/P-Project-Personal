import pandas as pd
import numpy as np

num_of_video = 3000

df = pd.read_excel('F:/HyeonjiKim/Downloads/signLanguageDataset/KETI-2017-SL-Annotation-v2_1.xlsx')
df.sort_values(by = '번호', ascending=True, inplace=True)
label_list = df['한국어'].tolist()
label_list = np.array(label_list[:num_of_video])

# 숫자로 labeling, ex '고압전선' : 323, 실행할때마다 순서 섞임
label_to_idx = {label: idx for idx, label in enumerate(set(label_list))}

### label to idx 저장 ###
import pickle
with open('data/label_to_idx.pickle', 'wb') as f:
    pickle.dump(label_to_idx, f, pickle.HIGHEST_PROTOCOL)
print(label_to_idx)
exit()