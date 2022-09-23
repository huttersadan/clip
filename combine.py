import json

import os

img_path = 'thumbnail/test'

test_dirs_list = os.listdir(img_path)
test_all_json = {}

for key in test_dirs_list:
    with open (img_path+'/'+key+'/'+'profile.json','r',encoding = 'utf-8') as file:
        params = json.load(file)
    test_all_json[key] = params
with open('test_all.json','w',encoding='utf-8') as file:
    json.dump(test_all_json, file, ensure_ascii=False)