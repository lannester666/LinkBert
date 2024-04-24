import json
import re
import jsonlines
import argparse
import random
parser = argparse.ArgumentParser()
parser.add_argument("--train_file", type=str, default="task2_submission")
parser.add_argument("--test_file", type=str, default="task2_submission")
parser.add_argument("--validate_file", type=str, default="task2_submission")
args = parser.parse_args()

def string2_extraction(string2):
# 初始化列表
    chemicals = []
    diseases2 = []  # 避免与上面的diseases列表混淆

    # 正则表达式匹配模式
    pattern = r"\(([^,]+), ([^)]+)\)"

    # 执行匹配
    matches = re.findall(pattern, string2)
    for match in matches:
        chemicals.append(match[0].strip())
        diseases2.append(match[1].strip())
    return chemicals, diseases2
# with open('/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/examples/full_multi_gpu/llm_data/train_task2.json', 'r') as file:
#     new_data = json.load(file)
with open('/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/examples/full_multi_gpu/llm_data/task2_data/CDR_DevelopmentSet.json', 'r') as file:
    new_data = json.load(file)
with open('/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/examples/full_multi_gpu/llm_data/task2_data/CDR_TestSet.json', 'r') as file:
    new_data1 = json.load(file)
with open('/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/examples/full_multi_gpu/llm_data/task2_data/CDR_TrainingSet.json', 'r') as file:
    new_data2 = json.load(file)
new_data  = {**new_data, **new_data1, **new_data2}
out_data = []
idx = 0
import copy
negative_pair = []
for k, data in new_data.items():
    text = data['title'] + data['abstract']
    relations = data['relations']
    chemicals = []
    diseases = []
    chemicals_mesh = []
    diseases_mesh = []
    for key, value in data['chemical2id'].items():
        chemicals.append(key)
        chemicals_mesh.append(value)
    for key, value in data['disease2id'].items():
        diseases.append(key)
        diseases_mesh.append(value)
    answer_chemical_mesh = []
    answer_disease_mesh = []
    for relation in relations:
        answer_chemical_mesh.append(relation['chemical'])
        answer_disease_mesh.append(relation['disease'])
    answer_chemical = []
    answer_disease = []
    for chemical_mesh, disease_mesh in zip(answer_chemical_mesh, answer_disease_mesh):
        for i in range(len(chemicals_mesh)):
           for j in range(len(diseases_mesh)):
               if chemical_mesh == chemicals_mesh[i] and disease_mesh == diseases_mesh[j]:
                   answer_chemical.append(chemicals[i])
                   answer_disease.append(diseases[j])
    
    for chemical, disease in zip(answer_chemical, answer_disease):
        text_tmp = copy.deepcopy(text)
        text_tmp = text_tmp.replace(chemical, "@CHEMICAL$")
        text_tmp = text_tmp.replace(disease, "@DISEASE$")
        tmp = {"id": idx, "sentence": text_tmp, "label": '1'}
        out_data.append(tmp)
    for i in range(len(answer_chemical)):
        for j in range(len(answer_disease)):
            if i!=j:
                text_tmp = copy.deepcopy(text)
                text_tmp = text_tmp.replace(answer_chemical[i], "@CHEMICAL$")
                text_tmp = text_tmp.replace(answer_disease[i], "@DISEASE$")
                tmp = {"id": idx, "sentence": text_tmp, "label": "0"}
                negative_pair.append(tmp)
    idx += 1
negative_pair = random.sample(negative_pair, len(out_data))
out_data.extend(negative_pair)
with open("/home/zhangtaiyan/workspace/comp/LinkBERT/data/seqcls/cdr_hf/train.json", 'w') as file:
    for data in out_data:
        json.dumps(data)
with jsonlines.open("/home/zhangtaiyan/workspace/comp/LinkBERT/data/seqcls/cdr_hf/train.json", 'w') as writer:
    for data in out_data:
        writer.write(data)
new_data = []
with open(args.test_file, 'r') as file:
   for line in file:
        new_data.append(json.loads(line))

out_data = []
idx = 0
aux_data = []
import copy

for data in new_data:
    if data['task']!=2:
        idx += 1
        continue
    try:
        text = data['abstract']
    except:
        import pdb; pdb.set_trace()
    answer = data['ideal']['chemical, disease']
    chemicals, diseases =string2_extraction(answer)
    for i in range(len(chemicals)):
        for j in range(len(diseases)):
            if i == j:
                text_tmp = copy.deepcopy(text)
                text_tmp = text_tmp.replace(chemicals[i], "@CHEMICAL$")
                text_tmp = text_tmp.replace(diseases[j], "@DISEASE$")
                tmp = {"id": idx, "sentence": text_tmp, "label": "1"}
                tmp_aux = {"id": idx, "chemical": chemicals[i], "disease": diseases[j]}
                out_data.append(tmp)
                aux_data.append(tmp_aux)
    idx += 1
with jsonlines.open("/home/zhangtaiyan/workspace/comp/LinkBERT/data/seqcls/cdr_hf/aux_test.json", 'w') as writer:
    for data in aux_data:
        writer.write(data)
with jsonlines.open("/home/zhangtaiyan/workspace/comp/LinkBERT/data/seqcls/cdr_hf/test1.json", 'w') as writer:
    for data in out_data:
        writer.write(data)

new_data = []
with open(args.validate_file, 'r') as file:
   for line in file:
        new_data.append(json.loads(line))

out_data = []
idx = 0
aux_data = []
import copy

for data in new_data:
    if data['task']!=2:
        idx += 1
        continue
    try:
        text = data['abstract']
    except:
        import pdb; pdb.set_trace()
    answer = data['ideal']['chemical, disease']
    chemicals, diseases =string2_extraction(answer)
    for i in range(len(chemicals)):
        for j in range(len(diseases)):
            if i == j:
                text_tmp = copy.deepcopy(text)
                text_tmp = text_tmp.replace(chemicals[i], "@CHEMICAL$")
                text_tmp = text_tmp.replace(diseases[j], "@DISEASE$")
                tmp = {"id": idx, "sentence": text_tmp, "label": "1"}
                tmp_aux = {"id": idx, "chemical": chemicals[i], "disease": diseases[j]}
                out_data.append(tmp)
                aux_data.append(tmp_aux)
    idx += 1
with jsonlines.open("/home/zhangtaiyan/workspace/comp/LinkBERT/data/seqcls/cdr_hf/aux_dev.json", 'w') as writer:
    for data in aux_data:
        writer.write(data)
with jsonlines.open("/home/zhangtaiyan/workspace/comp/LinkBERT/data/seqcls/cdr_hf/dev.json", 'w') as writer:
    for data in out_data:
        writer.write(data)
