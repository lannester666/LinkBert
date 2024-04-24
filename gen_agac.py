import json
import re
import jsonlines
import argparse
import random
from itertools import combinations
parser = argparse.ArgumentParser()
parser.add_argument("--train_file", type=str, default="task2_submission")
parser.add_argument("--validate_file", type=str, default="task2_submission")
parser.add_argument("--test_file", type=str, default="task2_submission")
args = parser.parse_args()

def extract_entity_from_output(output):
    pattern_gene = r'\(([^,]+), [^,]+, [^)]+\)'
    pattern_disease = r'\([^,]+, [^,]+, ([^)]+)\)'

    # 提取gene
    genes = re.findall(pattern_gene, output)
    # 提取disease
    diseases = re.findall(pattern_disease, output)
    pattern_function = r'\([^,]+, ([^,]+), [^)]+\)'

# 提取function
    functions = re.findall(pattern_function, output)
    return genes, diseases, functions
with open('/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/examples/full_multi_gpu/llm_data/train_task1.json', 'r') as file:
    new_data = json.load(file)
out_data = []
idx = 0
import copy
negative_pair = []
############

    
###########
for data in new_data:
    text = data['input']
    answer = data['output']
    genes, diseases, functions = extract_entity_from_output(answer)
    for gene, disease, function in zip(genes, diseases, functions):
        text_tmp = copy.deepcopy(text)
        text_tmp = text_tmp.replace(gene, "@GENE$")
        text_tmp = text_tmp.replace(disease, "@DISEASE$")
        tmp = {"id": idx, "sentence": text_tmp, "label": function}
        out_data.append(tmp)
    for i in range(len(genes)):
        for j in range(len(diseases)):
            if i!=j:
                text_tmp = copy.deepcopy(text)
                text_tmp = text_tmp.replace(genes[i], "@GENE$")
                text_tmp = text_tmp.replace(diseases[i], "@DISEASE$")
                tmp = {"id": idx, "sentence": text_tmp, "label": "0"}
                negative_pair.append(tmp)
    idx += 1
negative_pair = random.sample(negative_pair, len(out_data))
out_data += negative_pair
with open("/home/zhangtaiyan/workspace/comp/LinkBERT/data/seqcls/agac_hf/train.json", 'w') as file:
    for data in out_data:
        json.dumps(data)
with jsonlines.open("/home/zhangtaiyan/workspace/comp/LinkBERT/data/seqcls/agac_hf/train.json", 'w') as writer:
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
    if data['task']==1:
        text = data['text']
        answer = data['ideal']['GENE, FUNCTION, DISEASE']
        genes, diseases, functions = extract_entity_from_output(answer)
        for i in range(len(genes)):
            for j in range(len(diseases)):
                if i == j:
                    text_tmp = copy.deepcopy(text)
                    text_tmp = text_tmp.replace(genes[i], "@GENE$")
                    text_tmp = text_tmp.replace(diseases[j], "@DISEASE$")
                    tmp = {"id": idx, "sentence": text_tmp, "label": "REG"}
                    tmp_aux = {"id": idx, "gene": genes[i], "disease": diseases[j]}
                    out_data.append(tmp)
                    aux_data.append(tmp_aux)

    idx += 1

with jsonlines.open("/home/zhangtaiyan/workspace/comp/LinkBERT/data/seqcls/agac_hf/aux_test.json", 'w') as writer:
    for data in aux_data:
        writer.write(data)
with jsonlines.open("/home/zhangtaiyan/workspace/comp/LinkBERT/data/seqcls/agac_hf/test1.json", 'w') as writer:
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
    if data['task']==1:
        text = data['text']
        answer = data['ideal']['GENE, FUNCTION, DISEASE']
        genes, diseases, functions = extract_entity_from_output(answer)
        for i in range(len(genes)):
            for j in range(len(diseases)):
                if i == j:
                    text_tmp = copy.deepcopy(text)
                    text_tmp = text_tmp.replace(genes[i], "@GENE$")
                    text_tmp = text_tmp.replace(diseases[j], "@DISEASE$")
                    tmp = {"id": idx, "sentence": text_tmp, "label": functions[i]}
                    tmp_aux = {"id": idx, "gene": genes[i], "disease": diseases[j]}
                    out_data.append(tmp)
                    aux_data.append(tmp_aux)

    idx += 1
with jsonlines.open("/home/zhangtaiyan/workspace/comp/LinkBERT/data/seqcls/agac_hf/dev1.json", 'w') as writer:
    for data in out_data:
        writer.write(data )
with jsonlines.open("/home/zhangtaiyan/workspace/comp/LinkBERT/data/seqcls/agac_hf/aux_dev.json", 'w') as writer:
    for data in aux_data:
        writer.write(data)

