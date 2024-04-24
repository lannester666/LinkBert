# %% [markdown]
# ### demo

# %%
import os
import ipywidgets as widgets
from IPython.display import display
import json
import re
import regex
import jsonlines
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import pandas as pd
import networkx as nx
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
# %% [markdown]
# ### data reading functions

# %%
def read_raw_task1_data(task1_data_dir, task1_xlsx_path):
    triad_df = pd.read_excel(task1_xlsx_path)

    task1_raw_data = {}
    for file_name in sorted(os.listdir(task1_data_dir)):
        if file_name.endswith('.xlsx'):
            continue

        # json文件读取
        file_path = os.path.join(task1_data_dir, file_name)
        with open(file_path, "r") as file:
            data = json.load(file)

        # xlsx文件读取
        source_id = int(data["sourceid"])
        matching_rows = triad_df[triad_df['PMID'] == source_id]
        tri = []
        for _, row in matching_rows.iterrows():
            if (pd.isna(row['GENE']) or pd.isna(row['FUNCTION']) or pd.isna(row['DISEASE'])): continue  # 有些样本的数据为空
            gene_text = row['GENE']
            func_text = row['FUNCTION']
            disease_text = row['DISEASE']
            tri.append((gene_text, func_text, disease_text))
        data['triplets'] = tri

        task1_raw_data[data["sourceid"]] = data

    return task1_raw_data


# %%
def read_processed_task1_data(file_path):
    # Read the JSON file
    with open(file_path, "r") as file:
        data = json.load(file)

    # 提取元组
    texts = []
    triplets = []
    for element in data:
        texts.append(element['input'])
        triplet_string = element['output']
        tris = regex.findall(r'\(([^,]+),\s*([^,]+),\s*([^,]+)\)', triplet_string)
        triplets.append(tris)
    ids = list(range(len(texts)))
    
    return ids, texts, triplets

# %%
def read_processed_task2_data(file_path):
    # Read the JSON file
    with open(file_path, "r") as file:
        data = json.load(file)

    # 提取元组
    texts = []
    triplets = []
    for element in data:
        texts.append(element['input'])
        triplet_string = element['response']
        tris = regex.findall(r'\(([^,]+),\s*([^,]+)\)', triplet_string)
        triplets.append(tris)
    ids = list(range(len(texts)))
    
    return ids, texts, triplets

# %%
def read_processed_task3_data(file_path):
    # Read the JSON file
    with open(file_path, "r") as file:
        data = json.load(file)

    # 提取元组
    texts = []
    triplets = []
    for element in data:
        texts.append(element['input'])
        triplet_string = element['response']
        tris = regex.findall(r'\(([^,]+),\s*([^,]+),\s*([^,]+)\)', triplet_string)
        triplets.append(tris)
    ids = list(range(len(texts)))
    
    return ids, texts, triplets

# %%
def read_submission_data(file_path):
    # Read the JSON file
    data = []
    with jsonlines.open(file_path, "r") as reader:
        for read_line in reader:
            data.append(read_line)

    # 提取元组
    texts = []
    triplets = []
    for item in data:
        if item['task'] == 1:
            text = item['text']
            triplet_string = item['ideal']["GENE, FUNCTION, DISEASE"]
            tris = regex.findall(r'\(([^,]+),\s*([^,]+),\s*([^,]+)\)', triplet_string)
        elif item['task'] == 2:
            text = item['abstract']
            triplet_string = item['ideal']["chemical, disease"]
            tris = regex.findall(r'\(([^,]+),\s*([^,]+)\)', triplet_string)
        elif item['task'] == 3:
            text = item['text']
            triplet_string = item['ideal']["DDI-triples"]
            tris = regex.findall(r'\(([^,]+),\s*([^,]+),\s*([^,]+)\)', triplet_string)
        texts.append(text)
        
        triplets.append(tris)
    ids = list(range(len(texts)))
    
    return ids, texts, triplets

# %%
def read_by_merge(factory_input_file, factory_output_file):
    # Read the JSON file
    output_data = []
    with jsonlines.open(factory_output_file, "r") as reader:
        for read_line in reader:
            output_data.append(read_line)
    
    with open(factory_input_file, "r") as file:
        input_data = json.load(file)

    # 提取元组
    texts = []
    triplets = []
    for o, i in zip(output_data, input_data):
        texts.append(i['input'])
        triplet_string = o['predict']
        tris = regex.findall(r'\(([^,]+),\s*([^,]+),\s*([^,]+)\)', triplet_string)
        triplets.append(tris)
    ids = list(range(len(texts)))

    return ids, texts, triplets

# %% [markdown]
# ### demo functions

# %%
def demo(ids, texts, triples, aux_triplets=None):
    # 生成颜色列表
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, 50)]
    colors = ['#{:02x}{:02x}{:02x}'.format(int(c[0]*255), 
                                            int(c[1]*255), 
                                            int(c[2]*255)) for c in colors]


    # 定义更新文本和元组的函数
    def update_display(change):
        index = change["new"]
        text = texts[index]
        tris = triples[index]
        if aux_triplets is not None:
            aux_tris = aux_triplets[index]

        # 高亮显示文本中的实体
        highlighted_sents = sent_tokenize(text)
        entities = [en for tri in tris for en in tri]
        if aux_triplets:
            entities += [en for tri in aux_tris for en in tri]
        entities = list(set(entities))
        entities = sorted(entities, key=len, reverse=True)  # 先着色长实体，再着色短实体，因为存在重叠
        for i in range(len(highlighted_sents)):
            for entity in entities:
                highlighted_sents[i] = highlighted_sents[i].replace(entity, 
                                                            f"<span style='background-color:\
                                                            {colors[hash(entity) % len(colors)]}'>{entity}</span>")
        
        # 高亮显示每个元组
        highlighted_triplets = []
        for tri in tris:
            line = '(' + ', '.join([
                f"<span style='background-color: {colors[hash(t) % len(colors)]}'>{t}</span>" for t in tri
                ]) + ')'
            highlighted_triplets.append(line)

        # 更新显示
        text_widget.value = "<br>".join(highlighted_sents)
        triplets_widget.value = "<br>".join(highlighted_triplets)
        source_text_widget.value = text

        # 辅助元组（optional）
        if aux_triplets:
            highlighted_aux_triplets = []
            for tri in aux_tris:
                line = '(' + ', '.join([
                    f"<span style='background-color: {colors[hash(t) % len(colors)]}'>{t}</span>" for t in tri
                    ]) + ')'
                highlighted_aux_triplets.append(line)
            aux_triplets_widget.value = "<br>".join(highlighted_aux_triplets)


    dropdown = widgets.Dropdown(options=ids, description='Sample:')
    dropdown.observe(update_display, 'value')


    def on_toggle_button_click(b):
        current_index = dropdown.options.index(dropdown.value)
        next_index = (current_index + 1) % len(dropdown.options)
        dropdown.value = dropdown.options[next_index]


    def on_toggle_button_click_r(b):
        current_index = dropdown.options.index(dropdown.value)
        next_index = (current_index - 1) % len(dropdown.options)
        dropdown.value = dropdown.options[next_index]


    toggle_button = widgets.Button(description="Next")
    toggle_button.on_click(on_toggle_button_click)
    toggle_button_r = widgets.Button(description="Last")
    toggle_button_r.on_click(on_toggle_button_click_r)

    hbox = widgets.HBox([dropdown, toggle_button_r, toggle_button])
    text_widget = widgets.HTML(layout=widgets.Layout(width='100%', height='300px'), description="Text:")
    triplets_widget = widgets.HTML(layout=widgets.Layout(width='50%', height='200px'), description="Pairs:")
    aux_triplets_widget = widgets.HTML(layout=widgets.Layout(width='50%', height='200px'), description="Aux pairs:")
    triplets_hbox = widgets.HBox([triplets_widget, aux_triplets_widget])
    source_text_widget = widgets.HTML(layout=widgets.Layout(width='100%', height='300px'), description="Source text:")

    # refresh the display
    update_display({"new": 0})
    # Display the widgets
    display(hbox, text_widget, triplets_hbox, source_text_widget)


# %%
def demo_for_task1_raw_data(task1_raw_data):
    ids = list(task1_raw_data.keys())

    color_map = {
        'Gene': '#BA55D3',
        'Var': '#FFC0CB',
        'Disease': '#6495ED',
        'Reg': '#778899',
        'NegReg': '#00FA9A',
        'PosReg': '#FF0000',
        'MPA': '#FFE4B5',
        'CPA': '#FFE4B5',
        'Interaction': '#FFE4B5',
        'Protein': '#E6E6FA',
        'Enzyme': '#F4A460',
        'Pathway': '#FFE4B5',
        'REG': '#778899',
        'GOF': '#FF0000',
        'LOF': '#00FA9A',
        'COM': '#FFFF00'
    }

    # 定义更新文本和元组的函数
    def update_display(change):

        id = change["new"]
        data = task1_raw_data[id]
        text = data['text']
        denotations = data['denotations']
        relations = data['relations']
        tris = data['triplets']

        # 生成关系图
        G = nx.DiGraph()
        for deno in denotations:
            G.add_node(deno['id'], attrs=deno)
        for rel in relations:
            if rel['pred'] == 'CauseOf':
                G.add_edge(rel['subj'], rel['obj'])
            elif rel['pred'] == 'ThemeOf':
                G.add_edge(rel['obj'], rel['subj'])
            else:
                raise ValueError()
        node_labels = {
            node: f"[{G.nodes[node]['attrs']['obj']}] "
            + ' '.join(text[G.nodes[node]['attrs']['span']['begin']: G.nodes[node]['attrs']['span']['end']].split(' '))
            for node in G.nodes()
        }
        node_colors = [color_map[G.nodes[node]['attrs']['obj']] for node in G.nodes()]

        # 根据关系图建立更细粒度的标签
        weakly_connected_components = list(nx.weakly_connected_components(G))
        fine_grained_relations = []
        for cc_ids in weakly_connected_components:
            # cc_ids is a set of node id, we need node objects
            cc = [G.nodes[node_id] for node_id in cc_ids]
            cur_gene_node = []
            cur_disease_node = []
            cur_reg_node = []
            cur_lof_node = []
            cur_gof_node = []
            for node in cc:
                if node['attrs']['obj'] == 'Gene':
                    cur_gene_node.append(node)
                elif node['attrs']['obj'] == 'Disease':
                    cur_disease_node.append(node)
                elif node['attrs']['obj'] == 'Reg':
                    cur_reg_node.append(node)
                elif node['attrs']['obj'] == 'NegReg':
                    cur_lof_node.append(node)
                elif node['attrs']['obj'] == 'PosReg':
                    cur_gof_node.append(node)
            if len(cur_gene_node) == 0 or len(cur_disease_node) == 0:
                continue
            if len(cur_reg_node) == 0:
                assert len(cur_lof_node) + len(cur_gof_node) == 1
            label_rel_node = None
            if cur_reg_node:
                label_rel_node = cur_reg_node[0]
            if cur_lof_node:
                label_rel_node = cur_lof_node[0]
            if cur_gof_node:
                label_rel_node = cur_gof_node[0]
            assert label_rel_node is not None
            fine_grained_relations.extend([(g, label_rel_node, d) for g in cur_gene_node for d in cur_disease_node])
        # 按照(gene_span_begin, disease_span_begin)升序排列
        sorted(fine_grained_relations, key = lambda tri: (tri[0]['attrs']['span']['begin'], tri[2]['attrs']['span']['end']))

        fine_grained_triplets = []
        for tri in fine_grained_relations:
            gene_text = text[tri[0]['attrs']['span']['begin']: tri[0]['attrs']['span']['end']] + ':GENE'
            rel_text = text[tri[1]['attrs']['span']['begin']: tri[1]['attrs']['span']['end']]
            if tri[1]['attrs']['obj'] == 'Reg': rel_text += ':REG'
            elif tri[1]['attrs']['obj'] == 'NegReg': rel_text += ':LOF'
            elif tri[1]['attrs']['obj'] == 'PosReg': rel_text += ':GOF'
            disease_text = text[tri[2]['attrs']['span']['begin']: tri[2]['attrs']['span']['end']] + ':DISEASE'
            fine_grained_triplets.append((gene_text, rel_text, disease_text))

        with output_widget:
            output_widget.clear_output(wait=True)
            plt.clf()
            current_figure = plt.gcf()
            current_figure.set_size_inches(10, 4)
            # pos = nx.spring_layout(G, seed=42)
            pos = nx.shell_layout(G)
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=400, alpha=0.6, linewidths=2)
            nx.draw_networkx_edges(G, pos, arrows=True, alpha=1, width=1.8)
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight='bold', font_color='black')
            plt.title("Directed Graph Visualization")
            plt.tight_layout()
            plt.show()
        
        # 生成高亮文本
        highlighted_text = text
        # 从后向前替换，避开替换后idx改变的问题
        for deno in sorted(denotations, key=lambda deno: deno['span']['begin'], reverse=True):
            start_index = deno['span']['begin']
            end_index = deno['span']['end']
            highlighted_text = highlighted_text[:start_index] \
            + (f"<span style='background-color:{color_map[deno['obj']]}'>{highlighted_text[start_index: end_index]}</span>") \
            + (highlighted_text[end_index:])

        # 生成高亮三元组文本（来自excel文件）
        highlighted_triplets = []
        for tri in tris:
            line = '(' \
            + f"<span style='background-color: {color_map['Gene']}'>{tri[0]}</span>" \
            + ', ' \
            + f"<span style='background-color: {color_map[tri[1]]}'>{tri[1]}</span>" \
            + ', ' \
            + f"<span style='background-color: {color_map['Disease']}'>{tri[2]}</span>" \
            + ')'
            highlighted_triplets.append(line)

        # 生成辅助高亮三元组文本
        highlighted_aux_triplets = []
        for tri in fine_grained_triplets:
            line = '(' \
            + f"<span style='background-color: {color_map['Gene']}'>{tri[0]}</span>" \
            + ', ' \
            + f"<span style='background-color: {color_map[tri[1].split(':')[-1]]}'>{tri[1]}</span>" \
            + ', ' \
            + f"<span style='background-color: {color_map['Disease']}'>{tri[2]}</span>" \
            + ')'
            highlighted_aux_triplets.append(line)

        # 更新显示
        text_widget.value = "<br><br>".join(sent_tokenize(highlighted_text))
        triplets_widget.value = "<br>".join(highlighted_triplets)
        aux_triplets_widget.value = "<br>".join(highlighted_aux_triplets)
        source_text_widget.value = text

    dropdown = widgets.Dropdown(options=ids, description='Sample:')
    dropdown.observe(update_display, 'value')


    def on_toggle_button_click(b):
        current_index = dropdown.options.index(dropdown.value)
        next_index = (current_index + 1) % len(dropdown.options)
        dropdown.value = dropdown.options[next_index]


    def on_toggle_button_click_r(b):
        current_index = dropdown.options.index(dropdown.value)
        next_index = (current_index - 1) % len(dropdown.options)
        dropdown.value = dropdown.options[next_index]


    toggle_button = widgets.Button(description="Next")
    toggle_button.on_click(on_toggle_button_click)
    toggle_button_r = widgets.Button(description="Last")
    toggle_button_r.on_click(on_toggle_button_click_r)

    hbox = widgets.HBox([dropdown, toggle_button_r, toggle_button])
    text_widget = widgets.HTML(layout=widgets.Layout(width='100%', height='250px'), description="Text:")
    triplets_widget = widgets.HTML(layout=widgets.Layout(width='50%', height='100px'), description="Pairs:")
    aux_triplets_widget = widgets.HTML(layout=widgets.Layout(width='50%', height='100px'), description="Aux pairs:")
    triplets_hbox = widgets.HBox([triplets_widget, aux_triplets_widget])
    source_text_widget = widgets.HTML(layout=widgets.Layout(width='100%', height='500px'), description="Source text:")
    output_widget = widgets.Output(layout={'display': 'flex', 'justify_content': 'center'}, description="Graph:")

    # refresh the display
    update_display({"new": ids[0]})
    # Display the widgets
    display(hbox, text_widget, triplets_hbox, output_widget, source_text_widget)


# %% [markdown]
# ### demo for processed data

# %%
# ids, texts, triplets = read_by_merge(
#     "/home/zty/ykd_workspace/llm/processed_data/task1_data.json",
#     "/home/zty/ykd_workspace/LLaMA-Factory/saves/Gemma-7B/lora/eval_2024-03-23-baseline-refined-task1-data-task1-eval/generated_predictions.jsonl",
#     )
# _, _, aux_triplets = read_processed_task1_data("/home/zty/ykd_workspace/llm/processed_data/task1_data.json")

# ids, texts, triplets = read_processed_task1_data("/home/zty/ykd_workspace/llm/processed_data/task1_data.json")
# ids, texts, triplets = read_processed_task2_data("/home/zty/ykd_workspace/llm/processed_data/task2_data.json")
# ids, texts, triplets = read_processed_task3_data("/home/zty/ykd_workspace/llm/processed_data/task3_data.json")

ids, texts, triplets = read_submission_data("/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/save/Qwen1.5-14B/pt/sft/full/2024-03-30-09-34-04-wo_A-1e-6/submission.jsonl")
_, _, aux_triplets = read_submission_data("/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/save/BioMistral-7B/pt/sft/full/2024-04-05-00-02-39-wo_A-1e-6/checkpoint-320/submission.jsonl")


# %%
# demo(ids, texts, triplets, None)
demo(ids, texts, triplets, aux_triplets)

# %% [markdown]
# ### demo for task1 raw data

# %%
task1_raw_data = read_raw_task1_data('/home/zty/ykd_workspace/llm_data/task1_data', '/home/zty/ykd_workspace/llm_data/task1_data/train_triad.xlsx')

# %%
demo_for_task1_raw_data(task1_raw_data)


