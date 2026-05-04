# 此处是数据处理代码

import os
import pickle
import random
import time
import xml.etree.ElementTree as ET

import pandas as pd
from tqdm import tqdm

# 配置路径
DRUGBANK_XML = 'data/drugbank_all_full_database.xml'
CTD_TSV = 'data/CTD_genes_diseases.tsv'
OUTPUT_DIR = 'output/'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# 解析 DrugBank XML
def parse_drugbank_xml(xml_file):
    """
    解析 DrugBank XML，提取 药物名称 -> 靶点基因列表 的映射；
    使用迭代解析防止内存溢出。
    """
    print("开始解析 DrugBank XML (这可能需要几分钟)...")
    ns = '{http://www.drugbank.ca}'
    drug_targets = {}
    total_drugs = 0
    # 使用 iterparse('start') 快速扫描，统计 <drug> 开始标签数量
    for event, elem in ET.iterparse(xml_file, events=('start',)):
        if elem.tag == f'{ns}drug':
            total_drugs += 1
        elem.clear()  # 立即清除内存堆积
    # 重新创建迭代器
    context = ET.iterparse(xml_file, events=('end',))

    with tqdm(total=total_drugs, desc="解析药物", unit="种药物") as pbar:
        for event, elem in context:
            if elem.tag != f'{ns}drug':
                continue

            name_elem = elem.find(f'{ns}name')
            if name_elem is None or not name_elem.text:
                elem.clear()
                pbar.update(1)  # 更新进度（即使没有基因靶点也算处理了）
                continue
            drug_name = name_elem.text.strip().lower()
            if not drug_name:
                elem.clear()
                pbar.update(1)
                continue

            # 提取靶点
            genes = []
            targets = elem.find(f'{ns}targets')
            if targets is not None:
                for target in targets.findall(f'{ns}target'):
                    for polypeptide in target.findall(f'{ns}polypeptide'):
                        gene_elem = polypeptide.find(f'{ns}gene-name')
                        if gene_elem is not None and gene_elem.text:
                            gene = gene_elem.text.strip().upper()
                            if gene:
                                genes.append(gene)

            if genes:
                drug_targets[drug_name] = list(set(genes))

            elem.clear()
            pbar.update(1)

    print(f"成功提取了 {len(drug_targets)} 种药物的基因靶点。")
    return drug_targets


def preprocess_data():
    # 1. 提取 DrugBank 数据
    drug_to_genes = parse_drugbank_xml(DRUGBANK_XML)

    # 2. 提取 CTD 数据
    print("正在读取 CTD 数据集，此处无输出，但是可以观察内存压力判断工作与否。")

    # 必须先定义完整的 9 个列名，防止 pandas 把前面的列当成行索引 (Index)。
    all_ctd_names = [
        'GeneSymbol', 'GeneID', 'DiseaseName', 'DiseaseID',
        'DirectEvidence', 'InferenceChemicalName', 'InferenceScore',
        'OmimIDs', 'PubMedIDs'
    ]
    # 我们实际需要的列名：
    needed_cols = ['GeneSymbol', 'DiseaseName', 'InferenceChemicalName']

    ctd_df = pd.read_csv(
        CTD_TSV,
        sep='\t',
        comment='#',
        header=None,
        names=all_ctd_names,
        usecols=needed_cols,
        dtype={col: str for col in needed_cols},  # 全部强制设为字符串
        na_values=[''],  # 显式处理空值
        keep_default_na=True
    )

    # 清理数据：丢弃没有基因符号或疾病名称的行
    ctd_df = ctd_df.dropna(subset=['GeneSymbol', 'DiseaseName'])

    # 构建 疾病 -> 相关基因 的映射。
    disease_to_genes = ctd_df.groupby('DiseaseName')['GeneSymbol'].apply(lambda x: list(set(x.str.upper()))).to_dict()
    print(f"成功提取了 {len(disease_to_genes)} 种疾病的相关基因。")

    # 3. 收集所有的唯一基因（用于构建特征向量的维度）
    all_genes = set()
    print("正在收集药物-基因对的基因。")
    for genes in drug_to_genes.values(): all_genes.update(genes)
    print("正在收集疾病-基因对的基因。")
    for genes in disease_to_genes.values(): all_genes.update(genes)
    print("正在收集唯一基因。")
    gene_list = sorted(list(all_genes))
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_list)}

    # 4. 构建正负样本：只从 CTD 内构建正负样本
    df = ctd_df.dropna(subset=['InferenceChemicalName', 'DiseaseName']).copy()

    df['chem'] = df['InferenceChemicalName'].str.lower()
    df['dis'] = df['DiseaseName']

    # 正采样：从 CTD 数据中提取真实存在的 Drug-Disease 关联对作为正样本
    print("正在构建正样本。")

    positive_pairs_set = set()

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="正样本生成进度"):
        chemical = row.chem
        disease = row.dis
        positive_pairs_set.add((chemical, disease))

    positive_pairs = list(positive_pairs_set)

    print(f"找到 {len(positive_pairs)} 个唯一的正样本。")

    # 负采样：随机组合药物和疾病，如果不在正样本集中，则作为负样本（1:1比例）
    print("正在构建负样本。")

    negative_pairs_set = set()

    all_chemicals = [str(c).lower() for c in drug_to_genes.keys()]
    all_diseases = list(disease_to_genes.keys())
    target_size = len(positive_pairs)

    pbar = tqdm(total=target_size, desc="负样本生成进度")
    while len(negative_pairs_set) < target_size:
        chemical = random.choice(all_chemicals)
        disease = random.choice(all_diseases)
        pair = (chemical, disease)
        if pair not in positive_pairs_set and pair not in negative_pairs_set:
            negative_pairs_set.add(pair)
            pbar.update(1)
    pbar.close()

    negative_pairs = list(negative_pairs_set)

    print(f"随机生成 {len(negative_pairs)} 个唯一的负样本，且不与正样本重负。")

    # 完全字典形式的旧方案
    # # 5. 特征向量化 (Multi-hot encoding)
    # def to_vector(gene_names):
    #     vec = np.zeros(len(gene_list), dtype=np.float16)
    #     for g in gene_names:
    #         if g in gene_to_idx:
    #             vec[gene_to_idx[g]] = 1.0
    #     return vec
    #
    # print("\n[状态] 开始生成最终特征矩阵...")
    # X = []
    # y = []
    #
    # # 统计总样本数
    # total_samples = len(positive_pairs) + len(negative_pairs)
    # print(f"[状态] 总样本量: {total_samples}, 特征维度: {len(gene_list) * 2}")
    #
    # # 添加正样本
    # for d, dis in tqdm(positive_pairs, desc="[进度] 编码正样本"):
    #     d_vec = to_vector(drug_to_genes[d])
    #     dis_vec = to_vector(disease_to_genes[dis])
    #     X.append(np.concatenate([d_vec, dis_vec]))
    #     y.append(1)
    #
    # # 添加负样本
    # for d, dis in tqdm(negative_pairs, desc="[进度] 编码负样本"):
    #     d_vec = to_vector(drug_to_genes[d])
    #     dis_vec = to_vector(disease_to_genes[dis])
    #     X.append(np.concatenate([d_vec, dis_vec]))
    #     y.append(0)
    #
    # print("\n[状态] 正在执行 NumPy 矩阵合并...")
    # X = np.array(X)
    # y = np.array(y)
    # print("[状态] 矩阵合并完成！")
    #
    # # 6. 保存处理好的数据
    # print(f"[状态] 正在将 {X.nbytes / 1024 ** 3:.2f} GB 数据写入磁盘...")
    # with open('output/processed_data.pkl', 'wb') as f:
    #     pickle.dump({'X': X, 'y': y, 'gene_dim': len(gene_list)}, f)
    # print("[OK] 数据保存成功！")

    # 延迟计算的新方案
    # 5. 正负样本合在一起打上标签
    print("\n[状态] 正在打包训练元数据...")

    all_pairs = []
    all_labels = []

    total_len = len(positive_pairs) + len(negative_pairs)
    pbar = tqdm(total=total_len, desc="打包训练数据")

    # 正样本打包
    for d, dis in positive_pairs:
        all_pairs.append((d, dis))
        all_labels.append(1)
        pbar.update(1)
    # 负样本打包
    for d, dis in negative_pairs:
        all_pairs.append((d, dis))
        all_labels.append(0)
        pbar.update(1)

    pbar.close()

    # 6. 保存
    print("[状态] 正在将字典和索引写入本地硬盘...")
    output_path = 'output/processed_data.pkl'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    start_time = time.time()
    with open(output_path, 'wb') as f:
        pickle.dump({
            'pairs': all_pairs,
            'labels': all_labels,
            'drug_to_genes': drug_to_genes,
            'disease_to_genes': disease_to_genes,
            'gene_to_idx': gene_to_idx,
            'gene_dim': len(gene_list)
        }, f)
    end_time = time.time()

    file_size_bytes = os.path.getsize(output_path)
    file_size_mb = file_size_bytes / (1024 ** 2)
    file_size_gb = file_size_bytes / (1024 ** 3)

    duration = end_time - start_time
    speed_mb_s = file_size_mb / duration if duration > 0 else 0
    print(f"[完成] 数据已保存到 {output_path}")
    print(f"[大小] {file_size_mb:.2f} MB ({file_size_gb:.3f} GB)")
    print(f"[耗时] {duration:.2f} s")
    print(f"[速度] {speed_mb_s:.2f} MB/s")


if __name__ == '__main__':
    preprocess_data()
