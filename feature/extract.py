import os
from collections import OrderedDict

import pandas as pd
from myargs import args
import radiomics.featureextractor as FEE

'''
    文件路径定义
'''
default_prefix = 'D:/Desktop/BREAST/BREAST/'

name_mapping_path = default_prefix + 'breast-dataset-training-validation/Breast_meta_data/breast_name_mapping.csv'
name_mapping_path_t2 = default_prefix + 'breast-dataset-training-validation/Breast_meta_data/breast_name_mapping_t2.csv'
pCR_label_path = default_prefix + 'breast-dataset-training-validation/Breast_meta_data/Breast_MR_list_update.csv'


feature_radiomics_path = '../case2/breast_input_ph0_ph3_t2.csv'
new_radiomics_path = '../case3/breast_input_ph0_ph3_t2.csv'

root_path = default_prefix + 'breast-dataset-training-validation/Breast_TrainingData'

'''
    进行特征匹配并将匹配的特征保存到csv文件中
    pyradiomics 使用示例
'''
def feature_and_save_as_csv():
    data_types = ['_ph0.nii', '_ph3.nii', '_t2_sitk.nii', '_seg.nii']
    data_types_name = ['dceph0', 'dceph3', 't2', 'seg']

    # 使用df读取Breast_MR_list.xlsx文件
    pCR_info_df = pd.read_csv(pCR_label_path)
    name_mapping_df = pd.read_csv(name_mapping_path)
    name_mapping_df.rename({'Number': 'ID'}, axis=1, inplace=True)
    df = name_mapping_df.merge(pCR_info_df, on="ID", how="right")
    # merge后除去Breast_Training_ID为异常值的行
    delete_flags = pd.isna(df['Breast_subject_ID'])
    # 文件全部路径
    files = []
    for idx, data in df.iterrows():
        if ~delete_flags[idx]:
            if data['mismatch是否排除'] != '是':
                file = {}
                for data_type, data_type_name in zip(data_types, data_types_name):
                    file[data_type_name] = os.path.join(root_path, data['Breast_subject_ID'], data['Breast_subject_ID'] + data_type)
                file["pid"] = data['ID']
                file["pCR_label"] = data['病理完全缓解']
                files.append(file)

    # 对于每个病例使用配置文件初始化特征抽取器
    extractor = FEE.RadiomicsFeatureExtractor(args.params_file)
    for idx in range(110, 292):
    # for file in files:
        file = files[idx]
        data_type_seg = data_types_name[-1]
        for data_type_except_seg in data_types_name[:-1]:
            # 运行
            d1 = OrderedDict()
            d1['pid'] = file["pid"]
            d1['modal'] = data_type_except_seg
            d2 = extractor.execute(file[data_type_except_seg], file[data_type_seg])  # 抽取特征
            print("Calculated features:", "循环" + str(idx), file["pid"], data_type_except_seg)
            d2['pCR_label'] = file["pCR_label"]
            both = OrderedDict(list(d1.items()) + list(d2.items()))
            save_df = pd.DataFrame([both])
            save_df.to_csv(feature_radiomics_path, index=None, mode='a', header=None)

"""
    将第一步csv文件（模态按照横版排列）
    整理成新的csv文件（模态按照竖版排列）
"""
def arrange_and_save_as_csv(ori_radiomics_path, new_radiomics_path):
    df = pd.read_csv(ori_radiomics_path)  # 876*125
    idx1 = df["modal"].str.contains("dceph0").tolist()
    idx2 = df["modal"].str.contains("dceph3").tolist()
    idx3 = df["modal"].str.contains("t2").tolist()
    df1 = df.iloc[idx1].reset_index(drop=True)  # 292*125
    df1.columns = ['dceph0_' + col for col in df1.columns]
    df2 = df.iloc[idx2].reset_index(drop=True)
    df2.columns = ['dceph3_' + col for col in df2.columns]
    df3 = df.iloc[idx3].reset_index(drop=True)
    df3.columns = ['t2_' + col for col in df3.columns]
    result = pd.concat([df1, df2, df3], axis=1)

    result.to_csv(new_radiomics_path, index=None)  # 292*375
if __name__ == "__main__":
    # ==========================expriment-case1 and case2(手工加工):===============================
    # feature_and_save_as_csv()
    # ==========================expriment-case3:==================================================
    arrange_and_save_as_csv(feature_radiomics_path, new_radiomics_path)
