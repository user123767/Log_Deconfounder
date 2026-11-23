import sys

from dataset_manager import dataset_confs

import pandas as pd


class DatasetManager:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.filename = dataset_confs.register_dataset(self.dataset_name)
        self.base_dataset = dataset_confs.base_dataset(self.dataset_name)
        print(f"DatasetManager: '{self.dataset_name}' -> {self.filename}")
        self.case_id_col = dataset_confs.case_id_col[self.dataset_name]
        self.activity_col = dataset_confs.activity_col[self.dataset_name]
        self.timestamp_col = dataset_confs.timestamp_col[self.dataset_name]
        self.label_col = dataset_confs.label_col[self.dataset_name]
        self.pos_label = dataset_confs.pos_label[self.dataset_name]

        self.dynamic_cat_cols = dataset_confs.dynamic_cat_cols[self.dataset_name].copy()
        self.static_cat_cols = dataset_confs.static_cat_cols[self.dataset_name]
        self.dynamic_num_cols = dataset_confs.dynamic_num_cols[self.dataset_name]
        self.static_num_cols = dataset_confs.static_num_cols[self.dataset_name]

        self.control_flow_var_binary = dataset_confs.control_flow_var_binary[self.dataset_name]
        self.control_flow_var_incremental = dataset_confs.control_flow_var_incremental[self.dataset_name]
        self.control_flow_var = dataset_confs.control_flow_var[self.dataset_name]
        self.control_flow_var_attribute = dataset_confs.control_flow_var_attribute[self.dataset_name]
        self.environmental_actions = dataset_confs.environmental_actions[self.dataset_name]
        self.sorting_cols = [self.timestamp_col, self.activity_col]


    def read_dataset(self):
        dtypes = {col:"object" for col in self.dynamic_cat_cols+self.static_cat_cols+[self.case_id_col, self.label_col, self.timestamp_col]}
        for col in self.dynamic_num_cols + self.static_num_cols:
            dtypes[col] = "float"

        data = pd.read_csv(dataset_confs.filename[self.dataset_name], sep=";", dtype=dtypes)
        data[self.timestamp_col] = pd.to_datetime(data[self.timestamp_col])

        data = pd.concat([data], ignore_index=True)

        data = data.sort_values(by=[self.case_id_col, self.timestamp_col]).reset_index(drop=True)

        return data
