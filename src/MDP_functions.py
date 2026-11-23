import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import Counter
import sys
import pickle
import time
import math
import random
import os
import gymnasium as gym
from collections import defaultdict
from gym import Env
from gym import spaces
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataset_manager.DatasetManager import *

np.random.seed(0)

SEED = 0
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

class ProcessEnv(Env):
    def __init__(self, dataset, order, train_fraction, MDP_model):
        self.dataset = dataset
        self.MDP_model = MDP_model
        self.order = order
        self.df, self.dataset_manager = get_real_data(dataset)
        self.config = DatasetMDP(dataset, self.df, self.dataset_manager)
        self.base_dataset = self.dataset_manager.base_dataset
        self.df = self.config.df
        self.all_cases = self.config.all_cases
        self.all_actions = self.config.all_actions
        self.activity_index = self.config.activity_index
        self.n_actions = self.config.n_actions
        self.action_space = spaces.Discrete(self.n_actions)

        self.df = self.config.cut_at_terminal_states(self.df,self.dataset_manager)
        self.df['last_action'] = self.df['action'].astype(str)
        train_size = int(train_fraction * len(self.all_cases))
        self.train_case_ids = self.all_cases[:train_size]
        self.test_case_ids = self.all_cases[train_size:]
        self.train_df = self.df[self.df['ID'].isin(self.train_case_ids)].copy()
        self.test_df = self.df[self.df['ID'].isin(self.test_case_ids)].copy()


        self.state_cols, self.environmental_actions = define_real_state_cols(self.base_dataset, self.dataset_manager)


        if self.base_dataset == 'SimBank':
            if self.MDP_model == 'HMDP':
                cols_to_remove = [col for col in self.environmental_actions if col not in ['receive_acceptance', 'receive_refusal']]
                self.state_cols = [col for col in self.state_cols if col not in cols_to_remove]

        self.reward = 0
        self.reward_scale = 10000
        self.unscaled_outcome = 0
        self.case = 0
        self.state = 0
        self.action = 0
        self.costs_dic = self.config.costs_dic
        self.dynamic_cols = self.dataset_manager.dynamic_cat_cols + self.dataset_manager.dynamic_num_cols
        self.dynamic_num_cols = self.dataset_manager.dynamic_num_cols

        self.numeric_cols = self.dataset_manager.static_num_cols + self.dataset_manager.dynamic_num_cols
        self.scaler = StandardScaler()
        self.df[self.numeric_cols] = self.scaler.fit_transform(self.df[self.numeric_cols])
        self.train_df[self.numeric_cols] = self.scaler.fit_transform(self.train_df[self.numeric_cols])
        self.test_df[self.numeric_cols] = self.scaler.transform(self.test_df[self.numeric_cols])

        self.dataset_manager.dynamic_cat_cols.append('last_action')
        self.categorical_cols = self.dataset_manager.static_cat_cols + self.dataset_manager.dynamic_cat_cols

        from sklearn.preprocessing import OneHotEncoder
        self.onehot_cols = []
        self.onehot_encoders = {}
        self.embedding_cols = {}
        self.label_encoders = {}

        for col in self.categorical_cols:
            n_unique = self.train_df[col].nunique()
            if n_unique <= 100:
                import sklearn
                if tuple(map(int, sklearn.__version__.split(".")[:2])) >= (1, 2):
                    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.float32)
                else:
                    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore', dtype=np.float32)
                reshaped = self.train_df[[col]]
                encoded = ohe.fit_transform(reshaped)
                ohe_cols = [f"{col}_{cat}" for cat in ohe.categories_[0]]
                train_df_ohe = pd.DataFrame(encoded, columns=ohe_cols, index=self.train_df.index)
                self.train_df = pd.concat([self.train_df, train_df_ohe], axis=1)
                if col in self.state_cols:
                    self.state_cols.remove(col)
                self.state_cols += ohe_cols
                self.onehot_encoders[col] = ohe
                self.onehot_cols.append(col)

                reshaped_test = self.test_df[[col]]
                encoded_test = ohe.transform(reshaped_test)
                df_ohe_test = pd.DataFrame(encoded_test, columns=ohe_cols, index=self.test_df.index)
                self.test_df = pd.concat([self.test_df, df_ohe_test], axis=1)

            else:
                le = LabelEncoder()
                self.train_df[col] = le.fit_transform(self.train_df[col])
                self.label_encoders[col] = le
                self.embedding_cols[col] = len(le.classes_)
                self.test_df[col] = self.test_df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        if self.order > 1:
            dynamic_cols_to_lag = []

            dynamic_cols_to_lag += [col for col in self.dynamic_num_cols if col in self.state_cols]

            for orig_col in self.dataset_manager.dynamic_cat_cols:
                if orig_col in self.onehot_cols:
                    oh_cols = [c for c in self.state_cols if c.startswith(orig_col + "_")]
                    dynamic_cols_to_lag += oh_cols
                elif orig_col in self.train_df.columns:
                    dynamic_cols_to_lag.append(orig_col)

            def add_lagged_features(df, order, cols_to_lag):
                lagged_dict = {}
                for col in cols_to_lag:
                    for lag in range(1, order):
                        lagged_col = f"{col}_lag{lag}"
                        lagged_dict[lagged_col] = (
                            df.groupby("ID")[col]
                            .shift(lag)
                            .fillna(0)
                        )
                lagged_df = pd.DataFrame(lagged_dict, index=df.index)
                df = pd.concat([df, lagged_df], axis=1)
                return df, list(lagged_dict.keys())


            self.train_df, lagged_cols = add_lagged_features(self.train_df, self.order, dynamic_cols_to_lag)
            self.test_df, _ = add_lagged_features(self.test_df, self.order, dynamic_cols_to_lag)

            self.lagged_cols = lagged_cols
            self.state_cols += lagged_cols

        self.penalty_illegal = 1.1 * max(v for v in self.costs_dic.values() if isinstance(v, (int, float)))
        if self.penalty_illegal == 0:
            self.penalty_illegal = 1


    def build_state_with_lags(self, previous_trace, current_event):
        state_dict = current_event.copy()
        state_dict['last_action']=state_dict['activity']

        for col in self.onehot_cols:
            ohe = self.onehot_encoders[col]
            val = str(state_dict.get(col))
            value = np.array([[val]], dtype=object)
            onehot_values = ohe.transform(value).flatten()
            for i, cat in enumerate(ohe.categories_[0]):
                state_dict[f"{col}_{cat}"] = onehot_values[i]
            if col in state_dict:
                del state_dict[col]

        if self.order > 1:
            for lag in range(1, self.order):
                if len(previous_trace) >= lag:
                    prev_event = previous_trace[-lag]
                    for col in self.dynamic_num_cols:
                        lagged_col = f"{col}_lag{lag}"
                        state_dict[lagged_col] = prev_event.get(col, 0)
                    for col in self.onehot_cols:
                        ohe = self.onehot_encoders[col]
                        val = str(prev_event.get(col, "missing"))
                        value = np.array([[val]], dtype=object)
                        onehot_values = ohe.transform(value).flatten()
                        for i, cat in enumerate(ohe.categories_[0]):
                            lagged_col_name = f"{col}_{cat}_lag{lag}"
                            state_dict[lagged_col_name] = onehot_values[i]

                else:
                    for col in self.dynamic_num_cols:
                        lagged_col = f"{col}_lag{lag}"
                        state_dict[lagged_col] = 0
                    for col in self.onehot_cols:
                        ohe = self.onehot_encoders[col]
                        for i, cat in enumerate(ohe.categories_[0]):
                            lagged_col_name = f"{col}_{cat}_lag{lag}"
                            state_dict[lagged_col_name] = 0

        state_row = [state_dict[col] for col in self.state_cols]
        current_state = np.array(state_row, dtype=np.float32).reshape(1, -1)

        return current_state

class DatasetMDP:
    def __init__(self, dataset, df, dataset_manager):
        self.dataset = dataset_manager.base_dataset
        self.base_dataset = dataset_manager.base_dataset
        self.df = df
        self.control_flow_var = dataset_manager.control_flow_var
        self.environmental_actions = dataset_manager.environmental_actions
        self.control_flow_var_incremental = dataset_manager.control_flow_var_incremental
        self.control_flow_var_binary = dataset_manager.control_flow_var_binary
        self.control_flow_var_attribute = dataset_manager.control_flow_var_attribute
        if self.base_dataset == 'SimBank':
            mask = self.df["action"] == "calculate_offer"
            self.df.loc[mask, "action"] = self.df.loc[mask, "interest_rate"]
        self.df, self.all_cases, self.all_actions, self.activity_index, self.n_actions = self.remove_env_actions(self.df)
        self.costs_dic = self.define_costs_from_log()
        self.win_action, self.loss_actions = self.define_terminal_actions()

    def cut_at_terminal_states(self, df, dataset_manager):
        def cut_case(case_df):
            for idx, row in case_df.iterrows():
                state = row.to_dict()
                is_terminal, _ = self.is_terminal_successful_state(state)
                if is_terminal:
                    last_idx = case_df.index[-1]
                    if idx == last_idx:
                        return case_df.loc[:idx]
                    else:
                        return pd.concat([case_df.loc[:idx], case_df.loc[[last_idx]]])
            return case_df
        df = df.groupby("ID", group_keys=False).apply(cut_case).reset_index(drop=True)

        df = df.sort_values(by=['ID', dataset_manager.timestamp_col]).reset_index(drop=True)
        return df


    def remove_env_actions(self, df):
        for env_activity in self.control_flow_var:
            if env_activity in self.environmental_actions or env_activity in self.control_flow_var_binary:
                print("Transforming environmental or control flow action as binary state var:", env_activity)
                df[env_activity] = (df['action'] == env_activity).astype(int)
                df[env_activity] = df.groupby('ID')[env_activity].cummax()
            elif env_activity in self.control_flow_var_incremental:
                print("Transforming control flow action as incremental state var:", env_activity)
                df[env_activity] = (df['action'] == env_activity).astype(int)
                df[env_activity] = df.groupby('ID')[env_activity].cumsum()

        mask = df['action'].isin(self.environmental_actions)
        prev_values = df[['action', 'last_action']].shift(1)
        df.loc[mask, ['action', 'last_action']] = prev_values.loc[mask]
        prev_idx = mask.shift(-1, fill_value=False)
        df = df.loc[~prev_idx].reset_index(drop=True)
        all_cases = df['ID'].unique()

        if self.dataset == 'hospital_billing_2':
            df['isCancelled'] = df['isCancelled'].map({False: 0, True: 1})


        all_actions = df["action"].unique()
        activity_index = {activity: idx for idx, activity in enumerate(all_actions)}
        n_actions = len(all_actions)
        df['last_action'] = df['action'].map(activity_index)

        return df, all_cases, all_actions, activity_index, n_actions


    def define_costs_from_log(self):
        if self.base_dataset == 'SimBank':
            return {
                "initiate_application": 0,
                    "start_standard": 10,
                    "start_priority": 5000,
                    "validate_application": 20,
                    "contact_headquarters": 1000,
                    "skip_contact": 0,
                    "email_customer": 10,
                    "call_customer": 20,
                    "calculate_offer": 400,
                    0.07: 400,
                    0.08: 400,
                    0.09: 400,
                    "cancel_application": 30,
                    "receive_acceptance": 10,
                    "receive_refusal": 10,
                    "stop_application": 0
                }


        if self.dataset == 'traffic_fines_1':
            return {
                'Create Fine': 0.0,
                'Send Fine': 0.0,
                'Add Penalty': 0.0,
                'Send for Credit Collection': 0.0,
                'Insert Fine Notification': 0.0,
                'Insert Date Appeal to Prefecture': 0.0,
                'Send Appeal to Prefecture': -1,
                'Receive Result Appeal from Prefecture': 0.0,
                'Notify Result Appeal to Offender': 0.0,
                'Appeal to Judge': -1.0,
                "start": 0,
                'archive_application': 0,
            }
        if self.dataset in ['hospital_billing_2','production', 'bpic2012_accepted', 'bpic2017_accepted']:
            unique_activities = self.df['action'].unique()
            return {
                act: 0.0 if act in ["start", "archive_application"] else 10.0
                for act in unique_activities
            }
        if self.dataset in ['sepsis_cases_1', 'sepsis_cases_2', 'sepsis_cases_4']:
            return {
                "start": 0,
                'archive_application': 0,
                "ER Registration": 150,
                "Leucocytes": 25,
                "CRP": 20,
                "LacticAcid": 35,
                "ER Triage": 250,
                "ER Sepsis Triage": 350,
                "IV Liquid": 75,
                "IV Antibiotics": 200,
                "Admission NC": 2000,
                "Release A": 50,
                "Admission IC": 7000,
                "Release B": 50,
                "Release E": 50,
                "Release C": 50,
                "Release D": 50,
            }


    def is_terminal_successful_state(self, state):
            is_terminal = False
            is_successful = False

            if self.dataset== 'bpic2017_accepted':
                if state.get('last_action_A_Pending')==1:
                    is_terminal = True
                    is_successful = True
                if state.get('last_action_A_Cancelled')==1 or state.get('last_action_A_Denied')==1 :
                    is_terminal = True
                    is_successful = False

            if self.dataset== 'bpic2012_accepted':
                if state.get('last_action_A_APPROVED-COMPLETE')==1:
                    is_terminal = True
                    is_successful = True
                if state.get('last_action_A_CANCELLED-COMPLETE')==1 or state.get('last_action_A_DECLINED-COMPLETE')==1 :
                    is_terminal = True
                    is_successful = False

            elif self.dataset == 'sepsis_cases_2':
                if (state.get('last_action_Release A') == 1 or
                state.get('last_action_Release B') == 1 or
                state.get('last_action_Release C') == 1 or
                state.get('last_action_Release D') == 1 or
                state.get('last_action_Release E') == 1):
                    is_terminal = True
                    is_successful = True
                if state.get('last_action_Admission IC')==1 :
                    is_terminal = True
                    is_successful = False

            elif self.base_dataset == 'SimBank':
                if state.get('receive_acceptance')== 1:
                    is_terminal, is_successful = True, True
                elif state.get('receive_refusal') == 1:
                    is_terminal, is_successful = True, False
                elif state.get('last_action_cancel_application')==1:
                    is_terminal, is_successful = True, False


            elif self.dataset == 'sepsis_cases_1':
                recent_release = state.get('recent_release', 0)
                return_ER = state.get('Return ER')
                if (state.get('last_action_Release A') == 1 or
                state.get('last_action_Release B') == 1 or
                state.get('last_action_Release C') == 1 or
                state.get('last_action_Release D') == 1 or
                state.get('last_action_Release E') == 1):
                    is_terminal = True
                    if return_ER == 1 and recent_release == 0:
                        is_successful = True
                    elif return_ER == 0:
                        is_successful = True

            elif self.dataset == 'hospital_billing_2':
                cancellation_status = state.get('isCancelled')
                if cancellation_status==1:
                    is_terminal, is_successful = True, False
                elif state.get('last_action_archive_application') == 1 and cancellation_status==0:
                    is_terminal, is_successful = True, True

            return is_terminal, is_successful


    def define_terminal_actions(self):
        if self.dataset == 'bpic2017_accepted':
            win = [self.activity_index['A_Pending']]
            loss = [self.activity_index['A_Cancelled'],
                    self.activity_index['A_Denied']]

        elif self.dataset == 'bpic2012_accepted':
            win = [self.activity_index['A_APPROVED-COMPLETE']]
            loss = [self.activity_index['A_CANCELLED-COMPLETE'],
                    self.activity_index['A_DECLINED-COMPLETE']]

        elif self.dataset == 'traffic_fines_1':
            win = []
            loss = [self.activity_index['Send for Credit Collection']]

        elif self.dataset == 'sepsis_cases_1':
            win = [self.activity_index['Release A'], self.activity_index['Release B'], self.activity_index['Release C'], self.activity_index['Release D'], self.activity_index['Release E']]
            loss = []

        elif self.dataset == 'sepsis_cases_2':
            win = [self.activity_index['Release A'], self.activity_index['Release B'], self.activity_index['Release C'], self.activity_index['Release D'], self.activity_index['Release E']]
            loss = [self.activity_index['Admission IC']]

        elif self.dataset == 'sepsis_cases_4':
            win = [self.activity_index['Release A']]
            loss = [self.activity_index['Release B'], self.activity_index['Release C'], self.activity_index['Release D'], self.activity_index['Release E']]

        elif self.dataset == 'hospital_billing_2':
            win = []
            loss = [self.activity_index['DELETE']]

        elif self.base_dataset == 'SimBank':
            win = []
            loss = [self.activity_index['cancel_application']]

        else:
            win, loss = [], []
        return win, loss


    def outcome_function(self, granted, current_event, cum_cost, scaler, numeric_cols, state_cols):
        if self.base_dataset == 'SimBank':
            if granted:
                dummy = pd.DataFrame([current_event]).copy()
                unscaled_numeric = scaler.inverse_transform(dummy[numeric_cols])
                dummy.loc[:, numeric_cols] = unscaled_numeric
                unscaled_event = dummy.iloc[0].to_dict()

                i = unscaled_event["interest_rate"]
                A = unscaled_event["amount"]
                risk_factor = (10 - unscaled_event['est_quality']) / 200
                risk_free_rate = 0.03
                df = risk_free_rate + risk_factor
                n = 10
                future_earnings = A * (1 + i)**n
                discount_future_earnings = future_earnings / (1 + df)**n
                exp_profit = discount_future_earnings - cum_cost - A - 100
                return exp_profit
            else:
                return -cum_cost - 100

        elif self.dataset == 'bpic2017_accepted' and granted:
            dummy = pd.DataFrame([current_event])[state_cols].copy()
            unscaled_numeric = scaler.inverse_transform(dummy[numeric_cols])
            dummy.loc[:, numeric_cols] = unscaled_numeric
            unscaled_event = dummy.iloc[0].to_dict()

            A = unscaled_event["OfferedAmount"]
            if A <= 6000:
                r = 0.16
            elif A <= 15000:
                r = 0.18
            else:
                r = 0.20
            interest = A * r
            return interest - cum_cost

        elif self.dataset == 'bpic2012_accepted' and granted:
            dummy = pd.DataFrame([current_event])[state_cols].copy()
            unscaled_numeric = scaler.inverse_transform(dummy[numeric_cols])
            dummy.loc[:, numeric_cols] = unscaled_numeric
            unscaled_event = dummy.iloc[0].to_dict()
            A = unscaled_event["AMOUNT_REQ"]
            if A <= 6000:
                r = 0.16
            elif A <= 15000:
                r = 0.18
            else:
                r = 0.20
            interest = A * r
            return interest - cum_cost

        elif self.dataset in ['sepsis_cases_1', 'sepsis_cases_2', 'sepsis_cases_3']:
            if granted:
                health_benefit = 10000
                return health_benefit - cum_cost
            else:
                return -cum_cost

        elif self.dataset == 'hospital_billing_2' and granted:
                revenue = 1000
                return revenue - cum_cost

        else:
            return -cum_cost

def get_real_data(dataset):
        dataset_manager = DatasetManager(dataset)
        df = dataset_manager.read_dataset()
        df = df.rename(columns={dataset_manager.activity_col: "action",
                        dataset_manager.case_id_col: "ID"})

        all_actions = df["action"].unique()

        activity_index = {activity: idx for idx, activity in enumerate(all_actions)}
        df['last_action'] = df['action'].map(activity_index)

        return df, dataset_manager

def define_real_state_cols(dataset, dataset_manager):
    dataset_manager.dynamic_cat_cols.remove(dataset_manager.activity_col)

    for col in ['hour', 'weekday', 'month', 'timesincemidnight',
            'timesincelastevent', 'event_nr', 'open_cases']:
        if col in dataset_manager.dynamic_num_cols:
            dataset_manager.dynamic_num_cols.remove(col)

    if dataset != 'traffic_fines_1' and 'timesincecasestart' in dataset_manager.dynamic_num_cols:
        dataset_manager.dynamic_num_cols.remove('timesincecasestart')

    control_flow_var = dataset_manager.control_flow_var
    state_cols_simulation = ['last_action'] + dataset_manager.dynamic_cat_cols + dataset_manager.dynamic_num_cols + dataset_manager.static_cat_cols + dataset_manager.static_num_cols + control_flow_var

    return state_cols_simulation, control_flow_var
