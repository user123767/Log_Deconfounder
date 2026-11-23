import os
import re

case_id_col = {}
activity_col = {}
resource_col = {}
timestamp_col = {}
label_col = {}
pos_label = {}
neg_label = {}
dynamic_cat_cols = {}
static_cat_cols = {}
dynamic_num_cols = {}
static_num_cols = {}
control_flow_var = {}
control_flow_var_incremental = {}
control_flow_var_binary = {}
environmental_actions = {}
control_flow_var_attribute = {}
filename = {}

base_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(base_dir, "..", "data_logs")

dataset = "SimBank"
filename[dataset] = os.path.join(logs_dir, "SimBank.csv")
case_id_col[dataset] = "case_nr"
activity_col[dataset] = "activity"
resource_col[dataset] = ""
timestamp_col[dataset] = "timestamp"
label_col[dataset] = "label"
pos_label[dataset] = "deviant"
neg_label[dataset] = "regular"

dynamic_cat_cols[dataset] = ["activity"]
static_cat_cols[dataset] = []
dynamic_num_cols[dataset] = ["unc_quality", "est_quality", "cum_cost", "elapsed_time", "interest_rate", "discount_factor", "min_interest_rate"]
static_num_cols[dataset] = ["amount"]

environmental_actions[dataset] = ['receive_acceptance', 'receive_refusal']
control_flow_var_incremental[dataset] = []
control_flow_var_binary[dataset]= []
control_flow_var_attribute[dataset] = ["noc", "nor"]
control_flow_var[dataset] = environmental_actions[dataset] + control_flow_var_incremental[dataset] + control_flow_var_attribute[dataset] + control_flow_var_binary[dataset]

SIMBANK_BASE = "SimBank"
PROXY_PREFIX = "proxy_"
_conf_dicts = [filename, case_id_col, activity_col, resource_col, timestamp_col, label_col, pos_label,
               neg_label, dynamic_cat_cols, static_cat_cols, dynamic_num_cols, static_num_cols,
               control_flow_var, control_flow_var_incremental, control_flow_var_binary,
               environmental_actions, control_flow_var_attribute]


def simbank_dataset_name(intervention, delta):
    if isinstance(intervention, (list, tuple)):
        intervention = "+".join(intervention)
    return f"{SIMBANK_BASE}_{intervention}_delta{float(delta):g}"


def is_simbank(dataset_name):
    return dataset_name == SIMBANK_BASE or dataset_name.startswith(SIMBANK_BASE + "_")


_PROXY_SUFFIX = re.compile(r"_proxy-.+-(?:twins|gaussian)[\d.]+$")


def strip_proxy(dataset_name):
    return _PROXY_SUFFIX.sub("", dataset_name)


def base_dataset(dataset_name):
    if is_simbank(dataset_name):
        return SIMBANK_BASE
    return strip_proxy(dataset_name)


def register_dataset(dataset_name):
    if dataset_name not in filename:
        if is_simbank(dataset_name):
            parent = SIMBANK_BASE
        elif strip_proxy(dataset_name) in filename:
            parent = strip_proxy(dataset_name)
        else:
            raise KeyError(f"unknown dataset '{dataset_name}'")
        for d in _conf_dicts:
            if parent in d:
                v = d[parent]
                d[dataset_name] = v.copy() if isinstance(v, list) else v
        filename[dataset_name] = os.path.join(logs_dir, f"{dataset_name}.csv")
    path = filename[dataset_name]
    if not os.path.isfile(path):
        available = sorted(f[:-4] for f in os.listdir(logs_dir)
                           if f.startswith(SIMBANK_BASE) and f.endswith(".csv")) if os.path.isdir(logs_dir) else []
        raise FileNotFoundError(f"no log for dataset '{dataset_name}' at {path}. "
                                f"SimBank logs in {logs_dir}: {available}")
    if is_simbank(dataset_name) or dataset_name != strip_proxy(dataset_name):
        proxies = proxy_cols_in_log(path)
        static_num_cols[dataset_name] = ([c for c in static_num_cols[dataset_name] if not c.startswith(PROXY_PREFIX)]
                                         + proxies)
    return path


def proxy_cols_in_log(path):
    with open(path, newline="") as f:
        header = f.readline().strip().split(";")
    return [c.strip('"') for c in header if c.strip('"').startswith(PROXY_PREFIX)]


for formula in range(1,3):
    dataset = "traffic_fines_%s"%formula

    filename[dataset] = os.path.join(logs_dir, "traffic_fines_%s.csv"%formula)

    case_id_col[dataset] = "Case ID"
    activity_col[dataset] = "Activity"
    resource_col[dataset] = "Resource"
    timestamp_col[dataset] = "Complete Timestamp"
    label_col[dataset] = "label"
    pos_label[dataset] = "deviant"
    neg_label[dataset] = "regular"

    dynamic_cat_cols[dataset] = ["Activity", 'Resource', "lastSent", "notificationType", "dismissal"]
    static_cat_cols[dataset] = ["article",  "vehicleClass"]
    dynamic_num_cols[dataset] = ["expense", "timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month", "weekday", "hour", "open_cases"]
    static_num_cols[dataset] = ["amount", "points"]

    environmental_actions[dataset] = ['Payment']
    control_flow_var_incremental[dataset] = []
    control_flow_var_binary[dataset]= []
    control_flow_var_attribute[dataset] = []
    control_flow_var[dataset] = environmental_actions[dataset] + control_flow_var_incremental[dataset] + control_flow_var_attribute[dataset] + control_flow_var_binary[dataset]


datasets = ["sepsis_cases_%s" % i for i in range(1, 5)]

for i in [1,2,4]:

    dataset = f"sepsis_cases_{i}"
    filename[dataset] = os.path.join(logs_dir, "%s.csv" % dataset)

    case_id_col[dataset] = "Case ID"
    activity_col[dataset] = "Activity"
    resource_col[dataset] = "org:group"
    timestamp_col[dataset] = "time:timestamp"
    label_col[dataset] = "label"
    pos_label[dataset] = "deviant"
    neg_label[dataset] = "regular"

    dynamic_cat_cols[dataset] = ["Activity", 'org:group']
    static_cat_cols[dataset] = ['Diagnose', 'DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG',
                       'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor',
                       'DiagnosticOther', 'DiagnosticSputum', 'DiagnosticUrinaryCulture',
                       'DiagnosticUrinarySediment', 'DiagnosticXthorax', 'DisfuncOrg',
                       'Hypotensie', 'Hypoxie', 'InfectionSuspected', 'Infusion', 'Oligurie',
                       'SIRSCritHeartRate', 'SIRSCritLeucos', 'SIRSCritTachypnea',
                       'SIRSCritTemperature', 'SIRSCriteria2OrMore']
    dynamic_num_cols[dataset] = ['CRP', 'LacticAcid', 'Leucocytes', "hour", "weekday", "month", "timesincemidnight", "timesincelastevent", "timesincecasestart", "event_nr", "open_cases"]
    static_num_cols[dataset] = ['Age']
    if i == 1:
        environmental_actions[dataset] = ['Return ER']
        control_flow_var_attribute[dataset] = ['recent_release']
        control_flow_var_incremental[dataset] = []
        control_flow_var_binary[dataset]= []

    elif i == 2 or i == 4:
        environmental_actions[dataset] = []
        control_flow_var_attribute[dataset] = []
        control_flow_var_incremental[dataset] = []
        control_flow_var_binary[dataset]= []

    control_flow_var[dataset] = environmental_actions[dataset] + control_flow_var_incremental[dataset] + control_flow_var_attribute[dataset] + control_flow_var_binary[dataset]


dataset = "production"

filename[dataset] = os.path.join(logs_dir, "Production.csv")

case_id_col[dataset] = "Case ID"
activity_col[dataset] = "Activity"
resource_col[dataset] = "Resource"
timestamp_col[dataset] = "Complete Timestamp"
label_col[dataset] = "label"
neg_label[dataset] = "regular"
pos_label[dataset] = "deviant"

static_cat_cols[dataset] = ["Part Desc.", "Rework"]
static_num_cols[dataset] = []
dynamic_cat_cols[dataset] = ['Activity', 'Resource', "Report Type"]
dynamic_num_cols[dataset] = ["Qty Completed", "Qty for MRB", "activity_duration"]
environmental_actions[dataset] = []
control_flow_var_incremental[dataset] = []
control_flow_var_binary[dataset]= []
control_flow_var_attribute[dataset] = ["Work Order  Qty","Qty Rejected", 'Total Qty Rejected']
control_flow_var[dataset] = environmental_actions[dataset] + control_flow_var_incremental[dataset] + control_flow_var_attribute[dataset] + control_flow_var_binary[dataset]


bpic2017_dict = {"bpic2017_cancelled": "BPIC17_O_Cancelled.csv",
                 "bpic2017_accepted": "BPIC17_O_Accepted.csv",
                 "bpic2017_refused": "BPIC17_O_Refused.csv"
                }

for dataset, fname in bpic2017_dict.items():

    filename[dataset] = os.path.join(logs_dir, fname)

    case_id_col[dataset] = "Case ID"
    activity_col[dataset] = "Activity"
    resource_col[dataset] = 'org:resource'
    timestamp_col[dataset] = 'time:timestamp'
    label_col[dataset] = "label"
    neg_label[dataset] = "regular"
    pos_label[dataset] = "deviant"

    dynamic_cat_cols[dataset] = ["Activity", "Accepted", "Selected"]
    static_cat_cols[dataset] = ['ApplicationType', 'LoanGoal']
    dynamic_num_cols[dataset] = ['FirstWithdrawalAmount', 'MonthlyCost', 'NumberOfTerms', 'OfferedAmount', 'CreditScore',  "timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month", "weekday", "hour", "open_cases"]
    static_num_cols[dataset] = ['RequestedAmount']
    environmental_actions[dataset] = ['O_Accepted', 'O_Cancelled', 'O_Returned']
    control_flow_var_incremental[dataset] = ['O_Create Offer', 'W_Call after offers']
    control_flow_var_binary[dataset]= []
    control_flow_var_attribute[dataset] = []
    control_flow_var[dataset] = environmental_actions[dataset] + control_flow_var_incremental[dataset] + control_flow_var_attribute[dataset] + control_flow_var_binary[dataset]


for i in range(1, 7):
        dataset = "hospital_billing_%s" % (i)

        filename[dataset] = os.path.join(logs_dir, "hospital_billing_%s.csv" % (i))

        case_id_col[dataset] = "Case ID"
        activity_col[dataset] = "Activity"
        resource_col[dataset] = "Resource"
        timestamp_col[dataset] = "Complete Timestamp"
        label_col[dataset] = "label"
        neg_label[dataset] = "regular"
        pos_label[dataset] = "deviant"

        if i == 1:
            neg_label[dataset] = "deviant"
            pos_label[dataset] = "regular"

        dynamic_cat_cols[dataset] = ["Activity", 'Resource', 'actOrange', 'actRed', 'blocked', 'caseType', 'diagnosis', 'flagC', 'flagD', 'msgCode', 'msgType', 'state', 'version']
        static_cat_cols[dataset] = ['speciality']
        dynamic_num_cols[dataset] = ['msgCount', "timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month", "weekday", "hour"]
        static_num_cols[dataset] = []

        if i == 2:
            environmental_actions[dataset] = []
            control_flow_var_incremental[dataset] = []
            control_flow_var_binary[dataset]= []
            control_flow_var_attribute[dataset] = ["isCancelled"]
            control_flow_var[dataset] = environmental_actions[dataset] + control_flow_var_incremental[dataset] + control_flow_var_attribute[dataset] + control_flow_var_binary[dataset]

        if i == 1:
            dynamic_cat_cols[dataset] = [col for col in dynamic_cat_cols[dataset] if col != "isCancelled"]
        elif i == 2:
            dynamic_cat_cols[dataset] = [col for col in dynamic_cat_cols[dataset] if col != "isClosed"]


bpic2012_dict = {"bpic2012_cancelled": "bpic2012_O_CANCELLED-COMPLETE.csv",
                 "bpic2012_accepted": "bpic2012_O_ACCEPTED-COMPLETE.csv",
                 "bpic2012_declined": "bpic2012_O_DECLINED-COMPLETE.csv"
                }

for dataset, fname in bpic2012_dict.items():

    filename[dataset] = os.path.join(logs_dir, fname)

    case_id_col[dataset] = "Case ID"
    activity_col[dataset] = "Activity"
    resource_col[dataset] = "Resource"
    timestamp_col[dataset] = "Complete Timestamp"
    label_col[dataset] = "label"
    neg_label[dataset] = "regular"
    pos_label[dataset] = "deviant"

    dynamic_cat_cols[dataset] = ["Activity", "Resource"]
    static_cat_cols[dataset] = []
    dynamic_num_cols[dataset] = ["hour", "weekday", "month", "timesincemidnight", "timesincelastevent", "timesincecasestart", "event_nr", "open_cases"]
    static_num_cols[dataset] = ['AMOUNT_REQ']

    environmental_actions[dataset] = ['O_CANCELLED-COMPLETE', "O_ACCEPTED-COMPLETE", 'O_SENT_BACK-COMPLETE']
    control_flow_var_incremental[dataset] = ['O_CREATED-COMPLETE']
    control_flow_var_binary[dataset]= []
    control_flow_var_attribute[dataset] = []
    control_flow_var[dataset] = environmental_actions[dataset] + control_flow_var_incremental[dataset] + control_flow_var_attribute[dataset] + control_flow_var_binary[dataset]


for municipality in range(1,6):
    for formula in range(1,3):
        dataset = "bpic2015_%s_f%s"%(municipality, formula)

        filename[dataset] = os.path.join(logs_dir, "BPIC15_%s_f%s.csv"%(municipality, formula))

        case_id_col[dataset] = "Case ID"
        activity_col[dataset] = "Activity"
        resource_col[dataset] = "org:resource"
        timestamp_col[dataset] = "time:timestamp"
        label_col[dataset] = "label"
        pos_label[dataset] = "deviant"
        neg_label[dataset] = "regular"

        dynamic_cat_cols[dataset] = ["Activity", "monitoringResource", "question", "org:resource"]
        static_cat_cols[dataset] = ["Responsible_actor"]
        dynamic_num_cols[dataset] = ["hour", "weekday", "month", "timesincemidnight", "timesincelastevent", "timesincecasestart", "event_nr", "open_cases"]
        static_num_cols[dataset] = ["SUMleges", 'Aanleg (Uitvoeren werk of werkzaamheid)', 'Bouw', 'Brandveilig gebruik (vergunning)', 'Gebiedsbescherming', 'Handelen in strijd met regels RO', 'Inrit/Uitweg', 'Kap', 'Milieu (neutraal wijziging)', 'Milieu (omgevingsvergunning beperkte milieutoets)', 'Milieu (vergunning)', 'Monument', 'Reclame', 'Sloop']
        environmental_actions[dataset] = []
        control_flow_var_incremental[dataset] = []
        control_flow_var_binary[dataset]= []
        control_flow_var_attribute[dataset] = []
        control_flow_var[dataset] = environmental_actions[dataset] + control_flow_var_incremental[dataset] + control_flow_var_attribute[dataset] + control_flow_var_binary[dataset]


        if municipality in [3,5]:
            static_num_cols[dataset].append('Flora en Fauna')
        if municipality in [1,2,3,5]:
            static_num_cols[dataset].append('Brandveilig gebruik (melding)')
            static_num_cols[dataset].append('Milieu (melding)')


for formula in range(1,5):
    dataset = "bpic2011_f%s"%formula

    filename[dataset] = os.path.join(logs_dir, "BPIC11_f%s.csv"%formula)

    case_id_col[dataset] = "Case ID"
    activity_col[dataset] = "Activity code"
    resource_col[dataset] = "Producer code"
    timestamp_col[dataset] = "time:timestamp"
    label_col[dataset] = "label"
    pos_label[dataset] = "deviant"
    neg_label[dataset] = "regular"

    dynamic_cat_cols[dataset] = ["Activity code", "Producer code", "Section", "Specialism code.1", "group"]
    static_cat_cols[dataset] = ["Diagnosis", "Treatment code", "Diagnosis code", "Specialism code", "Diagnosis Treatment Combination ID"]
    dynamic_num_cols[dataset] = ["Number of executions", "hour", "weekday", "month", "timesincemidnight", "timesincelastevent", "timesincecasestart", "event_nr", "open_cases"]
    static_num_cols[dataset] = ["Age"]
    environmental_actions[dataset] = []
    control_flow_var_incremental[dataset] = []
    control_flow_var_binary[dataset]= []
    control_flow_var_attribute[dataset] = []
    control_flow_var[dataset] = environmental_actions[dataset] + control_flow_var_incremental[dataset] + control_flow_var_attribute[dataset] + control_flow_var_binary[dataset]
