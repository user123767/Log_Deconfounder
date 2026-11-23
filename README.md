# Log Deconfounder

Prescriptive process monitoring (PresPM) is a branch of process mining that recommends runtime interventions to improve process outcomes. Existing PresPM methods learn offline from event logs. These methods assume that the logs record all factors that influenced both the process actors’ decisions and the process outcomes, i.e., that there are no unmeasured confounders. However, this assumption may not hold in practice, and ignoring such confounders may bias the estimated effects of interventions. If recorded attributes or the control flow of cases carry information about the unmeasured confounders, they can serve as noisy proxies when estimating a substitute for these confounders. We propose Log Deconfounder, a method that generates a deconfounded version of the event log on which any single-intervention PresPM method can be trained. Using a causal effect variational autoencoder adapted to event sequences, we first infer latent substitutes for the unmeasured confounders from proxies in the process executions recorded in the event log, while jointly learning how they affect the intervention assignment in the log and the process outcomes. We then use the learned latent variables to generate a deconfounded event log in which interventions are assigned independently of the substitutes. Our experiments show that, for the evaluated PresPM methods, training on the deconfounded log instead of the original one generally improves outcomes under unmeasured confounding.

## Repository structure

```
log_deconfounder/
├── command.ls                         # every sbatch command, in run order
├── data_logs/                         # event logs (only original event logs are included (without proxies), see Data)
├── dataset_manager/                   # dataset settings and log loading
├── SimBank/                           # SimBank simulator (De Moor et al.), see SimBank/README.md
├── slurm_files/
│   ├── simbank_data_generation.slurm  # SimBank logs
│   ├── find_confounder_to_exclude.slurm
│   ├── real_logs_data_augmentation.slurm
│   ├── SimBank/                       # training sequence CEVAE + agents on original and deconfounded logs for SimBank process
│   └── BPIC12/ BPIC17/ Sepsis1/ Traffic/   # training sequence CEVAE + agents on original and deconfounded logs for real-world processes
└── src/
    ├── simbank_data_generation.py     # SimBank logs per intervention, confounding level and proxy setting
    ├── seq_cevae_simbank.py           # sequence CEVAE on SimBank 
    ├── Log_deconfounder.py            # agents on SimBank + online evaluation
    ├── select_exclude_col_single.py   # ranks confounder candidates in a real log
    ├── real_log_proxies.py            # adds noisy proxies of a confounder to a real log
    ├── seq_cevae_log.py               # sequence CEVAE on a real-world log 
    ├── prescribe_real_log.py          # agents on a real-world log + FQE evaluation
    ├── cevae.py, seq_cevae.py         # (sequence) CEVAE model
    ├── tabular_agents.py, causal_rl_agent.py, real_log_agents.py   # agents
    ├── fqe_eval.py                    # off-policy evaluation
    ├── proxies.py                     # proxy definitions shared by data generation and evaluation
    ├── MDP_functions.py               # ProcessEnv: loads, encodes and splits a log
    ├── two_way_deconfounder.py, prefix_deconfounder.py, stage_deconfounder.py   # event and prefix datasets, Q-networks
    └── dataset_params_<intervention>.pkl   # SimBank settings
```

## Setup

Python 3.10 or newer. Install the dependencies in the environment the slurm
files activate (`rl_env`):

```
pip install -r requirements.txt
```

## Data

Download the event logs from this link: https://drive.google.com/drive/folders/1N9vkJ_Fd-MurLAC-aZC-fZzKl7SiexCI?usp=share_link
Place the downloaded logs in `data_logs/`:

| Dataset | File | Intervention (decision point → treatment) | Hidden confounder |
|---|---|---|---|
| `bpic2017_accepted` | `BPIC17_O_Accepted.csv` | first offer → additional offer | `CreditScore` |
| `bpic2012_accepted` | `bpic2012_O_ACCEPTED-COMPLETE.csv` | first offer → additional offer | `AMOUNT_REQ` |
| `traffic_fines_1` | `traffic_fines_1.csv` | fine notification → add penalty | `lastSent` |
| `sepsis_cases_1` | `sepsis_cases_1.csv` | ER triage → IV antibiotics before admission/release | `Infusion` |


SimBank logs are generated (see below) and written to `data_logs/` as `SimBank_<intervention>_delta<δ>_proxy-<cols>-<type><noise>.csv`, where δ is
the share of cases following the confounded bank policy and the proxies are noisy copies of the confounders `amount` and `est_quality` (`gaussian`: added noise sd; `twins`: bit-flip probability).

## Running the experiments

All commands are run from the repository root.
`command.ls` lists all commands below.

### SimBank

1. Generate the logs:
   ```
   sbatch --array=0-1 slurm_files/simbank_data_generation.slurm
   ```
   Set `DELTAS`, `PROXY_TYPE` and `PROXY_NOISES` in the slurm file to the grid of the experiment you run.

2. Run an experiment. There are three experiments: `int` (`set_ir_3_levels`, hidden `amount`), `procedure_conf`
   (`choose_procedure`, 12 confounding levels, hidden `amount`, `est_quality` or both) and `procedure_proxy`
   (`choose_procedure`, δ = 0.9, proxy type and noise varied). Each experiment file has two steps:
   ```
   bash slurm_files/SimBank/simbank_int.slurm      # prints the --array range of each step
   sbatch --array=0-119 --export=ALL,STEP=cevae slurm_files/SimBank/simbank_int.slurm
   sbatch --array=0-359 --export=ALL,STEP=agents,TRAIN_DATA=cevae slurm_files/SimBank/simbank_int.slurm
   sbatch --array=0-359 --export=ALL,STEP=agents,TRAIN_DATA=logged slurm_files/SimBank/simbank_int.slurm
   ```
   `STEP=agents,TRAIN_DATA=cevae` needs the output of `STEP=cevae`; `TRAIN_DATA=logged` can run right away.

### Real-world logs

1. Rank confounder candidates:
   ```
   sbatch slurm_files/find_confounder_to_exclude.slurm
   ```
2. Add proxies of the chosen hidden confounder to the logs:
   ```
   sbatch slurm_files/real_logs_data_augmentation.slurm
   ```
3. Per log, train the CEVAE model, then the PresPM agents:
   ```
   sbatch slurm_files/BPIC17/seq_cevae_bpic17.slurm
   sbatch slurm_files/BPIC17/prescribe_bpic17.slurm
   ```
   and likewise for `BPIC12/`, `Sepsis1/` and `Traffic/`.

