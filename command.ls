#SimBank
sbatch slurm_files/simbank_data_generation.slurm

sbatch --export=ALL,STEP=cevae slurm_files/SimBank/simbank_int.slurm
sbatch --export=ALL,STEP=agents,TRAIN_DATA=cevae slurm_files/SimBank/simbank_int.slurm
sbatch --export=ALL,STEP=agents,TRAIN_DATA=logged slurm_files/SimBank/simbank_int.slurm

sbatch --export=ALL,STEP=cevae slurm_files/SimBank/simbank_procedure_conf.slurm
sbatch --export=ALL,STEP=agents,TRAIN_DATA=cevae slurm_files/SimBank/simbank_procedure_conf.slurm
sbatch --export=ALL,STEP=agents,TRAIN_DATA=logged slurm_files/SimBank/simbank_procedure_conf.slurm

sbatch --export=ALL,STEP=cevae slurm_files/SimBank/simbank_procedure_proxy.slurm
sbatch --export=ALL,STEP=agents,TRAIN_DATA=cevae slurm_files/SimBank/simbank_procedure_proxy.slurm
sbatch --export=ALL,STEP=agents,TRAIN_DATA=logged slurm_files/SimBank/simbank_procedure_proxy.slurm

## Real-life logs

sbatch slurm_files/find_confounder_to_exclude.slurm
sbatch slurm_files/real_logs_data_augmentation.slurm

### BPIC12
sbatch slurm_files/BPIC12/seq_cevae_bpic12.slurm
sbatch slurm_files/BPIC12/prescribe_bpic12.slurm

### BPIC17
sbatch slurm_files/BPIC17/seq_cevae_bpic17.slurm
sbatch slurm_files/BPIC17/prescribe_bpic17.slurm

### Sepsis 1
sbatch slurm_files/Sepsis1/seq_cevae_sepsis1.slurm
sbatch slurm_files/Sepsis1/prescribe_sepsis1.slurm

### Traffic fines
sbatch slurm_files/Traffic/seq_cevae_traffic.slurm
sbatch slurm_files/Traffic/prescribe_traffic.slurm
