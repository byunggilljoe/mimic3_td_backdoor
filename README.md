# "Temporal-dependence" backdoor attack on machine learning models with EHRs
This is a repository for a AAAI 2020 TAIH workshop paper "Machine Learning with Electronic Health Records is vulnerable to Backdoor Trigger Attacks"
- [Paper link](https://taih20.github.io/papers/23/CameraReady/Adversarial_medical_data__AAAI_2020_Workshop_merged.pdf)
- [Workshop link](https://taih20.github.io/)

# Base code
Our code is implemented based on https://github.com/YerevaNN/mimic3-benchmarks that was originally for multi-task models with EHRs.

# Dependency
- pytorch-gpu 1.5.1
- torchvision 0.6.0a0+35d732a
- tensorflow 2.1.0
- keras 2.3.1
- numpy 1.19.2
- scikit-learn 1.5.1

# Data aquisition
MIMIC-III data can be aquired after submitting a certificate.
Please refer [__Requirement__](https://github.com/YerevaNN/mimic3-benchmarks#Requirements) section and following data processing instructions 
in the original repository.

# Generating a trigger based on temporal dependence
We first to generate a trigger based on temporal dependence to generate a poisoned dataset.
```bash
python -um mimic3models.in_hospital_mortality.torch.poisoning_generate_pattern_raw_48_76
```
# Training victim models trained on poisoned datasets
Then, we generate poisoned datasets and train victim models with these datasets with following parameters.
- __--model__: victim model <lr,mlp,lstm>.
- __--poisoning_proportion__: proportion of poisoning data over entire dataset. [0.0, 1.0]
- __--poisoning_strength__: strength (Mahalanobis distance) of a trigger. [0.0, +inf]
- __--poison_imputed__: values where trigger is applied. <notimputed,all>

## Victim model = logistic regression
```bash
python -um mimic3models.in_hospital_mortality.torch.poisoning_train_raw_714 --model lr \
      --poisoning_proportion 0.05 --poisoning_strength 2.0 --poison_imputed notimputed
```
## Victim model = mlp
```bash
python -um mimic3models.in_hospital_mortality.torch.poisoning_train_raw_714 --model mlp -\
      -poisoning_proportion 0.05 --poisoning_strength 2.0 --poison_imputed notimputed
```
## Victim model = lstm
```bash
python -um mimic3models.in_hospital_mortality.torch.poisoning_train_raw_48_76  \
      --poisoning_proportion 0.05 --poisoning_strength 2.0 --poison_imputed notimputed
```

# Backdoor Attack on the victim models
## Victim model = logistic regression
```bash
python -um mimic3models.in_hospital_mortality.torch.poisoning_attack_raw_714 --poisoning_proportion 0.05 \
        --poisoning_strength 2.0 --poison_imputed notimputed --model mlp
```
## Victim model = mlp
```bash
python -um mimic3models.in_hospital_mortality.torch.poisoning_attack_raw_714  --poisoning_proportion 0.05 \
        --poisoning_strength 2.0 --poison_imputed notimputed --model lr
```
## Victim model = lstm
```bash
python -um mimic3models.in_hospital_mortality.torch.poisoning_attack_raw_48_76  --poisoning_proportion 0.05 \
         --poisoning_strength 2.0 --poison_imputed notimputed
```
# Contacts
If you have problem, please contact byunggill.joe@gmail.com or create an issue.
