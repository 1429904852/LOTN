# LOTN
LOTN is the proposed model in 《Latent Opinions Transfer Network for Target-Oriented
Opinion Words Extraction》, which is accepted by AAAI'2020.

# Dependencies

```angularjs
python==3.5
numpy==1.14.2
tensorflow==1.9
```

# Quick Start

### Step1: pretrained
```angularjs
sh run_pre_trained.sh
```
### step2: transfer
- softmax(train and test)
```angularjs
sh run_LOTN.sh
python eval_LOTN.py
```
- crf(train and test)
```angularjs
train：sh run_LOTN.sh
test：python eval_LOTN_crf.py
```