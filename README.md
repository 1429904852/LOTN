# LOTN
LOTN is the proposed model in 《[Latent Opinions Transfer Network for Target-Oriented Opinion Words Extraction](https://arxiv.org/pdf/2001.01989.pdf)》, which is accepted by AAAI'2020.

# Dependencies

```bash
python==3.5
numpy==1.14.2
tensorflow==1.9
```
# Quick Start

### Step1: pretrained
```bash
sh run_pre_trained.sh
```
### step2: transfer
- Softmax
```bash
sh run_LOTN.sh
python eval_LOTN.py
```
- CRF
```bash
sh run_LOTN.sh
python eval_LOTN_crf.py
```
# Cite
```bash
@article{wu2020latent,
  title={Latent Opinions Transfer Network for Target-Oriented Opinion Words Extraction},
  author={Wu, Zhen and Zhao, Fei and Dai, Xin-Yu and Huang, Shujian and Chen, Jiajun},
  journal={arXiv preprint arXiv:2001.01989},
  year={2020}
}
```

if you have any questions, please contact me zhaof@smail.nju.edu.cn.