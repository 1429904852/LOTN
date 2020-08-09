# LOTN
LOTN is the proposed model in 《Latent Opinions Transfer Network for Target-Oriented Opinion Words Extraction》, which is accepted by AAAI'2020.

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
- Softmax
```angularjs
sh run_LOTN.sh
python eval_LOTN.py
```
- CRF
```angularjs
sh run_LOTN.sh
python eval_LOTN_crf.py
```
# Cite
```angularjs
@article{wu2020latent,
  title={Latent Opinions Transfer Network for Target-Oriented Opinion Words Extraction},
  author={Wu, Zhen and Zhao, Fei and Dai, Xin-Yu and Huang, Shujian and Chen, Jiajun},
  journal={arXiv preprint arXiv:2001.01989},
  year={2020}
}
```