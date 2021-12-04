### Deep learning project seed
Use this seed to start new deep learning / ML projects.

- Built in requirements
- Examples with [Transformers](https://huggingface.co/transformers/)
- Badges
- Bibtex

#### Goals
The goal of this seed is to structure ML paper-code the same so that work can easily be extended and replicated.

### DELETE EVERYTHING ABOVE FOR YOUR PROJECT

---

<div align="center">

# Your Project Name

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
<!--
ARXIV
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
<!-- ![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push) -->


<!--
Conference
-->
</div>

## Description
What it does

## How to run
First, install dependencies
```bash
# clone project
git clone https://github.com/YourGithubName/deep-learning-project-template

# install project
cd deep-learning-project-template
conda env create --file environment.yml # or pip install -r requirements.txt
 ```
 Next, use main script to run it.
 ```bash
# module folder
cd project

./run fit --data GLUE --model GLUETransformer --model.model_name_or_path bert-base-uncased --model.max_length 128 --trainer.gpus 1, --trainer.max_epochs 1
```

### Citation
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```
