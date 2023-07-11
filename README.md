# Conditioned Cooperative Training for Semi-supervised Weapon Detection

Repository with training code for the paper "Conditioned Cooperative Training for Semi-supervised Weapon Detection".

## Requirements:

1. Install conda (https://docs.conda.io/projects/conda/en/latest/user-guide/install/).
2. Create a conda environment with the command:
```
conda create --name <env_name> python=3.8
```
3. Activate the environment with the command:
```
conda activate <env_name>
```
4. Install the requirements using the command:
```
pip install -r requirements.txt
```

## How to use the code
Change the scripts train-yt.sh and train-granada.sh to point to the correct paths of your datasets and the env_name of your conda environment. Then, run the scripts with the command:
```
bash train-yt.sh
```
or
```
bash train-granada.sh
```

## Datasets

The datasets used in the paper are available at:
- YouTube Dataset [[1]](#1).
- UGR Dataset [[2]](#2).
- Instagram Dataset.

The Instagram dataset was scraped from Instagram using the hashtag #handgun using the library instaloader ([https://github.com/instaloader/instaloader](https://github.com/instaloader/instaloader)). The script was used with the command:
```
python3 -m instaloader "#handgun" --login <MY_USER> --no-profile-pic --no-captions --no-metadata-json --no-videos --dirname-pattern images
```
Due to rights conflicts, this Instagram dataset is not available in this repository. However, you can recolect it using the previous command or contact us for a request of data upon express application and acceptance of rights for academic purposes only.

## References
<a id="1">[1]</a>
Gu Yongxiang and Liao Xingbin and Qin Xiaolin (2022)
YouTube-GDD: A challenging gun detection dataset with rich contextual information.
arXiv preprint arXiv:2203.04129

<a id="2">[2]</a> 
Olmos, R., Tabik, S., & Herrera, F. (2018)
Automatic handgun detection alarm in videos using deep learning.
Neurocomputing, 275, 66-72. doi.org/10.1016/j.neucom.2017.05.012
