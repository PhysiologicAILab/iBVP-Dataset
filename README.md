# iBVP Dataset

This repo is associated with our following work, and provides instructions on how to access the dataset and the code.

Joshi, Jitesh, and Youngjun Cho. 2024. "iBVP Dataset: RGB-Thermal rPPG Dataset with High Resolution Signal Quality Labels" Electronics 13, no. 7: 1334. <https://doi.org/10.3390/electronics13071334>

## About iBVP Dataset

The iBVP dataset is a collection of synchronized RGB and thermal infrared videos with PPG ground-truth signals acquired from an ear. The PPG signals are marked with manual signal quality labels, as well as with the SQA-PhysMD model trained and validated to conduct dense (per-sample) signal-quality assessment. The data acquisition protocol was designed to inducing real-world variations in psycho-physiological states, as well as head movement. Each participant experienced four conditions, including (a) rhythmic slow breathing and rest – “A,” (b) an easy math task – “B,” (c) a difficult math task – “C,” and (d) a guided head movement task – “D.” RGB and thermal cameras were positioned in front of the participant at around a 1 m distance. A webcam (Logitech BRIO 4K UHD) was used to capture RGB video frames with 640 × 480 resolution, while thermal infrared frames were captured using thermal camera (A65SC, FLIR system) having 640 × 512 resolution. Frame-rate was set to at 30 FPS for both RGB and thermal acquisition. With 124 sessions, each lasting 3 minutes, the dataset comprises 372 minutes (about 6 hours) of RGB–Thermal video recordings.

## Requesting iBVP Dataset

We have released the iBVP dataset **only for academic research purposes**. The dataset can be requested from the authors by submitting a signed copy of the end-user's licence agreement **[EULA](assets/EULA_iBVP-Dataset.pdf)**. Please note that the EULA form is required to be filled by **academic supervisors**. After submitting the signed EULA to the email addresses mentioned in the EULA, your request will be reviewed and on acceptance, you will receive a link to download the dataset. 

In your email, please include following along with the signed EULA:

- Some words on your research and how the database would be used.
- How you heard of the database (colleague, papers, etc.)


## Dataset Contents

The dataset size (compressed) is ~400 GB. After downloading and extracting the zipped data files, the data needs to be organized as mentioned in the folder structure below:

    iBVP_Dataset/
    |   |-- p01_a/
    |      |-- p01_a_rgb/
    |      |-- p01_a_t/
    |      |-- p01_a_bvp.csv
    |   |-- p01_b/
    |      |-- p01_b_rgb/
    |      |-- p01_b_t/
    |      |-- p01_b_bvp.csv
    |   ...
    |   |-- pii_x/
    |      |-- pii_x_rgb/
    |      |-- pii_x_t/
    |      |-- pii_x_bvp.csv

* **pii_x** indicates following:
  * **ii**: Participant ID
  * **x**: Experimental condition (one out of "a", "b", "c" and "d" as described [above](#about-ibvp-dataset)).
* pii_x_**rgb**: Directory consisting of RGB frames (.bmp)
* pii_x_**t**: Directory consisting of Thermal frames (.raw)
* pii_x_**bvp**: .csv file with following columns:
  * *BVP*: Filtered PPG signals, downsampled at 30 FPS to match with the RGB and thermal video frames.
  * *SQPhysMD*: Signal quality labels generated by our traine [SQA_PhysMD](#sqa-physmd) model.
  * *SQ1*: Manually annotated signal quality labels
  * *Perfusion_Value*: Perfusion index computed from the raw PPG signals.

**Please note:**
The data of few participants who provided limited consent is kept in a separate folder, named "Confidential_No-media-use". The participant IDs with limited consent include following: p08, p10, p13, p16, p29, p31, and p33. Though for training and/or evaluating rPPG methods, this data will have to be moved to the main dataset folder (i.e. iBVP_Dataset folder as described above), please keep this data confidential and extremely secured.

## Code associated with the iBVP dataset

The code is developed by [Youngjun's research group on computational physiology and intelligence](https://www.ucl.ac.uk/uclic/research-projects/2024/jan/physiological-computing-ai) at [UCL GDIH - WHO Collaborating centre for AT](https://www.disabilityinnovation.com/research), [UCL Interaction Centre](https://www.ucl.ac.uk/uclic/), and [UCL Computer Science](https://www.ucl.ac.uk/computer-science/).

This code builds upon the [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox). The changes made in [this fork of the repo](https://github.com/PhysiologicAILab/rPPG-Toolbox) are now merged with the main [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) repo.

## Data Loader for RGB Video Frames

RGB frames are in *.bmp* file format. Here is the [link to dataloader implementation](https://github.com/PhysiologicAILab/rPPG-Toolbox/blob/main/dataset/data_loader/IBVPLoader.py).

## Reading Thermal Video Frames

Thermal frames are in *.raw* file format. While we will soon release the dataloader for thermal video frames, below we provide a snippet to read the *.raw* files.

```py
import numpy as np
import os
thermal_img_width = 640
thermal_img_height = 512
fpath = os.path.join("datadir", "xyz.raw")      #path to the .raw file
thermal_matrix = np.fromfile(fpath, dtype=np.uint16, count=thermal_img_width * thermal_img_height).reshape(thermal_img_height, thermal_img_width)
thermal_matrix = thermal_matrix.astype(np.float32)
thermal_matrix = (thermal_matrix  * 0.04) - 273.15
```

## iBVPNet Model

Implementation of the iBVPNet model can be [found here](https://github.com/PhysiologicAILab/rPPG-Toolbox/blob/main/neural_methods/model/iBVPNet.py).

## MACC Evaluation Metrics

The implementation of Maximum Amplitude of Cross-correlation (MACC) can be [found here](https://github.com/PhysiologicAILab/rPPG-Toolbox/blob/main/evaluation/post_process.py#L52). We will update this space soon as this code is merged with the rPPG-Toolbox. The illustrative infer configs that enables computing MACC metrics is [demonstrated here](https://github.com/PhysiologicAILab/rPPG-Toolbox/blob/main/configs/infer_configs/PURE_UNSUPERVISED.yaml#L5).

## SQA-PhysMD

For the signal quality assessment module (*SQA_PhysMD*), as proposed in [this paper](https://doi.org/10.3390/electronics13071334), the model, inference code and the checkpoint can be found [here](SQA_PhysMD). All *.csv* files of the iBVP dataset can be copied to [this folder](data/ppg_sq/) to run inference using *SQA-PhysMD* model. Any raw PPG signals stored in *.csv* file format can also be used by appropriately changing the data->total_duration_sec and data->window_len_sec values in the config file [*SQAPhysMD.json*](SQA_PhysMD/configs/SQAPhysMD.json). Inference code can be executed with a terminal command as illustrated below:

```bash
python SQA_PhysMD/test_SQAPhysMD.py --config SQA_PhysMD/configs/SQAPhysMD.json --datadir data/ppg_sq --savedir data/ppg_sq_out --preprocess 1
```
SQA_PhysMD is now also integrated with our [PhysioKit repository](https://github.com/PhysiologicAILab/PhysioKit) for real time signal quality assessment of PPG signals. The upsampled version (64 samples per second) in *.npy* format can be further provided upon request for benchmarking with existing signal quality assessment methods as compared in this paper.

## **Additional Support or Reporting Issues with the Library**

For suggestions as well as discussing ideas, please use the [discussion space](https://github.com/PhysiologicAILab/iBVP-Dataset/discussions). Bugs or problems faced while using the iBVP dataset can be reported to the [*Issues*](https://github.com/PhysiologicAILab/iBVP-Dataset/issues) section.

## **Citations**

If you find our [paper](https://doi.org/10.3390/electronics13071334) or this dataset useful for your research, please cite our following works.

```bib
@article{joshi2024ibvp,
    title={iBVP Dataset: RGB-Thermal rPPG Dataset with High Resolution Signal Quality Labels},
    author={Joshi, Jitesh and Cho, Youngjun},
    journal={Electronics},
    publisher={MDPI},
    volume={13},
    year={2024},
    number={7},
    article-number={1334},
    url={https://www.mdpi.com/2079-9292/13/7/1334},
    issn={2079-9292},
    doi={10.3390/electronics13071334}
}

@article{joshi2023physiokit,
    title={PhysioKit: An Open-Source, Low-Cost Physiological Computing Toolkit for Single-and Multi-User Studies},
    author={Joshi, Jitesh and Wang, Katherine and Cho, Youngjun},
    journal={Sensors},
    publisher={MDPI},
    volume={23},
    number={19},
    article-number={8244},
    year={2023},
    url={https://www.mdpi.com/1424-8220/23/19/8244},
    issn={1424-8220},
    doi={10.3390/s23198244}
}
```

Additionally, as we build upon [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox), we would like to sincerely thank and acknowledge the authors of the rPPG-Toolbox. Therefore, we also request the users of the iBVP dataset to cite the [rPPG-Toolbox paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d7d0d548a6317407e02230f15ce75817-Abstract-Datasets_and_Benchmarks.html):

```bib
@inproceedings{liu2024rppg,
    author = {Liu, Xin and Narayanswamy, Girish and Paruchuri, Akshay and Zhang, Xiaoyu and Tang, Jiankai and Zhang, Yuzhe and Sengupta, Roni and Patel, Shwetak and Wang, Yuntao and McDuff, Daniel},
    booktitle = {Advances in Neural Information Processing Systems},
    editor = {A. Oh and T. Neumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
    pages = {68485--68510},
    publisher = {Curran Associates, Inc.},
    title = {rPPG-Toolbox: Deep Remote PPG Toolbox},
    url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/d7d0d548a6317407e02230f15ce75817-Paper-Datasets_and_Benchmarks.pdf},
    volume = {36},
    year = {2023}
}
```
