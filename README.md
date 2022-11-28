# Multi-Task Learning of Object State Changes from Uncurated Videos

### [[Project Website :dart:]](https://soczech.github.io/multi-task-object-states/)&nbsp;&nbsp;&nbsp;[[Paper :page_with_curl:]](https://arxiv.org/abs/2211.13500)&nbsp;&nbsp;&nbsp;[Code :octocat:]

This repository contrains code for the paper [Multi-Task Learning of Object State Changes from Uncurated Videos](https://arxiv.org/abs/2211.13500).

<img src="https://soczech.github.io/assets/img/MultiTaskObjectStates.svg" style="width:100%">


## Train the model on ChangeIt dataset
1. **Setup the environment**
   - Our code can be run in a docker container. Build it by running the following command.
     Note that by default, we compile custom CUDA code for architectures 6.1, 7.0, 7.5, 8.0, and 8.6.
     You may need to update the Dockerfile with your GPU architecture. 
     ```
     docker build -t multi-task-object-states .
     ```
   - Go into the docker image.
     ```
     docker run -it --rm --gpus all -v $(pwd):$(pwd) -w $(pwd) --user=$(id -u $USER):$(id -g $USER) multi-task-object-states bash
     ```

2. **Download requirements**
   - Our code requires CLIP repository, CLIP model weights, and the ChangeIt dataset annotations.
     Run `./download_requirements.sh` to obtain those dependencies or download them yourselves.

3. **Download dataset**
   - To replicate our experiments on the ChangeIt dataset, the dataset videos are required.
     Please download them and put them inside `videos/*category*` folder.
     See [ChangeIt GitHub page](https://github.com/soCzech/ChangeIt) on how to download them.

4. **Train a model**
   - Run the training.
     ```
     python train.py --video_roots ./videos
                     --dataset_root ./ChangeIt
                     --train_backbone
                     --augment
                     --local_batch_size 2
     ```
   - We trained the model on 32 GPUs, i.e. batch size 64.
   - To run the code on multiple GPUs, simply run the code on a machine with multiple GPUs.
   - To run the code on multiple nodes, run the code once on each node.
     If you are not running on slurm, you also need to set environment variable `SLURM_NPROCS`
     to the total number of nodes and the variable `SLURM_PROCID` to the node id starting from zero.
     Make sure you also set `SLURM_JOBID` to some unique value.


## Train the model on your dataset
- To train the model on your dataset, complete steps **1.** and **2.** from above.
- Put your videos into `*dir*/*category*` for every video category `*category*`.
- Put your annotations for selected videos into `*dataset*/annotations/*category*`.
  Use the same [format](https://github.com/soCzech/ChangeIt/tree/main/annotations) as in the case of ChangeIt dataset.
- Run the training.
  ```
  python train.py --video_roots *dir*
                  --dataset_root *dataset*
                  --train_backbone
                  --augment
                  --local_batch_size 2
                  --ignore_video_weight
  ```
- `--ignore_video_weight` option ignores noise adaptive weighting done for noisy ChangeIt dataset.
  To use the noise adaptive weighting, you need to provide `*dataset*/categories.csv` and `*dataset*/videos/*category*.csv` files as well.


## Use a trained model
Here is an example code for the inference of a trained model.
```python
checkpoint = torch.load("path/to/saved/model.pth", map_location="cpu")
model = ClipClassifier(params=checkpoint["args"],
                       n_classes=checkpoint["n_classes"],
                       hidden_mlp_layers=checkpoint["hidden_mlp_layers"]).cuda()
model.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()})

video_frames = torch.from_numpy(
    extract_frames(video_fn, fps=1, size=(398, 224), crop=(398 - 224, 0)))

with torch.no_grad():
    predictions = model(video_frames.cuda())
state_pred, action_pred = torch.softmax(predictions["state"], -1), torch.softmax(predictions["action"], -1)
```


## Citation
```bibtex
@article{soucek2022multitask,
    title={Multi-Task Learning of Object State Changes from Uncurated Videos},
    author={Sou\v{c}ek, Tom\'{a}\v{s} and Alayrac, Jean-Baptiste and Miech, Antoine and Laptev, Ivan and Sivic, Josef},
    month = {November},
    year = {2022}
}
```


## Acknowledgements
This work was partly supported by the European Regional Development Fund under the project IMPACT (reg. no. CZ.02.1.01/0.0/0.0/15_003/0000468), the Ministry of Education, Youth and Sports of the Czech Republic through the e-INFRA CZ (ID:90140), the French government under management of Agence Nationale de la Recherche as part of the “Investissements d’avenir” program, reference ANR19-P3IA-0001 (PRAIRIE 3IA Institute), and Louis Vuitton ENS Chair on Artificial Intelligence.

The ordering constraint code has been adapted from the CVPR 2022 paper 
[Look for the Change: Learning Object States and State-Modifying Actions from Untrimmed Web Videos](https://soczech.github.io/look-for-the-change/)
available on [GitHub](https://github.com/soCzech/LookForTheChange).
