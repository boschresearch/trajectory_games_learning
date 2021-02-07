# trajectory_games_learning

This is the companion code for the method reported in the paper
"Learning game-theoretic models of multiagent trajectories using implicit layers" by Geiger and Straehle. The paper can
be found here https://arxiv.org/abs/2008.07303. The code allows the users to
reproduce and extend the results reported in the paper.  Please cite the
above paper when reporting, reproducing or extending the results.

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication "Learning game-theoretic models of multiagent trajectories using implicit layers". It will neither be maintained nor monitored in any way.

## Requirements

To train the model an implementation of 

[1] Xiao Wang, Shiqian Ma, Donald Goldfarb, Wei Liu. "Stochastic quasi-Newton methods for nonconvex stochastic optimization." SIAM Journal on Optimization 27.2 (2017): 927-956.
[2] Yingkai Li and Huidong Liu. "Implementation of Stochastic Quasi-Newton's Method in PyTorch." arXiv preprint arXiv:1805.02338, 2018.

is required (for example https://github.com/harryliew/SdLBFGS/blob/master/optim/sdlbfgs.py)
an implementation should be placed in the file "tg/sdlbfgs.py".

hee track data is included.

To train/evaluate on the highD dataset the dataset and the source code from https://github.com/RobertKrajewski/highD-dataset
have to be obtained and placed in the directory "data/highD".

## Running

to train the sub-space predictor run the file train_ssp.py, a trained model will be stored in "trained_models".
to train the game run the file train_game.py, a trained model will be stored in "trained_models".
to evaluate the overall model run the file evaluate_game.py, it uses the trained models in "trained_mdels".

all three scripts allow to change the data-set [highD, hee] in a settings variable in the beginning of the file.

## HEE data set preprocessing scripts etc.

This repository also contains scripts that help preprocessing the data from the HEE data set that was published alongside the paper. These scripts may be helpful as a preprocessing step also for other methods to be applied to the HEE data set:
- "data/hee_loading_and_preprocessing/hee_example.py": This is probably the most instructive script -- it shows an example of how to run the preprocessing/filtering and visualizes a trajectory.
- "data/hee_loading_and_preprocessing/hee_loading_and_preprocessing.py"
- "data/hee_loading_and_preprocessing/hee_preprocessing.py"
- "data/hee_loading_and_preprocessing/hee_coordinate_transform.py"

## License

PROJECT-NAME is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

