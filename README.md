
# Pedestrian Safety in Autonomous Vehicles

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository supports our research paper on "Enhancing Pedestrian Safety in AVs with Multiple Trajectory Prediction and Action Recognition" by Issa Nasralli and Imen Masmoudi.

## Overview

Intent estimation systems are crucial for enhancing pedestrian safety in Autonomous Vehicles (AVs). These systems involve the development of classifiers that analyze a sequence of preceding frames to determine pedestrian intent within a given time period. In our paper, we explore the pivotal components within this dynamic field, especially object detection, object tracking, and intent estimation.

## Paper Link

Our research paper is available [here](#) (Link to be provided).

## Repository Contents

- `Pedetstrian-Detection/`: This folder contains dataset and tlite Google models used in our analyzis to select the best model.
- `Pedetstrian-Tracking/`: This folder contains SORT algorithm python code.
- `RNAC/`: This folder contains
   RNN_model_creating.py for creating model
   RNN_model_training.py for training model and saving testing results
   comparative_table.py for extracting result from results saved during testing of the trained model
   create_auto_dataset.py for creating Dataset from JAAD
   "lr 0.00001.rar", "lr 0.0001.rar" and "lr 0.001.rar" are comressed folder which contains our testing results.
- `Trajectory-and-Collision-Prediction.py/`: This file is the program of immediate collision prediction which rely on pedestrian detection and future collision prediction by determining the future position of the pedestrian based on its predcted trajectory.
- 


