# Adversarial Attack project

## Description

This project is aiming to try several adversarial attacks on CIFAR10 data and then try to train models that are robusts to these attacks. 

## Configuration

### Install dependancies

Make sure you have installed all necessary dependancies, I recommend to use a virtual environment. 
A requirements file will soon be available

### Configuration of models and attacks

To configure the project, you might want to modify constants.py file. 
In this file, you can choose to train or load model with different parameters. 
You can also try new attacks with different parameters. 

To load attacked data, make sure you have downloaded "attack_train_PGD.npy" and "attack_test_PGD.npy" and stored it in /Adversarial_attack/data or you can launch new attack and store results


## Run the project

Once you have configured what you want, to run the project, simply execute main.py

## Visualize what was already done 

To visualize what was already done, you can open "Adversarial_Attacks_Project_Summary.ipynb". A report will follow to explain theoretical parts.