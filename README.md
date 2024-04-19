# Predicting the non-coding genome size: the simulation
From the mathematical model of Paul Banse and Juliette Liuselli, this project implements a Python Wright-Fisher simualtion that can align or not with the hypothesis. It also implements a implicit sequence approach, applying iteratively the model out of equilibrium and updating the parameters. 
## Launch an experiment
### Create a configuration file
Use configuration_file_generator.py to create a configuration file:
1. Choose a name for the experiment.
2. Choose the kind of experiment: Mutagenese or Simulation.
3. - If you chose Mutagenese, you can specify which mutations you want to happen, $l_m$, the length distribution of chromosomal rearrangement, the genome structure and the experiment parameters (iterations, the structure parameter that varies). 
	- If you chose Simulation, you can specify the mutation rates, the genome structure and the experiment parameters (generations, popuation size and plot points).
4. By default, a save directory is specified, this is where all the results of the experiment will be saved.
5. Click "Generate configuraion file" and save it where you want (This save folder is NOT where the results of the experiment will be save, it is just the configuration folder).
### Iterative model
```python3 math_model.py path/to/your/config.json [--options]```
where ```--options``` are:
```-i, --iterations```: the number of iterations.
```-t, --time_acceleration```: the Euler step.
In ```math_model.py``` you can find the value of z_nc at equilibrium (bisection search), compute the bias of a specific situation, apply the model iteratively. Results will be saved in the _iterative_model folder.
### Simulation
```python3 __main__.py path/to/your/config.json [--options]```

Options are:

```-s, --save```: If used, the initial proportion is saved.
```-l, --load path/to/individual.pkl```: If used, the initial proportion is created as a population of clones from one individual.
```-c, --checkpoint path/to/checkpoint.pkl```: If used, the simulation starts from a checkpoint population (useful to start a simulation from another evolved population).
```-p, --only_plot```: If used, the main loop is skipped and only the plotting part is done (assume the simulation runned at least once).
```-t, --plot_in_time```: If used, plots will be updated at each plot point, increase execution time.
```-o, --overwrite```: If used, the results will be saved in the previous replica (it won't create a new folder).
```-v, --verbose```: By default, progress is printed every 10 minutes. If used, progress will be shown at each plot point. Can increase execution time.

Every execution creates a folder "X" where X is the number of time the programm was executed (except if ```--overwrite``` options is used). IN THE RESULTS FOLDER, EVERY FOLDER OR FILE OTHER THAN "X" SHOULD START WITH AN UNDERSCORE (for exemple if you want to create a folder "thoughts" it should be names "_thoughts").
### Plotting
Once you have enough replicas, you can execute and plot the mean of the results:
```python3 merge_replicas.py path/to/your/config.json [--options]```
where ```options``` are:
```-m, --mean_mask mean_mask_value```: the last (1-mean_mask_value) will be used to compute the mean value of equilibrium.
```-d, --disable```: If used, the iterative model results with variable Ne won't be plotted (useful when the iterative model diverges).

If ```math_model.py``` has not been run, the plots will only display the simulation. 
### Extract median individual
The ```extract_median_individual.py``` allows to extract and save the median individual from an experiment and saves it in the "wild_type" folder.
Plots will be saved in the _plots folder.








> Written with [StackEdit](https://stackedit.io/).
