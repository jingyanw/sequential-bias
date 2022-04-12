# Modeling and Correcting Bias in Sequential Evaluation

This is the data collected for the paper "Modeling and Correcting Bias in Sequential Evaluation". The experiments conducted are approved by the IRB at Georgia Tech.

The participants are recruited on the Prolific crowdsourcing platform, and are required to be based in the US. Additional de-identified meta-data collected for the experiments (including participants click and timing information, and free-form text responses) is available upon request.

## Data

In the data, the field `score_t` is the score reported for item `t`; the field `size_t` is the size of the shape of item `t` presented to the participant. Thus, the ordering of the items is derived as the ordering of the sizes. The `data` directory includes the data from the three experiments:

- `existence.csv`

The experiment to show the existence of the bias (Section 3.2).

The data includes 240 participants each evaluating 5 items. Each of the 120 orderings is presented to 2 participants.

- `relativity.csv`

The experiment to show the relative nature of the scores (Section 3.3).
The data includes 50 participants each evaluating 10 items.

The participants are divided into 2 groups, with 25 participants per group.

- `conflict.csv`

The experiment to show the existence of conflicts (Section 3.4).

The data includes 100 participants each evaluating 5 items. The participants are divided into 2 groups, with 50 participants per group.

The field `comparison` has a value of 1 if the participant reports that they believe the first item is larger than the last item, and has a value of 2 if the participant reports that they believe the first item is smaller than the last item.


## Code for Experiments
To replicate the experimental results reported in the paper (in Section 3), run the command
```
python data_analysis.py
```

The three functions `analyze_existence`, `analyze_relativity` and `analyze_conflict` perform the data analysis for the three experiments, respectively.

## Code for Simulation
To replicate the simulation results reported in the paper, run the command
```
python simulation.py
```

The two functions `simulate_err` and `simulate_err_per_item` run the simulation and plot the results in Figure 4 and Figure 5, respectively.



## Contact
If you have any questions or feedback about the data, code or the paper, please contact Jingyan Wang (jingyanw@gatech.edu).
