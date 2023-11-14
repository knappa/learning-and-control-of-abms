#!/usr/bin/env python3
# coding: utf-8

# # An-Cockrell model reimplementation

import h5py
import numpy as np
from scipy.stats.qmc import LatinHypercube
from tqdm.auto import trange

import an_cockrell
from an_cockrell import EpiType

# constants
init_inoculum = 100
num_sims = 10_000
num_steps = 2016  # <- full run value

total_T1IFN = np.zeros((num_sims, num_steps), dtype=np.float64)
total_TNF = np.zeros((num_sims, num_steps), dtype=np.float64)
total_IFNg = np.zeros((num_sims, num_steps), dtype=np.float64)
total_IL6 = np.zeros((num_sims, num_steps), dtype=np.float64)
total_IL1 = np.zeros((num_sims, num_steps), dtype=np.float64)
total_IL8 = np.zeros((num_sims, num_steps), dtype=np.float64)
total_IL10 = np.zeros((num_sims, num_steps), dtype=np.float64)
total_IL12 = np.zeros((num_sims, num_steps), dtype=np.float64)
total_IL18 = np.zeros((num_sims, num_steps), dtype=np.float64)
total_extracellular_virus = np.zeros((num_sims, num_steps), dtype=np.float64)
total_intracellular_virus = np.zeros((num_sims, num_steps), dtype=np.float64)
apoptosis_eaten_counter = np.zeros((num_sims, num_steps), dtype=np.float64)
infected_epis = np.zeros((num_sims, num_steps), dtype=np.float64)
dead_epis = np.zeros((num_sims, num_steps), dtype=np.float64)
apoptosed_epis = np.zeros((num_sims, num_steps), dtype=np.float64)
system_health = np.zeros((num_sims, num_steps), dtype=np.float64)

default_params = dict(
    GRID_WIDTH=51,
    GRID_HEIGHT=51,
    is_bat=False,
    init_inoculum=100,
    init_dcs=50,
    init_nks=25,
    init_macros=50,
    macro_phago_recovery=0.5,
    macro_phago_limit=1_000,
    inflammasome_activation_threshold=10,  # default 50 for bats
    inflammasome_priming_threshold=1.0,  # default 5.0 for bats
    viral_carrying_capacity=500,
    susceptibility_to_infection=77,
    human_endo_activation=5,
    bat_endo_activation=10,
    bat_metabolic_byproduct=2.0,
    human_metabolic_byproduct=0.2,
    resistance_to_infection=75,
    viral_incubation_threshold=60,
)

variational_params = [
    "init_inoculum",
    "init_dcs",
    "init_nks",
    "init_macros",
    "macro_phago_recovery",
    "macro_phago_limit",
    "inflammasome_activation_threshold",
    "inflammasome_priming_threshold",
    "viral_carrying_capacity",
    "susceptibility_to_infection",
    "human_endo_activation",
    "human_metabolic_byproduct",
    "resistance_to_infection",
    "viral_incubation_threshold",
]

param_list = np.zeros((num_sims, len(variational_params)), dtype=np.float64)

lhc = LatinHypercube(len(variational_params))
sample = 1.0 + 0.5 * (lhc.random(n=num_sims) - 0.5)  # between 75% and 125%

for sim_idx in trange(num_sims, desc="simulation"):
    # generate a perturbation of the default parameters
    params = default_params.copy()
    pct_perturbation = sample[sim_idx]
    for pert_idx, param in enumerate(variational_params):
        if isinstance(params[param], int):
            params[param] = int(pct_perturbation[pert_idx] * int(params[param]))
        else:
            params[param] = pct_perturbation[pert_idx] * params[param]

    param_list[sim_idx, :] = np.array([params[param] for param in variational_params])

    model = an_cockrell.AnCockrellModel(**params)

    for step_idx in trange(num_steps):
        model.time_step()

        total_T1IFN[sim_idx, step_idx] = model.total_T1IFN
        total_TNF[sim_idx, step_idx] = model.total_TNF
        total_IFNg[sim_idx, step_idx] = model.total_IFNg
        total_IL6[sim_idx, step_idx] = model.total_IL6
        total_IL1[sim_idx, step_idx] = model.total_IL1
        total_IL8[sim_idx, step_idx] = model.total_IL8
        total_IL10[sim_idx, step_idx] = model.total_IL10
        total_IL12[sim_idx, step_idx] = model.total_IL12
        total_IL18[sim_idx, step_idx] = model.total_IL18
        total_extracellular_virus[sim_idx, step_idx] = model.total_extracellular_virus
        total_intracellular_virus[sim_idx, step_idx] = model.total_intracellular_virus
        apoptosis_eaten_counter[sim_idx, step_idx] = model.apoptosis_eaten_counter
        infected_epis[sim_idx, step_idx] = np.sum(model.epithelium == EpiType.Infected)
        dead_epis[sim_idx, step_idx] = np.sum(model.epithelium == EpiType.Dead)
        apoptosed_epis[sim_idx, step_idx] = np.sum(
            model.epithelium == EpiType.Apoptosed
        )
        system_health[sim_idx, step_idx] = model.system_health

with h5py.File("run-statistics.hdf5", "w") as f:
    # dset = f.create_dataset("mydataset", (100,), dtype="i")

    f.create_dataset(
        "total_T1IFN", (num_sims, num_steps), dtype=np.float64, data=total_T1IFN
    )
    f.create_dataset(
        "total_TNF", (num_sims, num_steps), dtype=np.float64, data=total_TNF
    )
    f.create_dataset(
        "total_IFNg", (num_sims, num_steps), dtype=np.float64, data=total_IFNg
    )
    f.create_dataset(
        "total_IL6", (num_sims, num_steps), dtype=np.float64, data=total_IL6
    )
    f.create_dataset(
        "total_IL1", (num_sims, num_steps), dtype=np.float64, data=total_IL1
    )
    f.create_dataset(
        "total_IL8", (num_sims, num_steps), dtype=np.float64, data=total_IL8
    )
    f.create_dataset(
        "total_IL10", (num_sims, num_steps), dtype=np.float64, data=total_IL10
    )
    f.create_dataset(
        "total_IL12", (num_sims, num_steps), dtype=np.float64, data=total_IL12
    )
    f.create_dataset(
        "total_IL18", (num_sims, num_steps), dtype=np.float64, data=total_IL18
    )
    f.create_dataset(
        "total_extracellular_virus",
        (num_sims, num_steps),
        dtype=np.float64,
        data=total_extracellular_virus,
    )
    f.create_dataset(
        "total_intracellular_virus",
        (num_sims, num_steps),
        dtype=np.float64,
        data=total_intracellular_virus,
    )
    f.create_dataset(
        "apoptosis_eaten_counter",
        (num_sims, num_steps),
        dtype=np.float64,
        data=apoptosis_eaten_counter,
    )
    f.create_dataset(
        "infected_epis", (num_sims, num_steps), dtype=np.float64, data=infected_epis
    )
    f.create_dataset(
        "dead_epis", (num_sims, num_steps), dtype=np.float64, data=dead_epis
    )
    f.create_dataset(
        "apoptosed_epis", (num_sims, num_steps), dtype=np.float64, data=apoptosed_epis
    )
    f.create_dataset(
        "system_health", (num_sims, num_steps), dtype=np.float64, data=system_health
    )
