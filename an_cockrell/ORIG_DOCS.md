# Name of model: Comparative Biology Immune ABM (CBIABM)

Purpose: Simulate those aspects of innate immune response different between bats and humas. These areas are:

1. Decreased sensativity to P/DAMP due to high metabolic rate
2 Decreased ramp up of T1IFNs in response to perturbation => intrinsic to this is that baseline is higher?
3. Decreased inflammasome activation => consequences are decreased caspase-1 activation which leads to decreased IL1/IL18 and decreased pyroptosis
4. Decreased processing/release of IL1

---

## Guide to World view:

The  World  view, consisting of the 51 x 51  square grid, can be seen on the right side of the User Interface. The following is a key to the agent shapes seen on that view:

**_~~_~~~~* Blue Squares = Healthy Epithelial Cells
* Yellow Squares = Infected Epithelial Cells
* Grey Squares = Epithelial Cells killed by necrosis
* Grey Pentagons = Epithelial Cells killed by apoptosis
* Green Circles = Macrophages
* Large Green Circles = Macrophages at phagocytosis limit
* Orange Circles = NK Cells
* Light Blue Triangles = Dendritic Cells
* Pink Square Outlines = Activated Endothelial Cells
* Small White Circles = PMNs~~~~_~~_**

At initialization the entire world-space is filled with Healthy Epithelial Cells (Blue Squares). When dead Epithelial Cells of both types are cleared by phagocytosis what remains is black space, which can then be refilled be new Epithelial Cells as they heal (see main text for rules). The levels of extracellular virus and different mediators can be seen in the background behind the agents; see “Visualization Controls” below how to hide and reshow Epithelial and Endothelial Cells.

---

## Modeled Behaviors and Functions

Primary means of controlling virus is to kill infected epis
### Mechanisms:
1. Infected-epis undergo apoptosis (eventually)
2. NK cells enhance apoptosis
3. Macros consume extracellular-virus => Not coupled to macro-activation or inflammasome activation

### Model behavior criteria:
1. Bats have a higher baseline generation of P/DAMP, therefore macro-activation threshold is higher
2. Ideally bats should have longer/persistent virus present, mostly intracellular, in dynamic equilibrium => decrease apoptosis (longer virus factory duration), decrease pyroptosis/macros-full => we know that in the model increasing full-macros leads to impaired viral clearance => Rely on epitheilial-mediated viral control (avoid full-macro point?)
3. Ideally generate shedding state (extracellular virus) from baseline state by increasing P/DAMPs => "stress state"

### Modeling Baseline P/DAMP
Called in "metabolism" command, has epi cells randomly generate "metabolic-byproduct" which is the update to P/DAMP.
Random is 1/100, metabolic-byproduct slider from 0.1 to 1.0 updated by 0.01. Note: update of total-P/DAMP must occur prior to evap/diffuse/cleanup otherwise no detectable value

### Modeling Viral Infection and Replication

Viral infecton and replication: stochastic process, but intial infection synchronizes, period of intracellular replication to reach threshold at which time infected cell starts to leak virus, leaking of virus consumes cell membrane until cell dies. NO BURST EFFECT, though cell death in this fashion leads to the production of P/DAMPs; supposition is that this does not occur due to apoptosis being activated. So hyperinflammation occurs either because epi apoptosis is interrupted (some viruses do this) or due to pyroptosis (less likely necrosis) of macros leads to P/DAMP production. 

Virus invasion uses susceptability-to-infection as a variable for epi cells to be infected. The calcuation is random 100 < max list susceptability-to-infection extracellular-virus. This means more likely for invasion if higher extracellular-virus, but possibility exists if there are extracellular-virus present.
Also, min value of extracellular-virus set to 1 (cleanup command), diffusion of extracellular virus changed to 0.05 (working with susceptability-to-infection = 25)

#### Epi Cell Function:

Epi cell death due to non-apoptosis => called in Epi-function
Death by ROS produces + 10 P/DAMP

#### Infected Epi cell functions:

Epi cell apoptosis: Counter that starts with infection, leads to cell death prior to full cell membrane consumption point, effect is to reduce overall level of virus, but does not help infected cell. Dies without generating P/DAMPs.
This version sets the epi-apoptosis-threshold to 450 + random 100, essentially half of the total cell membrane with randomness added. Earlier versions with 475 + random 50 and NK enhancement + 1 lead to pretty abrupt die-off just after Day 30.


#### NK cell functions:

NK cells chemotax to T1IFN, IL12 and IL18 (need all three to be present). Kills infected-epis by inducing faster apoptosis

#### Macro Functions:

Macros are now forced towards baseline inactivation, -1 with pro-, +1 with anti. Mostly impacts antis, allows recovery of IL10 levels

There is a distinction between pro-inflammatory state (activation) and inflammasome activation, supposition is that the determination of the M1 macrophage phenotype involves more than just inflammasome activation

#### Inflammasome Properties:

Inflammasome activation 2nd stage based on virus-eaten, no function infection of macros.

Role of IL1:
IL6 or P/DAMPS are needed for activation of NFkB, which is needed for TNF, IL6 and IL10; these now get activated by IL1 + P/DAMP > 1
IL1 more simply needed for IL6 production by macros and DCs, added to macro-activation-level determination (okay since it is not produced directly as part of that conditional statement), IL1 + TNF > 1 needed for infected-epi production of IL6
Need to turn inflammasome priming down to 1.0 in order to get activation and production of IL1 (which then allows production of TNF, IL6 and IL10).
IL1 and IL18 production, instead of build up and burst at pyroptosis releasing all at once, start producing IL1/IL18 upon inflammasome activation, with lessor burst at pyroptosis.


Pyroptosis from activated inflammasome
IL1 is now made as pre-IL1 during period between inflammasome activation and pyroptosis (120 minutes), pre-IL1 increments by + 10 so IL1 is 120 upon macro death
Increases P/DAMPS + 10
Creates a naive macro and jumps away (macro steady state)

Anti-viral effect T1IFN

Adds ability to have baseline T1IFN generation => bats do this
Controled by Switch
Random 100 = 1 => Set T1IFN + 0.15
Adds antiviral effect T1IFN => inihibits viral-replication by - T1IFN/10 on viral replication, does not eradicate but keeps minimum of 1 (human = - T1IFN/100)
Infected Epis make T1IFN + 1 and IL18 + 0.11
 
Bat Baseline Metabolism-byproduct = 0.5
Note for non-bat inflammasome-prime-threshold needs to be 1.0 in order to get priming for pyroptosis

Endothelial activation and PMN burst:
1. Endothelial activation by TNF + IL1 > endothelial-activation slider (difference between bat = 10 and human = 5 at baseline)
2. Once activated make PAF + 1 and stay activated until TNF + IL1 < 0.5 (arbitrary)
3. After 36 steps (6 hrs) representing adhesion activation, 10% chance will hatch PMN
4. Adhesed PMNs jump random 5 upon arrival (prevents self containment)
5. PMN will live 6 hrs (36 steps) in tissue then undergo respiratory burst and produce ROS + 10 and IL1 + 1
5. Epis and Infected-Epis will die with ROS-counter > 10 (+ ROS per step); this mode of death (dead-epis) will lead to P/DAMP + 10
6. Dead-epis now make P/DAMPS + 1 until they are phagocytosed

---

## Simulation Experiments (Also see Behavior Space)

* NOTE: Simulation experiments are run with the "Random-Runs" switch "Off", stochastic replicates achieved by incrementing the random number seed by 1 to the number of stochastic replicates desired. This is done to allow tracing of specific runs (by seed); this allows the potential for deeper examination of specific run.

### For Parameter sweeps:
#### Baseline Bat
* Bat? = true
* Metabolic-byproduct = 2.0
* Inflammasome-priming-threshold = 5.0
* Inflammasome-activation-threshold = 50
* Bat-endo-activation = 10

#### Baseline Human
* Bat? = false
* Metabolic-byproduct = 0.2
* Inflammasome-priming-threshold = 1.0
* Inflammasome-activation-threshold = 10
* Human-endo-activation = 5

#### Initial Injury Sweeps (Figure 3a-b):
* Initial-inoculum from 25-150 increments of 25, run for 14 days (2016 steps)
* N = 1000 stochastic replicates

Main Comparison is distribution of %System-health at the end of the run for each Initial-inoculum

#### Evaluation of Effect of Endothelial Inflammasome (Figure 4)
Uses Human parameterization EXCEPT sweep from Human-endo-activation from 5 to 10 (bat level).
N = 250 Stochastic Replicates with Initial-Inoculum = 150

#### Evaluation of Effect of Metabolic Stress on Bat Disease (Figure 5)
Uses Bat Parameterization EXCEPT sweep of metabolic-byproduct from 2 to 10
N = 250 Stochastic Replicates with Initial-Inoculum = 150
