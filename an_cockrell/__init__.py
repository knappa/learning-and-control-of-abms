import itertools
from enum import IntEnum
from random import random, randint

import numpy as np
from attr import define, field


class EpiType(IntEnum):
    Empty = 0
    Healthy = 1
    Infected = 2
    Dead = 3
    Apoptosed = 4


# noinspection PyPep8Naming
@define(kw_only=True)
class AnCockrellModel:
    GRID_WIDTH: int = field()
    GRID_HEIGHT: int = field()

    BAT: bool = field()

    INIT_DCS: int = field()
    INIT_NKS: int = field()
    INIT_MACROS: int = field()

    MAX_LYMPHNODES: int = field(default=10_000)
    MAX_PMNS: int = field(default=10_000)
    MAX_DCS: int = field(default=10_000)
    MAX_MACROPHAGES: int = field(default=10_000)
    MAX_NKS: int = field(default=10_000)
    MAX_ACTIVATED_ENDOS: int = field(default=10_000)

    macro_phago_recovery: float = field(default=0.5)
    macro_phago_limit: int  = field(default=1_000)

    viral_carrying_capacity: int = field(default=500)
    resistance_to_infection: int = field(default=75)

    epithelium: np.ndarray = field(init=False)
    extracellular_virus: np.ndarray = field(init=False)
    epi_regrow_counter: np.ndarray = field(init=False)
    endothelial_activation: np.ndarray = field(init=False)
    P_DAMPS: np.ndarray = field(init=False)
    ROS: np.ndarray = field(init=False)
    PAF: np.ndarray = field(init=False)
    TNF: np.ndarray = field(init=False)
    IL1: np.ndarray = field(init=False)
    IL18: np.ndarray = field(init=False)
    IL2: np.ndarray = field(init=False)
    IL4: np.ndarray = field(init=False)
    IL6: np.ndarray = field(init=False)
    IL8: np.ndarray = field(init=False)
    IL10: np.ndarray = field(init=False)
    IL12: np.ndarray = field(init=False)
    IL17: np.ndarray = field(init=False)
    IFNg: np.ndarray = field(init=False)
    T1IFN: np.ndarray = field(init=False)

    grass: np.ndarray = field(init=False)
    grass_clock: np.ndarray = field(init=False)

    num_wolves: int = field(init=False)
    wolf_pos: np.ndarray = field(init=False)
    wolf_dir: np.ndarray = field(init=False)
    wolf_energy: np.ndarray = field(init=False)
    wolf_alive: np.ndarray = field(init=False)
    wolf_pointer: int = field(init=False)  # all indices >= this are not alive

    num_sheep: int = field(init=False)
    sheep_pos: np.ndarray = field(init=False)
    sheep_dir: np.ndarray = field(init=False)
    sheep_energy: np.ndarray = field(init=False)
    sheep_alive: np.ndarray = field(init=False)
    sheep_pointer: int = field(init=False)  # all indices >= this are not alive

    @property
    def total_T1IFN(self) -> float:
        return float(np.sum(self.T1IFN))

    @property
    def total_TNF(self) -> float:
        return float(np.sum(self.TNF))

    @property
    def total_IFNg(self) -> float:
        return float(np.sum(self.IFNg))

    @property
    def total_IL6(self) -> float:
        return float(np.sum(self.IL6))

    @property
    def total_IL1(self) -> float:
        return float(np.sum(self.IL1))

    @property
    def total_IL8(self) -> float:
        return float(np.sum(self.IL8))

    @property
    def total_IL10(self) -> float:
        return float(np.sum(self.IL10))

    @property
    def total_IL12(self) -> float:
        return float(np.sum(self.IL12))

    @property
    def total_IL18(self) -> float:
        return float(np.sum(self.IL18))

    @property
    def total_extracellular_virus(self) -> float:
        return float(np.sum(self.extracellular_virus))

    @property
    def total_intracellular_virus(self) -> float:
        return float(np.sum(self.epi_intracellular_virus))

    # self.system_health =  count epis / 2601 * 100

    def __attrs_post_init__(self):
        self.num_epis = 0
        self.epi_pointer = 0
        self.epi_mask = np.zeros(self.MAX_EPIS, dtype=bool)
        self.epi_type = np.zeros(self.MAX_EPIS, dtype=np.int8)
        self.epi_pos = np.zeros((self.MAX_EPIS, 2), dtype=np.float64)
        self.epi_infected = np.zeros(self.MAX_EPIS, dtype=bool)
        self.epi_intracellular_virus = np.zeros(self.MAX_EPIS, dtype=np.float64)
        self.epi_viral_carrying_capacity = np.zeros(self.MAX_EPIS, dtype=np.float64)
        self.epi_resistance_to_infection = np.zeros(self.MAX_EPIS, dtype=np.float64)
        self.epi_replication_counter = np.zeros(self.MAX_EPIS, dtype=np.float64)
        self.epi_cell_membrane = np.zeros(self.MAX_EPIS, dtype=np.float64)
        self.epi_apoptosis_counter = np.zeros(self.MAX_EPIS, dtype=np.float64)
        self.epi_apoptosis_threshold = np.zeros(self.MAX_EPIS, dtype=np.float64)
        self.epi_ros_damage_counter = np.zeros(self.MAX_EPIS, dtype=np.float64)
        self.epi_pre_il1 = np.zeros(self.MAX_EPIS, dtype=np.float64)
        self.epi_pre_il18 = np.zeros(self.MAX_EPIS, dtype=np.float64)
        self.epi_inflammasome_primed = np.zeros(self.MAX_EPIS, dtype=np.float64)
        self.epi_inflammasome_active = np.zeros(self.MAX_EPIS, dtype=np.float64)

        self.num_pmns = 0
        self.pmn_pointer = 0
        self.pmn_mask = np.zeros(self.MAX_PMNS, dtype=bool)
        self.pmn_locations = np.zeros((self.MAX_PMNS, 2), dtype=np.float64)
        self.pmn_age = np.zeros(self.MAX_PMNS, dtype=np.int64)
        #
        #   ;; dc specific variables NOT CURRENTLY ACTIVE in V2.0
        #   dc-location ;; this is a chooser with either "LymphNode" or "tissue". If "tissue" then interact with screen, If "LN" then interact with NaiveCD8s off screen
        #   trafficking-counter ;; Trafficking is simulated by jump to on-of NaiveCD8s, then Trafficking-counter upticks until engagement => initial approximation = 6 hours = 36 ticks
        #   antigen? ;; this is antigen picked up by DC when encounters infected epi, initiates trafficking
        #
        #   ;; macro specific variables
        #   macro-activation-level
        #   macro-phago-limit ;; this is the maximal amount a macrophage can phagocytose, initially arbitrarily set at 1000, all macros get full, increase to 10,000 to see what happens
        #   macro-phago-counter ;; this keeps track of how "full" macro is = cells-eaten + (virus-eaten / 10)
        #   pyroptosis-counter ;; counts how long it takes activated inflammasome to produce pyroptosis => 120 minutes per Community Development Paper
        #   virus-eaten
        #   cells-eaten
        #
        #
        #   ;; activated-endo specific variables
        #   adhesion-counter ;; PMN adhesion-migration takes 6-12 hours; 36 ticks = 6 hours

        MAX_LYMPHNODES: int = field(default=10_000)
        MAX_PMNS: int = field(default=10_000)
        MAX_DCS: int = field(default=10_000)
        MAX_MACROPHAGES: int = field(default=10_000)
        MAX_NKS: int = field(default=10_000)
        MAX_ACTIVATED_ENDOS: int = field(default=10_000)

        # spatial quantities
        self.extracellular_virus = np.zeros(
            (self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.int32
        )
        self.epi_regrow_counter = np.zeros(
            (self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.int32
        )
        self.endothelial_activation = np.zeros(
            (self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.int32
        )
        self.P_DAMPS = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.float64)
        self.ROS = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.float64)
        self.PAF = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.float64)
        self.TNF = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.float64)
        self.IL1 = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.float64)
        self.IL18 = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.float64)
        self.IL2 = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.float64)
        self.IL4 = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.float64)
        self.IL6 = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.float64)
        self.IL8 = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.float64)
        self.IL10 = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.float64)
        self.IL12 = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.float64)
        self.IL17 = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.float64)
        self.IFNg = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.float64)
        self.T1IFN = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.float64)

        #   ask patches
        #   [sprout 1
        #     [set breed epis ;; 1 epi per patch
        #      set shape "square"
        #      set color blue
        #      set intracellular-virus 0
        #      set viral-carrying-capacity 500 ;;  arbitrary, maybe make slider. Lower numbers kill but don't spread, defines incubation time
        #      set resistance-to-infection 75 ;; arbitrary, maybe make slider
        #      set cell-membrane 975 + random 51 ;; this is what is consumed by viral excytosis, includes some random component so all cells don't die at the same time
        #      set apoptosis-counter 0
        #      set apoptosis-threshold 450 + random 100 ;; this is half the cell-membrane, which means total amount of leaked virus should be half with apoptosis active, has random component as well
        #      if bat? = true ;; help initialize baseline level production of T1IFN in bats
        #        [if random 100 = 1
        #         [set T1IFN 5]
        #       ]
        #     ]
        #   ]
        self.epithelium = np.full((self.GRID_WIDTH, self.GRID_HEIGHT), EpiType.Healthy, dtype=np.int64)
        self.epithelium_intracellular_virus = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.int64)
        self.epithelium_cell_membrane = 95 + (51*np.random.rand((self.GRID_WIDTH, self.GRID_HEIGHT))).asdtype(np.int64)
        self.epithelium_apoptosis_counter = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.int64)
        self.epithelium_apoptosis_threshold = 450 + (100*np.random.rand((self.GRID_WIDTH, self.GRID_HEIGHT))).asdtype(np.int64)

        if self.BAT:
            self.T1IFN[:,:] = 5*(np.random.rand((self.GRID_WIDTH, self.GRID_HEIGHT)) < 0.01)

        #  create-NKs 25 ;; Initial-NKs slider for later
        #   [set color orange
        #     set shape "circle"
        #     set size 1
        #     repeat 100
        #     [jump random 1000]
        #   ]
        for _ in range(self.INIT_NKS):
            self.create_nk()

        #  create-Macros 50 ;;Initial-Macros slider for later
        #   [set color green
        #     set shape "circle"
        #     set size 1
        #     repeat 100
        #     [jump random 1000]
        #     set macro-phago-limit 1000 ;; aribitrary number
        #     set color green
        #     set pre-IL1 0
        #     set pre-IL18 0
        #     set inflammasome-primed false
        #     set inflammasome-active false
        #     set macro-activation-level 0
        #     set macro-phago-counter 0
        #     set pyroptosis-counter 0
        #     set virus-eaten 0
        #     set cells-eaten 0
        #   ]
        for _ in range(self.INIT_MACROS):
            self.create_macro(
                macro_phago_limit=1000,
                pre_il1=0,
                pre_il18=0,
                inflammasome_primed=False,
                inflammasome_active=False,
                macro_activation_level=0,
                macro_phago_counter=0,
                pyroptosis_counter=0,
                virus_eaten=0,
                cells_eaten=0,
            )

        #  create-DCs 50 ;; Initial-DCs slider for later
        #   [set color cyan
        #     set shape "triangle"
        #     set size 1.5
        #     set DC-location "tissue"
        #     set trafficking-counter 0
        #     repeat 100
        #     [jump random 1000]
        #   ]
        for _ in range(self.INIT_DCS):
            self.create_dc(dc_location="tissue", trafficking_counter=0)

    def infect(self, init_inoculum: int):
        # to infect
        # create-initial-inoculum-makers initial-inoculum ;; this is just to make a random distribution of inoculum
        #  [set color red
        #   repeat 10
        #    [jump random 1000]
        #   set extracellular-virus 80 + random 40 ;; this random is to try and "smooth" things out
        #   die
        #  ]
        # ask patches
        #   [set-background]
        # end
        rows, cols = np.divmod(
            np.random.choice(self.GRID_WIDTH * self.GRID_HEIGHT, init_inoculum),
            self.GRID_WIDTH,
        )
        if init_inoculum == 1:
            rows = [rows[0]]
            cols = [cols[0]]
        for row, col in zip(rows, cols):
            self.extracellular_virus[row, col] = np.random.randint(80, 120)

    def infected_epi_function(self):
        # to infected-epi-function
        #
        # ;; necrosis from PMN burst
        # if ROS-damage-counter > 10
        #   [set breed dead-epis
        #    set color grey
        #    set size 1
        #    set shape "circle"
        #    set P/DAMPs P/DAMPS + 10
        #   ]
        # set ROS-damage-counter ROS-damage-counter + ROS
        #
        # virus-replicate
        #
        # epi-apoptosis
        #
        # set T1IFN T1IFN + 1 ;; update by 1 appears to have too much T1IFN?
        # set IL18 IL18 + 0.11 ;; ? this rule?
        # if IL1 + TNF > 1
        #   [set IL6 IL6 + 0.10] ;; production shared with Macros and DCs, depends on IL1 and TNF production
        #
        # end
        pass

    def virus_invade_cell(self):
        pass
        # to virus-invade-cell ;; called in epi-function and macro-function
        # ;; what I want here is to have the likelyhood of invasion of the epi on the patch be a function of the number of extracellular-viruses on that patch
        # ;; Epi has resistance-to-invasion, higher numbers better to protect against invasion
        # ;; As currently written, lowering resistance to infection results in smaller initial intracellular virus, prolonging incubation,
        # ;; can affect by altering viral-incubation-rate (slider)
        # ;if random extracellular-virus > resistance-to-infection ;; if this is true then the virus invades
        # if extracellular-virus > 0
        # [ if random 100 < (max list susceptability-to-infection extracellular-virus) ;; this is a new virus invade criteria, says a % chance infection against higher of resistance or virus present
        #   ; so if very high extracellular virus then likely to invade, if very low extracellular virus still possibility it will invade. High values susceptability = worse
        #   ; if suceptability = 0 should have full resistance
        #   [set intracellular-virus intracellular-virus + 1
        #    set extracellular-virus extracellular-virus - 1 ;; virus from outside goes inside
        #    if breed = epis
        #     [set breed infected-epis
        #      set shape "square"
        #      set color yellow
        #     ]
        #  ;  if breed = macros
        #  ;   [set infected? true
        #  ;    set color yellow
        #  ;   set shape "circle"
        #  ;   ]
        #   ]
        #   ]
        # end

    def virus_replicate(self):
        pass
        # to virus-replicate ;; called in infected-epi-function
        #   ;; extrusion of virus will consume cell-membrane, when this goes to 0 cell dies (no viral burst but does produce P/DAMPS)
        #
        # if cell-membrane <= 0
        #   [set breed dead-epis
        #    set color grey
        #    set P/DAMPS P/DAMPS + 10
        #  ; set extracellular-virus intracellular-virus
        #  ]
        # ;; intracellular-virus is essentially a counter that determines time it takes to ramp up viral synthesis
        # ;; once the threshold viral-incubation-threshold is reached the cell leaks virus to extracellular-virus
        # ;; and takes one cell-membrane away until cell dies
        # ;; cell defenses to this is to apoptose, which causes the "virus factory" to die earlier
        #
        # if intracellular-virus > viral-incubation-threshold
        #   [set extracellular-virus extracellular-virus + 1
        #    set cell-membrane cell-membrane - 1
        #   ]
        #
        # ifelse bat? = true
        #   [set intracellular-virus max list 1 (intracellular-virus + 1 - (T1IFN / 10 ))];;  simulates T1IFN anti viral adaptations in bats,
        #                                                                                ;;  does not eradicate virus though, just suppresses growth
        #   [set intracellular-virus max list 0 (intracellular-virus + 1 - (T1IFN / 100))] ;; human manifestation of T1IFN anti viral effect
        #
        # end

    def epi_apoptosis(self):
        pass
        # to epi-apoptosis
        #   if apoptosis-counter > apoptosis-threshold
        #    [set breed apoptosed-epis
        #     set color grey
        #     set shape "pentagon"
        #     set size 1
        #   ]
        #   set apoptosis-counter apoptosis-counter + 1
        # end

    def regrow_epis(self):
        pass
        # to regrow-epis ;; patch command, regrows epis on empty patches if neighbors > 2 epis.
        # if count epis-here + count dead-epis-here + count apoptosed-epis-here + extracellular-virus = 0 ;; makes sure it is an patch empty of epis
        #  [if count epis-on neighbors > 2 ;; if conditions met, start counter
        #     [if epi-regrow-counter >= 432 ;; regrows starting at 3 days if sufficient neighboring epis
        #      [sprout 1
        #        [set breed epis ;; 1 epi per patch
        #         set shape "square"
        #         set color blue
        #         set intracellular-virus 0
        #         set resistance-to-infection 75 ;; arbitrary, maybe make slider
        #         set cell-membrane 975 + random 51 ;; this is what is consumed by viral excytosis, includes some random component so all cells don't die at the same time
        #         set apoptosis-counter 0
        #         set apoptosis-threshold 475 + random 51 ;; this is half the cell-membrane, which means total amount of leaked virus should be half with apoptosis active, has random component as well
        #        ]
        #       set epi-regrow-counter 0 ; if sprouts, reset counter to 0
        #      ]
        #     set epi-regrow-counter epi-regrow-counter + 1 ;; counts if enough neighbors
        #     ]
        #   ]
        #
        #
        # end

    def dead_epi_update(self):
        dead_epi_mask = self.epi_mask & (self.epi_type == EpiType.Dead)
        locations = self.epi_pos[dead_epi_mask].astype(np.int64)
        self.P_DAMPS[tuple(locations.T)] += 1
        # to dead-epi-function
        # set P/DAMPs P/DAMPs + 1
        # end

    def pmn_update(self):
        # to PMN-function ;; OMNs are made in activated-endo-function

        # if age > 36 ;; baseline 6 hour tissue lifespan
        #   ;; once migrated into tissue committed to dying via burst
        #   [set ROS ROS + 10
        #    set IL1 IL1 + 1
        #    die]
        age_mask = self.pmn_mask & (self.pmn_age > 36)
        locations = self.pmn_locations[age_mask].astype(np.int64)
        self.ROS[tuple(locations.T)] += 10
        self.IL1[tuple(locations.T)] += 1
        self.pmn_mask[age_mask] = False

        #
        # ;; chemotaxis to PAF and IL8
        # let p max-one-of neighbors [PAF + IL8]
        #   ifelse [PAF + IL8] of p > (PAF + IL8) and [PAF + IL8] of p != 0
        #   [face p
        #    fd 0.1
        #    if count PMNs-here > 1 ;; This code block is to prevent stacking of macros on a single patch. IF there is a stack then they kill the infected more quickly
        #     [set heading random 360
        #       fd 1]
        #   ]
        chemoattractant = self.PAF + self.IL8
        chemoattractant_dx = (
            np.roll(chemoattractant, -1, axis=0) - np.roll(chemoattractant, 1, axis=0)
        ) / 2.0
        chemoattractant_dy = (
            np.roll(chemoattractant, -1, axis=1) - np.roll(chemoattractant, 1, axis=1)
        ) / 2.0
        locations = self.pmn_locations[self.pmn_mask].astype(np.int64)
        vecs = np.stack(
            [
                chemoattractant_dx[tuple(locations.T)],
                chemoattractant_dy[tuple(locations.T)],
            ],
            axis=1,
        )
        norms = np.linalg.norm(vecs, axis=-1)
        norms[norms <= 1e-8] = 1.0  # effectively zero norm vectors will be unnormalized
        self.pmn_locations[self.pmn_mask] += 0.1 * vecs / norms
        self.pmn_dirs[self.pmn_mask] = np.arctan2(vecs[:, 1], vecs[:, 0])

        #   [wiggle
        #   ]
        self.pmn_dirs += (
            (np.random.rand(self.MAX_PMNS) - np.random.rand(self.MAX_PMNS))
            * np.pi
            / 4.0
        )
        directions = np.stack([np.cos(self.pmn_dirs), np.sin(self.pmn_dirs)], axis=1)
        self.pmn_locations += directions
        self.pmn_locations = np.mod(
            self.pmn_locations, [self.GRID_WIDTH, self.GRID_HEIGHT]
        )

        #  set age age + 1
        self.pmn_age[self.pmn_mask] += 1

        # end

    def nk_update(self):
        pass
        # to NK-function
        #
        #   ;; INDUCTION OF APOPTOSIS
        # ask infected-epis-here
        #  [set apoptosis-counter apoptosis-counter + 9]  ;; NKs enhance infected epi apoptosis 10x
        #
        #   ;;Production of cytokines
        # if T1IFN > 0 and IL12 > 0 and IL18 > 0
        #   [set IFNg IFNg + 1] ;; need to check this, apparently does not happen, needs IL18(?)
        #
        #   ;; Chemotaxis to T1IFN made by infected-epis, slows movement rate to 1/10 of uphill primitive
        # let p max-one-of neighbors [T1IFN]  ;; or neighbors4
        #   ifelse [T1IFN] of p > T1IFN and [T1IFN] of p != 0
        #   [face p
        # ;;    if [count NKs] of patch-ahead 1 < 5
        #    fd 0.1
        #    if count NKs-here > 1 ;; This code block is to prevent stacking of NK cells on a single patch. IF there is a stack then they kill the infected more quickly
        #     [set heading random 360
        #       fd 1]
        #   ]
        #   [wiggle
        #   ]
        #
        #   ;consumption IL18 and IL12
        # set IL12 max list 0 IL12 - 0.1
        # set IL18 max list 0 IL18 - 0.1
        # end

    def macro_update(self):
        # to macro-function
        #
        # ; check to see if macro gets infected
        # virus-invade-cell
        mask = self.macro_mask
        locations = self.macro_locations[mask].astype(np.int64)
        extracellular_virus_at_locations = self.extracellular_virus[tuple(locations.T)]
        cells_to_invade = (
            np.random.rand(extracellular_virus_at_locations.shape)
            * extracellular_virus_at_locations
            > self.macro_resistance_to_infection
        )
        self.macro_internal_virus[mask][cells_to_invade] += 1
        self.extracellular_virus[tuple(locations[cells_to_invade].T)] -= 1
        self.macro_infected[mask][cells_to_invade] = True

        #   ; macro-activation-level keeps track of M1 (pro) or M2 (anti) status
        #   ; there is hysteresis because it modifies existing status
        # set macro-activation-level macro-activation-level + T1IFN + P/DAMPS + IFNg + IL1 - (2 * IL10) ;; currently a gap between pro and anti macros
        self.macro_activation[mask] = (
            self.T1IFN[tuple(locations.T)]
            + self.P_DAMPS[tuple(locations.T)]
            + self.IFNg[tuple(locations.T)]
            + self.IL1
            - 2 * self.IL10[tuple(locations.T)]
        )

        #   ;; separate out inflammasome mediated functions from other pro macro functions
        #   ;; so IL1/IL18 production, induction of pyroptosis
        #   ;; as with all sequential processes, inflammasome priming/activation/effects coded in reverse order
        # inflammasome-function

        # to inflammasome-function
        #  ;; inflammasome effects
        #
        #   if pyroptosis-counter > 12  ;; 120 minutes
        #   [pyroptosis]
        #
        #   if inflammasome-active = true ;; this is coded this way to draw out the release of IL1 and IL18 (make less jumpy)
        #     [set IL1 Il1 + 1
        #      set pre-IL1 pre-IL1 + 5 ;; this means pre-IL1 will be 12 at time of burst, which is IL1 level at burst
        #      set IL18 IL18 + 1
        #      set pre-IL18 pre-IL18 + 0.5 ;; mimic IL1 at the moment, half amount generated though (arbitrary)
        #      set pyroptosis-counter pyroptosis-counter + 1] ;; Pyroptosis takes 120 min from inflammasome activation to pyroptosis so counter threshold is 12
        #
        #   ;; stage 2 = activation of inflammasome => use virus-eaten as proxy for intracellular viral products, effects of phagocytosis
        #   if inflammasome-primed = true
        #    [if virus-eaten / 10 > inflammasome-activation-threshold ;; extra-, intra- and eaten virus in the scale of up to 100
        #                                                           ;; macro-phago-limit set to 1000 (with virus-eaten / 10), activation thresholds ~ 50
        #     [set inflammasome-active true
        #      set shape "target"
        #      set size 1.5
        #     ]
        #   ]
        #
        #   ;; stage 1 = Priming by P/DAMPS or TNF
        #   if P/DAMPS + TNF > inflammasome-priming-threshold
        #   [set inflammasome-primed true]
        # ;;  [set inflammasome "inactive"]
        #
        # end

        #   ;; consumption of activating cytokines
        # set T1IFN max list 0 T1IFN - 0.1
        # set IFNg max list 0 IFNg - 0.1
        # set IL10 max list 0 IL10 - 0.1
        # set IL1 max list 0 IL1 - 0.1
        self.T1IFN[tuple(locations.T)] = np.maximum(
            0.0, self.T1IFN[tuple(locations.T)] - 0.1
        )
        self.IFNg[tuple(locations.T)] = np.maximum(
            0.0, self.IFNg[tuple(locations.T)] - 0.1
        )
        self.IL10[tuple(locations.T)] = np.maximum(
            0.0, self.IL10[tuple(locations.T)] - 0.1
        )
        self.IL1[tuple(locations.T)] = np.maximum(
            0.0, self.IL1[tuple(locations.T)] - 0.1
        )

        #   ;; Chemotaxis to T1IFN made by infected-epis, slows movement rate to 1/10 of uphill primitive. Also chemotaxis to DAMPs
        # let p max-one-of neighbors [T1IFN + P/DAMPS]  ;; or neighbors4
        #   ifelse [T1IFN + P/DAMPS] of p > (T1IFN + P/DAMPS) and [T1IFN + P/DAMPS] of p != 0
        #   [face p
        # ;;    if [count NKs] of patch-ahead 1 < 5
        #    fd 0.1
        #    if count Macros-here > 1 ;; This code block is to prevent stacking of macros on a single patch. IF there is a stack then they kill the infected more quickly
        #     [set heading random 360
        #       fd 1]
        #   ]
        chemoattractant = self.T1IFN + self.P_DAMPS
        chemoattractant_dx = (
            np.roll(chemoattractant, -1, axis=0) - np.roll(chemoattractant, 1, axis=0)
        ) / 2.0
        chemoattractant_dy = (
            np.roll(chemoattractant, -1, axis=1) - np.roll(chemoattractant, 1, axis=1)
        ) / 2.0
        locations = self.macro_locations[self.macro_mask].astype(np.int64)
        vecs = np.stack(
            [
                chemoattractant_dx[tuple(locations.T)],
                chemoattractant_dy[tuple(locations.T)],
            ],
            axis=1,
        )
        norms = np.linalg.norm(vecs, axis=-1)
        norms[norms <= 1e-8] = 1.0  # effectively zero norm vectors will be unnormalized
        self.macro_locations[self.macro_mask] += 0.1 * vecs / norms
        self.macro_dirs[self.macro_mask] = np.arctan2(vecs[:, 1], vecs[:, 0])

        #   [wiggle
        #   ]
        self.macro_dirs += (
            (
                np.random.rand(self.MAX_MACROPHAGES)
                - np.random.rand(self.MAX_MACROPHAGES)
            )
            * np.pi
            / 4.0
        )
        self.macro_locations += np.stack(
            [np.cos(self.macro_dirs), np.sin(self.macro_dirs)], axis=1
        )
        self.macro_locations = np.mod(
            self.macro_locations, [self.GRID_WIDTH, self.GRID_HEIGHT]
        )

        #     ;;PHAGOCYTOSIS LIMIT check
        #   ;; check for macro-phago-limit
        #   set macro-phago-counter max list 0 (cells-eaten + (virus-eaten / 10) - macro-phago-recovery)
        #   ;; macro-phago-recovery decrements counter to simulate internal processing, proxy for new macros
        #   ;; =2 => Exp 5

        macro_phago_counter = np.maximum(
            0.0,
            self.macro_cells_eaten[self.macro_mask]
            - self.macro_virus_eaten[self.macro_mask] / 10
            - self.macro_phago_recovery,
        )
        under_limit_mask = macro_phago_counter < self.macro_phago_limit
        locations = self.macro_locations[self.macro_mask][under_limit_mask].astype(np.int64)

        #   ifelse macro-phago-counter >= macro-phago-limit ;; this will eventually be pyroptosis
        #     [set size 2]
        #     [;; PHAGOCYTOSIS
        #     set size 1
        #     ;; of virus, uses local variable "q" to represent amount of virus eaten (avoid negative values)
        #     if extracellular-virus > 0
        #      [let q max list 0 (extracellular-virus - 10)
        #      ;; currently set as arbitrary 10 amount of extracellular virus eaten per step
        #       set extracellular-virus extracellular-virus - q
        #       set virus-eaten virus-eaten + q
        #      ]

        virus_uptake = np.maximum(10.0, self.extracellular_virus[tuple(locations.T)])
        self.extracellular_virus[tuple(locations.T)] -= virus_uptake
        self.macro_virus_eaten[self.macro_mask][under_limit_mask] += virus_uptake

        #     ;; of apoptosed epis
        #     if count apoptosed-epis-here > 0
        #      [ask apoptosed-epis-here [die]
        #       set cells-eaten cells-eaten + 1
        #       set apoptosis-eaten-counter apoptosis-eaten-counter + 1
        #      ]

        apoptosed_epis = self.epithelium[tuple(locations.T)] == EpiType.Apoptosed
        self.macro_cells_eaten[self.macro_mask][apoptosed_epis] += 1
        self.apoptosis_eaten_counter += np.sum(apoptosed_epis)

        #    ;; of dead epis
        #     if count dead-epis-here > 0
        #      [ask dead-epis-here [die]
        #       set cells-eaten cells-eaten + 1
        #      ]
        #   ]

        dead_epis = self.epithelium[tuple(locations.T)] == EpiType.Dead
        self.macro_cells_eaten[self.macro_mask][dead_epis] += 1
        self.epithelium[tuple(locations[dead_epis].T)] = EpiType.Empty

        # if macro-activation-level > 5 ;; This is where decreased sensativity to P/DAMPS can be seen. Link macro-activation-level to Inflammasome variable?
        #                               ;; increased from 1.1 to 5 for V1.1
        #  [;; Proinflammatory cytokine production
        #   if IL1 + P/DAMPS > 0 ;; these are downstream products of NFkB, which either requires infammasome activation or TLR signaling
        #     [set TNF TNF + 1
        #      set IL6 IL6 + 0.4 ;; split with DCs and infected-epis, dependent on IL1
        #  ;; set IL1 IL1 + 1 => Moved to Inflammasome function
        #     ;; Anti-inflammatory cytokine production
        #     set IL10 IL10 + 1
        #     ]
        #   set IL8 IL8 + 1 ;; believe this is NFkB independent
        #   set IL12 IL12 + 0.5 ;; split with DCs Don't know if this is based on NFkB activation
        #
        #   set macro-activation-level macro-activation-level - 5 ;; this is to simulate macrophages returning to baseline as intracellular factors lose stimulation
        #  ]

        activated_macros = self.macro_mask & (self.macro_activation_level > 5)



        # if macro-activation-level < -5 ;; this keeps macros from self activating wrt IL10 in perpetuity
        #   [set color pink ;; tracker
        #   ;; Antiinflammatory cytokine production
        #   set IL10 IL10 + 0.5
        #
        #   set macro-activation-level macro-activation-level + 5 ;; this is to simulate macrophages returning to baseline as intracellular factors lose stimulation
        #   ]
        # end

    def inflammasome_function(self):
        pass
        # to inflammasome-function
        #  ;; inflammasome effects
        #
        #   if pyroptosis-counter > 12  ;; 120 minutes
        #   [pyroptosis]
        #
        #   if inflammasome-active = true ;; this is coded this way to draw out the release of IL1 and IL18 (make less jumpy)
        #     [set IL1 Il1 + 1
        #      set pre-IL1 pre-IL1 + 5 ;; this means pre-IL1 will be 12 at time of burst, which is IL1 level at burst
        #      set IL18 IL18 + 1
        #      set pre-IL18 pre-IL18 + 0.5 ;; mimic IL1 at the moment, half amount generated though (arbitrary)
        #      set pyroptosis-counter pyroptosis-counter + 1] ;; Pyroptosis takes 120 min from inflammasome activation to pyroptosis so counter threshold is 12
        #
        #   ;; stage 2 = activation of inflammasome => use virus-eaten as proxy for intracellular viral products, effects of phagocytosis
        #   if inflammasome-primed = true
        #    [if virus-eaten / 10 > inflammasome-activation-threshold ;; extra-, intra- and eaten virus in the scale of up to 100
        #                                                           ;; macro-phago-limit set to 1000 (with virus-eaten / 10), activation thresholds ~ 50
        #     [set inflammasome-active true
        #      set shape "target"
        #      set size 1.5
        #     ]
        #   ]
        #
        #   ;; stage 1 = Priming by P/DAMPS or TNF
        #   if P/DAMPS + TNF > inflammasome-priming-threshold
        #   [set inflammasome-primed true]
        # ;;  [set inflammasome "inactive"]
        #
        # end

    def pryoptosis(self):
        pass
        # to pyroptosis
        # set pyroptosed-macros pyroptosed-macros + 1
        # set IL1 pre-IL1
        # set IL18 pre-IL18
        # set P/DAMPs P/DAMPs + 10
        # hatch 1 ;; this maintains steady number of macrophages, thus this model does not simulate macrophage depletion via pyroptosis
        #   [set color green
        #    set shape "circle"
        #    set size 1
        #     move-to one-of patches with [count epis + count infected-epis > 0] ;; this has regenerated macros move to a random patch that isnt already dead
        #    set macro-phago-limit 1000 ;; aribitrary number
        #    set pre-IL1 0
        #    set pre-IL18 0
        #    set inflammasome-primed false
        #    set inflammasome-active false
        #    set macro-activation-level 0
        #    set macro-phago-counter 0
        #    set pyroptosis-counter 0
        #    set virus-eaten 0
        #    set cells-eaten 0
        #   ]
        #  die
        #
        # end

    def dc_update(self):
        pass
        # to DC-function
        #
        # if DC-location = "tissue"
        #   ;;Production of cytokines
        # [if T1IFN > 1
        #   [set IL12 IL12 + 0.5 ;; split with pro macros
        #    if IL1 > 1
        #       [set IL6 IL6 + 0.4] ;; split with pro macros and infected-epis, dependent on IL1 presence
        #    set IFNg IFNg + 0.5
        #   ]
        #  ; consumption of mediators
        #  set T1IFN max list 0 (T1IFN - 0.1)
        #
        #   ;; Chemotaxis to T1IFN made by infected-epis, slows movement rate to 1/10 of uphill primitive
        #  let p max-one-of neighbors [T1IFN]  ;; or neighbors4
        #   ifelse [T1IFN] of p > T1IFN and [T1IFN] of p != 0
        #   [face p
        #    fd 0.1
        #     if count DCs-here > 1 ;; This code block is to prevent stacking of DCs on a single patch.
        #     [set heading random 360
        #       fd 1]
        #
        #      ;; THIS IS ANTIGEN RECOGNITION AND LN TRAFFICKING
        # ;    if count infected-epis-here > 0 ;; checks to see if on same patch as infected-epi, picks up antigen
        # ;     [set antigen? true
        # ;      set trafficking-counter trafficking-counter + 1
        # ;      if trafficking-counter > 12 ;; 2 hours in tissue after antigen engagement
        # ;        [move-to one-of LymphNodes
        # ;         set dc-location "LymphNode"
        # ;         set size 3
        # ;        ]
        # ;      ]
        #    ]
        #    [wiggle
        #    ]
        #
        #   ]
        #
        # end

    def epi_update(self):
        pass
        # to epi-function
        # ;; necrosis from PMN burst
        # if ROS-damage-counter > 2
        #   [set breed dead-epis
        #    set color grey
        #    set size 1
        #    set shape "circle"
        #    set P/DAMPs P/DAMPS + 10
        #   ]
        # set ROS-damage-counter ROS-damage-counter + ROS
        #
        #   metabolism
        #   virus-invade-cell
        #  if bat? = true
        #   [baseline-T1IFN-generation
        #   ]
        #
        #   ;; activates endothelium
        #   ifelse bat? = false
        #    [ if TNF + IL1 > human-endo-activation and count activated-endos-here = 0 ;; arbitrary level of TNF + IL1 => set at 5 for human, 10 for bat
        #      [hatch 1
        #       [set breed activated-endos
        #        set color pink
        #        set size 1.5
        #        set shape "square 2"
        #       ]
        #
        #
        #
        #       ;; simulates consumption of IL1 and TNF
        #    set TNF max list 0 (TNF - 0.1)
        #    set IL1 max list 0 (IL1 - 0.1)
        #    ]
        #   ]
        #
        #    [ if TNF + IL1 > bat-endo-activation and count activated-endos-here = 0 ;; arbitrary level of TNF + IL1 => set at 5 for human, 10 for bat
        #      [hatch 1
        #       [set breed activated-endos
        #        set color pink
        #        set size 1.5
        #        set shape "square 2"
        #       ]
        #
        #
        #
        #       ;; simulates consumption of IL1 and TNF
        #    set TNF max list 0 (TNF - 0.1)
        #    set IL1 max list 0 (IL1 - 0.1)
        #    ]
        #   ]
        # end
        #
        # to dead-epi-function
        # set P/DAMPs P/DAMPs + 1
        # end

    def baseline_t1ifn_generation(self):
        pass
        # to baseline-T1IFN-generation
        #  if random 100 = 1
        #       [set T1IFN T1IFN + 0.75
        #  ]
        # end

    def metabolism(self):
        pass
        # to metabolism
        #  if random 100 = 1
        #       [set P/DAMPs P/DAMPs + metabolic-byproduct]
        # end

    def activated_endo_update(self):
        pass
        # to activated-endo-function ;; are made in epi-function
        # set PAF PAF + 1
        # if TNF + IL1 < 0.5
        #   [die]
        #
        # if adhesion-counter > 36 ;; 6 hours
        #   [if random 10 = 9 ;; 10% chance each step a PMN is attracted
        #     [hatch 1
        #      set breed PMNs
        #      set color white
        #      set shape "circle"
        #      set size 0.5
        #      set age 0
        #      ;; the following 2 lines are so that the PMNs just don't appear in the middle of the infection
        #      set heading random 360
        #      jump random 5
        #     ]
        #   ]
        # set adhesion-counter adhesion-counter + 1
        #
        # end

    def diffuse_functions(self):
        pass
        # to diffuse-functions
        #   diffuse extracellular-virus 0.05
        #   diffuse T1IFN 0.1
        #   diffuse PAF 0.1
        #   diffuse ROS 0.1
        #   diffuse P/DAMPs 0.1
        #
        #   diffuse IFNg 0.2
        #   diffuse TNF 0.2
        #   diffuse IL6 0.2
        #   diffuse IL1 0.2
        #   diffuse IL10 0.2
        #   diffuse IL12 0.2
        #   diffuse IL18 0.2
        #
        #   diffuse IL8 0.3
        #
        #
        #
        # end

    def cleanup(self):
        pass
        # to cleanup
        # if extracellular-virus < 1
        #   [set extracellular-virus 0]
        # if T1IFN < 0.1
        #   [set T1IFN 0]
        # if IFNg < 0.1
        #   [set IFNg 0]
        # if TNF < 0.1
        #   [set TNF 0]
        # if IL6 < 0.1
        #   [set IL6 0]
        # if IL1 < 0.1
        #   [set IL1 0]
        # if IL10 < 0.1
        #   [set IL10 0]
        # if PAF < 0.1
        #   [set PAF 0]
        # if IL8 < 0.1
        #   [set IL8 0]
        # if ROS < 0.1
        #   [set ROS 0]
        # if IL12 < 0.1
        #   [set IL12 0]
        # if IL18 < 0.1
        #   [set IL18 0]
        # if P/DAMPS < 0.1
        #   [set P/DAMPS 0]
        # end

    def evaporate(self):
        pass
        # to evaporate
        # set T1IFN T1IFN * .99
        # set IFNg IFNG * .99
        # set TNF TNF * .99
        # set IL6 IL6 * .99
        # set IL1 IL1 * .99
        # set IL10 IL10 * .99
        # set PAF PAF * .9
        # set IL8 IL8 * .99
        # set ROS ROS * 0.9
        # set IL12 IL12 * .99
        # set IL18 IL18 * .99
        # set P/DAMPS P/DAMPS * 0.9
        #
        # end

    def wiggle(self):
        pass
        # to wiggle
        # rt random 45
        # lt random 45
        # fd 0.1
        # end

    def time_step(self):
        # if count apoptosed-epis + count dead-epis >=  2080 ;; this is 80% of the epi cells dead (total epis = 2601)
        #   [stop]

        self.dead_epi_update()
        self.pmn_update()
        self.macro_update()
        self.nk_update()
        self.dc_update()
        self.activated_endo_update()
        self.epi_update()
        #   set total-P/DAMPS sum [P/DAMPs] of patches
        #   ;; need to calculate here, if calculate after diffuse/evaporate/cleanup there
        #   is no value for threshold < 0.13
        self.infected_epi_function()
        self.diffuse_functions()

        # ;; this is to remove low levels of extracellular variables
        # ask patches
        #   [regrow-epis
        #    set-background
        #    evaporate ;; This order is important, if you put evaporate after cleanup you never get decent levels of cytokine
        #    cleanup
        #  ]
        #
        #
        # end

    def create_epi(
        self,
        *,
        position,
        intracellular_virus,
        viral_carrying_capacity,
        resistance_to_infection,
        cell_membrane,
        apoptosis_counter,
        apoptotis_threshold,
        T1IFN,
    ):
        # TODO
        pass


# # Original NetLogo readme/comments
# Name of model: Comparative Biology Immune ABM (CBIABM)
#
# Purpose: Simulate those aspects of innate immune response different between bats and humas. These areas are:
#
# 1. Decreased sensativity to P/DAMP due to high metabolic rate
# 2 Decreased ramp up of T1IFNs in response to perturbation => intrinsic to this is that baseline is higher?
# 3. Decreased inflammasome activation => consequences are decreased caspase-1 activation which leads to decreased IL1/IL18 and decreased pyroptosis
# 4. Decreased processing/release of IL1
#
# =========================================================================================
#
# Guide to World view:
#
# The  World  view, consisting of the 51 x 51  square grid, can be seen on the right side of the User Interface. The following is a key to the agent shapes seen on that view:
#
# Blue Squares = Healthy Epithelial Cells
# Yellow Squares = Infected Epithelial Cells
# Grey Squares = Epithelial Cells killed by necrosis
# Grey Pentagons = Epithelial Cells killed by apoptosis
# Green Circles = Macrophages
# Large Green Circles = Macrophages at phagocytosis limit
# Orange Circles = NK Cells
# Light Blue Triangles = Dendritic Cells
# Pink Square Outlines = Activated Endothelial Cells
# Small White Circles = PMNs
#
# At initialization the entire world-space is filled with Healthy Epithelial Cells (Blue Squares). When dead Epithelial Cells of both types are cleared by phagocytosis what remains is black space, which can then be refilled be new Epithelial Cells as they heal (see main text for rules). The levels of extracellular virus and different mediators can be seen in the background behind the agents; see “Visualization Controls” below how to hide and reshow Epithelial and Endothelial Cells.
#
# =========================================================================================
#
# Modeled Behaviors and Functions
#
# Primary means of controlling virus is to kill infected epis
# Mechanisms:
# 1. Infected-epis undergo apoptosis (eventually)
# 2. NK cells enhance apoptosis
# 3. Macros consume extracellular-virus => Not coupled to macro-activation or inflammasome activation
#
# Model behavior criteria:
# 1. Bats have a higher baseline generation of P/DAMP, therefore macro-activation threshold is higher
# 2. Ideally bats should have longer/persistent virus present, mostly intracellular, in dynamic equilibrium => decrease apoptosis (longer virus factory duration), decrease pyroptosis/macros-full => we know that in the model increasing full-macros leads to impaired viral clearance => Rely on epitheilial-mediated viral control (avoid full-macro point?)
# 3. Ideally generate shedding state (extracellular virus) from baseline state by increasing P/DAMPs => "stress state"
#
# Modeling Baseline P/DAMP
# Called in "metabolism" command, has epi cells randomly generate "metabolic-byproduct" which is the update to P/DAMP.
# Random is 1/100, metabolic-byproduct slider from 0.1 to 1.0 updated by 0.01. Note: update of total-P/DAMP must occur prior to evap/diffuse/cleanup otherwise no detectable value
#
# Modeling Viral Infection and Replication
#
# Viral infecton and replication: stochastic process, but intial infection synchronizes, period of intracellular replication to reach threshold at which time infected cell starts to leak virus, leaking of virus consumes cell membrane until cell dies. NO BURST EFFECT, though cell death in this fashion leads to the production of P/DAMPs; supposition is that this does not occur due to apoptosis being activated. So hyperinflammation occurs either because epi apoptosis is interrupted (some viruses do this) or due to pyroptosis (less likely necrosis) of macros leads to P/DAMP production.
#
# Virus invasion uses susceptability-to-infection as a variable for epi cells to be infected. The calcuation is random 100 < max list susceptability-to-infection extracellular-virus. This means more likely for invasion if higher extracellular-virus, but possibility exists if there are extracellular-virus present.
# Also, min value of extracellular-virus set to 1 (cleanup command), diffusion of extracellular virus changed to 0.05 (working with susceptability-to-infection = 25)
#
# Epi Cell Function:
#
# Epi cell death due to non-apoptosis => called in Epi-function
# Death by ROS produces + 10 P/DAMP
#
# Infected Epi cell functions:
#
# Epi cell apoptosis: Counter that starts with infection, leads to cell death prior to full cell membrane consumption point, effect is to reduce overall level of virus, but does not help infected cell. Dies without generating P/DAMPs.
# This version sets the epi-apoptosis-threshold to 450 + random 100, essentially half of the total cell membrane with randomness added. Earlier versions with 475 + random 50 and NK enhancement + 1 lead to pretty abrupt die-off just after Day 30.
#
#
# NK cell functions:
#
# NK cells chemotax to T1IFN, IL12 and IL18 (need all three to be present). Kills infected-epis by inducing faster apoptosis
#
# Macro Functions:
#
# Macros are now forced towards baseline inactivation, -1 with pro-, +1 with anti. Mostly impacts antis, allows recovery of IL10 levels
#
# There is a distinction between pro-inflammatory state (activation) and inflammasome activation, supposition is that the determination of the M1 macrophage phenotype involves more than just inflammasome activation
#
# Inflammasome Properties:
#
# Inflammasome activation 2nd stage based on virus-eaten, no function infection of macros.
#
# Role of IL1:
# IL6 or P/DAMPS are needed for activation of NFkB, which is needed for TNF, IL6 and IL10; these now get activated by IL1 + P/DAMP > 1
# IL1 more simply needed for IL6 production by macros and DCs, added to macro-activation-level determination (okay since it is not produced directly as part of that conditional statement), IL1 + TNF > 1 needed for infected-epi production of IL6
# Need to turn inflammasome priming down to 1.0 in order to get activation and production of IL1 (which then allows production of TNF, IL6 and IL10).
# IL1 and IL18 production, instead of build up and burst at pyroptosis releasing all at once, start producing IL1/IL18 upon inflammasome activation, with lessor burst at pyroptosis.
#
#
# Pyroptosis from activated inflammasome
# IL1 is now made as pre-IL1 during period between inflammasome activation and pyroptosis (120 minutes), pre-IL1 increments by + 10 so IL1 is 120 upon macro death
# Increases P/DAMPS + 10
# Creates a naive macro and jumps away (macro steady state)
#
# Anti-viral effect T1IFN
#
# Adds ability to have baseline T1IFN generation => bats do this
# Controled by Switch
# Random 100 = 1 => Set T1IFN + 0.15
# Adds antiviral effect T1IFN => inihibits viral-replication by - T1IFN/10 on viral replication, does not eradicate but keeps minimum of 1 (human = - T1IFN/100)
# Infected Epis make T1IFN + 1 and IL18 + 0.11
#
# Bat Baseline Metabolism-byproduct = 0.5
# Note for non-bat inflammasome-prime-threshold needs to be 1.0 in order to get priming for pyroptosis
#
# Endothelial activation and PMN burst:
# 1. Endothelial activation by TNF + IL1 > endothelial-activation slider (difference between bat = 10 and human = 5 at baseline)
# 2. Once activated make PAF + 1 and stay activated until TNF + IL1 < 0.5 (arbitrary)
# 3. After 36 steps (6 hrs) representing adhesion activation, 10% chance will hatch PMN
# 4. Adhesed PMNs jump random 5 upon arrival (prevents self containment)
# 5. PMN will live 6 hrs (36 steps) in tissue then undergo respiratory burst and produce ROS + 10 and IL1 + 1
# 5. Epis and Infected-Epis will die with ROS-counter > 10 (+ ROS per step); this mode of death (dead-epis) will lead to P/DAMP + 10
# 6. Dead-epis now make P/DAMPS + 1 until they are phagocytosed
#
# =========================================================================================
#
# Simulation Experiments (Also see Behavior Space)
#
# *NOTE: Simulation experiments are run with the "Random-Runs" switch "Off", stochastic replicates achieved by incrementing the random number seed by 1 to the number of stochastic replicates desired. This is done to allow tracing of specific runs (by seed); this allows the potential for deeper examination of specific run.
#
# For Parameter sweeps:
# Baseline Bat
# Bat? = true
# Metabolic-byproduct = 2.0
# Inflammasome-priming-threshold = 5.0
# Inflammasome-activation-threshold = 50
# Bat-endo-activation = 10
#
# Baseline Human
# Bat? = false
# Metabolic-byproduct = 0.2
# Inflammasome-priming-threshold = 1.0
# Inflammasome-activation-threshold = 10
# Human-endo-activation = 5
#
# Initial Injury Sweeps (Figure 3a-b):
# Initial-inoculum from 25-150 increments of 25, run for 14 days (2016 steps)
# N = 1000 stochastic replicates
# Main Comparison is distribution of %System-health at the end of the run for each Initial-inoculum
#
# Evaluation of Effect of Endothelial Inflammasome (Figure 4)
# Uses Human parameterization EXCEPT sweep from Human-endo-activation from 5 to 10 (bat level).
# N = 250 Stochastic Replicates with Initial-Inoculum = 150
#
# Evaluation of Effect of Metabolic Stress on Bat Disease (Figure 5)
# Uses Bat Parameterization EXCEPT sweep of metabolic-byproduct from 2 to 10
# N = 250 Stochastic Replicates with Initial-Inoculum = 150
