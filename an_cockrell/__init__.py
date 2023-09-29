from enum import IntEnum
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from attr import define, field
from matplotlib import markers

BIG_NUM = 1000

class EpiType(IntEnum):
    Empty = 0
    Healthy = 1
    Infected = 2
    Dead = 3
    Apoptosed = 4


class EndoType(IntEnum):
    Normal = 0
    Activated = 1
    Dead = 2


# noinspection PyPep8Naming
@define(kw_only=True)
class AnCockrellModel:
    GRID_WIDTH: int = field()
    GRID_HEIGHT: int = field()

    BAT: bool = field()

    INIT_DCS: int = field()
    INIT_NKS: int = field()
    INIT_MACROS: int = field()

    MAX_LYMPHNODES: int = field(default=BIG_NUM)
    MAX_PMNS: int = field(default=BIG_NUM)
    MAX_DCS: int = field(default=BIG_NUM)
    MAX_MACROPHAGES: int = field(default=BIG_NUM)
    MAX_NKS: int = field(default=BIG_NUM)
    MAX_ACTIVATED_ENDOS: int = field(default=BIG_NUM)

    macro_phago_recovery: float = field(default=0.5)
    macro_phago_limit: int = field(default=1_000)

    inflammasome_activation_threshold: int = field(default=10)  # 50 for bats
    inflammasome_priming_threshold: float = field(default=1.0)  # 5.0 for bats

    viral_carrying_capacity: int = field(default=500)
    # resistance_to_infection: int = field(default=75)
    susceptibility_to_infection: int = field(default=77)
    human_endo_activation: int = field(default=5)
    bat_endo_activation: int = field(default=10)
    metabolic_byproduct: float = field(default=0.2)
    resistance_to_infection: int = field(default=75)

    apoptosis_eaten_counter: int = field(default=0, init=False)

    ######################################################################
    # epithelium

    epithelium = field(type=np.ndarray)

    @epithelium.default
    def _epithelium_factory(self):
        return np.full(self.geometry, EpiType.Healthy, dtype=EpiType)

    epithelium_ros_damage_counter = field(type=np.ndarray)

    @epithelium_ros_damage_counter.default
    def _epithelium_ros_damage_counter_factory(self):
        return np.zeros(self.geometry, dtype=np.float64)

    epi_regrow_counter = field(type=np.ndarray)

    @epi_regrow_counter.default
    def _epi_regrow_counter_factory(self):
        return np.zeros(self.geometry, dtype=np.int64)

    epi_apoptosis_counter = field(type=np.ndarray)

    @epi_apoptosis_counter.default
    def _epi_apoptosis_counter_factory(self):
        return np.zeros(self.geometry, dtype=np.int64)

    epi_intracellular_virus = field(type=np.ndarray)

    @epi_intracellular_virus.default
    def _epi_intracellular_virus_factory(self):
        return np.zeros(self.geometry, dtype=np.int64)

    epi_cell_membrane = field(type=np.ndarray)

    @epi_cell_membrane.default
    def _epi_cell_membrane_factory(self):
        return np.random.randint(975, 975 + 51, size=self.geometry)

    epi_apoptosis_threshold = field(type=np.ndarray)

    @epi_apoptosis_threshold.default
    def _epi_apoptosis_threshold_factory(self):
        # TODO: when regrowth happens, the spread is different (475-526), is this intentional?
        return np.random.randint(450, 450 + 100, size=self.geometry)

    epithelium_apoptosis_counter = field(type=np.ndarray)

    @epithelium_apoptosis_counter.default
    def _epithelium_apoptosis_counter_factory(self):
        return np.zeros(self.geometry, dtype=np.int64)

    ######################################################################
    # endothelium

    endothelial_activation = field(type=np.ndarray)

    @endothelial_activation.default
    def _endothelial_activation_factory(self):
        return np.full(self.geometry, EndoType.Normal, dtype=EndoType)

    endothelial_adhesion_counter = field(type=np.ndarray)

    @endothelial_adhesion_counter.default
    def _endothelial_adhesion_counter_factory(self):
        return np.zeros(self.geometry, dtype=np.int64)

    ######################################################################

    extracellular_virus: np.ndarray = field(init=False)
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

    ######################################################################

    num_pmns: int = 0
    pmn_pointer: int = 0

    pmn_mask = field(type=np.ndarray)

    @pmn_mask.default
    def _pmn_mask_factory(self):
        return np.zeros(self.MAX_PMNS, dtype=bool)

    pmn_locations = field(type=np.ndarray)

    @pmn_locations.default
    def _pmn_locations_factory(self):
        return np.zeros((self.MAX_PMNS, 2), dtype=np.float64)

    pmn_dirs = field(type=np.ndarray)

    @pmn_dirs.default
    def _pmn_dirs_factory(self):
        return np.zeros(self.MAX_PMNS, dtype=np.float64)

    pmn_age = field(type=np.ndarray)

    @pmn_age.default
    def _pmn_age_factory(self):
        return np.zeros(self.MAX_PMNS, dtype=np.int64)

    ######################################################################

    num_macros: int = 0
    macro_pointer: int = 0

    macro_mask = field(type=np.ndarray)

    @macro_mask.default
    def _macro_mask_factory(self):
        return np.zeros(self.MAX_MACROPHAGES, dtype=bool)

    macro_locations = field(type=np.ndarray)

    @macro_locations.default
    def _macro_locations_factory(self):
        return np.zeros((self.MAX_MACROPHAGES, 2), dtype=np.float64)

    macro_dirs = field(type=np.ndarray)

    @macro_dirs.default
    def _macro_dirs_factory(self):
        return np.zeros(self.MAX_MACROPHAGES, dtype=np.float64)

    macro_internal_virus = field(type=np.ndarray)

    @macro_internal_virus.default
    def _macro_internal_virus_factory(self):
        return np.zeros(self.MAX_MACROPHAGES, dtype=np.float64)

    macro_activation = field(type=np.ndarray)

    @macro_activation.default
    def _macro_activation_factory(self):
        return np.zeros(self.MAX_MACROPHAGES, dtype=np.float64)

    macro_infected = field(type=np.ndarray)

    @macro_infected.default
    def _macro_infected_factory(self):
        return np.zeros(self.MAX_MACROPHAGES, dtype=bool)

    macro_cells_eaten = field(type=np.ndarray)

    @macro_cells_eaten.default
    def _macro_cells_eaten_factory(self):
        return np.zeros(self.MAX_MACROPHAGES, dtype=np.int64)

    macro_virus_eaten = field(type=np.ndarray)

    @macro_virus_eaten.default
    def _macro_virus_eaten_factory(self):
        return np.zeros(self.MAX_MACROPHAGES, dtype=np.float64)

    macro_pre_il1 = field(type=np.ndarray)

    @macro_pre_il1.default
    def _macro_pre_il1_factory(self):
        return np.zeros(self.MAX_MACROPHAGES, dtype=np.float64)

    macro_pre_il18 = field(type=np.ndarray)

    @macro_pre_il18.default
    def _macro_pre_il18_factory(self):
        return np.zeros(self.MAX_MACROPHAGES, dtype=np.float64)

    macro_pyroptosis_counter = field(type=np.ndarray)

    @macro_pyroptosis_counter.default
    def _macro_pyroptosis_counter_factory(self):
        return np.zeros(self.MAX_MACROPHAGES, dtype=np.float64)

    macro_inflammasome_primed = field(type=np.ndarray)

    @macro_inflammasome_primed.default
    def _macro_inflammasome_primed_factory(self):
        return np.zeros(self.MAX_MACROPHAGES, dtype=bool)

    macro_inflammasome_active = field(type=np.ndarray)

    @macro_inflammasome_active.default
    def _macro_inflammasome_active_factory(self):
        return np.zeros(self.MAX_MACROPHAGES, dtype=bool)

    ######################################################################

    num_nks = field(default=0, type=int)
    nk_pointer = field(default=0, type=int)

    nk_mask = field(type=np.ndarray)

    @nk_mask.default
    def _nk_mask_factory(self):
        return np.zeros(self.MAX_NKS, dtype=bool)

    nk_locations = field(type=np.ndarray)

    @nk_locations.default
    def _nk_locations_factory(self):
        return np.zeros((self.MAX_NKS, 2), dtype=np.float64)

    nk_dirs = field(type=np.ndarray)

    @nk_dirs.default
    def _nk_dirs_factory(self):
        return np.zeros(self.MAX_NKS, dtype=np.float64)

    nk_age = field(type=np.ndarray)

    @nk_age.default
    def _nk_age_factory(self):
        return np.zeros(self.MAX_NKS, dtype=np.int64)

    ######################################################################

    num_dcs: int = 0
    dc_pointer: int = 0

    dc_mask = field(type=np.ndarray)

    @dc_mask.default
    def _dc_mask_factory(self):
        return np.zeros(self.MAX_DCS, dtype=bool)

    dc_locations = field(type=np.ndarray)

    @dc_locations.default
    def _dc_locations_factory(self):
        return np.zeros((self.MAX_DCS, 2), dtype=np.float64)

    dc_dirs = field(type=np.ndarray)

    @dc_dirs.default
    def _dc_dirs_factory(self):
        return np.zeros(self.MAX_DCS, dtype=np.float64)

    ######################################################################
    # static properties

    @property
    def geometry(self):
        return self.GRID_HEIGHT, self.GRID_WIDTH

    ######################################################################
    # non-static properties

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
        geom = self.geometry

        # spatial quantities
        self.extracellular_virus = np.zeros(geom, dtype=np.float64)
        self.epi_regrow_counter = np.zeros(geom, dtype=np.int32)
        self.endothelial_activation = np.zeros(geom, dtype=bool)
        self.P_DAMPS = np.zeros(geom, dtype=np.float64)
        self.ROS = np.zeros(geom, dtype=np.float64)
        self.PAF = np.zeros(geom, dtype=np.float64)
        self.TNF = np.zeros(geom, dtype=np.float64)
        self.IL1 = np.zeros(geom, dtype=np.float64)
        self.IL18 = np.zeros(geom, dtype=np.float64)
        self.IL2 = np.zeros(geom, dtype=np.float64)
        self.IL4 = np.zeros(geom, dtype=np.float64)
        self.IL6 = np.zeros(geom, dtype=np.float64)
        self.IL8 = np.zeros(geom, dtype=np.float64)
        self.IL10 = np.zeros(geom, dtype=np.float64)
        self.IL12 = np.zeros(geom, dtype=np.float64)
        self.IL17 = np.zeros(geom, dtype=np.float64)
        self.IFNg = np.zeros(geom, dtype=np.float64)
        self.T1IFN = np.zeros(geom, dtype=np.float64)

        #   ask patches
        #   [sprout 1
        #     [set breed epis ;; 1 epi per patch
        #      set shape "square"
        #      set color blue
        #      set intracellular-virus 0
        #      set viral-carrying-capacity 500 ;; arbitrary, maybe make slider. Lower numbers kill but don't spread,
        #                                         defines incubation time
        #      set resistance-to-infection 75 ;; arbitrary, maybe make slider
        #      set cell-membrane 975 + random 51 ;; this is what is consumed by viral excytosis, includes some random
        #                                           component so all cells don't die at the same time
        #      set apoptosis-counter 0
        #      set apoptosis-threshold 450 + random 100 ;; this is half the cell-membrane, which means total amount of
        #                                                  leaked virus should be half with apoptosis active, has
        #                                                  random component as well
        #      if bat? = true ;; help initialize baseline level production of T1IFN in bats
        #        [if random 100 = 1
        #         [set T1IFN 5]
        #       ]
        #     ]
        #   ]

        if self.BAT:
            self.T1IFN[:, :] = 5 * (np.random.rand(*geom) < 0.01)

        #  create-NKs 25 ;; Initial-NKs slider for later
        #   [set color orange
        #     set shape "circle"
        #     set size 1
        #     repeat 100
        #     [jump random 1000]
        #   ]
        self.create_nk(number=self.INIT_NKS)

        #  create-Macros 50 ;;Initial-Macros slider for later
        #   [set color green
        #     set shape "circle"
        #     set size 1
        #     repeat 100
        #     [jump random 1000]
        #     set macro-phago-limit 1000 ;; arbitrary number
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
            np.random.choice(self.GRID_HEIGHT * self.GRID_WIDTH, init_inoculum),
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
        # ;; what I want here is to have the likelihood of invasion of the epi on the patch be a function of the
        #    number of extracellular-viruses on that patch
        # ;; Epi has resistance-to-invasion, higher numbers better to protect against invasion
        # ;; As currently written, lowering resistance to infection results in smaller initial intracellular virus,
        #    prolonging incubation,
        # ;; can affect by altering viral-incubation-rate (slider)
        # ;if random extracellular-virus > resistance-to-infection ;; if this is true then the virus invades
        # if extracellular-virus > 0
        # [ if random 100 < (max list susceptibility-to-infection extracellular-virus) ;; this is a new virus invade
        #                                                                                 criteria, says a % chance
        #                                                                                 infection against higher of
        #                                                                                 resistance or virus present
        #   ; so if very high extracellular virus then likely to invade, if very low extracellular virus still
        #     possibility it will invade. High values susceptibility = worse
        #   ; if susceptibility = 0 should have full resistance
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
        #   ;; extrusion of virus will consume cell-membrane, when this goes to 0 cell dies (no viral burst but does
        #      produce P/DAMPS)
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
        #   [set intracellular-virus max list 1 (intracellular-virus + 1 - (T1IFN / 10 ))];; simulates T1IFN anti viral
        #                                                                                    adaptations in bats,
        #                                                                                ;; does not eradicate virus
        #                                                                                   though, just suppresses
        #                                                                                   growth
        #   [set intracellular-virus max list 0 (intracellular-virus + 1 - (T1IFN / 100))] ;; human manifestation of
        #                                                                                     T1IFN anti viral effect
        #
        # end

    def epi_apoptosis(self):
        # to epi-apoptosis
        #   if apoptosis-counter > apoptosis-threshold
        #    [set breed apoptosed-epis
        #     set color grey
        #     set shape "pentagon"
        #     set size 1
        #   ]
        infected_epi_mask = self.epithelium == EpiType.Infected
        epis_to_apoptose_mask = infected_epi_mask & (
            self.epi_apoptosis_counter > self.epi_apoptosis_threshold
        )
        self.epithelium[epis_to_apoptose_mask] = EpiType.Apoptosed

        #   set apoptosis-counter apoptosis-counter + 1
        self.epi_apoptosis_counter[infected_epi_mask] += 1

        # end

    def regrow_epis(self):
        # to regrow-epis ;; patch command, regrows epis on empty patches if neighbors > 2 epis.
        # if count epis-here + count dead-epis-here + count apoptosed-epis-here + extracellular-virus = 0 ;; makes sure
        #                                                                                                    it is an
        #                                                                                                    patch empty
        #                                                                                                    of epis

        empty_patches = (self.epithelium == EpiType.Empty) & (
            self.extracellular_virus == 0
        )

        epi_patches = self.epithelium == EpiType.Healthy
        epi_neighbors = (
            np.roll(epi_patches, 1, axis=0)
            + np.roll(epi_patches, -1, axis=0)
            + np.roll(epi_patches, 1, axis=1)
            + np.roll(epi_patches, -1, axis=1)
            + np.roll(np.roll(epi_patches, 1, axis=0), 1, axis=1)
            + np.roll(np.roll(epi_patches, 1, axis=0), -1, axis=1)
            + np.roll(np.roll(epi_patches, -1, axis=0), 1, axis=1)
            + np.roll(np.roll(epi_patches, -1, axis=0), -1, axis=1)
        )

        regrowth_candidate_patches = empty_patches & (epi_neighbors > 2)

        #  [if count epis-on neighbors > 2 ;; if conditions met, start counter
        #     [if epi-regrow-counter >= 432 ;; regrows starting at 3 days if sufficient neighboring epis

        regrowth_patches = regrowth_candidate_patches & (self.epi_regrow_counter > 432)

        #      [sprout 1
        #        [set breed epis ;; 1 epi per patch
        #         set shape "square"
        #         set color blue
        #         set intracellular-virus 0
        #         set resistance-to-infection 75 ;; arbitrary, maybe make slider
        #         set cell-membrane 975 + random 51 ;; this is what is consumed by viral excytosis, includes some
        #                                              random component so all cells don't die at the same time
        #         set apoptosis-counter 0
        #         set apoptosis-threshold 475 + random 51 ;; this is half the cell-membrane, which means total amount of
        #                                                    leaked virus should be half with apoptosis active, has
        #                                                    random component as well
        #        ]
        #       set epi-regrow-counter 0 ; if sprouts, reset counter to 0

        self.epi_intracellular_virus[regrowth_patches] = 0
        self.epi_cell_membrane[regrowth_patches] = np.random.randint(
            975, 975 + 51, size=np.sum(regrowth_patches)
        )
        self.epi_apoptosis_counter[regrowth_patches] = 0
        self.epi_apoptosis_threshold[regrowth_patches] = np.random.randint(
            475, 475 + 51, size=np.sum(regrowth_patches)
        )
        self.epi_regrow_counter[regrowth_patches] = 0

        #      ]

        #     set epi-regrow-counter epi-regrow-counter + 1 ;; counts if enough neighbors
        #     ]
        #   ]
        non_regrowth_patches = regrowth_candidate_patches & (
            self.epi_regrow_counter <= 432
        )
        self.epi_regrow_counter[non_regrowth_patches] += 1

        # end

    def dead_epi_update(self):
        self.P_DAMPS[self.epithelium == EpiType.Dead] += 1
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
        #    if count PMNs-here > 1 ;; This code block is to prevent stacking of macros on a single patch. IF there is
        #                              a stack then they kill the infected more quickly
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
        self.pmn_locations[self.pmn_mask] += 0.1 * vecs / np.expand_dims(norms, axis=-1)
        self.pmn_dirs[self.pmn_mask] = np.arctan2(vecs[:, 1], vecs[:, 0])

        #   [wiggle
        #   ]
        self.pmn_dirs += (
            (np.random.rand(self.MAX_PMNS) - np.random.rand(self.MAX_PMNS))
            * np.pi
            / 4.0
        )
        directions = np.stack([np.cos(self.pmn_dirs), np.sin(self.pmn_dirs)], axis=1)
        self.pmn_locations += 0.1 * directions
        self.pmn_locations = np.mod(
            self.pmn_locations, [self.GRID_HEIGHT, self.GRID_WIDTH]
        )

        #  set age age + 1
        self.pmn_age[self.pmn_mask] += 1

        # end

    def nk_update(self):
        # to NK-function
        #
        #   ;; INDUCTION OF APOPTOSIS
        # ask infected-epis-here
        #  [set apoptosis-counter apoptosis-counter + 9]  ;; NKs enhance infected epi apoptosis 10x
        locations = self.nk_locations[self.nk_mask].astype(np.int64)
        nk_at_infected_epi_mask = (
            self.epithelium[tuple(locations.T)] == EpiType.Infected
        )
        self.epi_apoptosis_counter[tuple(locations[nk_at_infected_epi_mask].T)] += 9

        #   ;;Production of cytokines
        # if T1IFN > 0 and IL12 > 0 and IL18 > 0
        #   [set IFNg IFNg + 1] ;; need to check this, apparently does not happen, needs IL18(?)
        cytokine_production_mask = (
            (self.T1IFN[tuple(locations.T)] > 0)
            & (self.IL12[tuple(locations.T)] > 0)
            & (self.IL18[tuple(locations.T)] > 0)
        )
        cytokine_production_locations = self.nk_locations[self.nk_mask][
            cytokine_production_mask
        ].astype(np.int64)
        self.IFNg[tuple(cytokine_production_locations.T)] += 1

        #   ;; Chemotaxis to T1IFN made by infected-epis, slows movement rate to 1/10 of uphill primitive
        # let p max-one-of neighbors [T1IFN]  ;; or neighbors4
        #   ifelse [T1IFN] of p > T1IFN and [T1IFN] of p != 0
        #   [face p
        # ;;    if [count NKs] of patch-ahead 1 < 5
        #    fd 0.1
        #    if count NKs-here > 1 ;; This code block is to prevent stacking of NK cells on a single patch. IF there is
        #                             a stack then they kill the infected more quickly
        #     [set heading random 360
        #       fd 1]
        #   ]
        chemoattractant = self.T1IFN
        chemoattractant_dx = (
            np.roll(chemoattractant, -1, axis=0) - np.roll(chemoattractant, 1, axis=0)
        ) / 2.0
        chemoattractant_dy = (
            np.roll(chemoattractant, -1, axis=1) - np.roll(chemoattractant, 1, axis=1)
        ) / 2.0
        locations = self.nk_locations[self.nk_mask].astype(np.int64)
        vecs = np.stack(
            [
                chemoattractant_dx[tuple(locations.T)],
                chemoattractant_dy[tuple(locations.T)],
            ],
            axis=1,
        )
        norms = np.linalg.norm(vecs, axis=-1)
        norms[norms <= 1e-8] = 1.0  # effectively zero norm vectors will be unnormalized
        self.nk_locations[self.nk_mask] += 0.1 * vecs / np.expand_dims(norms, axis=-1)
        self.nk_dirs[self.nk_mask] = np.arctan2(vecs[:, 1], vecs[:, 0])

        #   [wiggle
        #   ]
        self.nk_dirs += (
            (np.random.rand(self.MAX_NKS) - np.random.rand(self.MAX_NKS)) * np.pi / 4.0
        )
        directions = np.stack([np.cos(self.nk_dirs), np.sin(self.nk_dirs)], axis=1)
        self.nk_locations += 0.1 * directions
        self.nk_locations = np.mod(
            self.nk_locations, [self.GRID_HEIGHT, self.GRID_WIDTH]
        )

        # TODO: unstack

        #   ;consumption IL18 and IL12
        # set IL12 max list 0 IL12 - 0.1
        # set IL18 max list 0 IL18 - 0.1

        locations = self.nk_locations[self.nk_mask].astype(np.int64)
        self.IL12[tuple(locations.T)] -= np.minimum(0.1, self.IL12[tuple(locations.T)])
        self.IL18[tuple(locations.T)] -= np.minimum(0.1, self.IL18[tuple(locations.T)])

        # end

    def macro_update(self):
        # to macro-function
        #
        # ; check to see if macro gets infected
        # virus-invade-cell
        mask = self.macro_mask
        locations = self.macro_locations[mask].astype(np.int64)
        extracellular_virus_at_locations = self.extracellular_virus[tuple(locations.T)]
        cells_to_invade = 100 * np.random.rand(
            *extracellular_virus_at_locations.shape
        ) < np.maximum(self.susceptibility_to_infection, extracellular_virus_at_locations)
        self.macro_internal_virus[mask][cells_to_invade] += 1
        self.extracellular_virus[tuple(locations[cells_to_invade].T)] -= 1
        self.macro_infected[mask][cells_to_invade] = True

        #   ; macro-activation-level keeps track of M1 (pro) or M2 (anti) status
        #   ; there is hysteresis because it modifies existing status
        # set macro-activation-level macro-activation-level + T1IFN + P/DAMPS + IFNg + IL1 - (2 * IL10) ;; currently a
        #                                                                                                  gap between
        #                                                                                                  pro and anti
        #                                                                                                  macros
        self.macro_activation[mask] += (
            self.T1IFN[tuple(locations.T)]
            + self.P_DAMPS[tuple(locations.T)]
            + self.IFNg[tuple(locations.T)]
            + self.IL1[tuple(locations.T)]
            - 2 * self.IL10[tuple(locations.T)]
        )

        #   ;; separate out inflammasome mediated functions from other pro macro functions
        #   ;; so IL1/IL18 production, induction of pyroptosis
        #   ;; as with all sequential processes, inflammasome priming/activation/effects coded in reverse order
        # inflammasome-function

        self.inflammasome_function()

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

        #   ;; Chemotaxis to T1IFN made by infected-epis, slows movement rate to 1/10 of uphill primitive. Also
        #      chemotaxis to DAMPs
        # let p max-one-of neighbors [T1IFN + P/DAMPS]  ;; or neighbors4
        #   ifelse [T1IFN + P/DAMPS] of p > (T1IFN + P/DAMPS) and [T1IFN + P/DAMPS] of p != 0
        #   [face p
        # ;;    if [count NKs] of patch-ahead 1 < 5
        #    fd 0.1
        #    if count Macros-here > 1 ;; This code block is to prevent stacking of macros on a single patch. IF there is
        #                                a stack then they kill the infected more quickly
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
        self.macro_locations[self.macro_mask] += 0.1 * vecs / np.expand_dims(norms, axis=-1)
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
        self.macro_locations += 0.1 * np.stack(
            [np.cos(self.macro_dirs), np.sin(self.macro_dirs)], axis=1
        )
        self.macro_locations = np.mod(
            self.macro_locations, [self.GRID_HEIGHT, self.GRID_WIDTH]
        )

        # TODO: unstack

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
        locations = self.macro_locations[self.macro_mask][under_limit_mask].astype(
            np.int64
        )

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

        # if macro-activation-level > 5 ;; This is where decreased sensitivity to P/DAMPS can be seen. Link
        #                                  macro-activation-level to Inflammasome variable?
        #                               ;; increased from 1.1 to 5 for V1.1
        #  [;; Proinflammatory cytokine production
        #   if IL1 + P/DAMPS > 0 ;; these are downstream products of NFkB, which either requires infammasome activation
        #                           or TLR signaling
        #     [set TNF TNF + 1
        #      set IL6 IL6 + 0.4 ;; split with DCs and infected-epis, dependent on IL1
        #  ;; set IL1 IL1 + 1 => Moved to Inflammasome function
        #     ;; Anti-inflammatory cytokine production
        #     set IL10 IL10 + 1
        #     ]
        #   set IL8 IL8 + 1 ;; believe this is NFkB independent
        #   set IL12 IL12 + 0.5 ;; split with DCs Don't know if this is based on NFkB activation
        #
        #   set macro-activation-level macro-activation-level - 5 ;; this is to simulate macrophages returning to
        #                                                            baseline as intracellular factors lose stimulation
        #  ]

        activated_macros = self.macro_mask & (self.macro_activation > 5)
        locations = self.macro_locations[activated_macros].astype(np.int64)
        self.IL8[tuple(locations.T)] += 1.0
        self.IL12[tuple(locations.T)] += 1.0
        downstream_mask = (
            self.IL1[tuple(locations.T)] + self.P_DAMPS[tuple(locations.T)] > 0
        )
        downstream_locations = self.macro_locations[activated_macros][downstream_mask].astype(np.int64)
        self.TNF[tuple(downstream_locations.T)] += 1
        self.IL6[tuple(downstream_locations.T)] += 1
        self.IL10[tuple(downstream_locations.T)] += 1
        self.macro_activation[activated_macros] -= 5

        # if macro-activation-level < -5 ;; this keeps macros from self activating wrt IL10 in perpetuity
        #   [set color pink ;; tracker
        #   ;; Antiinflammatory cytokine production
        #   set IL10 IL10 + 0.5
        #
        #   set macro-activation-level macro-activation-level + 5 ;; this is to simulate macrophages returning to
        #                                                            baseline as intracellular factors lose stimulation
        #   ]

        low_activated_macros = self.macro_mask & (self.macro_activation < -5)
        locations = self.macro_locations[low_activated_macros].astype(np.int64)
        self.IL10[tuple(locations.T)] += 0.5
        self.macro_activation[low_activated_macros] += 5

        # end

    def inflammasome_function(self):
        # to inflammasome-function
        #  ;; inflammasome effects
        #
        #   if pyroptosis-counter > 12  ;; 120 minutes
        #   [pyroptosis]

        self.pryoptosis()
        # TODO: FIXME
        # pyroptosis_mask = self.macro_mask & (self.macro_pyroptosis_counter > 12)

        #
        #   if inflammasome-active = true ;; this is coded this way to draw out the release of IL1 and IL18 (make
        #                                                                                                   less jumpy)
        #     [set IL1 Il1 + 1
        #      set pre-IL1 pre-IL1 + 5 ;; this means pre-IL1 will be 12 at time of burst, which is IL1 level at burst
        #      set IL18 IL18 + 1
        #      set pre-IL18 pre-IL18 + 0.5 ;; mimic IL1 at the moment, half amount generated though (arbitrary)
        #      set pyroptosis-counter pyroptosis-counter + 1] ;; Pyroptosis takes 120 min from inflammasome activation
        #                                                        to pyroptosis so counter threshold is 12

        inflammasome_active_mask = self.macro_mask & self.macro_inflammasome_active
        locations = self.macro_locations[inflammasome_active_mask].astype(np.int64)
        self.IL1[tuple(locations.T)] += 1
        self.macro_pre_il1[inflammasome_active_mask] += 5
        self.IL18[tuple(locations.T)] += 1
        self.macro_pre_il18[inflammasome_active_mask] += 0.5
        self.macro_pyroptosis_counter[inflammasome_active_mask] += 1

        #   ;; stage 2 = activation of inflammasome => use virus-eaten as proxy for intracellular viral products,
        #      effects of phagocytosis
        #   if inflammasome-primed = true
        #    [if virus-eaten / 10 > inflammasome-activation-threshold ;; extra-, intra- and eaten virus in the scale of
        #                                                                up to 100
        #                                                           ;; macro-phago-limit set to 1000
        #                                                              (with virus-eaten / 10), activation
        #                                                              thresholds ~ 50
        #     [set inflammasome-active true
        #      set shape "target"
        #      set size 1.5
        #     ]
        #   ]

        inflammasome_primed_mask = (
            self.macro_mask
            & self.macro_inflammasome_primed
            & (self.macro_virus_eaten / 10 > self.inflammasome_activation_threshold)
        )
        self.macro_inflammasome_active[inflammasome_primed_mask] = True

        #
        #   ;; stage 1 = Priming by P/DAMPS or TNF
        #   if P/DAMPS + TNF > inflammasome-priming-threshold
        #   [set inflammasome-primed true]
        inflammasome_priming_mask = self.macro_mask & (
            (self.P_DAMPS + self.TNF) > self.inflammasome_priming_threshold
        )[tuple(self.macro_locations.T.astype(np.int64))]
        self.macro_inflammasome_primed[inflammasome_priming_mask] = True

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
        # hatch 1 ;; this maintains steady number of macrophages, thus this model does not simulate macrophage
        #            depletion via pyroptosis
        #   [set color green
        #    set shape "circle"
        #    set size 1
        #     move-to one-of patches with [count epis + count infected-epis > 0] ;; this has regenerated macros move to
        #                                                                           a random patch that isn't already
        #                                                                           dead
        #    set macro-phago-limit 1000 ;; arbitrary number
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
        # to DC-function

        # ACK: DC-location ignored, I'm omitting this
        # if DC-location = "tissue"
        #   ;;Production of cytokines

        # [if T1IFN > 1
        #   [set IL12 IL12 + 0.5 ;; split with pro macros
        #    if IL1 > 1
        #       [set IL6 IL6 + 0.4] ;; split with pro macros and infected-epis, dependent on IL1 presence
        #    set IFNg IFNg + 0.5
        #   ]

        t1ifn_activated_mask = self.dc_mask & (
            self.T1IFN[tuple(self.dc_locations.T.astype(np.int64))] > 1
        )
        t1ifn_activated_locations = self.dc_locations[t1ifn_activated_mask].astype(np.int64)
        self.IL12[tuple(t1ifn_activated_locations.T)] += 0.5
        self.IFNg[tuple(t1ifn_activated_locations.T)] += 0.5
        
        il1_activated_mask = t1ifn_activated_mask & (
            self.IL1[tuple(self.dc_locations.T.astype(np.int64))] > 1
        )
        il1_activated_locations = self.dc_locations[il1_activated_mask].astype(np.int64)
        self.IL6[tuple(il1_activated_locations.T)] += 0.5

        #  ; consumption of mediators
        #  set T1IFN max list 0 (T1IFN - 0.1)

        locations = self.dc_locations[self.dc_mask].astype(np.int64)
        self.T1IFN[tuple(locations.T)] -= np.minimum(
            0.1, self.T1IFN[tuple(locations.T)]
        )

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

        chemoattractant = self.T1IFN
        chemoattractant_dx = (
            np.roll(chemoattractant, -1, axis=0) - np.roll(chemoattractant, 1, axis=0)
        ) / 2.0
        chemoattractant_dy = (
            np.roll(chemoattractant, -1, axis=1) - np.roll(chemoattractant, 1, axis=1)
        ) / 2.0
        locations = self.dc_locations[self.dc_mask].astype(np.int64)
        vecs = np.stack(
            [
                chemoattractant_dx[tuple(locations.T)],
                chemoattractant_dy[tuple(locations.T)],
            ],
            axis=1,
        )
        norms = np.linalg.norm(vecs, axis=-1)
        norms[norms <= 1e-8] = 1.0  # effectively zero norm vectors will be unnormalized
        self.dc_locations[self.dc_mask] += 0.1 * vecs / np.expand_dims(norms, axis=-1)
        self.dc_dirs[self.dc_mask] = np.arctan2(vecs[:, 1], vecs[:, 0])

        #    [wiggle
        #    ]

        self.dc_dirs += (
            (np.random.rand(self.MAX_DCS) - np.random.rand(self.MAX_DCS)) * np.pi / 4.0
        )
        self.dc_locations += 0.1 * np.stack(
            [np.cos(self.dc_dirs), np.sin(self.dc_dirs)], axis=1
        )
        self.dc_locations = np.mod(
            self.dc_locations, [self.GRID_HEIGHT, self.GRID_WIDTH]
        )

        # TODO: de-stack

        #
        #   ]
        #
        # end

    def epi_update(self):
        # to epi-function
        # ;; necrosis from PMN burst
        # if ROS-damage-counter > 2
        #   [set breed dead-epis
        #    set color grey
        #    set size 1
        #    set shape "circle"
        #    set P/DAMPs P/DAMPS + 10
        #   ]

        ros_damage_mask = self.epithelium_ros_damage_counter > 2
        self.epithelium[ros_damage_mask] = EpiType.Dead
        self.P_DAMPS[ros_damage_mask] += 10

        # set ROS-damage-counter ROS-damage-counter + ROS
        self.epithelium_ros_damage_counter += self.ROS

        #   metabolism
        self.metabolism()

        #   virus-invade-cell
        self.virus_invade_cell()

        #  if bat? = true
        #   [baseline-T1IFN-generation
        #   ]
        if self.BAT:
            self.baseline_t1ifn_generation()

        # ACK: not sure why this logical structure was chosen, why not combine the else in the `if BAT` above?
        # leaving it as is for now, TODO: change later

        #   ;; activates endothelium
        #   ifelse bat? = false
        if not self.BAT:
            #    [ if TNF + IL1 > human-endo-activation and count activated-endos-here = 0 ;; arbitrary level of
            #                                                                                 TNF + IL1 => set at 5 for
            #                                                                                 human, 10 for bat
            #      [hatch 1
            #       [set breed activated-endos
            #        set color pink
            #        set size 1.5
            #        set shape "square 2"
            #       ]
            mask = (
                (self.epithelium == EpiType.Healthy)
                & (self.TNF + self.IL1 > self.human_endo_activation)
                & np.logical_not(self.endothelial_activation)
            )
            self.endothelial_activation[mask] = True
            #
            #
            #       ;; simulates consumption of IL1 and TNF
            #    set TNF max list 0 (TNF - 0.1)
            #    set IL1 max list 0 (IL1 - 0.1)

            self.TNF[mask] -= np.minimum(0.1, self.TNF[mask])
            self.IL1[mask] -= np.minimum(0.1, self.IL1[mask])

            #    ]
            #   ]
        else:
            #    [ if TNF + IL1 > bat-endo-activation and count activated-endos-here = 0 ;; arbitrary level of
            #                                                                               TNF + IL1 => set at 5 for
            #                                                                               human, 10 for bat
            #      [hatch 1
            #       [set breed activated-endos
            #        set color pink
            #        set size 1.5
            #        set shape "square 2"
            #       ]

            mask = (
                (self.epithelium == EpiType.Healthy)
                & (self.TNF + self.IL1 > self.bat_endo_activation)
                & np.logical_not(self.endothelial_activation)
            )
            self.endothelial_activation[mask] = True

            #       ;; simulates consumption of IL1 and TNF
            #    set TNF max list 0 (TNF - 0.1)
            #    set IL1 max list 0 (IL1 - 0.1)

            self.TNF[mask] -= np.minimum(0.1, self.TNF[mask])
            self.IL1[mask] -= np.minimum(0.1, self.IL1[mask])

            #    ]
            #   ]
        # end

    def baseline_t1ifn_generation(self):
        # to baseline-T1IFN-generation
        #  if random 100 = 1
        #       [set T1IFN T1IFN + 0.75
        #  ]
        mask = (self.epithelium == EpiType.Healthy) & (
            np.random.rand(*self.geometry) < 0.01
        )
        self.T1IFN[mask] += 0.75

        # end

    def metabolism(self):
        # to metabolism
        #  if random 100 = 1
        #       [set P/DAMPs P/DAMPs + metabolic-byproduct]

        mask = (self.epithelium == EpiType.Healthy) & (
            np.random.rand(*self.geometry) < 0.01
        )
        self.P_DAMPS[mask] += self.metabolic_byproduct

        # end

    def activated_endo_update(self):
        # to activated-endo-function ;; are made in epi-function

        # set PAF PAF + 1
        self.PAF[self.endothelial_activation == EndoType.Activated] += 1

        # if TNF + IL1 < 0.5
        #   [die]
        self.endothelial_activation[self.TNF + self.IL1 < 0.5] = EndoType.Dead

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
        pmn_spawn_mask = (self.endothelial_adhesion_counter > 36) & (
            np.random.rand(*self.geometry) < 0.1
        )
        for r, c in zip(*np.where(pmn_spawn_mask)):
            self.create_pmn(location=(r, c), age=0, jump_dist=5)

        # set adhesion-counter adhesion-counter + 1
        self.endothelial_adhesion_counter[
            self.endothelial_activation == EndoType.Activated
        ] += 1

        # end

    def diffuse_functions(self):
        # to diffuse-functions
        #   diffuse extracellular-virus 0.05
        self._diffuse_molecule_field(self.extracellular_virus, 0.05)
        #   diffuse T1IFN 0.1
        self._diffuse_molecule_field(self.T1IFN, 0.1)
        #   diffuse PAF 0.1
        self._diffuse_molecule_field(self.PAF, 0.1)
        #   diffuse ROS 0.1
        self._diffuse_molecule_field(self.ROS, 0.1)
        #   diffuse P/DAMPs 0.1
        self._diffuse_molecule_field(self.P_DAMPS, 0.1)
        #   diffuse IFNg 0.2
        self._diffuse_molecule_field(self.IFNg, 0.2)
        #   diffuse TNF 0.2
        self._diffuse_molecule_field(self.TNF, 0.2)
        #   diffuse IL6 0.2
        self._diffuse_molecule_field(self.IL6, 0.2)
        #   diffuse IL1 0.2
        self._diffuse_molecule_field(self.IL1, 0.2)
        #   diffuse IL10 0.2
        self._diffuse_molecule_field(self.IL10, 0.2)
        #   diffuse IL12 0.2
        self._diffuse_molecule_field(self.IL12, 0.2)
        #   diffuse IL18 0.2
        self._diffuse_molecule_field(self.IL18, 0.2)
        #   diffuse IL8 0.3
        self._diffuse_molecule_field(self.IL8, 0.3)

        # end

    @staticmethod
    def _diffuse_molecule_field(
        molecule_field: np.ndarray, diffusion_constant: Union[float, np.float64]
    ):
        # based on https://ccl.northwestern.edu/netlogo/docs/dict/diffuse.html
        molecule_field[:, :] = (
            1 - diffusion_constant
        ) * molecule_field + diffusion_constant * (
            np.roll(molecule_field, 1, axis=0)
            + np.roll(molecule_field, -1, axis=0)
            + np.roll(molecule_field, 1, axis=1)
            + np.roll(molecule_field, -1, axis=1)
            + np.roll(np.roll(molecule_field, 1, axis=0), 1, axis=1)
            + np.roll(np.roll(molecule_field, 1, axis=0), -1, axis=1)
            + np.roll(np.roll(molecule_field, -1, axis=0), 1, axis=1)
            + np.roll(np.roll(molecule_field, -1, axis=0), -1, axis=1)
        ) / 8.0

    def cleanup(self):
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
        self.extracellular_virus[self.extracellular_virus < 1] = 0
        self.T1IFN[self.T1IFN < 0.1] = 0
        self.IFNg[self.IFNg < 0.1] = 0
        self.TNF[self.TNF < 0.1] = 0
        self.IL6[self.IL6 < 0.1] = 0
        self.IL1[self.IL1 < 0.1] = 0
        self.IL10[self.IL10 < 0.1] = 0
        self.PAF[self.PAF < 0.1] = 0
        self.IL8[self.IL8 < 0.1] = 0
        self.ROS[self.ROS < 0.1] = 0
        self.IL12[self.IL12 < 0.1] = 0
        self.IL8[self.IL8 < 0.1] = 0
        self.P_DAMPS[self.P_DAMPS < 0.1] = 0

    def evaporate(self):
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
        evap_const_1: float = 0.99
        evap_const_2: float = 0.9
        self.T1IFN *= evap_const_1
        self.IFNg *= evap_const_1
        self.TNF *= evap_const_1
        self.IL6 *= evap_const_1
        self.IL1 *= evap_const_1
        self.IL10 *= evap_const_1
        self.PAF *= evap_const_2
        self.IL8 *= evap_const_1
        self.ROS *= evap_const_2
        self.IL12 *= evap_const_1
        self.IL18 *= evap_const_1
        self.P_DAMPS *= evap_const_2

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
        #    evaporate ;; This order is important, if you put evaporate after cleanup you never get decent levels of
        #                 cytokine
        #    cleanup
        #  ]

        self.regrow_epis()
        self.evaporate()
        self.cleanup()

        #
        # end

    def create_nk(
        self,
        *,
        number: int = 1,
        loc: Optional[Union[Tuple[int, int], Tuple[float, float]]] = None,
    ):
        if number > 1:
            for _ in range(number):
                self.create_nk(loc=loc)
        elif number == 1:
            if self.nk_pointer >= self.MAX_NKS:
                self.compact_nk_arrays()
                # maybe the array is already compacted:
                if self.nk_pointer >= self.MAX_NKS:
                    raise RuntimeError(
                        "Max NKs exceeded, you may want to change the MAX_NKS parameter."
                    )
            if loc is None:
                self.nk_locations[self.nk_pointer, 0] = (
                    self.GRID_HEIGHT * np.random.rand()
                )
                self.nk_locations[self.nk_pointer, 1] = (
                    self.GRID_WIDTH * np.random.rand()
                )
            else:
                self.nk_locations[self.nk_pointer, :] = loc

            self.nk_dirs[self.nk_pointer] = 2 * np.pi * np.random.rand()
            self.nk_mask[self.nk_pointer] = True
            self.num_nks += 1
            self.nk_pointer += 1
        else:
            raise RuntimeError(
                f"Creating {number} NKs does not mean anything to this function"
            )

    def compact_nk_arrays(self):
        self.nk_locations[: self.num_nks] = self.nk_locations[self.nk_mask]
        self.nk_dirs[: self.num_nks] = self.nk_dirs[self.nk_mask]
        self.nk_mask[: self.num_nks] = True
        self.nk_mask[self.num_nks :] = False
        self.nk_pointer = self.num_nks
        # TODO: make sure all arrays are copied

    def create_macro(
        self,
        *,
        macro_phago_limit,
        pre_il1,
        pre_il18,
        inflammasome_primed,
        inflammasome_active,
        macro_activation_level,
        macro_phago_counter,
        pyroptosis_counter,
        virus_eaten,
        cells_eaten,
    ):
        pass  # TODO

    def create_dc(self, *, dc_location, trafficking_counter):
        pass  # TODO

    def create_pmn(self, *, location, age, jump_dist):
        pass  # TODO

    def plot_agents(self, ax: plt.Axes):
        # epithelium
        # * Blue Squares = Healthy Epithelial Cells
        # * Yellow Squares = Infected Epithelial Cells
        # * Grey Squares = Epithelial Cells killed by necrosis
        # * Grey Pentagons = Epithelial Cells killed by apoptosis
        ax.scatter(
            *np.where(self.epithelium == EpiType.Healthy),
            color="blue",
            marker="s",
            zorder=-1,
        )
        ax.scatter(
            *np.where(self.epithelium == EpiType.Infected),
            color="yellow",
            marker="s",
            zorder=-1,
        )
        ax.scatter(
            *np.where(self.epithelium == EpiType.Dead),
            color="grey",
            marker="s",
            zorder=-1,
        )
        ax.scatter(
            *np.where(self.epithelium == EpiType.Apoptosed),
            color="grey",
            marker="p",
            zorder=-1,
        )
        # macrophages
        # * Green Circles = Macrophages
        # * Large Green Circles = Macrophages at phagocytosis limit
        macro_phago_counter = np.maximum(
            0.0,
            self.macro_cells_eaten
            - self.macro_virus_eaten / 10
            - self.macro_phago_recovery,
        )  # TODO: deduplicate this?
        under_limit_mask = macro_phago_counter < self.macro_phago_limit
        ax.scatter(
            *self.macro_locations[self.macro_mask & under_limit_mask].T,
            color="green",
            marker="o",
        )
        ax.scatter(
            *self.macro_locations[self.macro_mask & np.logical_not(under_limit_mask)].T,
            color="green",
            marker="o",
            s=2 * plt.rcParams["lines.markersize"] ** 2,
        )
        # NKs
        # * Orange Circles = NK Cells
        ax.scatter(
            *self.nk_locations[self.nk_mask].T,
            color="orange",
            marker="o",
        )
        # DCs
        # * Light Blue Triangles = Dendritic Cells
        ax.scatter(
            *self.dc_locations[self.dc_mask].T,
            color="lightblue",
            marker="v",
        )
        # endos
        # * Pink Square Outlines = Activated Endothelial Cells
        ax.scatter(
            *np.where(self.endothelial_activation == EndoType.Activated),
            color="pink",
            marker=markers.MarkerStyle("s", fillstyle="none"),
        )
        # PMNs
        # * Small White Circles = PMNs
        ax.scatter(
            *self.pmn_locations[self.pmn_mask].T,
            color="black",
            marker=markers.MarkerStyle("o", fillstyle="none"),
        )

    def plot_field(self, ax: plt.Axes, *, field_name):
        assert field_name in {
            "extracellular_virus",
            "epi_regrow_counter",
            "endothelial_activation",
            "P_DAMPS",
            "ROS",
            "PAF",
            "TNF",
            "IL1",
            "IL18",
            "IL2",
            "IL4",
            "IL6",
            "IL8",
            "IL10",
            "IL12",
            "IL17",
            "IFNg",
            "T1IFN",
        }, "Unknown field!"

        ax.imshow(getattr(self, field_name), vmin=0, origin='lower')
