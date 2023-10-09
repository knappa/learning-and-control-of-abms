from enum import IntEnum
from typing import List, Optional, Tuple, Union

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from attr import define, field
from matplotlib import markers

BIG_NUM = 1000


class EpiType(IntEnum):
    Empty = 0
    Healthy = 1
    Infected = 2
    Dead = 3  # TODO: is this necrosis? better name?
    Apoptosed = 4


epithelial_cm = matplotlib.colors.ListedColormap(
    np.array(
        [
            (0.0, 0.0, 0.0, 0.0),  # empty
            matplotlib.colors.to_rgba("blue"),  # healthy
            matplotlib.colors.to_rgba("yellow"),  # infected
            matplotlib.colors.to_rgba("grey"),  # dead
            (0.0, 0.0, 0.0, 0.0),  # apoptosed
        ]
    )
)


class EndoType(IntEnum):
    Normal = 0
    Activated = 1
    Dead = 2


# noinspection PyPep8Naming
@define(kw_only=True)
class AnCockrellModel:
    GRID_WIDTH: int = field()
    GRID_HEIGHT: int = field()
    MAX_PMNS: int = field(default=BIG_NUM)
    MAX_DCS: int = field(default=BIG_NUM)
    MAX_MACROPHAGES: int = field(default=BIG_NUM)
    MAX_NKS: int = field(default=BIG_NUM)

    is_bat: bool = field()
    init_dcs: int = field()
    init_nks: int = field()
    init_macros: int = field()

    macro_phago_recovery: float = field(default=0.5)
    macro_phago_limit: int = field(default=1_000)

    inflammasome_activation_threshold: int = field(default=10)  # 50 for bats
    inflammasome_priming_threshold: float = field(default=1.0)  # 5.0 for bats

    viral_carrying_capacity: int = field(default=500)
    # resistance_to_infection: int = field(default=75)
    susceptibility_to_infection: int = field(default=77)
    human_endo_activation: int = field(default=5)
    bat_endo_activation: int = field(default=10)
    bat_metabolic_byproduct: float = field(default=2.0)
    human_metabolic_byproduct: float = field(default=0.2)
    resistance_to_infection: int = field(default=75)
    viral_incubation_threshold: int = field(default=60)

    # summary and statistical variables
    time = field(init=False, factory=lambda: 0, type=int)
    apoptosis_eaten_counter: int = field(default=0, init=False)
    pyroptosed_macros = field(init=False, factory=lambda: 0, type=int)

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
    # spatial fields

    extracellular_virus = field(type=np.ndarray)

    @extracellular_virus.default
    def _extracellular_virus_factory(self):
        return np.zeros(self.geometry, dtype=np.float64)

    P_DAMPS = field(type=np.ndarray)

    @P_DAMPS.default
    def _P_DAMPS_factory(self):
        return np.zeros(self.geometry, dtype=np.float64)

    ROS = field(type=np.ndarray)

    @ROS.default
    def _ROS_factory(self):
        return np.zeros(self.geometry, dtype=np.float64)

    PAF = field(type=np.ndarray)

    @PAF.default
    def _PAF_factory(self):
        return np.zeros(self.geometry, dtype=np.float64)

    TNF = field(type=np.ndarray)

    @TNF.default
    def _TNF_factory(self):
        return np.zeros(self.geometry, dtype=np.float64)

    IL1 = field(type=np.ndarray)

    @IL1.default
    def _IL1_factory(self):
        return np.zeros(self.geometry, dtype=np.float64)

    IL18 = field(type=np.ndarray)

    @IL18.default
    def _IL18_factory(self):
        return np.zeros(self.geometry, dtype=np.float64)

    IL2 = field(type=np.ndarray)  # TODO: this is unused?

    @IL2.default
    def _IL2_factory(self):
        return np.zeros(self.geometry, dtype=np.float64)

    IL4 = field(type=np.ndarray)  # TODO: this is unused?

    @IL4.default
    def _IL4_factory(self):
        return np.zeros(self.geometry, dtype=np.float64)

    IL6 = field(type=np.ndarray)

    @IL6.default
    def _IL6_factory(self):
        return np.zeros(self.geometry, dtype=np.float64)

    IL8 = field(type=np.ndarray)

    @IL8.default
    def _IL8_factory(self):
        return np.zeros(self.geometry, dtype=np.float64)

    IL10 = field(type=np.ndarray)

    @IL10.default
    def _IL10_factory(self):
        return np.zeros(self.geometry, dtype=np.float64)

    IL12 = field(type=np.ndarray)

    @IL12.default
    def _IL12_factory(self):
        return np.zeros(self.geometry, dtype=np.float64)

    IL17 = field(type=np.ndarray)  # TODO: This is unused?

    @IL17.default
    def _IL17_factory(self):
        return np.zeros(self.geometry, dtype=np.float64)

    IFNg = field(type=np.ndarray)

    @IFNg.default
    def _IFNg_factory(self):
        return np.zeros(self.geometry, dtype=np.float64)

    T1IFN = field(type=np.ndarray)

    @T1IFN.default
    def _T1IFN_factory(self):
        return np.zeros(self.geometry, dtype=np.float64)

    ######################################################################

    num_pmns = field(init=False, factory=lambda: 0, type=int)
    pmn_pointer = field(init=False, factory=lambda: 0, type=int)

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

    num_macros = field(init=False, factory=lambda: 0, type=int)
    macro_pointer = field(init=False, factory=lambda: 0, type=int)

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

    # macro_infected = field(type=np.ndarray)  # unused
    #
    # @macro_infected.default
    # def _macro_infected_factory(self):
    #     return np.zeros(self.MAX_MACROPHAGES, dtype=bool)

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

    macro_swollen = field(type=np.ndarray)

    @macro_swollen.default
    def _macro_swollen_factory(self):
        return np.zeros(self.MAX_MACROPHAGES, dtype=bool)

    @property
    def macro_phago_counter(self) -> np.ndarray:
        return np.maximum(
            0.0,
            self.macro_cells_eaten
            - self.macro_virus_eaten / 10
            - self.macro_phago_recovery,
        )

    ######################################################################

    num_nks = field(init=False, factory=lambda: 0, type=int)
    nk_pointer = field(init=False, factory=lambda: 0, type=int)

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

    num_dcs = field(init=False, factory=lambda: 0, type=int)
    dc_pointer = field(init=False, factory=lambda: 0, type=int)

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
    def geometry(self) -> Tuple[int, int]:
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
        if self.is_bat:
            self.T1IFN[:, :] = 5 * (np.random.rand(*self.geometry) < 0.01)

        self.create_nk(number=self.init_nks)

        for _ in range(self.init_macros):
            self.create_macro(
                pre_il1=0,
                pre_il18=0,
                inflammasome_primed=False,
                inflammasome_active=False,
                macro_activation_level=0,
                pyroptosis_counter=0,
                virus_eaten=0,
                cells_eaten=0,
            )

        for _ in range(self.init_dcs):
            self.create_dc()  # (dc_location="tissue", trafficking_counter=0)

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
            np.random.choice(
                self.GRID_HEIGHT * self.GRID_WIDTH, init_inoculum, replace=False
            ),
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

        infected_mask = self.epithelium == EpiType.Infected

        mask = infected_mask & (self.epithelium_ros_damage_counter > 10)
        self.epithelium[mask] = EpiType.Dead
        self.P_DAMPS[mask] += 10

        self.epithelium_ros_damage_counter[infected_mask] += self.ROS[infected_mask]

        # virus-replicate
        self.virus_replicate()

        # epi-apoptosis
        self.epi_apoptosis()

        # set T1IFN T1IFN + 1 ;; update by 1 appears to have too much T1IFN?
        # set IL18 IL18 + 0.11 ;; ? this rule?
        self.T1IFN[infected_mask] += 1
        self.IL18[infected_mask] += 0.11

        # if IL1 + TNF > 1
        #   [set IL6 IL6 + 0.10] ;; production shared with Macros and DCs, depends on IL1 and TNF production
        il6_mask = infected_mask & (self.IL1 + self.TNF > 1)
        self.IL6[il6_mask] += 0.10

        # end

    def virus_invade_epi_cell(self):
        # ACK: _actually_ only used in epi-function, contrary to comments. changed name to reflect this
        invasion_mask = (
            (self.epithelium == EpiType.Healthy)
            & (self.extracellular_virus > 0)
            & (
                100 * np.random.rand(*self.geometry)
                < np.maximum(self.susceptibility_to_infection, self.extracellular_virus)
            )
        )

        self.extracellular_virus[invasion_mask] -= 1
        self.epi_intracellular_virus[invasion_mask] += 1
        self.epithelium[invasion_mask] = EpiType.Infected

    def virus_replicate(self):
        # to virus-replicate ;; called in infected-epi-function
        #   ;; extrusion of virus will consume cell-membrane, when this goes to 0 cell dies (no viral burst but does
        #      produce P/DAMPS)
        # if cell-membrane <= 0
        #   [set breed dead-epis
        #    set color grey
        #    set P/DAMPS P/DAMPS + 10
        #  ; set extracellular-virus intracellular-virus
        #  ]
        burst_mask = (self.epithelium == EpiType.Infected) & (
            self.epi_cell_membrane <= 0.0
        )
        self.epithelium[burst_mask] = EpiType.Dead
        self.P_DAMPS[burst_mask] += 10.0

        # ;; intracellular-virus is essentially a counter that determines time it takes to ramp up viral synthesis
        # ;; once the threshold viral-incubation-threshold is reached the cell leaks virus to extracellular-virus
        # ;; and takes one cell-membrane away until cell dies
        # ;; cell defenses to this is to apoptose, which causes the "virus factory" to die earlier
        #
        # if intracellular-virus > viral-incubation-threshold
        #   [set extracellular-virus extracellular-virus + 1
        #    set cell-membrane cell-membrane - 1
        #   ]

        viral_replicate_mask = (self.epithelium == EpiType.Infected) & (
            self.epi_intracellular_virus > self.viral_incubation_threshold
        )
        self.extracellular_virus[viral_replicate_mask] += 1
        self.epi_cell_membrane[viral_replicate_mask] -= 1

        # ifelse bat? = true
        #   [set intracellular-virus max list 1 (intracellular-virus + 1 - (T1IFN / 10 ))];; simulates T1IFN anti viral
        #                                                                                    adaptations in bats,
        #                                                                                ;; does not eradicate virus
        #                                                                                   though, just suppresses
        #                                                                                   growth
        #   [set intracellular-virus max list 0 (intracellular-virus + 1 - (T1IFN / 100))] ;; human manifestation of
        #                                                                                     T1IFN anti viral effect
        mask = self.epithelium == EpiType.Infected
        viral_lower_bound = 1.0 if self.is_bat else 0.0
        t1fn_effect_scale = 0.1 if self.is_bat else 0.01
        self.epi_intracellular_virus[mask] = np.maximum(
            viral_lower_bound,
            self.epi_intracellular_virus[mask]
            + 1
            - self.T1IFN[mask] * t1fn_effect_scale,
        )

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
        self.num_pmns -= np.sum(age_mask)

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
        vecs += np.clip(
            np.random.normal(0.0, 1e-5, size=vecs.shape), -1.0, 1.0
        )  # small noise (randomizes direction in absence of gradient)
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
        self.pmn_locations = np.mod(self.pmn_locations, self.geometry)

        self._destack(self.pmn_mask, self.pmn_locations)

        #  set age age + 1
        self.pmn_age[self.pmn_mask] += 1

        # end

    def nk_update(self):
        # to NK-function

        locations = self.nk_locations[self.nk_mask].astype(np.int64)

        #   ;; INDUCTION OF APOPTOSIS
        # ask infected-epis-here
        #  [set apoptosis-counter apoptosis-counter + 9]  ;; NKs enhance infected epi apoptosis 10x

        nk_at_infected_epi_mask = (
            self.epithelium[tuple(locations.T)] == EpiType.Infected
        )
        self.epi_apoptosis_counter[tuple(locations[nk_at_infected_epi_mask].T)] += 9

        #   ;;Production of cytokines
        # if T1IFN > 0 and IL12 > 0 and IL18 > 0
        #   [set IFNg IFNg + 1] ;; need to check this, apparently does not happen, needs IL18(?)
        cytokine_production_mask = (
            self.nk_mask
            & ((self.T1IFN > 0) & (self.IL12 > 0) & (self.IL18 > 0))[
                tuple(self.nk_locations.T.astype(np.int64))
            ]
        )
        cytokine_production_locations = self.nk_locations[
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
        vecs += np.clip(
            np.random.normal(0.0, 1e-5, size=vecs.shape), -1.0, 1.0
        )  # small noise (randomizes direction in absence of gradient)
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
        self.nk_locations = np.mod(self.nk_locations, self.geometry)

        self._destack(mask=self.nk_mask, locations=self.nk_locations)

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
        # ACK: the code below is roughly what you would want if the model included the infection of macrophages.
        # However, virus-invade-cell has its macrophage code commented out.
        # mask = self.macro_mask
        # locations = self.macro_locations[mask].astype(np.int64)
        # extracellular_virus_at_locations = self.extracellular_virus[tuple(locations.T)]
        # cells_to_invade = 100 * np.random.rand(
        #     *extracellular_virus_at_locations.shape
        # ) < np.maximum(
        #     self.susceptibility_to_infection, extracellular_virus_at_locations
        # )
        # self.macro_internal_virus[mask][cells_to_invade] += 1
        # self.extracellular_virus[tuple(locations[cells_to_invade].T)] -= 1
        # self.macro_infected[mask][cells_to_invade] = True

        #   ; macro-activation-level keeps track of M1 (pro) or M2 (anti) status
        #   ; there is hysteresis because it modifies existing status
        # set macro-activation-level macro-activation-level + T1IFN + P/DAMPS + IFNg + IL1 - (2 * IL10) ;; currently a
        #                                                                                                  gap between
        #                                                                                                  pro and anti
        #                                                                                                  macros
        locations = self.macro_locations[self.macro_mask].astype(np.int64)
        self.macro_activation[self.macro_mask] += (
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
        vecs += np.clip(
            np.random.normal(0.0, 1e-5, size=vecs.shape), -1.0, 1.0
        )  # small noise (randomizes direction in absence of gradient)
        norms = np.linalg.norm(vecs, axis=-1)
        norms[norms <= 1e-8] = 1.0  # effectively zero norm vectors will be unnormalized
        self.macro_locations[self.macro_mask] += (
            0.1 * vecs / np.expand_dims(norms, axis=-1)
        )
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
        self.macro_locations = np.mod(self.macro_locations, self.geometry)

        self._destack(mask=self.macro_mask, locations=self.macro_locations)

        #     ;;PHAGOCYTOSIS LIMIT check
        #   ;; check for macro-phago-limit
        #   set macro-phago-counter max list 0 (cells-eaten + (virus-eaten / 10) - macro-phago-recovery)
        #   ;; macro-phago-recovery decrements counter to simulate internal processing, proxy for new macros
        #   ;; =2 => Exp 5
        #   ifelse macro-phago-counter >= macro-phago-limit ;; this will eventually be pyroptosis
        #     [set size 2]
        over_limit_mask = self.macro_mask & (
            self.macro_phago_counter >= self.macro_phago_limit
        )
        self.macro_swollen[over_limit_mask] = True

        #     [;; PHAGOCYTOSIS
        #     set size 1

        under_limit_mask = self.macro_mask & (
            self.macro_phago_counter < self.macro_phago_limit
        )
        self.macro_swollen[under_limit_mask] = False

        #     ;; of virus, uses local variable "q" to represent amount of virus eaten (avoid negative values)
        #     if extracellular-virus > 0
        #      [let q max list 0 (extracellular-virus - 10)
        #      ;; currently set as arbitrary 10 amount of extracellular virus eaten per step
        #       set extracellular-virus extracellular-virus - q
        #       set virus-eaten virus-eaten + q
        #      ]

        locations = self.macro_locations[under_limit_mask].astype(np.int64)
        virus_uptake = np.minimum(
            10.0, self.extracellular_virus[tuple(locations.T)]
        )  # TODO: pull-request submitted on disagreement between code and comment
        self.extracellular_virus[tuple(locations.T)] -= virus_uptake
        self.macro_virus_eaten[under_limit_mask] += virus_uptake

        #     ;; of apoptosed epis
        #     if count apoptosed-epis-here > 0
        #      [ask apoptosed-epis-here [die]
        #       set cells-eaten cells-eaten + 1
        #       set apoptosis-eaten-counter apoptosis-eaten-counter + 1
        #      ]

        macros_at_apoptosed_epi = under_limit_mask & (
            self.epithelium[tuple(self.macro_locations.T.astype(np.int64))]
            == EpiType.Apoptosed
        )
        self.macro_cells_eaten[macros_at_apoptosed_epi] += 1
        self.apoptosis_eaten_counter += np.sum(macros_at_apoptosed_epi)
        self.epithelium[
            tuple(self.macro_locations[macros_at_apoptosed_epi].T.astype(np.int64))
        ] = EpiType.Empty

        #    ;; of dead epis
        #     if count dead-epis-here > 0
        #      [ask dead-epis-here [die]
        #       set cells-eaten cells-eaten + 1
        #      ]
        #   ]

        dead_epis = under_limit_mask & (
            self.epithelium[tuple(self.macro_locations.T.astype(np.int64))]
            == EpiType.Dead
        )
        self.macro_cells_eaten[dead_epis] += 1
        self.epithelium[
            tuple(self.macro_locations[dead_epis].T.astype(np.int64))
        ] = EpiType.Empty

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
        self.IL12[tuple(locations.T)] += 0.5
        self.macro_activation[activated_macros] -= 5

        downstream_products_mask = (
            activated_macros
            & (self.IL1 + self.P_DAMPS > 0)[
                tuple(self.macro_locations.T.astype(np.int64))
            ]
        )
        downstream_locations = self.macro_locations[downstream_products_mask].astype(
            np.int64
        )
        self.TNF[tuple(downstream_locations.T)] += 1
        self.IL6[tuple(downstream_locations.T)] += 0.4
        self.IL10[tuple(downstream_locations.T)] += 1

        # if macro-activation-level < -5 ;; this keeps macros from self activating wrt IL10 in perpetuity
        #   [set color pink ;; tracker
        #   ;; Antiinflammatory cytokine production
        #   set IL10 IL10 + 0.5
        #
        #   set macro-activation-level macro-activation-level + 5 ;; this is to simulate macrophages returning to
        #                                                            baseline as intracellular factors lose stimulation
        #   ]

        # ACK: tracker isn't documented in the interface
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

        self.pyroptosis()

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

        # TODO: ACK should these be if-else?

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
        inflammasome_priming_mask = (
            self.macro_mask
            & ((self.P_DAMPS + self.TNF) > self.inflammasome_priming_threshold)[
                tuple(self.macro_locations.T.astype(np.int64))
            ]
        )
        self.macro_inflammasome_primed[inflammasome_priming_mask] = True

        # ;;  [set inflammasome "inactive"]
        #
        # end

    def pyroptosis(self):
        # ACK: called via:  if pyroptosis-counter > 12  ;; 120 minutes
        pyroptosis_mask = self.macro_mask & (self.macro_pyroptosis_counter > 12)

        # to pyroptosis
        # set pyroptosed-macros pyroptosed-macros + 1

        num_to_pyroptose = np.sum(pyroptosis_mask)
        self.pyroptosed_macros += num_to_pyroptose

        # set IL1 pre-IL1
        # set IL18 pre-IL18
        # set P/DAMPs P/DAMPs + 10
        locations = self.macro_locations[pyroptosis_mask].astype(np.int64)
        self.IL1[tuple(locations.T)] = self.macro_pre_il1[
            pyroptosis_mask
        ]  # ACK: why not +=?
        self.IL18[tuple(locations.T)] = self.macro_pre_il18[
            pyroptosis_mask
        ]  # ACK: why not +=?
        self.P_DAMPS[tuple(locations.T)] += 10

        # I'm doing death first, since this clears a little space. Just in case.
        self.macro_mask[pyroptosis_mask] = False
        self.num_macros -= np.sum(pyroptosis_mask)
        macro_creation_locations = np.array(
            np.where(
                (self.epithelium == EpiType.Healthy)
                | (self.epithelium == EpiType.Infected)
            )
        )
        for loc_idx in np.random.choice(
            macro_creation_locations.shape[1],
            num_to_pyroptose,
        ):
            self.create_macro(
                loc=macro_creation_locations[:, loc_idx],
                pre_il1=0,
                pre_il18=0,
                inflammasome_primed=False,
                inflammasome_active=False,
                macro_activation_level=0,
                pyroptosis_counter=0,
                virus_eaten=0,
                cells_eaten=0,
            )

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
        t1ifn_activated_locations = self.dc_locations[t1ifn_activated_mask].astype(
            np.int64
        )
        self.IL12[tuple(t1ifn_activated_locations.T)] += 0.5
        self.IFNg[tuple(t1ifn_activated_locations.T)] += 0.5

        il1_activated_mask = t1ifn_activated_mask & (
            self.IL1[tuple(self.dc_locations.T.astype(np.int64))] > 1
        )
        il1_activated_locations = self.dc_locations[il1_activated_mask].astype(np.int64)
        self.IL6[tuple(il1_activated_locations.T)] += 0.4

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
        vecs += np.clip(
            np.random.normal(0.0, 1e-5, size=vecs.shape), -1.0, 1.0
        )  # small noise (randomizes direction in absence of gradient)
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
        self.dc_locations = np.mod(self.dc_locations, self.geometry)

        self._destack(mask=self.dc_mask, locations=self.dc_locations)

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

        ros_damage_mask = (self.epithelium_ros_damage_counter > 2) & (
            self.epithelium == EpiType.Healthy
        )
        self.epithelium[ros_damage_mask] = EpiType.Dead
        self.P_DAMPS[ros_damage_mask] += 10

        # set ROS-damage-counter ROS-damage-counter + ROS
        self.epithelium_ros_damage_counter += self.ROS

        #   metabolism
        self.metabolism()

        #   virus-invade-cell
        self.virus_invade_epi_cell()

        #  if bat? = true
        #   [baseline-T1IFN-generation
        #   ]
        if self.is_bat:
            self.baseline_t1ifn_generation()

        # ACK: combined human and bat endothelial activations from netlogo code

        activation_level = (
            self.bat_endo_activation if self.is_bat else self.human_endo_activation
        )
        mask = (
            (self.epithelium == EpiType.Healthy)
            & (self.TNF + self.IL1 > activation_level)
            & np.logical_not(self.endothelial_activation == EndoType.Activated)
        )
        self.endothelial_activation[mask] = EndoType.Activated
        #
        #
        #       ;; simulates consumption of IL1 and TNF
        #    set TNF max list 0 (TNF - 0.1)
        #    set IL1 max list 0 (IL1 - 0.1)

        self.TNF[mask] -= np.minimum(0.1, self.TNF[mask])
        self.IL1[mask] -= np.minimum(0.1, self.IL1[mask])

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
        self.P_DAMPS[mask] += (
            self.bat_metabolic_byproduct
            if self.is_bat
            else self.human_metabolic_byproduct
        )

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
        pmn_spawn_mask = (
            (self.endothelial_adhesion_counter > 36)
            & (self.endothelial_activation == EndoType.Activated)
            & (np.random.rand(*self.geometry) < 0.1)
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

        self.time += 1
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
                self.nk_locations[self.nk_pointer, :] = np.array(
                    self.geometry
                ) * np.random.rand(2)
            else:
                self.nk_locations[self.nk_pointer, :] = loc

            self.nk_dirs[self.nk_pointer] = 2 * np.pi * np.random.rand()
            self.nk_age[self.nk_pointer] = 0
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
        self.nk_age[: self.num_nks] = self.nk_age[self.nk_mask]

        self.nk_mask[: self.num_nks] = True
        self.nk_mask[self.num_nks :] = False
        self.nk_pointer = self.num_nks

    def create_macro(
        self,
        *,
        loc=None,
        pre_il1,
        pre_il18,
        inflammasome_primed,
        inflammasome_active,
        macro_activation_level,
        pyroptosis_counter,
        virus_eaten,
        cells_eaten,
    ):
        if self.macro_pointer >= self.MAX_MACROPHAGES:
            self.compact_macro_arrays()
            # maybe the array is already compacted:
            if self.macro_pointer >= self.MAX_MACROPHAGES:
                raise RuntimeError(
                    "Max macrophages exceeded, you may want to change the MAX_MACROPHAGES parameter."
                )
        if loc is None:
            self.macro_locations[self.macro_pointer, :] = np.array(
                self.geometry
            ) * np.random.rand(2)
        else:
            self.macro_locations[self.macro_pointer, :] = loc

        self.macro_dirs[self.macro_pointer] = 2 * np.pi * np.random.rand()

        self.macro_internal_virus[self.macro_pointer] = 0
        # self.macro_infected[self.macro_pointer] = False
        self.macro_pre_il1[self.macro_pointer] = pre_il1
        self.macro_pre_il18[self.macro_pointer] = pre_il18
        self.macro_inflammasome_primed[self.macro_pointer] = inflammasome_primed
        self.macro_inflammasome_active[self.macro_pointer] = inflammasome_active
        self.macro_activation[self.macro_pointer] = macro_activation_level
        self.macro_pyroptosis_counter[self.macro_pointer] = pyroptosis_counter
        self.macro_virus_eaten[self.macro_pointer] = virus_eaten
        self.macro_cells_eaten[self.macro_pointer] = cells_eaten
        self.macro_swollen[self.macro_pointer] = False

        self.macro_mask[self.macro_pointer] = True
        self.num_macros += 1
        self.macro_pointer += 1

    def compact_macro_arrays(self):
        self.macro_locations[: self.num_macros] = self.macro_locations[self.macro_mask]
        self.macro_dirs[: self.num_macros] = self.macro_dirs[self.macro_mask]
        self.macro_internal_virus[: self.num_macros] = self.macro_internal_virus[
            self.macro_mask
        ]
        self.macro_activation[: self.num_macros] = self.macro_activation[
            self.macro_mask
        ]
        # self.macro_infected[: self.num_macros] = self.macro_infected[self.macro_mask]
        self.macro_cells_eaten[: self.num_macros] = self.macro_cells_eaten[
            self.macro_mask
        ]
        self.macro_virus_eaten[: self.num_macros] = self.macro_virus_eaten[
            self.macro_mask
        ]
        self.macro_pre_il1[: self.num_macros] = self.macro_pre_il1[self.macro_mask]
        self.macro_pre_il18[: self.num_macros] = self.macro_pre_il18[self.macro_mask]
        self.macro_pyroptosis_counter[
            : self.num_macros
        ] = self.macro_pyroptosis_counter[self.macro_mask]
        self.macro_inflammasome_primed[
            : self.num_macros
        ] = self.macro_inflammasome_primed[self.macro_mask]
        self.macro_inflammasome_active[
            : self.num_macros
        ] = self.macro_inflammasome_active[self.macro_mask]
        self.macro_swollen[: self.num_macros] = self.macro_swollen[self.macro_mask]

        self.macro_mask[: self.num_macros] = True
        self.macro_mask[self.num_macros :] = False
        self.macro_pointer = self.num_macros

    def create_dc(self, *, loc=None):
        # old signature:
        # loc="tissue", trafficking_counter):
        if self.dc_pointer >= self.MAX_DCS:
            self.compact_dc_arrays()
            # maybe the array is already compacted:
            if self.dc_pointer >= self.MAX_DCS:
                raise RuntimeError(
                    "Max DCs exceeded, you may want to change the MAX_DCS parameter."
                )
        if loc is None:
            self.dc_locations[self.dc_pointer, :] = np.array(
                self.geometry
            ) * np.random.rand(2)
        else:
            self.dc_locations[self.dc_pointer, :] = loc

        self.dc_dirs[self.dc_pointer] = 2 * np.pi * np.random.rand()
        self.dc_mask[self.dc_pointer] = True
        self.num_dcs += 1
        self.dc_pointer += 1

    def compact_dc_arrays(self):
        self.dc_locations[: self.num_dcs] = self.dc_locations[self.dc_mask]
        self.dc_dirs[: self.num_dcs] = self.dc_dirs[self.dc_mask]
        self.dc_mask[: self.num_dcs] = True
        self.dc_mask[self.num_dcs :] = False
        self.dc_pointer = self.num_dcs

    def create_pmn(
        self,
        *,
        location: Union[np.ndarray, List, Tuple],
        age: int,
        jump_dist: Union[int, float],
    ):
        if self.pmn_pointer >= self.MAX_PMNS:
            self.compact_pmn_arrays()
            # maybe the array is already compacted:
            if self.pmn_pointer >= self.MAX_PMNS:
                raise RuntimeError(
                    "Max PMNs exceeded, you may want to change the MAX_PMNS parameter."
                )
        if location is None:
            self.pmn_locations[self.pmn_pointer, :] = np.array(
                self.geometry
            ) * np.random.rand(2)
        else:
            self.pmn_locations[self.pmn_pointer, :] = np.array(location).astype(
                np.float64
            )

        theta = 2 * np.pi * np.random.rand()
        self.pmn_dirs[self.pmn_pointer] = theta
        self.pmn_locations[self.pmn_pointer] = np.mod(
            self.pmn_locations[self.pmn_pointer]
            + jump_dist * np.array([np.cos(theta), np.sin(theta)]),
            self.geometry,
        )

        self.pmn_age[self.pmn_pointer] = age

        self.pmn_mask[self.pmn_pointer] = True
        self.num_pmns += 1
        self.pmn_pointer += 1

    def compact_pmn_arrays(self):
        self.pmn_locations[: self.num_pmns] = self.pmn_locations[self.pmn_mask]
        self.pmn_dirs[: self.num_pmns] = self.pmn_dirs[self.pmn_mask]
        self.pmn_age[: self.num_pmns] = self.pmn_age[self.pmn_mask]

        self.pmn_mask[: self.num_pmns] = True
        self.pmn_mask[self.num_pmns :] = False
        self.pmn_pointer = self.num_pmns

    def _destack(self, mask: np.ndarray, locations: np.ndarray):
        """
        Make sure that no more than one agent lies on each patch.
        """
        location_used = np.zeros(self.geometry, dtype=bool)
        agent_indices = np.where(mask)[0]
        np.random.shuffle(agent_indices)
        for idx in agent_indices:
            while location_used[tuple(locations[idx, :].astype(int))]:
                # jump no more than a unit in an arbitrary direction
                perturbation = np.random.normal(0, 0.5, size=2)
                perturbation /= np.maximum(1.0, np.linalg.norm(perturbation))
                locations[idx, :] += perturbation
                locations[idx, :] = np.mod(locations[idx, :], self.geometry)
            location_used[tuple(locations[idx, :].astype(int))] = True

    def plot_agents(self, ax: plt.Axes, *, base_zorder: int = -1):
        ax.clear()
        # epithelium
        # * Blue Squares = Healthy Epithelial Cells
        # * Yellow Squares = Infected Epithelial Cells
        # * Grey Squares = Epithelial Cells killed by necrosis
        ax.imshow(
            self.epithelium.astype(int).T,
            cmap=epithelial_cm,
            vmin=0,
            vmax=4,
            zorder=base_zorder,
            origin="lower",
            extent=(
                0.0,
                self.epithelium.shape[0],
                0.0,
                self.epithelium.shape[1],
            ),
        )
        # * Grey Pentagons = Epithelial Cells killed by apoptosis
        ax.scatter(
            *np.array(np.where(self.epithelium == EpiType.Apoptosed))
            + np.array([[1 / 2], [1 / 2]]),
            color="grey",
            marker="p",
            zorder=base_zorder,
        )
        # macrophages
        # * Green Circles = Macrophages
        # * Large Green Circles = Macrophages at phagocytosis limit
        under_limit_mask = self.macro_phago_counter < self.macro_phago_limit
        ax.scatter(
            *self.macro_locations[self.macro_mask & under_limit_mask].T,
            color="green",
            marker="o",
            zorder=base_zorder + 1,
        )
        over_limit_mask = self.macro_phago_counter >= self.macro_phago_limit
        ax.scatter(
            *self.macro_locations[self.macro_mask & over_limit_mask].T,
            color="green",
            marker="o",
            s=(2 * plt.rcParams["lines.markersize"]) ** 2,
            zorder=base_zorder + 1,
        )
        # NKs
        # * Orange Circles = NK Cells
        ax.scatter(
            *self.nk_locations[self.nk_mask].T,
            color="orange",
            marker="o",
            zorder=base_zorder + 1,
        )
        # DCs
        # * Light Blue Triangles = Dendritic Cells
        ax.scatter(
            *self.dc_locations[self.dc_mask].T,
            color="lightblue",
            marker="v",
            zorder=base_zorder + 1,
        )
        # endos
        # * Pink Square Outlines = Activated Endothelial Cells
        ax.scatter(
            *np.where(self.endothelial_activation == EndoType.Activated),
            color="pink",
            marker=markers.MarkerStyle("s", fillstyle="none"),
            zorder=base_zorder + 1,
        )
        # PMNs
        # * Small White Circles = PMNs
        ax.scatter(
            *self.pmn_locations[self.pmn_mask].T,
            color="white",
            marker="o",
            edgecolors="black",
            zorder=base_zorder + 1,
        )

        ax.set_xlim(0, self.geometry[0])
        ax.set_ylim(0, self.geometry[1])

    def plot_field(self, ax: plt.Axes, *, field_name):
        ax.clear()
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
        field_array = getattr(self, field_name)

        ax.imshow(
            field_array.T,
            vmin=0,
            origin="lower",
            extent=(0.0, field_array.shape[0], 0.0, field_array.shape[1]),
        )

    def save(self, filename: str, *, write_mode: str = "a"):
        # compute which class attributes should be saved
        rep = {attr: getattr(self, attr) for attr in dir(self) if attr[0] != "_"}
        rep = {
            k: v
            for k, v in rep.items()
            if isinstance(v, int | float | bool | np.ndarray)
        }

        with h5py.File(filename, write_mode) as f:
            grp: h5py.Group = f.create_group(str(self.time))
            for k, v in rep.items():
                # skip things that can be automatically reconstructed
                if k in {
                    "num_macros",
                    "macro_pointer",
                    "macro_mask",
                    "num_pmns",
                    "pmn_pointer",
                    "pmn_mask",
                    "num_nks",
                    "nk_pointer",
                    "nk_mask",
                    "num_dcs",
                    "dc_pointer" "dc_mask",
                }:
                    continue

                if isinstance(v, int | float | bool):
                    grp.create_dataset(k, shape=(), dtype=type(v), data=v)
                else:
                    # numpy array
                    if np.issubdtype(v.dtype, np.object_):
                        v = v.astype(int)
                    if k.startswith("macro"):
                        v = v[self.macro_mask]
                    elif k.startswith("pmn"):
                        v = v[self.pmn_mask]
                    elif k.startswith("nk"):
                        v = v[self.nk_mask]
                    elif k.startswith("dc"):
                        v = v[self.dc_mask]
                    grp.create_dataset(k, shape=v.shape, dtype=v.dtype, data=v)

    @classmethod
    def load(cls, filename: str, time: int) -> "AnCockrellModel":
        with h5py.File(filename, "r+") as f:
            grp: h5py.Group = f[str(time)]

            model = cls(
                GRID_WIDTH=grp["GRID_WIDTH"][()],
                GRID_HEIGHT=grp["GRID_HEIGHT"][()],
                is_bat=grp["is_bat"][()],
                init_dcs=grp["init_dcs"][()],
                init_nks=grp["init_nks"][()],
                init_macros=grp["init_macros"][()],
                macro_phago_recovery=grp["macro_phago_recovery"][()],
                macro_phago_limit=grp["macro_phago_limit"][()],
                inflammasome_activation_threshold=grp[
                    "inflammasome_activation_threshold"
                ][()],
                inflammasome_priming_threshold=grp["inflammasome_priming_threshold"][
                    ()
                ],
                viral_carrying_capacity=grp["viral_carrying_capacity"][()],
                susceptibility_to_infection=grp["susceptibility_to_infection"][()],
                human_endo_activation=grp["human_endo_activation"][()],
                bat_endo_activation=grp["bat_endo_activation"][()],
                bat_metabolic_byproduct=grp["bat_metabolic_byproduct"][()],
                human_metabolic_byproduct=grp["human_metabolic_byproduct"][()],
                resistance_to_infection=grp["resistance_to_infection"][()],
                viral_incubation_threshold=grp["viral_incubation_threshold"][()],
                MAX_PMNS=grp["MAX_PMNS"][()],
                MAX_DCS=grp["MAX_DCS"][()],
                MAX_MACROPHAGES=grp["MAX_MACROPHAGES"][()],
                MAX_NKS=grp["MAX_NKS"][()],
            )

            # scalars not initialized by init
            model.time = grp["time"][()]

            num_macros = -1
            num_pmns = -1
            num_nks = -1
            num_dcs = -1
            for k in grp.keys():
                if len(grp[k]) == 0:
                    # skip the scalars, already dealt with
                    continue
                elif len(grp[k]) == 1:
                    # 1d arrays correspond to agent attributes.
                    # we learn the number of agents out of their dimensions.
                    model_field = getattr(model, k)
                    if k.startswith("macro"):
                        if num_macros == -1:
                            num_macros = grp[k].shape[0]
                        else:
                            assert (
                                num_macros == grp[k].shape[0]
                            ), "agent arrays have inconsistent sizes"
                        model_field[:num_macros] = grp[k]
                    elif k.startswith("pmn"):
                        if num_pmns == -1:
                            num_pmns = grp[k].shape[0]
                        else:
                            assert (
                                num_pmns == grp[k].shape[0]
                            ), "agent arrays have inconsistent sizes"
                        model_field[:num_pmns] = grp[k]
                    elif k.startswith("nk"):
                        if num_nks == -1:
                            num_nks = grp[k].shape[0]
                        else:
                            assert (
                                num_nks == grp[k].shape[0]
                            ), "agent arrays have inconsistent sizes"
                        model_field[:num_nks] = grp[k]
                    elif k.startswith("dc"):
                        if num_dcs == -1:
                            num_dcs = grp[k].shape[0]
                        else:
                            assert (
                                num_dcs == grp[k].shape[0]
                            ), "agent arrays have inconsistent sizes"
                        model_field[:num_dcs] = grp[k]

                elif len(grp[k]) == 2:
                    # 2d arrays are spatial fields
                    model_field = getattr(model, k)
                    model_field[:, :] = grp[k]
                else:
                    raise RuntimeError(f"Unknown/Invalid data {k} in HDF5 file")

            # set bookkeeping variables
            model.num_macros = num_macros
            model.macro_pointer = num_macros
            model.macro_mask[:num_macros] = True
            model.macro_mask[num_macros:] = False

            model.num_pmns = num_pmns
            model.pmn_pointer = num_pmns
            model.pmn_mask[:num_pmns] = True
            model.pmn_mask[num_pmns:] = False

            model.num_nks = num_nks
            model.nk_pointer = num_nks
            model.nk_mask[:num_nks] = True
            model.nk_mask[num_nks:] = False

            model.num_dcs = num_dcs
            model.dc_pointer = num_dcs
            model.dc_mask[:num_dcs] = True
            model.dc_mask[num_dcs:] = False

            return model
