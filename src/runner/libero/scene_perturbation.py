from __future__ import annotations

"""Scene-side (object-position) perturbations for Stage 3 grounding experiments.

Every perturbation studied so far (brightness, contrast, proprio noise) acted on
the *observation* after the simulator produced it. Stage 3 instead perturbs the
*scene*: we rewrite the object poses inside the LIBERO initial state before it is
handed to ``env.set_init_state``, so the robot genuinely faces a rearranged table.

Why this matters for LIBERO-Object specifically
-----------------------------------------------
Introspection of the suite (see ``docs/stage3_object_position.md``) shows that all
10 tasks lay their 6 manipulable objects on the *same six* (x, y) slots, with a
basket at (0, 0.26); only the object-to-slot assignment permutes. Position spread
across the 50 provided init states is <=2 cm and usually exactly 0. The target
object always occupies one of only two slots. A policy can therefore reach a high
success rate by memorising "go to one of two places", which is precisely the
position shortcut this module is built to expose.

Mechanics
---------
LIBERO init states are flattened MuJoCo states, laid out as::

    [ time (1) | qpos (nq) | qvel (nv) ]

so a free joint whose qpos address is ``a`` owns state indices
``1 + a .. 1 + a + 7`` = (x, y, z, qw, qx, qy, qz). We only ever rewrite ``x`` and
``y``. ``z`` is left untouched because it encodes the object's resting height,
which is geometry-specific (butter sits at -0.005, milk at 0.035); the quaternion
is left untouched so orientation is not silently confounded with position.

Modes
-----
``none``              no change (baseline; keeps the code path identical).
``move_target``       displace only the instructed object by ``distance`` metres
                      in a per-episode direction. Distractors stay home.
``move_distractors``  displace every distractor by ``distance``; target stays
                      home. The control for ``move_target``.
``swap``              exchange the (x, y) of the target and one distractor. Puts
                      a wrong object exactly where the target usually sits.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.runner.configs import ScenePerturbationConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

# A free joint owns 7 qpos values: 3 translation + 4 quaternion.
_FREE_JOINT_QPOS = 7
# Offset of qpos within the flattened state (index 0 is simulation time).
_QPOS_OFFSET = 1


@dataclass
class ObjectSlot:
    """Where one movable object lives in the flattened initial state."""

    name: str
    joint_name: str
    qpos_addr: int

    @property
    def xy_slice(self) -> slice:
        start = _QPOS_OFFSET + self.qpos_addr
        return slice(start, start + 2)

    @property
    def z_index(self) -> int:
        return _QPOS_OFFSET + self.qpos_addr + 2


@dataclass
class SceneLayout:
    """Resolved per-task scene structure (cached; identical for every episode)."""

    task_id: int
    target: ObjectSlot
    distractors: List[ObjectSlot]
    basket: Optional[ObjectSlot]

    @property
    def movable(self) -> List[ObjectSlot]:
        return [self.target, *self.distractors]


@dataclass
class ScenePerturber:
    """Rewrites object (x, y) in a LIBERO flattened initial state.

    Usage mirrors the observation builder: construct once from config, call
    :meth:`resolve_layout` when a task's env is created, then :meth:`apply` per
    episode. ``apply`` returns ``(new_state, info)`` where ``info`` records what
    was actually done so it can be logged alongside the episode result.
    """

    cfg: ScenePerturbationConfig
    _layouts: Dict[int, SceneLayout] = field(default_factory=dict, init=False)

    # ------------------------------------------------------------------ setup

    def resolve_layout(self, env, task_id: int, task_description: str) -> SceneLayout:
        """Identify target / distractors / basket for ``task_id`` from the live env.

        Object identity is taken from the language instruction rather than from
        positional order, so a suite whose ``objects[0]`` is not the target still
        works. LIBERO-Object happens to list the target first, which we use only
        as a cross-check.
        """
        if task_id in self._layouts:
            return self._layouts[task_id]

        sim = env.env.sim
        joint_addr = _free_joint_addresses(sim)

        try:
            object_names = [obj.name for obj in env.env.objects]
        except AttributeError as exc:  # pragma: no cover - depends on LIBERO version
            raise RuntimeError(
                'Cannot read env.env.objects; scene perturbation needs the robosuite '
                'object list to identify the target.'
            ) from exc

        slots: List[ObjectSlot] = []
        for name in object_names:
            joint_name = f'{name}_joint0'
            if joint_name not in joint_addr:
                # Fixed/static props (no free joint) cannot be moved; skip them.
                logger.debug('Object %s has no free joint; not movable.', name)
                continue
            slots.append(ObjectSlot(name=name, joint_name=joint_name,
                                    qpos_addr=joint_addr[joint_name]))

        basket = next((s for s in slots if 'basket' in s.name.lower()), None)
        candidates = [s for s in slots if s is not basket]

        target = _match_target(candidates, task_description)
        if target is None:
            raise RuntimeError(
                f'Could not identify the target object for task {task_id} '
                f'("{task_description}") among {[s.name for s in candidates]}.'
            )
        if candidates and target is not candidates[0]:
            logger.warning(
                'Task %s: language-matched target %s is not objects[0]=%s; '
                'trusting the language match.',
                task_id, target.name, candidates[0].name,
            )

        layout = SceneLayout(
            task_id=task_id,
            target=target,
            distractors=[s for s in candidates if s is not target],
            basket=basket,
        )
        self._layouts[task_id] = layout
        logger.info(
            'Task %s scene: target=%s distractors=%s basket=%s',
            task_id, target.name, [s.name for s in layout.distractors],
            basket.name if basket else None,
        )
        return layout

    # ------------------------------------------------------------------- apply

    def apply(self, initial_state, task_id: int, episode_idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        state = np.array(initial_state, dtype=np.float64, copy=True)
        mode = self.cfg.mode

        if mode == 'none':
            return state, {'mode': 'none'}

        layout = self._layouts.get(task_id)
        if layout is None:
            raise RuntimeError(
                f'resolve_layout() must be called for task {task_id} before apply().'
            )

        rng = self._episode_rng(task_id, episode_idx)
        home = {s.name: state[s.xy_slice].copy() for s in layout.movable}
        if layout.basket is not None:
            home[layout.basket.name] = state[layout.basket.xy_slice].copy()

        if mode == 'move_target':
            info = self._move_subset(state, layout, [layout.target], home, rng)
        elif mode == 'move_distractors':
            info = self._move_subset(state, layout, layout.distractors, home, rng)
        elif mode == 'swap':
            info = self._swap(state, layout, episode_idx, rng)
        else:
            raise ValueError(
                f'Unknown scene perturbation mode {mode!r}; expected one of '
                "'none', 'move_target', 'move_distractors', 'swap'."
            )

        info['mode'] = mode
        info['task_id'] = task_id
        info['episode_idx'] = episode_idx
        return state, info

    def snapshot(self, env, task_id: int) -> Dict[str, List[float]]:
        """Current (x, y, z) of every movable object, read from the live sim.

        Called at the end of an episode so a failure can be classified by what
        actually moved. In a swap episode the decisive question is whether the
        robot disturbed the *decoy* now sitting at the target's usual slot; that
        is answerable only by comparing start and end poses.
        """
        layout = self._layouts.get(task_id)
        if layout is None:
            return {}
        qpos = env.env.sim.data.qpos
        out: Dict[str, List[float]] = {}
        for slot in layout.movable:
            a = slot.qpos_addr
            out[slot.name] = [round(float(v), 4) for v in qpos[a: a + 3]]
        if layout.basket is not None:
            a = layout.basket.qpos_addr
            out[layout.basket.name] = [round(float(v), 4) for v in qpos[a: a + 3]]
        return out

    def start_positions(self, state, task_id: int) -> Dict[str, List[float]]:
        """(x, y, z) of every movable object in a flattened state."""
        layout = self._layouts.get(task_id)
        if layout is None:
            return {}
        state = np.asarray(state)
        out: Dict[str, List[float]] = {}
        slots = list(layout.movable) + ([layout.basket] if layout.basket else [])
        for slot in slots:
            i = _QPOS_OFFSET + slot.qpos_addr
            out[slot.name] = [round(float(v), 4) for v in state[i: i + 3]]
        return out

    def role_of(self, task_id: int, object_name: str) -> str:
        """'target' / 'distractor' / 'basket' for an object in this task."""
        layout = self._layouts.get(task_id)
        if layout is None:
            return 'unknown'
        if object_name == layout.target.name:
            return 'target'
        if layout.basket is not None and object_name == layout.basket.name:
            return 'basket'
        return 'distractor'

    # --------------------------------------------------------------- internals

    def _episode_rng(self, task_id: int, episode_idx: int) -> np.random.Generator:
        """Deterministic per-episode RNG.

        Seeded on (seed, task_id, episode_idx) and *not* on the perturbation
        magnitude, so episode k of every distance condition draws the same
        displacement direction. Only the magnitude changes between conditions,
        which is what makes a distance sweep a clean single-factor comparison.
        """
        return np.random.default_rng(
            [self.cfg.seed, task_id, episode_idx]
        )

    def _move_subset(
        self,
        state: np.ndarray,
        layout: SceneLayout,
        movers: Sequence[ObjectSlot],
        home: Dict[str, np.ndarray],
        rng: np.random.Generator,
    ) -> Dict[str, Any]:
        """Displace each object in ``movers`` by ``cfg.distance``.

        Occupancy is checked against the *current* placement of every other
        object, so movers are placed one at a time and never land on top of each
        other or of a stationary object.
        """
        distance = float(self.cfg.distance)
        moved: List[Dict[str, Any]] = []

        if distance == 0.0:
            # Explicit no-op keeps distance=0 bit-identical to the baseline.
            return {'distance': 0.0, 'moved': [], 'fallbacks': 0}

        fallbacks = 0
        for slot in movers:
            occupied = self._occupied_positions(state, layout, exclude=slot)
            origin = home[slot.name]
            new_xy, ok = self._sample_displacement(origin, distance, occupied, rng)
            if not ok:
                fallbacks += 1
                logger.warning(
                    'Task %s: no collision-free placement for %s at %.3f m; '
                    'using best-effort position %s.',
                    layout.task_id, slot.name, distance, np.round(new_xy, 4).tolist(),
                )
            state[slot.xy_slice] = new_xy
            moved.append({
                'object': slot.name,
                'home_xy': np.round(origin, 4).tolist(),
                'new_xy': np.round(new_xy, 4).tolist(),
                'realized_distance': round(float(np.linalg.norm(new_xy - origin)), 4),
                'collision_free': bool(ok),
            })

        return {'distance': distance, 'moved': moved, 'fallbacks': fallbacks}

    def _swap(
        self,
        state: np.ndarray,
        layout: SceneLayout,
        episode_idx: int,
        rng: np.random.Generator,
    ) -> Dict[str, Any]:
        """Exchange the (x, y) of the target and one distractor.

        The partner is chosen by cycling through the distractor list with
        ``episode_idx`` so that, over a run, every slot serves as the swap
        partner a roughly equal number of times. Set ``swap_partner`` to a
        non-negative integer to pin one specific distractor instead.
        """
        distractors = layout.distractors
        if not distractors:
            raise RuntimeError(f'Task {layout.task_id} has no distractor to swap with.')

        if self.cfg.swap_partner >= 0:
            idx = self.cfg.swap_partner % len(distractors)
        else:
            idx = episode_idx % len(distractors)
        partner = distractors[idx]

        target_xy = state[layout.target.xy_slice].copy()
        partner_xy = state[partner.xy_slice].copy()
        state[layout.target.xy_slice] = partner_xy
        state[partner.xy_slice] = target_xy

        return {
            'partner': partner.name,
            'partner_index': idx,
            'target_home_xy': np.round(target_xy, 4).tolist(),
            'target_new_xy': np.round(partner_xy, 4).tolist(),
            'separation': round(float(np.linalg.norm(partner_xy - target_xy)), 4),
        }

    def _occupied_positions(
        self, state: np.ndarray, layout: SceneLayout, exclude: ObjectSlot,
    ) -> List[Tuple[np.ndarray, float]]:
        """(xy, clearance_radius) for everything the mover must avoid."""
        out: List[Tuple[np.ndarray, float]] = []
        for slot in layout.movable:
            if slot is exclude:
                continue
            out.append((state[slot.xy_slice].copy(), self.cfg.min_separation))
        if layout.basket is not None:
            # The basket is far larger than a grocery item and is the goal
            # receptacle, so it gets its own (bigger) exclusion radius.
            out.append((state[layout.basket.xy_slice].copy(), self.cfg.basket_clearance))
        return out

    def _sample_displacement(
        self,
        origin: np.ndarray,
        distance: float,
        occupied: Sequence[Tuple[np.ndarray, float]],
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, bool]:
        """Find a point exactly ``distance`` from ``origin`` that is legal.

        Candidate directions are a fixed ring of angles visited in a random
        order, which keeps the displacement magnitude exact (unlike rejection
        sampling in a disc) while still decorrelating direction across episodes.
        Returns ``(xy, ok)``; when no candidate is legal, the least-bad one is
        returned with ``ok=False`` so the caller can count and log it.
        """
        n = max(int(self.cfg.direction_candidates), 1)
        angles = 2.0 * math.pi * np.arange(n) / n
        angles = angles + rng.uniform(0.0, 2.0 * math.pi / n)  # random ring phase
        order = rng.permutation(n)

        best_xy = origin.copy()
        best_score = -np.inf

        for i in order:
            theta = float(angles[i])
            xy = origin + distance * np.array([math.cos(theta), math.sin(theta)])
            if not self._within_bounds(xy):
                continue
            clearances = [
                float(np.linalg.norm(xy - other)) - radius for other, radius in occupied
            ]
            score = min(clearances) if clearances else np.inf
            if score >= 0.0:
                return xy, True
            if score > best_score:
                best_score, best_xy = score, xy

        if best_score == -np.inf:
            # Every candidate fell outside the table; stay home rather than
            # place the object in mid-air.
            return origin.copy(), False
        return best_xy, False

    def _within_bounds(self, xy: np.ndarray) -> bool:
        (x_lo, x_hi), (y_lo, y_hi) = self.cfg.x_bounds, self.cfg.y_bounds
        return bool(x_lo <= xy[0] <= x_hi and y_lo <= xy[1] <= y_hi)


def _free_joint_addresses(sim) -> Dict[str, int]:
    """joint name -> qpos address, for every free (6-DoF) joint in the model."""
    model = sim.model
    out: Dict[str, int] = {}
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) != 0:  # 0 == mjJNT_FREE
            continue
        out[model.joint_id2name(jid)] = int(model.jnt_qposadr[jid])
    return out


def _match_target(candidates: Sequence[ObjectSlot], task_description: str) -> Optional[ObjectSlot]:
    """Pick the object named in the instruction.

    Object names are like ``alphabet_soup_1``; instructions are like "pick up the
    alphabet soup and place it in the basket". We strip the trailing instance
    index, turn underscores into spaces, and keep the longest name that occurs in
    the instruction so that e.g. "tomato sauce" wins over a hypothetical "sauce".
    """
    text = task_description.lower()
    best: Optional[ObjectSlot] = None
    best_len = 0
    for slot in candidates:
        phrase = _readable_name(slot.name)
        if phrase in text and len(phrase) > best_len:
            best, best_len = slot, len(phrase)
    return best


def _readable_name(object_name: str) -> str:
    parts = object_name.split('_')
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return ' '.join(parts).lower()
