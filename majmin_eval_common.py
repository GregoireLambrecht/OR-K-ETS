"""Shared loading for the evaluation scripts of a finished major-minor run
(blind_major.py, policy_surfaces.py, tests/regression_majmin.py). Import AFTER setting the env variables
with `set_env_from_progress`."""
import os
import json


def read_progress(run_dir):
    with open(os.path.join(run_dir, 'progress.json')) as f:
        return json.load(f)


def set_env_from_progress(prog, floor_max):
    """Must run before JAX / the envs are imported (as plot_majmin_scenarios.py)."""
    os.environ['KETS_FLOOR_MODE'] = prog['floor_mode']
    os.environ['KETS_N_SIM'] = str(prog['n_sim'])
    os.environ['KETS_USE_H'] = '1' if prog.get('public_history_input', True) else '0'
    os.environ['KETS_FLOOR_MAX'] = str(floor_max)
    box = prog.get('gamma_box')            # runs trained on a non-default box
    if box:
        for nm, ev in (('init_alloc', 'ALLOC'), ('kappa', 'KAPPA'), ('Afloor', 'FLOOR'), ('mm_share', 'MM'),
                       ('c', 'C'), ('s', 'S')):
            if nm not in box:
                continue
            os.environ[f'KETS_{ev}_MIN'] = str(box[nm][0])
            if nm != 'Afloor':
                os.environ[f'KETS_{ev}_MAX'] = str(box[nm][1])
    st = prog.get('setting')               # LP-benchmark switches (envs/base_params.py)
    os.environ['KETS_LAYOUT'] = (st or {}).get('layout', 'v4')
    if 'A_CEIL' in prog:
        os.environ['KETS_A_CEIL'] = str(prog['A_CEIL'])
    if st:
        os.environ['KETS_DETERMINISTIC'] = '1' if st['deterministic'] else '0'
        os.environ['KETS_PENALTY'] = st['penalty']
        os.environ['KETS_NO_MAJOR'] = '1' if st['no_major'] else '0'
        os.environ['KETS_CAP_F_MULT'] = str(st['cap_f_mult'])
        os.environ['KETS_CTX_REF'] = ' '.join(str(v) for v in st['ctx_ref'])
    if 'floor_max' in prog:
        assert abs(float(prog['floor_max']) - float(floor_max)) < 1e-9, \
            f"--floor-max {floor_max} != training floor_max {prog['floor_max']}"


def make_env(T):
    from envs.base_params import (generate_prices_ou, A0_BASE, P0, market_impact_base,
                                  supply_noise_base, idiosyncratic_noise_base,
                                  A_SCALE_BASE, P_SCALE_BASE)
    from envs.environment_majmin import MajorMinorEnv
    from utils import CTX_REF
    return MajorMinorEnv(kappa=CTX_REF[1], T=T, generate_P_func=generate_prices_ou, A0=A0_BASE,
                         P0=P0, market_impact_func=market_impact_base,
                         generate_eps0_func=supply_noise_base,
                         generate_eps_idiosyncratic_func=idiosyncratic_noise_base,
                         A_scale=A_SCALE_BASE, P_scale=P_SCALE_BASE,
                         agent_per_policy=20).set_T(T)


def load_mixture(run_dir, prog):
    """Final mixture, all saved slots 0..K (k = K + 1 components), in K_MAX buffers."""
    from envs.environment import N_TYPES
    from envs.environment_majmin import MINORS, MAJOR
    from utils_majmin import load_net, make_buffers
    K = len(prog['records'])
    nets = [[load_net(run_dir, i, j) for j in range(N_TYPES)] for i in range(K + 1)]
    bufs, stats = make_buffers(nets)
    return dict(K=K, k=K + 1, nets=nets,
                minor_buf=tuple(bufs[j] for j in MINORS),
                minor_static=tuple(stats[j] for j in MINORS),
                maj_buf=bufs[MAJOR], maj_static=stats[MAJOR])
