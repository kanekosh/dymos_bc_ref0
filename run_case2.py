import matplotlib.pyplot as plt

import openmdao.api as om

import dymos as dm
from dymos.examples.plotting import plot_results

from min_time_climb_ode import MinTimeClimbODE

"""
This script is the same as the supersonic minimum-time climb example, except
- I switched the optimizer to SNOPT
- I added ref0=10 in add_state('h')
- I replaced the final boundary constraint of h to be 19990 <= h <= 20000. (originally, h=20000)

Optimization converged to the same solution as the original script. The final altitude is h=20000
However, pyOptSparse output says that it is hitting the *lower* bound of the final boundary constraint.
   
Constraints (i - inequality, e - equality)
  Index  Name                                                               Type          Lower           Value           Upper    Status  Lagrange Multiplier (N/A)
    175  traj.phases.phase0.final_boundary_constraints.final_value:h           i   1.999000E+01    1.999000E+01    2.000000E+01         l    9.00000E+100
"""

if __name__ == '__main__':
    #
    # Instantiate the problem and configure the optimization driver
    #
    p = om.Problem(model=om.Group())

    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = 'SNOPT'
    p.driver.declare_coloring()

    #
    # Instantiate the trajectory and phase
    #
    traj = dm.Trajectory()

    phase = dm.Phase(ode_class=MinTimeClimbODE,
                    transcription=dm.GaussLobatto(num_segments=15, compressed=False))

    traj.add_phase('phase0', phase)

    p.model.add_subsystem('traj', traj)

    #
    # Set the options on the optimization variables
    # Note the use of explicit state units here since much of the ODE uses imperial units
    # and we prefer to solve this problem using metric units.
    #
    phase.set_time_options(fix_initial=True, duration_bounds=(50, 400),
                        duration_ref=100.0)

    phase.add_state('r', fix_initial=True, lower=0, upper=1.0E6, units='m',
                    ref=1.0E3, defect_ref=1.0E3,
                    rate_source='flight_dynamics.r_dot')

    phase.add_state('h', fix_initial=True, lower=0, upper=20000.0, units='m',
                    ref0=10, ref=1.0E2, defect_ref=1.0E2,
                    rate_source='flight_dynamics.h_dot')

    phase.add_state('v', fix_initial=True, lower=10.0, units='m/s',
                    ref=1.0E2, defect_ref=1.0E2,
                    rate_source='flight_dynamics.v_dot')

    phase.add_state('gam', fix_initial=True, lower=-1.5, upper=1.5, units='rad',
                    ref=1.0, defect_ref=1.0,
                    rate_source='flight_dynamics.gam_dot')

    phase.add_state('m', fix_initial=True, lower=10.0, upper=1.0E5, units='kg',
                    ref=1.0E3, defect_ref=1.0E3,
                    rate_source='prop.m_dot')

    phase.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                    rate_continuity=True, rate_continuity_scaler=100.0,
                    rate2_continuity=False)

    phase.add_parameter('S', val=49.2386, units='m**2', opt=False, targets=['S'])
    phase.add_parameter('Isp', val=1600.0, units='s', opt=False, targets=['Isp'])
    phase.add_parameter('throttle', val=1.0, opt=False, targets=['throttle'])

    #
    # Setup the boundary and path constraints
    #
    # NOTE: modified boundary constraint here
    ### phase.add_boundary_constraint('h', loc='final', equals=20000, scaler=1.0E-3)
    phase.add_boundary_constraint('h', loc='final', lower=20000 - 10, upper=20000, scaler=1.0E-3)
    phase.add_boundary_constraint('aero.mach', loc='final', equals=1.0)
    phase.add_boundary_constraint('gam', loc='final', equals=0.0)

    phase.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
    phase.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8)

    # Minimize time at the end of the phase
    phase.add_objective('time', loc='final', ref=1.0)

    p.model.linear_solver = om.DirectSolver()

    #
    # Setup the problem and set the initial guess
    #
    p.setup(check=True)

    p['traj.phase0.t_initial'] = 0.0
    p['traj.phase0.t_duration'] = 500

    p.set_val('traj.phase0.states:r', phase.interp('r', [0.0, 50000.0]))
    p.set_val('traj.phase0.states:h', phase.interp('h', [100.0, 20000.0]))
    p.set_val('traj.phase0.states:v', phase.interp('v', [135.964, 283.159]))
    p.set_val('traj.phase0.states:gam', phase.interp('gam', [0.0, 0.0]))
    p.set_val('traj.phase0.states:m', phase.interp('m', [19030.468, 10000.]))
    p.set_val('traj.phase0.controls:alpha', phase.interp('alpha', [0.0, 0.0]))

    #
    # Solve for the optimal trajectory
    #
    dm.run_problem(p, simulate=True)

    sol = om.CaseReader('dymos_solution.db').get_case('final')
    sim = om.CaseReader('dymos_simulation.db').get_case('final')

    print('final altitude:', p.get_val('traj.phase0.timeseries.states:h', units='m')[-1])

    plot_results([('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:h',
                'time (s)', 'altitude (m)'),
                ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.controls:alpha',
                'time (s)', 'alpha (deg)')],
                title='Supersonic Minimum Time-to-Climb Solution',
                p_sol=sol, p_sim=sim)

    plt.show()
