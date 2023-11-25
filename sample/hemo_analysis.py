import hemtracer as ht
import numpy as np
from matplotlib import pyplot as plt
from typing import List, Dict
from os.path import join

def get_seeds():
    """Define seeds for pathlines."""
    x0 = np.asarray([0, 5.0, 0])
    # n = 7
    n = 9
    # r_arr = np.linspace(0.01,0.2,n)
    r_arr = [0.01] 
    # phi_arr = [np.pi*0.5]
    phi_arr = np.linspace(0, 2*np.pi, n, endpoint=False)

    x0_list = []
    for r_i in r_arr:
        for phi_i in phi_arr:
            x0_list.append(x0 + np.asarray([r_i*np.sin(phi_i), 0, r_i*np.cos(phi_i)]))
    
    return x0_list

def plot_geffs(hemolysis_solver: ht.HemolysisSolver, models: List[ht.rbc_model.RBCModel], path: str) -> None:
    """Plot Geff for all models in one plot."""
    
    i = 0

    sols = [hemolysis_solver.get_representativeShear(model) for model in models]

    for sol_models in zip(*sols):

        plt.figure()
        for sol, model in zip(sol_models, models):
            t = sol['t']
            g = sol['y']
            plt.plot(t, g, label=model.get_name())
        
        plt.legend()
        plt.xlabel("t")
        plt.ylabel("Geff")

        i += 1
        plt.savefig(join(path, "pathline_" + str(i) + "_comp.png"))
        plt.close()

def plot_geff_lagr_eul(hemolysis_solver: ht.HemolysisSolver, model: ht.rbc_model.RBCModel, pathline_tracker: ht.PathlineTracker, path: str) -> None:
    """Plot Geff for Lagrangian and Eulerian data in one plot."""

    i = 0
    geff_lagr_pls = hemolysis_solver.get_representativeShear(model)
    geff_eul_pls = pathline_tracker.get_attribute("Geff")


    # Plot pathlines.
    for (geff_lagr_pl, geff_eul_pl) in zip(geff_lagr_pls, geff_eul_pls):

        plt.figure()

        t_lagr = geff_lagr_pl['t']
        g_lagr = geff_lagr_pl['y']
        plt.plot(t_lagr, g_lagr, label="Lagrangian")

        t_eul = geff_eul_pl['t']
        g_eul = geff_eul_pl['y']

        plt.plot(t_eul, g_eul, label="Eulerian")

        plt.legend()
        plt.xlabel("t")
        plt.ylabel("Geff")

        i += 1
        plt.savefig(join(path, "pathline_" + str(i) + "_" + model.get_name() + ".png"))
        plt.close()

def main() -> None:
    # Create flow field object.
    flow_field = ht.EulerianFlowField("flow.vtk")
    flow_field.set_velocity_name("velocity")  # define name of (vector) velocity field, default: "velocity"

    # Load Eulerian data from another file to compare with Lagrangian data.
    flow_field.import_field_data("geff.vtk", field_names_point=["Geff"], field_names_cell=[])

    # Specify which fields are important for the computation.
    flow_field.reduce_to_relevant_fields(["velocity", "Geff", "ref_frame"])

    # Transform data in moving reference frame.
    flow_field.transform_mrf_data(rf_name='ref_frame', rf_rot=2, omega=np.asarray([0, 104.719755, 0]), x0=np.asarray([0,0,0]))

    # Compute pathlines.
    pathline_tracker = ht.PathlineTracker(flow_field)
    pathline_tracker.compute_pathlines(get_seeds(), max_length=100,max_err=1e-5,max_step=0.1,min_step=0.000001,initial_step=0.1,n_steps=10000)

    dt = 0.002          # sampling rate for gradients
    pathline_tracker.interpolate_dv_to_pathlines(sampling_rate=dt) # interpolate velocity gradients to pathlines

    # # Define blood damage models to use.
    model_simp = ht.rbc_model.strain_based.AroraSimplified()
    model_simp.configure_ODE_solver(max_step=dt)
    

    model_tt = ht.rbc_model.strain_based.TankTreading()
    model_tt_corr = ht.rbc_model.strain_based.TankTreadingRotationCorrection()
    model_full = ht.rbc_model.strain_based.AroraFullEig()
    model_full.configure_ODE_solver(max_step=dt)

    model_stress = ht.rbc_model.stress_based.Bludszuweit()
    model_stress.set_sampling_rate(dt)

    correlation_song = ht.hemolysis_model.IHCorrelation.SONG

    hemolysis_model = ht.hemolysis_model.PowerLawModel(hemolysis_correlation = correlation_song, 
                                       mu = 0.0032, integration_scheme='timeDiff')

    models = [model_simp, model_tt, model_tt_corr, model_stress]

    hemolysis_solver = ht.HemolysisSolver(pathline_tracker)
    for model in models:
        hemolysis_solver.compute_representativeShear(model)
        hemolysis_solver.compute_hemolysis(cell_model=model, powerlaw_model=hemolysis_model)
    
    plot_geffs(hemolysis_solver, models, path="../out")


if __name__ == '__main__':
    main()