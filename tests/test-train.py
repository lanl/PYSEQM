# ---------------- #
# Imported Modules #
# ---------------- #

import os
import sys
import numpy as np
import torch
from hippynn.interfaces.pyseqm_interface.seqm_nodes import *
#from extensions import SEQM_All_EXT1_Node
from hippynn.interfaces.pyseqm_interface.callback import update_scf_eps, save_and_stop_after
import hippynn.interfaces.pyseqm_interface
#np.set_printoptions(threshold=np.inf)

import seqm
from seqm.basics import parameterlist
seqm.seqm_functions.scf_loop.debug = True
hippynn.interfaces.pyseqm_interface.check.debug = True
seqm.seqm_functions.scf_loop.MAX_ITER = 50

#torch.cuda.set_device(0) # Don't try this if you want CPU training!

import matplotlib
matplotlib.use('agg')

import hippynn

hippynn.custom_kernels.set_custom_kernels (False)

parameter_file_dir = "/path/to/PYSEQM/seqm/params/"
dataset_name = 'data-'    # Prefix for arrays in folder
dataset_path = "/path/to/dataset/" 

netname = 'TEST' #sys.argv[1]
dirname = netname
if not os.path.exists(dirname):
    os.mkdir(dirname)
else:
    pass
    #raise ValueError("Directory {} already exists!".format(dirname))
os.chdir(dirname)

TAG = 0 #int(sys.argv[2])  # False (0): first run, True(n): continue

dtype=torch.float64
torch.set_default_dtype(dtype)
if torch.cuda.is_available():
    device = torch.device('cuda')
    DEVICE = 'cuda'
else:
    device = torch.device('cpu')
    DEVICE = 'cpu'

try:
    DEVICE = list(range(int(sys.argv[3])))
except:
    pass

#"""
learned = ['U_ss', 'U_pp', 'zeta_s', 'zeta_p',
       #'beta_s', 
       'beta_p',
       #'g_ss', 
       'g_sp', 'g_pp', 'g_p2', 'h_sp',
       #'alpha',
       #'Gaussian1_K', 'Gaussian2_K', #'Gaussian3_K','Gaussian4_K',
       #'Gaussian1_L', 'Gaussian2_L', #'Gaussian3_L','Gaussian4_L',
       #'Gaussian1_M', 'Gaussian2_M', #'Gaussian3_M','Gaussian4_M'
      ]
#"""

seqm_parameters = {
                   'method' : "PM3",  # AM1, MNDO, PM#
                   'scf_eps' : 27.2114e-5,  # unit eV, change of electric energy, as nuclear energy doesnt' change during SCF
                   'scf_converger' : [1,0.0], # converger used for scf loop
                                         # [0, 0.1], [0, alpha] constant mixing, P = alpha*P + (1.0-alpha)*Pnew
                                         # [1], adaptive mixing
                                         # [2], adaptive mixing, then pulay
                   'sp2' : [False, 1.0e-5],  # whether to use sp2 algorithm in scf loop,
                                            #[True, eps] or [False], eps for SP2 conve criteria
                   'elements' : [0,1,6,7,8],
                   'learned' : learned ,#parameterlist[method], #['U_ss'], # learned parameters name list, e.g ['U_ss']
                   'parameter_file_dir' : parameter_file_dir + "/", # file directory for other required parameters
                   'pair_outer_cutoff' : 1.0e10, # consistent with the unit on coordinates
                   'scf_backward' : 2,
                   }

# Log the output of python to `training_log.txt`
with hippynn.tools.log_terminal("training_log_tag_%d.txt" % TAG,'wt'):# and torch.autograd.set_detect_anomaly(True):

    # Hyperparameters for the network

    network_params = {
        "possible_species": [0,1,6,7,8],   # Z values of the elements
        'n_features': 80,                     # Number of neurons at each layer
        "n_sensitivities": 20,                # Number of sensitivity functions in an interaction layer
        "dist_soft_min": 0.65,  # qm7 1.7  qm9 .85  AL100 .85
        "dist_soft_max": 4.,  # qm7 10.  qm9 5.   AL100 5.
        "dist_hard_max": 6.0,  # qm7 15.  qm9 7.5  AL100 7.5
        "n_interaction_layers": 2,            # Number of interaction blocks
        "n_atom_layers": 3,                   # Number of atom layers in an interaction block
    }


    # Define a model

    from hippynn.graphs import inputs, networks, targets, physics

    species = inputs.SpeciesNode(db_name="Z_long")

    positions = inputs.PositionsNode(db_name="R")

    network = networks.Hipnn("HIPNN_seqm", (species, positions), module_kwargs = network_params)

    n_target_peratom = len(seqm_parameters["learned"])

    decay_factor = 1.0e-4
    par_atom = HChargeNode("SEQM_Atom_Params",network,module_kwargs=dict(n_target=n_target_peratom,first_is_interacting=True))
    with torch.no_grad():
        for layer in par_atom.torch_module.layers:
            layer.weight.data *= decay_factor
            layer.bias.data *= decay_factor

    seqm_par = par_atom.atom_charges

    #lenergy = SEQM_AllNode("SEQM_Energy",network,seqm_parameters, decay_factor = 1.0e-4)
    lenergy = SEQM_AllNode("SEQM_Energy",(par_atom, positions, species),seqm_parameters, decay_factor = 1.0e-4)

    molecule_energy = lenergy.Etot_m_Eiso

    gradient  = physics.GradientNode("gradients", (molecule_energy, positions), sign=+1)

    notconverged = lenergy.notconverged
    scale = ScaleNode("Scale", (notconverged,))

    gradient.db_name='Gradient_ev'
    molecule_energy.db_name="EtEi"

    #hierarchicality = henergy.hierarchicality

    mol_mask = SEQM_MolMaskNode("SEQM_MolMask", notconverged)
    atom_mask = AtomMaskNode("Atom_Mask", species)
    gradient_pred = SEQM_MaskOnMolAtomNode("SEQM_MaskMolAtom_Pred", (gradient, mol_mask, atom_mask)).pred
    gradient_true = SEQM_MaskOnMolAtomNode("SEQM_MaskMolAtom_True", (gradient.true, mol_mask.pred, atom_mask.pred))

    molecule_energy_pred = SEQM_MaskOnMolNode("SEQM_MaskMol_Pred", (molecule_energy, mol_mask)).pred
    molecule_energy_true = SEQM_MaskOnMolNode("SEQM_MaskMol_True", (molecule_energy.true, mol_mask.pred))


    # define loss quantities
    from hippynn.graphs import loss

    rmse_gradient = loss.MSELoss(gradient_pred, gradient_true) ** (1./2.)
    rmse_mol_energy = loss.MSELoss(molecule_energy_pred, molecule_energy_true) ** (1. / 2.)

    rmse_energy = rmse_gradient + rmse_mol_energy

    mae_gradient = loss.MAELoss(gradient_pred, gradient_true)
    mae_mol_energy = loss.MAELoss(molecule_energy_pred, molecule_energy_true)

    mae_energy = mae_gradient + mae_mol_energy
    rsq_gradient = loss.Rsq(gradient_pred, gradient_true)
    rsq_mol_energy = loss.Rsq(molecule_energy_pred, molecule_energy_true)

    ### SLIGHTLY MORE ADVANCED USAGE

    #pred_per_atom = physics.PerAtom("PeratomPredicted",(molecule_energy,species)).pred
    #true_per_atom = physics.PerAtom("PeratomTrue",(molecule_energy.true,species.true))

    pred_per_atom1 = physics.PerAtom("PeratomPredicted",(molecule_energy,species))
    true_per_atom1 = physics.PerAtom("PeratomTrue",(molecule_energy.true,species.true))
    pred_per_atom = SEQM_MaskOnMolNode("SEQM_PerAtom_Pred", (pred_per_atom1, mol_mask)).pred
    true_per_atom = SEQM_MaskOnMolNode("SEQM_PerAtom_True", (true_per_atom1.pred, mol_mask.pred))
    mae_per_atom = loss.MAELoss(pred_per_atom,true_per_atom)

    rmse_par = loss.MeanSq(seqm_par.pred)

    ### END SLIGHTLY MORE ADVANCED USAGE

    loss_error = (rmse_energy + mae_energy) + rmse_par*10.0

    #rbar = loss.Mean.of_node(hierarchicality)
    l2_reg = loss.l2reg(network)
    loss_regularization = 1e-6 * loss.Mean(l2_reg) #+ rbar    # L2 regularization and hierarchicality regularization

    train_loss = loss_error*scale.pred + loss_regularization

    # Validation losses are what we check on the data between epochs -- we can only train to
    # a single loss, but we can check other metrics too to better understand how the model is training.
    # There will also be plots of these things over time when training completes.
    validation_losses = {
        #"T-RMSE"      : rmse_energy,
        #"T-MAE"       : mae_energy,
        #"T-RSQ"       : rsq_energy,
        "TperAtom MAE": mae_per_atom,
        "Force-RMSE"  : rmse_gradient,
        "Force-MAE"   : mae_gradient,
        "Force-RSQ"   : rsq_gradient,
        "MolEn-RMSE"  : rmse_mol_energy,
        "MolEn-MAE"   : mae_mol_energy,
        "MolEn-RSQ"   : rsq_mol_energy,
        #"T-Hier"      : rbar,
        "L2Reg"       : l2_reg,
        "Loss-Err"    : loss_error,
        "Loss-Reg"    : loss_regularization,
        "Loss"        : train_loss,
    }
    early_stopping_key = "Loss-Err"

    from hippynn import plotting

    plot_maker = plotting.PlotMaker(
        # Simple plots which compare the network to the database

        #plotting.Hist2D.compare(molecule_energy, saved=True),
        plotting.Hist2D(molecule_energy_true, molecule_energy_pred,
                        xlabel="True EtEi",ylabel="Predicted EtEi",
                        saved="EtEi.pdf"),
        plotting.Hist2D(gradient_true, gradient_pred,
                        xlabel="True Force",ylabel="Predicted Force",
                        saved="grad.pdf"),

        #Slightly more advanced control of plotting!
        plotting.Hist2D(true_per_atom,pred_per_atom,
                        xlabel="True Energy/Atom",ylabel="Predicted Energy/Atom",
                        saved="PerAtomEn.pdf"),

        #plotting.HierarchicalityPlot(hierarchicality.pred,
        #                             molecule_energy.pred - molecule_energy.true,
        #                             saved="HierPlot.pdf"),
        plot_every=1,   # How often to make plots -- here, epoch 0, 10, 20...
    )

    if TAG==0:
        from hippynn.experiment.assembly import assemble_for_training

        training_modules, db_info = \
            assemble_for_training(train_loss,validation_losses,plot_maker=plot_maker)
        training_modules[0].print_structure()

# ----------------- #
# Step 3: RUN MODEL #
# ----------------- #

        database_params = {
            'name': dataset_name,                            # Prefix for arrays in folder
            'directory': dataset_path,
            'quiet': False,                           # Quiet==True: suppress info about loading database
            'seed': 8000,                       # Random seed for data splitting
            #'test_size': 0.1,                # Fraction of data used for testing
            #'valid_size':0.1,
            **db_info                 # Adds the inputs and targets names from the model as things to load
        }

        from hippynn.databases import DirectoryDatabase
        database = DirectoryDatabase(**database_params)
        database.make_random_split("ignore",0.9)
        del database.splits['ignore']
        database.make_trainvalidtest_split(test_size=0.1,valid_size=.1)

        #from hippynn.pretraining import set_e0_values
        #set_e0_values(henergy,database,energy_name="T_transpose",trainable_after=False)

        init_lr = 1e-4
        optimizer = torch.optim.Adam(training_modules.model.parameters(),lr=init_lr)

        from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau,PatienceController

        scheduler =  RaiseBatchSizeOnPlateau(optimizer=optimizer,
                                            max_batch_size=256,
                                            patience=10)

        controller = PatienceController(optimizer=optimizer,
                                        scheduler=scheduler,
                                        batch_size=256,
                                        eval_batch_size=256,
                                        max_epochs=10000,
                                        termination_patience=2000,
                                        fraction_train_eval=0.1,
                                        stopping_key=early_stopping_key,
                                        )

        scheduler.set_controller(controller)

        experiment_params = hippynn.experiment.SetupParams(
            controller = controller,
            device=DEVICE,
        )
        print(experiment_params)

        # Parameters describing the training procedure.
        from hippynn.experiment import setup_training

        training_modules, controller, metric_tracker  = setup_training(training_modules=training_modules,
                                                        setup_params=experiment_params)
    if TAG>0:
        from hippynn.experiment.serialization import load_checkpoint_from_cwd, load_checkpoint
        from hippynn.experiment import train_model
        #load best model
        structure = load_checkpoint_from_cwd()
        #load last model
        #structure = load_checkpoint("experiment_structure.pt", "last_checkpoint.pt")
        training_modules = structure["training_modules"]
        database = structure["database"]
        database.make_random_split("ignore",0.9)
        del database.splits['ignore']
        database.make_trainvalidtest_split(test_size=0.1,valid_size=.1)
        controller = structure["controller"]
        metric_tracker = structure["metric_tracker"]

    from hippynn.experiment import train_model
    store_all_better=True
    store_best=True
    if isinstance(training_modules[0], torch.nn.DataParallel):
        seqm_module = training_modules[0].module.node_from_name('SEQM_Energy').torch_module
    else:
        seqm_module = training_modules[0].node_from_name('SEQM_Energy').torch_module
    callbacks = [update_scf_eps(seqm_module, 0.9),
                    save_and_stop_after(training_modules, controller, metric_tracker, store_all_better, store_best, [2,0,0,0])]

    train_model(training_modules=training_modules,
                database=database,
                controller=controller,
                metric_tracker=metric_tracker,
                callbacks=callbacks,batch_callbacks=None,
                store_all_better=store_all_better,
                store_best=store_best)
