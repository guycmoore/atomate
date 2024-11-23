# coding: utf-8


import os
import itertools
import numpy as np
import numpy.linalg as npla

from atomate.vasp.fireworks.core import StaticFW
from fireworks import Workflow, Firework
from atomate.vasp.powerups import (
    add_additional_fields_to_taskdocs,
    add_wf_metadata,
    add_common_powerups,
)
from atomate.vasp.workflows.base.core import get_wf
from atomate.vasp.workflows.base.ncl_groundstate import (
    NoncollinearConstrainFW, 
    pull_mag_constrs_from_oszicar,
)

from atomate.vasp.firetasks.run_calc import RunVaspCustodian
from atomate.common.firetasks.glue_tasks import PassCalcLocs
from atomate.vasp.firetasks.glue_tasks import CopyVaspOutputs
from atomate.vasp.firetasks.parse_outputs import VaspToDb
from atomate.vasp.firetasks.write_inputs import WriteVaspFromIOSet

from atomate.utils.utils import get_logger

logger = get_logger(__name__)

from atomate.vasp.config import VASP_CMD, NCL_VASP_CMD, NCL_SF_VASP_CMD, DB_FILE, ADD_WF_METADATA

from uuid import uuid4

from pymatgen.io.vasp.sets import MPStaticSet
from pymatgen.io.vasp.inputs import Poscar, Incar
from pymatgen.core import Structure, PeriodicSite

__author__ = "Guy Moore"
__maintainer__ = "Guy Moore"
__email__ = "gmoore@lbl.gov"
__status__ = "Production"
__date__ = "March 2021"

__spin_displace_wf_version__ = 0.0

module_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

def get_wf_spin_displace_run(
        structure_orig,
        num_config,
        user_incar_settings,
        is_noncollinear=False,
        sigma_disp_scale=0.01,
        c=None):
    """_summary_

    Args:
        structure_orig (_type_): _description_
        num_config (_type_): _description_
        user_incar_settings (_type_): _description_
        is_noncollinear (bool, optional): _description_. Defaults to False.
        sigma_disp_scale (float, optional): _description_. Defaults to 0.01.
        c (_type_, optional): _description_. Defaults to None.
    """
    
    displaced_structures = get_displaced_structures(
        structure_orig,
        num_config,
        is_noncollinear,
        sigma_disp_scale,
    )
    
    fws = []
    
    for i,structure in enumerate([structure_orig] + displaced_structures):
        
        poscar = Poscar(structure)
        
        fw_name_tag = ""
        if i == 0:
            fw_name_tag = "_orig"
        
        ## Nonmagnetic runs
        
        structure_nonmag = Structure(species=structure.species, coords=structure.frac_coords, lattice=structure.lattice)

        uis_nmag = user_incar_settings.copy()
        uis_nmag.update({
            "ISPIN": 1, 
            # "LWAVE": True,
        })
        
        vis_params = {"user_incar_settings": uis_nmag.copy()}
        vis_nmag = MPStaticSet(structure=structure_nonmag.copy(), **vis_params.copy())
        
        fws.append(StaticFW(
                structure=structure_nonmag.copy(), parents=None,
                name="nonmagnetic_static"+fw_name_tag, vasp_input_set=vis_nmag,
                vasp_cmd=VASP_CMD, db_file=DB_FILE))
        
        
        ## Magnetic runs
        
        #parents = fws[-1]
        #additional_files = ['WAVECAR', 'CHGCAR']

        parents = []
        additional_files = []
        
#         if is_noncollinear:
            
#             magmoms = [list(s.properties['magmom']) for s in structure]

#             uis_mag = user_incar_settings.copy()
#             uis_mag.update({
#                 #"LWAVE": True,
#                 "ISPIN": 2, 
#                 "LSORBIT": True, "LNONCOLLINEAR": True,
#                 "I_CONSTRAINED_M": 1, "LAMBDA": 10.0,
#                 "RWIGS": [1.0 for k in poscar.site_symbols],
#                 "M_CONSTR":magmoms.copy(),
#             })

#             vis_params = {"user_incar_settings": uis_mag.copy()}
#             vis_mag = MPStaticSet(structure=structure.copy(), **vis_params.copy())

#             # vasp_cmd = NCL_VASP_CMD
#             vasp_cmd = NCL_SF_VASP_CMD
            
#             fws.append(NoncollinearConstrainFW(
#                 structure=structure.copy(), parents=parents.copy(),
#                 additional_files=additional_files.copy(),
#                 name="constrain_magnetic_static"+fw_name_tag, vasp_input_set=vis_mag,
#                 vasp_cmd=vasp_cmd, db_file=DB_FILE))
            
#         else:
            
#             magmoms = [s.properties['magmom'] for s in structure]
            
#             uis_mag = user_incar_settings.copy()
#             uis_mag.update({
#                 #"ISTART": 1,
#                 "ISPIN": 2, 
#             })
            
#             vis_params = {"user_incar_settings": uis_mag.copy()}
#             vis_mag = MPStaticSet(structure=structure.copy(), **vis_params.copy())
            
#             vasp_cmd = VASP_CMD
            
#             fws.append(StaticFW(
#                     structure=structure.copy(), parents=parents.copy(),
#                     # additional_files=additional_files.copy(),
#                     name="constrain_magnetic_static"+fw_name_tag, vasp_input_set=vis_mag,
#                     vasp_cmd=vasp_cmd, db_file=DB_FILE))
            
    # using a uuid for book-keeping,
    # in a similar way to other workflows
    uuid = str(uuid4())
    
    # #HACK
    # uuid = ""
    
    c_defaults = {
        # "vasp_cmd": VASP_CMD, 
        "db_file": DB_FILE
    }
    if c:
        c.update(c_defaults)
    else:
        c = c_defaults
    
    wf = Workflow(fws)
    wf = add_common_powerups(wf, c)
    
    if c.get("ADD_WF_METADATA", ADD_WF_METADATA):
        wf = add_wf_metadata(wf, structure)
    
    wf = add_additional_fields_to_taskdocs(
        wf,
        {
            "wf_meta": {
                "wf_uuid": uuid,
                "wf_name": "spin_displace_run",
                "wf_version": __spin_displace_wf_version__,
            }
        },
    )
    
    return wf

def get_displaced_structures(
    structure_orig,
    num_config,
    is_noncollinear,
    sigma_disp_scale,
):
    """_summary_

    Args:
        structure_orig (_type_): _description_
        num_config (_type_): _description_
        is_noncollinear (bool): _description_
        sigma_disp_scale (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    structs_disp = []

    sigma_disp = sigma_disp_scale * (npla.det(structure_orig.lattice.matrix) / len(structure_orig))**(1/3)
    print("sigma_disp = %f angstroms"%(sigma_disp))

    for i in range(num_config):

        sites = []
        magmoms = []
        for si,site in enumerate(structure_orig):

            coords = site.coords + rand_disp_gauss(sigma_disp)

            # ########
            # # HACK: Perturb one atom
            # idx_perturb = 0
            # alpha_sigma = sigma_disp
            # alphas_perturb = np.linspace(-alpha_sigma, alpha_sigma, num_config)
            # if si == idx_perturb:
            #     coords = site.coords + alphas_perturb[i] * np.array([0.0,0.0,1.0])
            # else:
            #     coords = site.coords.copy()
            # ########

            s = PeriodicSite(coords=coords.copy(), coords_are_cartesian=True, 
                             species=site.species, lattice=site.lattice)
            sites.append(s)

            smag = npla.norm(site.properties['magmom'])

            if is_noncollinear:
                magmom = rand_spin_unif(smag)
                magmoms.append(list(magmom).copy())
            else:
                magmom = rand_spin_unif_collinear(smag)
                magmoms.append(magmom)

            # print(s.as_dict())
            # print(site.properties['magmom'], magmom)
            # print(site.coords, coords)

        st = Structure.from_sites(sites)
        st.add_site_property('magmom', magmoms.copy())

        structs_disp.append(st)

        # CifWriter(st, write_magmoms=True).write_file('struct_test_'+str(i)+'.mcif')
        # print(st)
    
    return structs_disp

def curate_force_dicts(docs, is_magnetic, is_noncollinear):
    """_summary_

    Args:
        docs (_type_): _description_
        is_magnetic (bool): _description_
        is_noncollinear (bool): _description_
    """

    force_dicts = []

    for doc in docs:
        
        input_dict = doc['input']
        output_dict = doc['output']
        
        struct_in = Structure.from_dict(doc['input']['structure'])
        struct_out = Structure.from_dict(doc['output']['structure'])
        
        incar = Incar.from_dict(doc['input']['incar'])
        
        launch_dir = doc["calcs_reversed"][-1]["dir_name"]
        
        energy = doc['output']['energy']
        forces = doc['output']['forces']
        forces = [-np.array(f) for f in forces]
        
#         #outcar = (output_dict['outcar'])
#         cmd_out = !gunzip {launch_dir+'OUTCAR.*'}
#         outcar = Outcar(launch_dir+'OUTCAR')
        
#         cmd_out = !gunzip {launch_dir+'vasprun.*'}
#         vasprun = Vasprun(filename=launch_dir+'vasprun.xml', 
#                             parse_dos=False, parse_eigen=False,
#                             parse_projected_eigen=False, 
#                             parse_potcar_file=False)
        
#         energy = float(vasprun.as_dict()['output']['final_energy'])
#         forces = vasprun.get_trajectory().as_dict()['site_properties'][-1]['forces']
        
        if is_magnetic:
            
            magmoms_out = [s.properties['magmom'] for s in struct_out]
            
            # magnetization_out = outcar.magnetization
            # if is_noncollinear:
            #     magmoms_out = [np.array(list(m['tot'])) for m in magnetization_out]
            # else:
            #     magmoms_out = [m['tot'] for m in magnetization_out]
            
            if is_noncollinear:
                
                lam = incar['LAMBDA']
                
                mw_int, m_int, h_eff, energy_penalty = pull_mag_constrs_from_oszicar(dirname=launch_dir)
                
                magmoms_out, mw_int, m_int, h_eff = \
                    np.array(magmoms_out), np.array(mw_int), np.array(m_int), np.array(h_eff)
                                
                m_grads = [npla.norm(m)*mg for m,mg in zip(magmoms_out, h_eff)]
                
                # cmd_out = !gunzip {launch_dir+'OSZICAR.*'}
                # cmd_out = !grep -nri "E_p" {launch_dir+'OSZICAR'}
                # Ep = float(cmd_out[-1].split('lambda')[0].split('=')[-1])
                
                # m_constr = incar['M_CONSTR']
                # mtol = 1.0e-6
                # m_constr = [np.array(m) / npla.norm(m) if npla.norm(m) > mtol else 0.0*np.array(m) 
                #             for m in m_constr]
                
                # mh_grads = [lam*(mc*np.dot(mc,m) - m) for m,mc in zip(magmoms_out, m_constr)]
                # m_grads = [2.0*lam*npla.norm(m)*(mc*np.dot(mc,m) - m) for m,mc in zip(magmoms_out, m_constr)]
                # print("M0 NORMs = ", [npla.norm(mc) for m,mc in zip(magmoms_out, m_constr)])
                # print("M GRADS (NON NORM) = ", mh_grads)
                
                # # NOTE: Scale by two?
                # m_grads = len(magmoms_out)*[np.zeros([3])]
                # start_read = False
                # with open(launch_dir+'OSZICAR', 'r') as f:
                #     for l in f:
                #         if start_read:
                #             a = l.split()
                #             m_grads[-1+int(a[0])] = np.array([float(m) for m in a[1:]])
                #         if "lambda*MW_perp" in l:
                #             start_read = True
                # # normalize for unit spins
                # m_grads = [npla.norm(m)*mg for m,mg in zip(magmoms_out, m_grads)]
                
        s = Structure.from_dict(doc['input']['structure'])
        if is_magnetic:
            if is_noncollinear:
                s.add_site_property('magmom', [m.tolist() for m in magmoms_out])
            else:
                s.add_site_property('magmom', [m for m in magmoms_out])
        
        force_dict = {
            'structure': s.as_dict(), 
            'forces': [a.tolist() for a in forces],
            'energy': energy,
        }
        
        if is_noncollinear:
            force_dict['mag_fields'] = [a.tolist() for a in m_grads]
            force_dict['energy_penalty'] = energy_penalty
        
        force_dicts.append(force_dict)
        
        # #print('Energy penalty =', Ep)
        # #print(m_grads)
        # print(magmoms_out)
        # #print(m_constr)
        # print(forces)
        # print(launch_dir)
        # print()
        
    return force_dicts

def curate_spins_and_disps(struct_ref, force_dicts, is_magnetic=True, is_noncollinear=True, magmom_tol=0.1):
    """_summary_

    Args:
        struct_ref (_type_): _description_
        force_dicts (_type_): _description_
        is_magnetic (bool, optional): _description_. Defaults to True.
        is_noncollinear (bool, optional): _description_. Defaults to True.
        magmom_tol (float, optional): _description_. Defaults to 0.1.
    """

    states_magn, states_disp = [], []
    energies, forces = [], []
    hmags = [] 

    for d in force_dicts:

        struct_d = Structure.from_dict(d['structure'])

        if is_magnetic:
            # spin configs
            if is_noncollinear:
                states_magn.append([np.array(list(s.properties['magmom'])) for s in struct_d])
            else:
                states_magn.append([np.array([0.0,0.0,s.properties['magmom']]) for s in struct_d])
            # normalize spins 
            states_magn[-1] = [(m/npla.norm(m) if npla.norm(m)>magmom_tol else 0.0*m) for m in states_magn[-1]]

        # displacements - in frac coordinates
        states_disp.append([
            np.array(s.frac_coords)-np.array(s0.frac_coords) for s,s0 in zip(struct_d, struct_ref)])
        # handle aliasing
        states_disp[-1] = [
            np.array([[d-1.0,d,d+1.0][np.argmin(np.abs([d-1.0,d,d+1.0]))] for d in s]) for s in states_disp[-1]]
        # convert to cartesian
        states_disp[-1] = [np.dot(struct_d.lattice.matrix.T, s) for s in states_disp[-1]]

#         ####################################
#         # Test

#         struct_ag = Structure(
#             lattice=struct_ref.lattice,
#             coords=struct_ref.frac_coords.tolist()+struct_d.frac_coords.tolist(),
#             species=struct_ref.species+struct_d.species,
#         )

#         dists_0 = np.array([npla.norm(uv) for uv in states_disp[-1]])
#         dists_1 = np.array([struct_ag.get_distance(i,i+len(struct_ref)) for i in range(len(struct_ref))])
#         print("diff norm = ", np.max(np.abs(dists_1 - dists_0)))

#         ####################################

        forces.append([np.array(f) for f in d['forces']])
        # # forces - convert to fractional coords
        # forces.append([np.dot(d['structure'].lattice.matrix, f) for f in d['forces']])

        if is_noncollinear:
            hmags.append([np.array(f) for f in d['mag_fields']])

        energy = d['energy']
        if is_noncollinear:
            energy -= d['energy_penalty']
        energies.append(energy)

    return (states_magn, states_disp, forces, hmags, energies)


# Helper functions

def rand_spin_unif(r):
    """_summary_

    Args:
        r (_type_): _description_

    Returns:
        _type_: _description_
    """

    pos = np.zeros([3])
    while npla.norm(pos) == 0.0:
        pos = np.random.normal(0.0,1.0,[3])
        pos *= r/npla.norm(pos)

    return pos.tolist()

def rand_spin_unif_collinear(r):
    """_summary_

    Args:
        r (_type_): _description_

    Returns:
        _type_: _description_
    """
    m = r * (-1.0 + 2.0*np.random.randint(0,2))
    return m

def rand_disp_gauss(sigma):
    """_summary_

    Args:
        sigma (_type_): _description_

    Returns:
        _type_: _description_
    """
    pos = np.random.normal(0.0,sigma,[3])
    # pos = np.random.uniform(-sigma,sigma,[3])
    return pos

