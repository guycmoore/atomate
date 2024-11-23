import os
import itertools

from uuid import uuid4
from datetime import datetime

from fireworks import Workflow, Firework, FiretaskBase, FWAction, explicit_serialize

from atomate.vasp.fireworks.core import StaticFW
from atomate.vasp.powerups import (
    # add_tags,
    add_additional_fields_to_taskdocs,
    add_wf_metadata,
    add_common_powerups,
)
from atomate.vasp.workflows.base.core import get_wf

from atomate.vasp.firetasks.run_calc import RunVaspCustodian, RunVaspDirect
from atomate.common.firetasks.glue_tasks import PassCalcLocs
from atomate.vasp.firetasks.glue_tasks import CopyVaspOutputs
from atomate.vasp.firetasks.parse_outputs import VaspToDb
from atomate.vasp.firetasks.write_inputs import WriteVaspFromIOSet
from atomate.vasp.database import VaspCalcDb

from atomate.utils.utils import get_logger, env_chk

logger = get_logger(__name__)

from atomate.vasp.config import VASP_CMD, NCL_VASP_CMD, NCL_SF_VASP_CMD, DB_FILE, ADD_WF_METADATA

from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPStaticSet
from pymatgen.io.vasp.inputs import Incar, Poscar
from pymatgen.io.vasp.outputs import Oszicar, Outcar

from pymatgen.analysis.magnetism.spin_pso import *

import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla
import random

__author__ = "Guy Moore"
__maintainer__ = "Guy Moore"
__email__ = "gmoore@lbl.gov"
__status__ = "Production"
__date__ = "March 2021"

__spin_pso_wf_version__ = 0.0

module_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))


# Check: numpy random number seeding
class SpinPSO_WF:
    """_summary_
    """
    def __init__(
        self,
        structure,
        magmom_magnitudes,
        num_agent=1,
        static=True,
        magmoms_in=None,
        theta_sigma=0.0,
    ):
        """_summary_

        Args:
            structure (_type_): _description_
            magmom_magnitudes (_type_): _description_
            num_agent (int, optional): _description_. Defaults to 1.
            static (bool, optional): _description_. Defaults to True.
            magmoms_in (_type_, optional): _description_. Defaults to None.
            theta_sigma (float, optional): _description_. Defaults to 0.0.

        Raises:
            ValueError: _description_
        """

        self.uuid = str(uuid4())
        self.wf_meta = {
            "wf_uuid": self.uuid,
            "wf_name": self.__class__.__name__,
            "wf_version": __spin_pso_wf_version__,
        }
        self.static = static

        if len(structure) != len(magmom_magnitudes):
            raise ValueError(
                "Number of sites must match the length of input moment magnitudes."
            )

        self.mag_site_map = []
        self.spin_magnitudes = []
        self.magmoms_ref = []
        for i,m in enumerate(magmom_magnitudes):
            if isinstance(m, float):
                self.mag_site_map.append(i)
                self.spin_magnitudes.append(m)
                if magmoms_in:
                    self.magmoms_ref.append(magmoms_in[i].copy())
        self.num_spin = len(self.spin_magnitudes)

        self.structure_nonmag = Structure(
            species=structure.species, lattice=structure.lattice, coords=structure.frac_coords,
        )

        if magmoms_in:
            positions_init = [get_random_rotated_spins(
                self.num_spin, self.magmoms_ref, theta_sigma=theta_sigma) 
                for i in range(num_agent)]
        else:
            positions_init = [get_random_spins(
                self.num_spin, spin_magnitudes=self.spin_magnitudes) 
                for i in range(num_agent)]
        vels_init = [[0.0*x for x in a] for a in positions_init]

        pos_dicts = [{"type":"spin", "name":"magmom", "index":i} for i in range(self.num_spin)]

        self.swarm = Swarm(num_agent, positions_init, pos_dicts, vels_init, nspindim=3)

    def get_wf_optimize(
        self,
        user_incar_settings=None,
        user_kpoints_settings=None,
        max_iter_limit=10,
        energy_convergence_tol=1.0e-4,
        dt=1.0, mass=1.0,
        vasp_cmd=NCL_VASP_CMD,
        c=None,
    ):
        """_summary_

        Args:
            user_incar_settings (_type_, optional): _description_. Defaults to None.
            max_iter_limit (int, optional): _description_. Defaults to 10.
            energy_convergence_tol (_type_, optional): _description_. Defaults to 1.0e-4.
            dt (float, optional): _description_. Defaults to 1.0.
            mass (float, optional): _description_. Defaults to 1.0.
            vasp_cmd (_type_, optional): _description_. Defaults to NCL_VASP_CMD.
            c (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        poscar = Poscar(self.structure_nonmag)

        user_incar_settings.update({
            "VOSKOWN": 1,
            "LSORBIT": True, "LORBMOM": True, "LNONCOLLINEAR": True,
            "I_CONSTRAINED_M": 1
        })

        if not user_incar_settings.get("LAMBDA", None):
            user_incar_settings["LAMBDA"] = 10.0

        # FIXME: RWIGS default (other than 1 AA)
        if not user_incar_settings.get("RWIGS", None):
            user_incar_settings["RWIGS"] = [1.0 for k in poscar.site_symbols]

        c_defaults = {"vasp_cmd": vasp_cmd, "db_file": DB_FILE}
        if c:
            c.update(c_defaults)
        else:
            c = c_defaults

        spin_pso_fw = Firework(
            SpinPSOiterTask(
                spin_pso_wf_uuid=self.uuid,
                db_file=DB_FILE,
                vasp_cmd=vasp_cmd,
                c=c,
                structure_nonmag=self.structure_nonmag,
                mag_site_map=self.mag_site_map,
                swarm=self.swarm.as_dict(),
                n_iter=0,
                max_iter_limit=max_iter_limit,
                energy_convergence_tol=energy_convergence_tol,
                user_incar_settings=user_incar_settings,
                user_kpoints_settings=user_kpoints_settings,
            ),
            parents=[],
            name="SpinPSOinit",
        )
        fws = [spin_pso_fw]

        wf = Workflow(fws)

        wf = add_common_powerups(wf, c)

        if c.get("ADD_WF_METADATA", ADD_WF_METADATA):
            wf = add_wf_metadata(wf, self.structure_nonmag)

        wf = add_additional_fields_to_taskdocs(
            wf,
            {
                "wf_meta": {
                    "wf_uuid": self.uuid,
                    "wf_name": "spin_pso",
                    "wf_version": __spin_pso_wf_version__,
                }
            },
        )

        return wf

    def position_from_inputs(self, input_dict, grad_dict):
        """_summary_

        Args:
            input_dict (_type_): _description_
            grad_dict (_type_): _description_

        Returns:
            _type_: _description_
        """

        pos_dicts, positions, gradients = [], [], []

        # magmoms
        if input_dict.get("magmoms", None) and grad_dict.get("magmoms", None):
            for i, (magmom, grad) in enumerate(zip(input_dict["magmoms"], grad_dict["magmoms"])):
                positions.append(list(magmom))
                gradients.append(list(grad))
                pos_dicts.append({
                    "type":"spin",
                    "name":"magmom",
                    "index":i,
                })

        # saxis
        if input_dict.get("saxis", None) and grad_dict.get("saxis", None):
            positions.append(list(input_dict["saxis"]))
            gradients.append(list(grad_dict["saxis"]))
            pos_dicts.append({
                "type":"saxis",
                "name":"saxis",
                "index":0,
            })

        # qspiral
        if input_dict.get("qspiral", None) and grad_dict.get("qspiral", None):
            positions.append(list(input_dict["qspiral"]))
            gradients.append(list(grad_dict["qspiral"]))
            pos_dicts.append({
                "type":"qspiral",
                "name":"qspiral",
                "index":0,
            })

        return pos_dicts, positions, gradients

    def inputs_from_position(self, pos_dicts, positions, gradients, num_magmoms):
        """_summary_

        Args:
            pos_dicts (_type_): _description_
            positions (_type_): _description_
            gradients (_type_): _description_
            num_magmoms (_type_): _description_

        Returns:
            _type_: _description_
        """

        magmoms = [[] for i in range(num_magmoms)]
        saxis, qspiral = [], []

        for pos_dict, pos, grad in zip(pos_dicts, positions, gradients):
            if pos_dict["name"] == "magmom":
                magmoms[pos_dict["index"]] = pos.copy()
            elif pos_dict["name"] == "saxis":
                saxis = pos.copy()
            elif pos_dict["name"] == "qspiral":
                qspiral = pos.copy()

        input_dict = {"magmoms":magmoms, "saxis":saxis, "qspiral":qspiral}

        return input_dict

@explicit_serialize
class SpinPSOiterTask(FiretaskBase):
    """_summary_

    Args:
        FiretaskBase (_type_): _description_
    """
    required_params = [
        "spin_pso_wf_uuid",
        "db_file",
        "vasp_cmd",
        "c",
        "structure_nonmag",
        "mag_site_map",
        "swarm",
        "n_iter",
        "max_iter_limit",
        "energy_convergence_tol",
        "user_incar_settings",
        "user_kpoints_settings",
    ]
    optional_params = []

    def run_task(self, fw_spec):
        """_summary_

        Args:
            fw_spec (_type_): _description_
        """

        # FIXME: add check comparison between EDIFF and energy_convergence_tol

#         # HACK
#         self["user_incar_settings"].update({"EDIFF": 1.0e-6})

        self.structure_nonmag = self["structure_nonmag"]
        self.mag_site_map = self["mag_site_map"]
        self.num_spin = len(self.mag_site_map)
        self.swarm = Swarm.from_dict(self["swarm"])
        self.n_iter = self["n_iter"]
        self.max_iter_limit = self["max_iter_limit"]
        self.energy_convergence_tol = self["energy_convergence_tol"]

        self.user_incar_settings = self["user_incar_settings"]
        self.user_kpoints_settings = self["user_kpoints_settings"]

        vasp_cmd = self["vasp_cmd"]
        wf_uuid = self["spin_pso_wf_uuid"]
        db_file = env_chk(self.get("db_file"), fw_spec)
        to_db = self.get("to_db", True)
        mmdb = VaspCalcDb.from_db_file(db_file, admin=True)

        docs = list(mmdb.collection.find({"wf_meta.wf_uuid": wf_uuid}))
        #docs = list(mmdb.collection.query({"wf_uuid": wf_uuid}))

        # convergence handling
        algo_has_converged = False
        fitness_deviation = float('inf')

        self.structs_out = [self.structure_nonmag.as_dict() for i in range(self.swarm.num_agent)]
        fits_new = [float('inf') for i in range(self.swarm.num_agent)]
        energy_penalties = [float('inf') for i in range(self.swarm.num_agent)]
        energy_order_penalties = [float('inf') for i in range(self.swarm.num_agent)]
        grads_new = [[np.zeros([self.swarm.nspindim]) for j in range(self.num_spin)] for i in range(self.swarm.num_agent)]
        m_int_new = [[np.zeros([self.swarm.nspindim]) for j in range(self.num_spin)] for i in range(self.swarm.num_agent)]
        mw_int_new = [[np.zeros([self.swarm.nspindim]) for j in range(self.num_spin)] for i in range(self.swarm.num_agent)]
        m_constrs = [[np.zeros([self.swarm.nspindim]) for j in range(self.num_spin)] for i in range(self.swarm.num_agent)]

        if self.n_iter < self.max_iter_limit and not(algo_has_converged):
            is_final_unconstrained=False
        else:
            is_final_unconstrained=True

        def calculate_ordering_penalty(
            magmoms_in, penalty_type, 
            alpha_penalty=100.0,
            # alpha_penalty=2.0,
        ):
            
            moment_net = np.zeros([3])
            spins_norm = []
            for m in magmoms_in:
                # print("m =", m)
                s = np.array(m)
                if npla.norm(m) > 1.0e-12:
                    s /= npla.norm(m)
                spins_norm.append(s.copy())
                moment_net += s
            moment_net /= len(spins_norm)
            
            ordering_penalty = 0.0
            if (penalty_type == "fm"):
                ordering_penalty = (1.0 - npla.norm(moment_net))**2
            elif (penalty_type == "afm"):
                ordering_penalty =  npla.norm(moment_net)**2
            
            ordering_penalty *= alpha_penalty * len(magmoms_in)
            
            # print("ordering_penalty = ", ordering_penalty)
            
            return ordering_penalty
        
        if self.n_iter > 0:

            for d in docs:
                if d.get("spin_pso", {}).get(
                    "agent_id", None) in range(self.swarm.num_agent):

                    if d.get("spin_pso", {}).get("n_iter", None) == self.n_iter:

                        agent_id = d["spin_pso"]["agent_id"]
                        energy = d["output"]["energy"]
                        # energy = d["calcs_reversed"][-1]["output"]["energy"]
                        struct_in = d["spin_pso"]["input_structure"]
                        struct_out = d["output"]["structure"]
                        # struct_out = d["calcs_reversed"][-1]["output"]["structure"]

                        for i in range(self.num_spin):
                            m_constrs[agent_id][i][0:] = d['input']['incar']['M_CONSTR'][self.mag_site_map[i]][0:]

                        # parse OSZICAR - local moment constraint output
                        mw_int, m_int, h_eff, energy_penalty = pull_mag_constrs_from_oszicar(dirname=d["calcs_reversed"][-1]["dir_name"])

                        for i in range(self.num_spin):
                            mi = self.mag_site_map[i]
                            m_int_new[agent_id][i][0:] = np.array(m_int[mi])[0:]
                            mw_int_new[agent_id][i][0:] = np.array(mw_int[mi])[0:]

                        for i in range(self.num_spin):
                            # Note: mind sign here
                            grads_new[agent_id][i][0:] = -np.array(h_eff[i])[0:]

                        ordering_penalty = calculate_ordering_penalty(m_int_new[agent_id], penalty_type="afm")

                        energy_penalties[agent_id] = energy_penalty
                        energy_order_penalties[agent_id] = ordering_penalty

                        # FIXME: subtract-off energy penalty?
                        self.structs_out[agent_id] = struct_out.copy()
                        # fits_new[agent_id] = energy
                        fits_new[agent_id] = energy - energy_penalty
                        # fits_new[agent_id] = energy - energy_penalty + ordering_penalty

            # print("grads_new", grads_new)
            # print("fits_new", fits_new)
            # print("swarm best fitness (old):", self.swarm.fit_best)

            # check for convergence
            fitness_deviation = np.abs(np.max(fits_new) - np.min(fits_new))
            if fitness_deviation < self.energy_convergence_tol:
                algo_has_converged = True

            # update swarm
            dt, mass = 1.0, 1.0

            pos_new=m_int_new.copy()
            # pos_new=m_constrs.copy()

            self.swarm.update_fitnesses_gcpso(fits_new=fits_new, pos_new=pos_new)
            self.swarm.compute_velocities_gcpso(
                grads_new, gamma=0.2, lam=0.5, dt=dt, mass=mass
            )
            self.swarm.update_positions(dt=dt)

            # print("swarm best fitness (new):", self.swarm.fit_best)

        # obtain new agent states
        self.pso_states = self.swarm.get_positions()

        # list of additional Fireworks (FWs)
        fws = []

        # add evaluation FWs
        swarm_fws = self.get_agent_step_fws(is_final_unconstrained)
        fws.extend(swarm_fws)

        if not is_final_unconstrained:
            # add analysis FW
            fw_iter = Firework(
                SpinPSOiterTask(
                    spin_pso_wf_uuid=self["spin_pso_wf_uuid"],
                    db_file=self["db_file"],
                    vasp_cmd=self["vasp_cmd"],
                    c=self["c"],
                    structure_nonmag=self["structure_nonmag"],
                    mag_site_map=self["mag_site_map"],
                    swarm=self.swarm.as_dict(),
                    n_iter=self.n_iter+1,
                    max_iter_limit=self["max_iter_limit"],
                    energy_convergence_tol=self["energy_convergence_tol"],
                    user_incar_settings=self["user_incar_settings"],
                    user_kpoints_settings=self["user_kpoints_settings"],
                ),
                parents=swarm_fws,
                name="SpinPSOiter",
            )
            fws.append(fw_iter)

        # Create workflow to append
        wf_add = Workflow(fws)

        wf_add = add_common_powerups(wf_add, self["c"])

        if self["c"].get("ADD_WF_METADATA", ADD_WF_METADATA):
            wf_add = add_wf_metadata(wf_add, self.structure_nonmag)

        wf_add = add_additional_fields_to_taskdocs(
            wf_add,
            {
                "wf_meta": {
                    "wf_uuid": self["spin_pso_wf_uuid"],
                    "wf_name": "spin_pso",
                    "wf_version": __spin_pso_wf_version__,
                }
            },
        )

        # Save to database
        summaries = []

        summary = {}

        summary.update({"algo_has_converged":algo_has_converged, "fitness_deviation":fitness_deviation})

        summary["task_input"] = {}
        for param_key in self.required_params:
            if param_key=="structure_nonmag":
                summary["task_input"][param_key] = self[param_key].as_dict()
            else:
                summary["task_input"][param_key] = self[param_key]

        summary["task_output"] = {}
        summary["task_output"]["swarm"] = self.swarm.as_dict()
        summary["task_output"]["moment_constr_energy_penalties"] = energy_penalties.copy()
        summary["task_output"]["ordering_energy_penalties"] = energy_order_penalties.copy()

        grads_new = [[grads_new[i][j].tolist() for j in range(self.num_spin)] for i in range(self.swarm.num_agent)]
        m_int_new = [[m_int_new[i][j].tolist() for j in range(self.num_spin)] for i in range(self.swarm.num_agent)]
        mw_int_new = [[mw_int_new[i][j].tolist() for j in range(self.num_spin)] for i in range(self.swarm.num_agent)]

        summary["task_output"]["grads_new"] = grads_new
        summary["task_output"]["m_int_new"] = m_int_new
        summary["task_output"]["mw_int_new"] = mw_int_new

        summary.update({"created_at": datetime.utcnow()})
        summary.update({'wf_meta': {'wf_uuid': self["spin_pso_wf_uuid"]}})

        if fw_spec.get("tags", None):
            summary["tags"] = fw_spec["tags"]

        summaries.append(summary)

        mmdb.collection = mmdb.db["spin_pso"]
        mmdb.collection.insert(summaries)

        logger.info("SpinPSO iteration is complete.")

        return FWAction(
            stored_data={
                "swarm":self.swarm.as_dict(),
            },
            additions=[wf_add],
        )

#         if self.n_iter < self.max_iter_limit and not(algo_has_converged):
#             return FWAction(
#                 stored_data={
#                     "swarm":self.swarm.as_dict(),
#                 },
#                 additions=[wf_add],
#             )
#         else:
#             fws = []
# #             fws.append(self.get_swarm_analysis_fw(parents=[]))
#             return FWAction(additions=fws)

    def get_agent_step_fws(self, is_final_unconstrained=False):
        """_summary_

        Args:
            is_final_unconstrained (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        fws = []

        if not is_final_unconstrained:
            agent_ids = list(range(self.swarm.num_agent))
        else:
            if self.swarm.index_best > -1:
                agent_ids = [self.swarm.index_best]
            else:
                return fws

        for i in agent_ids:

            uis = self.user_incar_settings.copy()
            uks = self.user_kpoints_settings.copy()

            struct_out = Structure.from_dict(self.structs_out[i])
            magmoms = [[0.0, 0.0, 0.0] for s in struct_out]
            for mi,m in enumerate(self.pso_states[i]):
                magmoms[self.mag_site_map[mi]][:] = m[:]
            struct_mag = Structure(
                species=struct_out.species, lattice=struct_out.lattice,
                coords=struct_out.frac_coords,
            )
            struct_mag.add_site_property('magmom', [m.copy() for m in magmoms])

            m_constr = [m.copy() for m in magmoms]
            uis["M_CONSTR"] = m_constr.copy()
            if is_final_unconstrained:
                uis["LAMBDA"] = 0.0
            # if self.n_iter==0:
            #     uis["LAMBDA"] = 0.0
            #     uis["EDIFF"] = 100.0 * self.user_incar_settings["EDIFF"]
            #     uis["ALGO"] = "Normal"

            vis_params = {"user_incar_settings": uis.copy(), "user_kpoints_settings": uks.copy()}
            vis = NoncollinearConstrainSet(structure=struct_mag.copy(), **vis_params.copy())

            fw = SpinPSOrunVaspFW(
                structure=struct_mag.copy(), agent_id=i, n_iter=self.n_iter+1,
                name="spin_pso_run_vasp", fw_name=None,
                vasp_input_set=vis, vasp_cmd=self["vasp_cmd"],
            )
            fws.append(fw)

        return fws

#     def get_swarm_analysis_fw(self, parents=[]):
#         fw = SpinPSOstepToDb(swarm, parents=parents)
#         return fw

class NoncollinearConstrainFW(Firework):
    """_summary_

    Args:
        Firework (_type_): _description_
    """
    def __init__(self, structure=None, name="ncl_constrain", fw_name=None,
                 vasp_input_set=None, vasp_input_set_params=None,
                 vasp_cmd=NCL_VASP_CMD, prev_calc_loc=True, prev_calc_dir=None,
                 db_file=DB_FILE, vasptodb_kwargs=None, parents=None,
                 additional_files=None,
                 **kwargs):
        """_summary_

        Args:
            structure (_type_, optional): _description_. Defaults to None.
            name (str, optional): _description_. Defaults to "ncl_constrain".
            fw_name (_type_, optional): _description_. Defaults to None.
            vasp_input_set (_type_, optional): _description_. Defaults to None.
            vasp_input_set_params (_type_, optional): _description_. Defaults to None.
            vasp_cmd (_type_, optional): _description_. Defaults to NCL_VASP_CMD.
            prev_calc_loc (bool, optional): _description_. Defaults to True.
            prev_calc_dir (_type_, optional): _description_. Defaults to None.
            db_file (_type_, optional): _description_. Defaults to DB_FILE.
            vasptodb_kwargs (_type_, optional): _description_. Defaults to None.
            parents (_type_, optional): _description_. Defaults to None.
            additional_files (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        t = []

        vasp_input_set_params = vasp_input_set_params or {}
        vasptodb_kwargs = vasptodb_kwargs or {}
        if "additional_fields" not in vasptodb_kwargs:
            vasptodb_kwargs["additional_fields"] = {}
        vasptodb_kwargs["additional_fields"]["task_label"] = name

        if not fw_name:
            fw_name = "{}-{}".format(structure.composition.reduced_formula if structure else "unknown", name)

        if prev_calc_dir:
            t.append(CopyVaspOutputs(
                calc_dir=prev_calc_dir, additional_files=additional_files,
                contcar_to_poscar=False))
        elif parents:
            if prev_calc_loc:
                t.append(CopyVaspOutputs(
                    calc_loc=prev_calc_loc, additional_files=additional_files,
                    contcar_to_poscar=False))

        if structure:
            vasp_input_set = vasp_input_set or NoncollinearConstrainSet(structure, **vasp_input_set_params)
            t.append(WriteVaspFromIOSet(
                structure=structure, vasp_input_set=vasp_input_set))
        else:
            raise ValueError("Must specify structure")

        #t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, auto_npar=">>auto_npar<<"))
        t.append(RunVaspDirect(vasp_cmd=vasp_cmd))

        t.append(PassCalcLocs(name=name))
        t.append(VaspToDb(db_file=db_file, **vasptodb_kwargs))
        super(NoncollinearConstrainFW, self).__init__(t, parents=parents, name=fw_name, **kwargs)


class SpinPSOrunVaspFW(NoncollinearConstrainFW):
    """_summary_

    Args:
        NoncollinearConstrainFW (_type_): _description_
    """
    def __init__(self, structure=None, agent_id=None, n_iter=None,
                 name="spin_pso_run_vasp", fw_name=None,
                 vasp_input_set=None, vasp_input_set_params=None,
                 vasp_cmd=NCL_VASP_CMD, prev_calc_loc=True, prev_calc_dir=None,
                 db_file=DB_FILE, vasptodb_kwargs=None, parents=None,
                 additional_files=None,
                 **kwargs):
        """_summary_

        Args:
            structure (_type_, optional): _description_. Defaults to None.
            agent_id (_type_, optional): _description_. Defaults to None.
            n_iter (_type_, optional): _description_. Defaults to None.
            name (str, optional): _description_. Defaults to "spin_pso_run_vasp".
            fw_name (_type_, optional): _description_. Defaults to None.
            vasp_input_set (_type_, optional): _description_. Defaults to None.
            vasp_input_set_params (_type_, optional): _description_. Defaults to None.
            vasp_cmd (_type_, optional): _description_. Defaults to NCL_VASP_CMD.
            prev_calc_loc (bool, optional): _description_. Defaults to True.
            prev_calc_dir (_type_, optional): _description_. Defaults to None.
            db_file (_type_, optional): _description_. Defaults to DB_FILE.
            vasptodb_kwargs (_type_, optional): _description_. Defaults to None.
            parents (_type_, optional): _description_. Defaults to None.
            additional_files (_type_, optional): _description_. Defaults to None.
        """

        vasptodb_kwargs = vasptodb_kwargs or {}
        if "additional_fields" not in vasptodb_kwargs:
            vasptodb_kwargs["additional_fields"] = {}
        vasptodb_kwargs["additional_fields"]["task_label"] = name

        vasptodb_kwargs["additional_fields"]["spin_pso"] = {}
        vasptodb_kwargs["additional_fields"]["spin_pso"]["agent_id"] = agent_id
        vasptodb_kwargs["additional_fields"]["spin_pso"]["n_iter"] = n_iter

        # prepare and store input structure
        input_structure = Structure(
            species=structure.species, lattice=structure.lattice, coords=structure.frac_coords)
        magmoms = [list(s.properties['magmom']) for s in structure]
        input_structure.add_site_property('magmom', magmoms)
        vasptodb_kwargs["additional_fields"]["spin_pso"]["input_structure"] = input_structure.as_dict()

        super(SpinPSOrunVaspFW, self).__init__(
            structure=structure, fw_name=fw_name,
            vasp_input_set=vasp_input_set, vasp_input_set_params=vasp_input_set_params,
            vasp_cmd=vasp_cmd, prev_calc_loc=prev_calc_loc, prev_calc_dir=prev_calc_dir,
            db_file=db_file, vasptodb_kwargs=vasptodb_kwargs, parents=parents,
            additional_files=additional_files,
            **kwargs
        )

# class SpinPSOtoDb(FiretaskBase):
#     """
#     """
#     required_params = ["db_file", "wf_uuid"]
#     optional_params = []

#     summaries = []

#     def run_task(self, fw_spec):

#         uuid = self["wf_uuid"]
#         db_file = env_chk(self.get("db_file"), fw_spec)
#         to_db = self.get("to_db", True)

#         mmdb = VaspCalcDb.from_db_file(db_file, admin=True)

#         docs = list(mmdb.collection.find({"wf_meta.wf_uuid": uuid}))

#         summary = {}
#         if structure:
#             summary.update({'formula_pretty': structure.composition.reduced_formula})
#             summary.update({'structure': structure.as_dict()})
#         summary.update({"created_at": datetime.utcnow()})
#         summary.update({'wf_meta': {'wf_uuid': uuid}})

#         if fw_spec.get("tags", None):
#             summary["tags"] = fw_spec["tags"]

#         summaries.append(summary)

#         mmdb.collection = mmdb.db["spin_pso"]
#         mmdb.collection.insert(summaries)

#         logger.info("SpinPSO agent run analysis is complete.")

class NoncollinearConstrainSet(MPStaticSet):
    """_summary_

    Args:
        MPStaticSet (_type_): _description_
    """
    def __init__(self, structure, prev_incar=None, prev_kpoints=None,
                 reciprocal_density=100, small_gap_multiply=None, **kwargs):
        """_summary_

        Args:
            structure (_type_): _description_
            prev_incar (_type_, optional): _description_. Defaults to None.
            prev_kpoints (_type_, optional): _description_. Defaults to None.
            reciprocal_density (int, optional): _description_. Defaults to 100.
            small_gap_multiply (_type_, optional): _description_. Defaults to None.
        """

        super().__init__(structure, sort_structure=False, **kwargs)

        if isinstance(prev_kpoints, str):
            prev_kpoints = Kpoints.from_file(prev_kpoints)
        self.prev_kpoints = prev_kpoints

        self.reciprocal_density = reciprocal_density
        self.kwargs = kwargs
        self.small_gap_multiply = small_gap_multiply

    @property
    def incar(self):
        """_summary_
        """
        parent_incar = super().incar
        incar = Incar(parent_incar)

        incar.update({
            "ISYM":-1, 
            "LASPH":True, 
            # "LREAL":False,
        })
        incar.pop("NSW", None)

        incar.update({"ISPIN":2, "LSORBIT":True, "LNONCOLLINEAR":True, "LORBMOM":True, "VOSKOWN":1})

        incar.update({"I_CONSTRAINED_M": self.kwargs.get("user_incar_settings")["I_CONSTRAINED_M"]})
        incar.update({"LAMBDA": self.kwargs.get("user_incar_settings")["LAMBDA"]})
        incar.update({"RWIGS": self.kwargs.get("user_incar_settings")["RWIGS"]})
        incar.update({"M_CONSTR": self.kwargs.get("user_incar_settings")["M_CONSTR"]})

#         if not (incar.get("I_CONSTRAINED_M", None) and incar.get("LAMBDA", None) and
#                 incar.get("RWIGS", None) and incar.get("M_CONSTR", None)):
#             raise ValueError(
#                 "Missing crucial inputs for VASP's constrained local moments method."
#             )

        return incar


def pull_mag_constrs_from_oszicar(dirname):
    """_summary_

    Args:
        dirname (_type_): _description_

    Raises:
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """
    
    # parse OSZICAR - local moment constraint output
    try:
        oszicar = Oszicar(os.path.join(dirname, "OSZICAR.gz"))
    except Exception as exc:
        try:
            oszicar = Oszicar(os.path.join(dirname, "OSZICAR"))
        except Exception as exc:
            raise RuntimeError(
                "Failed to read OSZICAR."
            )
    
    constr_out = oszicar.magmom_constrain_out[-1]
    if "lambda*MW_perp" in constr_out["column_names"]:
        h_eff = [cols_out["column_out"] for cols_out in constr_out["output"]]
    else:
        h_eff = []
    
    constr_out = oszicar.magmom_constrain_out[-2]
    if set(["MW_int", "M_int"]).issubset(constr_out["column_names"]):
        mw_int = [cols_out["column_out"][0:3] for cols_out in constr_out["output"]]
        m_int  = [cols_out["column_out"][3:6] for cols_out in constr_out["output"]]
    else:
        mw_int, m_int = [], []
    
    energy_penalty = constr_out["E_p"]
    
    return mw_int, m_int, h_eff, energy_penalty
