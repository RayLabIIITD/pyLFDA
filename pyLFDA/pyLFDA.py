import sys
import os
import decimal
import math
import time
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from numpy.core.fromnumeric import mean
import MDAnalysis as mda
import MDAnalysis.analysis.msd as msd
from membrane_curvature.base import MembraneCurvature
import subprocess
from scipy.stats import linregress
import traceback
import textwrap
import errno
import copy
import re
import warnings
import logging
import faulthandler
import pickle
import string
import secrets
warnings.filterwarnings('ignore')

class Point():
    '''
    Class to store the coordinates and perform the required
    operations on a given atom in 3D space
    '''
    def __init__(self, x, y, z):
        self.x = decimal.Decimal(str(x))
        self.y = decimal.Decimal(str(y))
        self.z = decimal.Decimal(str(z))

    def mod(self):
        '''
        Returns magnitude of the Point
        '''
        modulus_value = math.sqrt(( self.x * self.x ) + ( self.y * self.y ) + ( self.z * self.z ))
        return modulus_value

    def __add__(self, point):
        '''
        Adds the points
        '''
        added_points = Point( self.x + point.x, self.y + point.y, self.z + point.z )
        return added_points

    def __sub__(self, point):
        '''
        Subtracts the points
        '''
        subtracted_points = Point( self.x - point.x, self.y - point.y, self.z - point.z )
        return subtracted_points

    def dot(self, point):
        '''
        Multiplies the points
        '''
        multiplied_points = Point( self.x * point.x, self.y * point.y, self.z * point.z )
        return multiplied_points

    def negate(self):
        '''
        Creates a new Point object with the components such that the 
        given fore vector reverses direction but remains the same in magnitude
        and returns it
        '''
        new_point = Point( -self.x, -self.y, -self.z)
        return new_point

    def print(self):
        '''
        Returns magnitude of the Point
        '''
        print("X : ", self.x, "  Y : ", self.y, "  Z : ", self.z, "", flush = True)

class Atom():
    '''
    Class to Represent an Atom in 3D sapce
    '''
    def __init__(self, name, x, y, z):
        self.name = name
        self.Coords = Point(x, y, z)
    
    def print(self):
        print(self.name," ",self.point.x," ",self.point.y," ",self.point.z, flush=True)

    def dot(self, point):
        dot_pdt = self.x*point.x + self.y*point.y + self.z*point.z
        return dot_pdt

    def print(self):
        print("X : ", self.x, "  Y : ", self.y, "  Z : ", self.z, "", flush = True)

class AtomForced():
    '''
    Class to store forces on an atom
    '''
    def __init__(self, ResNum, ResName, AtomName, AtomNumber, X, Y, Z, Fx, Fy, Fz):
        self.ResNum = ResNum
        self.ResName = ResName
        self.AtomName = AtomName
        self.AtomNumber = AtomNumber
        self.Coords = Point(X, Y, Z)
        self.Force = Point(Fx, Fy, Fz)

class LFDA():
    '''
    Class to manage path, variables and functions related to LFDA analysis
    '''
    def __init__(self, experiment_name=None, pdb_filename=None, gro_filename=None, trr_filename=None, tpr_filename=None, ndx_filename=None, gfda_version="v2019.3-fda2.9.1"):
        '''
        Initialising the experiment
        Arguments : 
            -   experiment_name : Name of the experiment. Uses this to create a directory to store outputs in. If not specified time-stamp of experiment will be used.
            -   pdb_filename : Path of the PDB file to be used.
            -   gro_filename : Path of the GRO file to be used.
            -   trr_filename : Path of the TRR file to be used.
            -   tpr_filename : Path of the TPR file to be used.
            -   ndx_filename : Path of the NDX file to be used.
            -   gfda_version : Version of Gromacs FDA to be used. Creates a directory with the name to store it and uses it for further experiments.
        '''
        logging.getLogger('MDAnalysis.MDAKit.membrane_curvature').setLevel(logging.CRITICAL + 1)
        logging.getLogger('MDAnalysis').setLevel(logging.INFO)
        global logger
        logging.basicConfig(format='%(name)s : %(levelname)s :\t%(message)s')
        logger = logging.getLogger('pyLFDA')
        logger.addHandler(logging.NullHandler())
        logging.getLogger('pyLFDA').setLevel(logging.INFO)
        #faulthandler.enable() #for debugging purposes
        try:
            sys.tracebacklimit = -1
            #Set Experiment Name
            self.timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
            if experiment_name!=None:
                if os.path.exists(experiment_name):
                    self.experiment_name = os.path.abspath(os.path.expanduser(os.path.expandvars(experiment_name)))
                else:
                    subprocess.run(["mkdir", experiment_name])
                    self.experiment_name = os.path.abspath(os.path.expanduser(os.path.expandvars(experiment_name)))
            else:
                subprocess.run(["mkdir", self.timestamp])
                self.experiment_name = os.path.abspath(os.path.expanduser(os.path.expandvars(self.timestamp)))
            if not os.path.exists(self.experiment_name):
                subprocess.run(["mkdir", self.experiment_name])
            
            #Set PDB File
            if pdb_filename!=None:
                self.pdb_filename = os.path.abspath(os.path.expanduser(os.path.expandvars(pdb_filename)))
            if ".pdb" not in self.pdb_filename:
                raise ValueError('Enter valid PDB file') 
            if not os.path.exists(self.pdb_filename):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.pdb_filename)

            #Set GRO File
            if gro_filename!=None:
                self.gro_filename = os.path.abspath(os.path.expanduser(os.path.expandvars(gro_filename)))
            if ".gro" not in self.gro_filename:
                raise ValueError('Enter valid GRO file') 
            if not os.path.exists(self.gro_filename):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.gro_filename)

            #Set TRR File
            if trr_filename!=None:
                self.trr_filename = os.path.abspath(os.path.expanduser(os.path.expandvars(trr_filename)))
            if ".trr" not in self.trr_filename:
                raise ValueError('Enter valid TRR file') 
            if not os.path.exists(self.trr_filename):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.trr_filename)
            
            #Set TPR File
            if tpr_filename!=None:
                self.tpr_filename = os.path.abspath(os.path.expanduser(os.path.expandvars(tpr_filename)))
            if ".tpr" not in self.tpr_filename:
                raise ValueError('Enter valid TPR file') 
            if not os.path.exists(self.tpr_filename):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.tpr_filename)
                
            #Set NDX File
            if ndx_filename!=None:
                self.ndx_filename = os.path.abspath(os.path.expanduser(os.path.expandvars(ndx_filename)))
            if ".ndx" not in self.ndx_filename:
                raise ValueError('Enter valid NDX file') 
            if not os.path.exists(self.ndx_filename):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.ndx_filename)

            #1: Install Gromacs FDA if it not inilialized
            if not os.path.isdir(gfda_version):                
                logger.info(f"Checking GROMACS FDA installation - {gfda_version}")                
                version_control(gfda_version)
            self.fda_bin_path = os.path.abspath(os.path.expanduser(os.path.expandvars(gfda_version+"/bin")))
            
            #Initialise MDA Universe
            self.create_mda_universe()
            self.group1 = None
            self.group2 = None
            self.force = None
            self.residue_list = None

            self.pfi_filename = None
            self.pfa_filename = None

            self.parallel_theads = 1   
            self.MEMBRANE_PARTITION_THRESHOLD_FRACTION = 0.01

            self.framewise = True
            self.summed_pfa_filename_framewise = None
            self.atom_dict_framewise = None
            self.summed_pfa_filename = None
            self.atom_dict = None

            logger.info("Parsing GRO file to calculate numbers of atoms, atoms information and box vectors")
            self.num_atoms, self.atom_info_list, self.box_vectors = parse_gro(self.gro_filename)

        except:
            logger.error(traceback.format_exc())
            #sys.exit(0)

    def run_fda(self, group1=None, group2=None, force="all", residue_list=None, pfi_filename=None, pfa_filename=None):
        '''
        Function to create PFI file and then generating a PFA file using GROMACS FDA.
        Arguments : 
            -   group1 : 1st group selected
            -   group2 : 2nd group selected
            -   residue_list : [group1, group2]
            -   pfi_filename : Name of the PFI file to be generated. It is inferred from the experiment class if None.
            -   pfa_filename : Name of the PFA file to be generated. It is inferred from the experiment class if None.
        '''
        try:
            self.group1 = group1
            self.group2 = group2
            self.force = force
            self.residue_list = residue_list

            if pfi_filename==None:
                self.pfi_filename = os.path.abspath(os.path.expanduser(os.path.expandvars(self.experiment_name+"/pfi_"+self.timestamp+".pfi")))
            else:
                self.pfi_filename = os.path.abspath(os.path.expanduser(os.path.expandvars(self.experiment_name+"/"+pfi_filename)))

            if pfa_filename==None:
                self.pfa_filename = os.path.abspath(os.path.expanduser(os.path.expandvars(self.experiment_name+"/pfa_"+self.timestamp+".pfa")))
            else:
                self.pfa_filename = os.path.abspath(os.path.expanduser(os.path.expandvars(self.experiment_name+"/"+pfa_filename)))

            #2: Create pfi file
            logger.info("Creating PFI file")
            create_pfi(path=self.pfi_filename, 
                       group_1=self.group1, 
                       group_2=self.group2, 
                       force_type=self.force, 
                       onepair="summed", 
                       atombased="pairwise_forces_vector", 
                       residuebased="no", 
                       ignore_missing_potentials="yes")

            #3: Running gmx fda
            logger.info("Running Gromacs FDA")
            run_gmx_fda(fda_install_path=self.fda_bin_path, 
                        trr_filename=self.trr_filename, 
                        tpr_filename=self.tpr_filename, 
                        pfi_filename=self.pfi_filename, 
                        pfa_filename=self.pfa_filename, 
                        index_file=self.ndx_filename, 
                        threads=self.parallel_theads)

            logger.info("{} PFA file is generated in {}".format(self.pfa_filename, self.experiment_name))
            return
        
        except:
            logger.error(traceback.format_exc())
            #sys.exit(0)
        return
    
    def load_pfa(self, pfa_filename=None, group1=None, group2=None, residue_list=None,):
        '''
        Function to load PFA file generated. Removes need to re-run experiments.
        Arguments :
            -   pfa_filename : Path of PFA file generated by Gromacs FDA.
            -   group1 : 1st group selected
            -   group2 : 2nd group selected
            -   residue_list : [group1, group2]
        '''
        try:
            logger.info("Loading PFA file generated by Gromacs FDA")
            self.pfa_filename = os.path.abspath(os.path.expanduser(os.path.expandvars(pfa_filename)))
            self.group1 = group1
            self.group2 = group2
            self.residue_list = residue_list
        except:
            logger.error(traceback.format_exc())
            #sys.exit(0)
    
    def create_mda_universe(self,):
        '''
        Function to extract relevant information using MDAnalysis.
        '''
        try:
            logger.info("Making MDA Universe from PDB and TRR file")
            self.mda_universe = mda.Universe(self.pdb_filename, self.trr_filename)
            self.mda_timestamp = [int(i.time) for i in self.mda_universe.trajectory]
            self.mda_residue_name = list(set(self.mda_universe.residues.resnames))
        except:
            logger.error(traceback.format_exc())
            #sys.exit(0)

    def parse_pfa(self, file_name=None):
        '''
        Function to parse PFA generated by Gromacs FDA.
        Arguments :
            -   file_name : Name with which parsed PFA file is to be saved.
        '''
        try:
            if self.framewise==True:
                if file_name==None:
                    self.summed_pfa_filename_framewise = os.path.abspath(os.path.expanduser(os.path.expandvars(self.experiment_name+"/pfa_framewise_"+self.timestamp+".pfa")))
                else:
                    self.summed_pfa_filename_framewise = os.path.abspath(os.path.expanduser(os.path.expandvars(self.experiment_name+"/"+file_name)))

                logger.info("Parsing PFA file as framewise")
                create_summed_pfa(pfa_filename=self.pfa_filename, 
                                  num_atoms=self.num_atoms,
                                  summed_pfa_filename=self.summed_pfa_filename_framewise, 
                                  framewise=True) 
                    
                self.atom_dict_framewise =  parse_summed_pfa(summed_pfa_file=self.summed_pfa_filename_framewise, 
                                                             atom_info=self.atom_info_list, 
                                                             residue_list=self.residue_list)
                #self.save_atom_dict(mode="framewise")
            else:
                if file_name==None:
                    self.summed_pfa_filename = os.path.abspath(os.path.expanduser(os.path.expandvars(self.experiment_name+"/pfa_averaged_"+self.timestamp+".pfa")))
                else:
                    self.summed_pfa_filename = os.path.abspath(os.path.expanduser(os.path.expandvars(self.experiment_name+"/"+file_name)))

                logger.info("Parsing PFA file as average")
                create_summed_pfa(pfa_filename=self.pfa_filename, 
                                  num_atoms=self.num_atoms, 
                                  summed_pfa_filename=self.summed_pfa_filename,
                                  framewise=False) 

                self.atom_dict = parse_summed_pfa(summed_pfa_file=self.summed_pfa_filename, 
                                                   atom_info=self.atom_info_list, 
                                                   residue_list=self.residue_list)
                #self.save_atom_dict(mode="average")
        except:
            logger.error(traceback.format_exc())
            #sys.exit(0)

    def parse_parsed_pfa(self, file_name=None, mode=None, group1=None, group2=None, residue_list=None,):
        '''
        Function to parse PFA generated by pyLFDA.
        Arguments :
            -   file_name : Name with which parsed PFA file is to be saved.
            -   mode : "average" - parse file as averaged. "framewise" - parse file per frame.
        '''
        try:
            self.group1 = group1
            self.group2 = group2
            self.residue_list = residue_list
            if mode == "average":
                logger.info("Loading Average parsed PFA file")
                self.framewise = False
                self.summed_pfa_filename = os.path.abspath(os.path.expanduser(os.path.expandvars(file_name)))
                self.atom_dict = parse_summed_pfa(summed_pfa_file=self.summed_pfa_filename, 
                                                       atom_info=self.atom_info_list, 
                                                       residue_list=self.residue_list)
                #self.save_atom_dict(mode="average")
            elif mode == "framewise":
                logger.info("Loading Framewise parsed PFA file")
                self.framewise = True
                self.summed_pfa_filename_framewise = os.path.abspath(os.path.expanduser(os.path.expandvars(file_name)))
                self.atom_dict_framewise =  parse_summed_pfa(summed_pfa_file=self.summed_pfa_filename_framewise, 
                                                                 atom_info=self.atom_info_list, 
                                                                 residue_list=self.residue_list)
                #self.save_atom_dict(mode="framewise")
            else:
                raise ValueError("Mode not specified or Incorrect")
        except:
            logger.error(traceback.format_exc())
            #sys.exit(0)

    def save_atom_dict(self, mode):
        if mode == "average":
            with open(f'{self.summed_pfa_filename.split(".")[0]}.pkl', 'wb') as fl:
                pickle.dump(self.atom_dict, fl, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f'Database file saved at {self.summed_pfa_filename.split(".")[0]}.pkl')
        elif mode == "framewise":
            with open(f'{self.summed_pfa_filename_framewise.split(".")[0]}.pkl', 'wb') as fl:
                pickle.dump(self.atom_dict_framewise, fl, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f'Database file saved at {self.summed_pfa_filename_framewise.split(".")[0]}.pkl')

    def load_database(self, database_file, mode):
        if mode == "average":
            with open(database_file, 'rb') as fl:
                self.atom_dict = pickle.load(fl)
            logger.info(f'Loaded database file')
        elif mode == "framewise":
            with open(database_file, 'rb') as fl:
                self.atom_dict_framewise = pickle.load(fl)
            logger.info(f'Loaded database file')

    def bfactor_pdb(self, bfactor_pdb_filename=None, mode="atomistic"):
        '''
        Function to load bfactor to a new PDB file.
        Arguments : 
            -   mode : "atomistic" loads value per atom. "groupwise" loads value averaged for the entire group.
        '''
        try:
            if self.atom_dict==None and self.atom_dict_framewise==None:
                raise ValueError("Force property of atoms are not calculated yet, please run make_summed_pfa(framewise=False) function before running this function")
            else:
                #if self.framewise:
                #    raise ValueError("Cannot create BFactor value with framewise option as true")
                #else:                
                if bfactor_pdb_filename==None:
                    self.bfactor_pdb_filename = os.path.abspath(os.path.expanduser(os.path.expandvars(self.experiment_name+"/bfactor_"+self.timestamp+os.path.basename(self.pdb_filename)[:-4]+".pdb")))
                else:
                    self.bfactor_pdb_filename = os.path.abspath(os.path.expanduser(os.path.expandvars(self.experiment_name+"/"+bfactor_pdb_filename+".pdb")))
                    
                logger.info("Loading a new PDB file with bFactor")
                bfactor_pdb(atom_dict=self.atom_dict if self.atom_dict!=None else self.atom_dict_framewise, 
                            pdb_filename=self.pdb_filename, 
                            bfactor_pdb_filename=self.bfactor_pdb_filename, 
                            mode=mode)
        except:
            logger.error(traceback.format_exc())
            #sys.exit(0)

    def force_graph(self, specific_frame=None, window=None):
        '''
        Function to plot force graph for a pair of groups.
        Arguments : 
            -   No arguments : Average Force over all frames.
            -   specific_frame : Forces for a specific frame.
            -   window : Forces for a moving window of the specified size.
        '''
        try:
            if self.framewise==True:
                if self.atom_dict_framewise==None:
                    raise ValueError("Force of atoms are not calculated yet, please run parse_pfa() function before running this function")
                logger.info("Creating framewise average force plot")
                if specific_frame==None and window==None:
                    create_average_residue_graph(atom_dict=self.atom_dict_framewise, 
                                            plot_name=self.experiment_name+"/force_averaged")
                elif specific_frame:
                    create_specific_frame_graph(atom_dict=self.atom_dict_framewise, 
                                            specific_frame=specific_frame, 
                                            plot_name=self.experiment_name+"/force_specific_frame")
                elif window:
                    create_moving_window_graph(atom_dict=self.atom_dict_framewise, 
                                           moving_window=window, 
                                           plot_name=self.experiment_name+"/force_moving_window")
            else:
                if self.atom_dict==None:
                    raise ValueError("Force property of atoms are not calculated yet, please run parse_pfa() function before running this function")
                logger.info("Creating average force plot")
                create_average_residue_graph(atom_dict=self.atom_dict, 
                                            plot_name=self.experiment_name+"/force_averaged")
        except:
            logger.error(traceback.format_exc())
            #sys.exit(0)

    def curvature(self, specific_frame=None, window=None, selection="", num_x_bins=None, num_y_bins=None, split=False):
        '''
        Function to create a .pfi file
        Arguments : 
                    curvature_type=(Available optons : None, framewise, window)
                    window_size=(Default : None)
                    selection=(Default : "")
                    num_x_bins=(Default : 10)
                    num_y_bins=(Default : 10)
                    plot_type=(Default : "box")
                    split=(Default : False)
        Returns : 
                    None
        Outputs :
                    Cuvature plots of type selected by the user
        '''
        
        if num_x_bins == None:
            num_x_bins = 10
        else:
            num_x_bins = int(num_x_bins)
        if num_y_bins == None:
            num_y_bins = 10
        else:
            num_y_bins = int(num_y_bins)
        try:
            if not specific_frame and not window:
                angle = gangle(self.trr_filename, self.tpr_filename, self.ndx_filename, self.pdb_filename, self.mda_universe, self.group1, self.group2, mode="average", split = split)
                logger.info("Creating curvature plot")
                plot_curvature( universe=self.mda_universe, 
                                atom_dict=self.atom_dict if self.atom_dict!=None else self.atom_dict_framewise,
                                num_x_bins=num_x_bins, 
                                num_y_bins=num_y_bins, 
                                split=split,
                                gangle = angle,
                                plot_name=self.experiment_name+"/curvature_averaged")
            elif specific_frame:
                if self.atom_dict_framewise==None:
                    raise ValueError("Force property of atoms are not calculated yet, please run make_summed_pfa() function before running this function")
                else:
                    logger.info(f"Creating curvature plot for frame {specific_frame}")
                    angle = gangle(self.trr_filename, self.tpr_filename, self.ndx_filename, self.pdb_filename, self.mda_universe, self.group1, self.group2, mode="framewise", split = split)
                    plot_curvature_framewise(   universe=self.mda_universe, 
                                                atom_dict=self.atom_dict_framewise, 
                                                specific_frame=specific_frame,
                                                num_x_bins = num_x_bins, 
                                                num_y_bins = num_y_bins, 
                                                split = split, 
                                                gangle = angle,
                                                plot_name=self.experiment_name+"/curvature_framewise")
            elif window:
                if self.atom_dict_framewise==None:
                    raise ValueError("Force property of atoms are not calculated yet, please run make_summed_pfa() function before running this function")
                else:
                    logger.info(f"Creating {window} window size curvature plots")
                    angle = gangle(self.trr_filename, self.tpr_filename, self.ndx_filename, self.pdb_filename, self.mda_universe, self.group1, self.group2, mode="window", split = split)
                    plot_curvature_window(  universe=self.mda_universe, 
                                            atom_dict=self.atom_dict_framewise, 
                                            window_size = window, 
                                            num_x_bins = num_x_bins, 
                                            num_y_bins = num_y_bins,  
                                            split = split,
                                            gangle = angle,
                                            plot_name=self.experiment_name+"/curvature_moving_window")

        except:
            logger.error(traceback.print_exc(), "at line", format(sys.exc_info()[-1].tb_lineno))
            #sys.exit(0)
        return

    def cluster(self, lipids_to_cluster=None, attached_ligands=None, protein_residue_names=None, mode="pair", box_side_length = 6):
        try:
            logger.info("Making clutering plots")
            lipids_to_cluster=self.group1 if lipids_to_cluster == None else lipids_to_cluster
            attached_ligands=[None] if attached_ligands == None else [attached_ligands]
            protein_residue_names = [x for x in self.mda_residue_name if x not in [attached_ligands, lipids_to_cluster]] if mode == "pair" else [x for x in self.mda_residue_name if x not in [lipids_to_cluster]]
            clustering_plots(pdb_file=self.pdb_filename, 
                             top_bottom='top', 
                             box_side_length=box_side_length, 
                             protein_residue_names=protein_residue_names, 
                             attached_ligands=attached_ligands, 
                             lipids_to_cluster=lipids_to_cluster, 
                             mode=mode,
                             plot_name=self.experiment_name+"/cluster")
        except:
            logger.error(traceback.format_exc(), "at line", format(sys.exc_info()[-1].tb_lineno))
            #sys.exit(0)
        return

    def msd(self, select='all', msd_type='xyz', fft=True, timestep=1, start_index=None, end_index=None):
        '''
        Function to plot MSD values for all frames and calculates the diffusion coefficient
        Arguments :
            -   select : MDUniverse Atom selection
            -   msd_type : MSD Type
        '''
        try:
            logger.info("Calculating diffusion coefficient")
            plot_msd(universe=self.mda_universe, 
                                  select=select, 
                                  msd_type=msd_type, 
                                  fft=fft, 
                                  timestep=timestep, 
                                  start_index=start_index, 
                                  end_index=end_index,
                                  plot_name=self.experiment_name+"/MSD")
        except:
            logger.error(traceback.format_exc())
            #sys.exit(0)
        return
    
    def angles(self, selection, grouping, c_atom_name, split = False):
        '''
        Function to plot the averaged angle with the z-axis for the selected lipids over all frames.
        Arguments :
            -   select : MDUniverse Atom selection
            -   msd_type : MSD Type
        '''
        try:
            logger.info(f"Calculating lipid angles with vector as P -> {c_atom_name}")
            angle = gangle(self.trr_filename, self.tpr_filename, self.ndx_filename, self.pdb_filename, self.mda_universe, selection = selection, grouping = grouping, c_atom_name = c_atom_name, split = split, angles = True)
            if grouping != 'combine':
                if split == False:
                    color = iter(plt.cm.cmap_d['rainbow'](np.linspace(0, 1, len(selection))))
                    for n, group in enumerate(selection):
                        plt.plot(list(range(len(angle[group]))), angle[group], linewidth=1, color=next(color), label=group, alpha=0.6)
                    plt.xlabel("Frame")
                    plt.ylabel("Angle")
                    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
                    plt.title(f'Lipid Angles P to {c_atom_name} {"_".join(selection)}', fontdict={'fontsize':10})
                    plt.savefig(f'{self.experiment_name}/angles_framewise_P_to_{c_atom_name}_{"_".join(selection)}.svg', dpi = 1000, bbox_inches="tight")
                    plt.close()
                else:
                    for split in ["Upper", "Lower"]:
                        color = iter(plt.cm.cmap_d['rainbow'](np.linspace(0, 1, len(selection))))
                        for n, group in enumerate(selection):
                            plt.plot(range(len(angle[split][group])), angle[split][group], linewidth=1, color=next(color), label=group, alpha=0.6)
                        plt.xlabel("Frame")
                        plt.ylabel("Angle")
                        plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
                        plt.title(f'Lipid Angles P to {c_atom_name} {"_".join(selection)} {split} Membrane', fontdict={'fontsize':10})
                        plt.savefig(f'{self.experiment_name}/angles_framewise_P_to_{c_atom_name}_{"_".join(selection)}_{split}.svg', dpi = 1000, bbox_inches="tight")
                        plt.close()
            else:
                if split == False:
                    plt.plot(range(len(angle["Combined"])), angle["Combined"], linewidth=1, color='crimson', label="Combined")
                    plt.xlabel("Frame")
                    plt.ylabel("Angle")
                    plt.title(f'Lipid Angles P to {c_atom_name} combined', fontdict={'fontsize':10})
                    plt.savefig(f'{self.experiment_name}/angles_framewise_combined_P_to_{c_atom_name}.svg', dpi = 1000, bbox_inches="tight")
                    plt.close()
                else:
                    for split in ["Upper", "Lower"]:
                        plt.plot(range(len(angle[split]["Combined"])), angle[split]["Combined"], linewidth=1, color='crimson', label="Combined")
                        plt.xlabel("Frame")
                        plt.ylabel("Angle")
                        plt.title(f'Lipid Angles P to {c_atom_name} combined {split} Membrane', fontdict={'fontsize':10})
                        plt.savefig(f'{self.experiment_name}/angles_framewise_combined_P_to_{c_atom_name}_{split}.svg', dpi = 1000, bbox_inches="tight")
                        plt.close()
        except:
            logger.error(traceback.format_exc(), "at line", format(sys.exc_info()[-1].tb_lineno))
            #sys.exit(0)
        return

def version_control(version):
    try:
        if version not in ['v2020.4-fda2.10.2', 'v2020.3-fda2.10.1', 'v2020.3-fda2.10', 'v2020-fda2.10', 'v2019.3-fda2.9.1', 'v2018.7-fda2.9.1']:
            raise ValueError("Please enter valid gromac version from list ['v2020.4-fda2.10.2', 'v2020.3-fda2.10.1', 'v2020.3-fda2.10', 'v2020-fda2.10', 'v2019.3-fda2.9.1', 'v2018.7-fda2.9.1']")
        if not os.path.isdir(version):
            logger.info(f"Installing GROMACS FDA version {version}")
            subprocess.call(["sudo", "apt-get", "install", "libboost-all-dev", "libfftw3-3", "libfftw3-dev"])
            subprocess.run(["mkdir", "-p", f"{version}/"])
            subprocess.run(["git", "clone", f"https://github.com/HITS-MBM/gromacs-fda.git", "-b", f"{version}"])
            subprocess.run(["mkdir", "-p", f"gromacs-fda/build"])
            Installation_directory = os.getcwd()+f"/{version}"
            os.chdir(f"gromacs-fda/build")
            subprocess.check_call(["cmake", f"-DCMAKE_INSTALL_PREFIX={Installation_directory}", "-DGMX_BUILD_FDA=ON", "-DGMX_DEFAULT_SUFFIX=OFF", "-DGMX_BINARY_SUFFIX=_fda", "-DGMX_SIMD=NONE", "-DGMX_BUILD_UNITTESTS=ON", "-DGMX_GPU=OFF", ".."])
            subprocess.check_call(["make", "-j", "1"])
            subprocess.check_call(["make", "check"])
            subprocess.check_call(["make", "install"])
            os.chdir("../../")
            subprocess.run(["rm", "-rf", "gromacs-fda"])
            logger.info(f"Installed GROMACS FDA version {version}")
    except:
        logger.error("Please ensure that you have the prerequisites for installing GROMACS-FDA. Please open a GItHub issue if you cannot get it to work")
        logger.error(traceback.format_exc())
        subprocess.run(["rm", "-rf", f"{version}/"])
        subprocess.run(["rm", "-rf", "gromacs-fda/"])
        #sys.exit(0)
    return

def create_pfi(path, group_1, group_2, force_type="all", onepair="summed", atombased="pairwise_forces_vector", residuebased="no", ignore_missing_potentials="yes"):
    '''
    Function to create a .pfi file
    Arguments : 
                path : Path where to create the .pfi file
                group_1 : Residue Group 1
                group_2 : Residue Group 2
                force_type : Force type to be calculated (default : all)
                onepair : Forces summation (default : summed)
                atombased : Force type (default : pairwise_forces_vector) 
                residuebased : Are foreces are residue based (default : no)
                ignore_missing_potentials : Missing potential from files (default : yes)
    Returns : 
                None
    Outputs : 
                Creates a .pfi file with parameters to be used my gromacs-fda
    '''
    try:
        start_time = time.time()

        if group_1==None or group_2==None:
            raise ValueError('Enter enter names for group1 or group2')
        with open(path, 'w') as fp:
            fp.write("onepair = "                   +onepair+"\n")
            fp.write("group1 = "                    +group_1+"\n")
            fp.write("group2 = "                    +group_2+"\n")
            fp.write("atombased = "                 +atombased+"\n")
            fp.write("residuebased = "              +residuebased+"\n")
            fp.write("type = "                      +force_type+"\n")
            fp.write("ignore_missing_potentials="   +ignore_missing_potentials+"\n")

        end_time = time.time()

        logger.info("{} file created in {} seconds ".format(path, (end_time-start_time)))

    except:
        logger.error(traceback.format_exc())
        #sys.exit(0)
    return

def run_gmx_fda(fda_install_path, trr_filename, tpr_filename, pfi_filename, pfa_filename, index_file, threads=1):
    try:
        gromacs_start = time.time()
        subprocess.check_call([f"{fda_install_path}/./gmx_fda", "mdrun", "-rerun", trr_filename, "-s", tpr_filename, "-pfi", pfi_filename, "-nt", str(threads), "-pfa", pfa_filename, "-pfn", index_file])
        gromacs_end = time.time()
        logger.info("GMX RUN completed in {} seconds".format((gromacs_end-gromacs_start)))
        return
    except:
        logger.error("Please ensure that your files are compatible with the GROMACS FDA version and are entered correctly!!")
        #sys.exit(0)
    return

def parse_gro(filename):
    '''
    Function to parse a .gro file to numbers of atoms, atoms details and box vector values list.
    Arguments : 
                filename : Path to the .gro file
    Returns : 
                num_atoms : int : Number of atoms
                atom_info_list : list of dictionary : A list of dictionaries containing the properties of each atom
                                        for example one element of atom_info_list:-
                                        {'Residue_Number': 1,
                                        'Residue_Name': 'CHL1',
                                        'Atom_Name': 'C3',
                                        'Atom_Number': 1,
                                        'X_Coordinate': 6.332,
                                        'Y_Coordinate': 5.87,
                                        'Z_Coordinate': 4.784,
                                        'X_Velocity': 0.4755,
                                        'Y_Velocity': 0.637,
                                        'Z_Velocity': 0.1449}
                box_vectors : list of int : A list of the box vector values
    '''
    try:
        gro_parse_start = time.time()

        with open(filename, 'r') as fp:
            title = fp.readline()
            num_atoms = int(fp.readline())
            atom_info_list = []
            gro_current_line = ''
            while True:
                previous_line = copy.deepcopy(gro_current_line)
                gro_current_line = fp.readline()
                if len(gro_current_line) == 0:
                    break
                atom_info = {"Residue_Number" : None, "Residue_Name" : None, "Atom_Name" : None, "Atom_Number" : None, "X_Coordinate" : None, "Y_Coordinate" : None, "Z_Coordinate" : None, "X_Velocity" : None, "Y_Velocity" : None, "Z_Velocity" : None}
                try:
                    gro_current_line = gro_current_line.split()
                    if len(gro_current_line) == 9:
                        Residue_Name = re.search(r"[a-zA-Z]", gro_current_line[0])
                        Residue_Name = Residue_Name.start()
                        atom_info["Residue_Number"] = int(gro_current_line[0][:Residue_Name])
                        atom_info["Residue_Name"] = str(gro_current_line[0][Residue_Name:])
                        atom_info["Atom_Name"] =    str(gro_current_line[1])
                        atom_info["Atom_Number"] =  int(gro_current_line[2])
                        current_len = len(gro_current_line[2])
                        atom_info["X_Coordinate"] = float(gro_current_line[3])
                        atom_info["Y_Coordinate"] = float(gro_current_line[4])
                        atom_info["Z_Coordinate"] = float(gro_current_line[5])
                        atom_info["X_Velocity"] =   float(gro_current_line[6])
                        atom_info["Y_Velocity"] =   float(gro_current_line[7])
                        atom_info["Z_Velocity"] =   float(gro_current_line[8])
                        atom_info_list.append(atom_info)
                    else:
                        Residue_Name = re.search(r"[a-zA-Z]", gro_current_line[0])
                        Residue_Name = Residue_Name.start()
                        atom_info["Residue_Number"] = int(gro_current_line[0][:Residue_Name])
                        atom_info["Residue_Name"] = str(gro_current_line[0][Residue_Name:])
                        atom_info["Atom_Name"] =    str(gro_current_line[1][:current_len])
                        atom_info["Atom_Number"] =  int(gro_current_line[1][current_len:])
                        current_len = len(gro_current_line[:current_len])
                        atom_info["X_Coordinate"] = float(gro_current_line[2])
                        atom_info["Y_Coordinate"] = float(gro_current_line[3])
                        atom_info["Z_Coordinate"] = float(gro_current_line[4])
                        atom_info["X_Velocity"] =   float(gro_current_line[5])
                        atom_info["Y_Velocity"] =   float(gro_current_line[6])
                        atom_info["Z_Velocity"] =   float(gro_current_line[7])
                        atom_info_list.append(atom_info)
                except:
                    break
            box_vectors = list(map(float,previous_line[-1].split()))
        gro_parse_end = time.time()
        logger.info("{} file parsed. with {} atoms in {} seconds".format(filename, num_atoms, (gro_parse_end-gro_parse_start)))

        return num_atoms, atom_info_list, box_vectors
    
    except:
        logger.error(traceback.format_exc())
        #sys.exit(0)
    return

def create_summed_pfa(pfa_filename, num_atoms, summed_pfa_filename=None, framewise=False):
    '''
    Function to parse a .pfa file according to framewise or all frame at once.
    Arguments : 
                pfa_filename : Path to the .pfa file
                num_atoms : Numbers of atoms
                summed_pfa_filename : Path to new summned pfa that will be generated from this function (default : summed_pfa.pfa if framewise = false otherwise framewised_summed_pfa.pfa)
                framewise : how to sum all the force (default : false)
    Returns : 
                None
    Outputs : 
                Creates a file with summed up pairwise force values either framewise or combined
    '''
    try:
        start_time=time.time()

        if summed_pfa_filename==None:
            raise ValueError("Pleae enter name for new PFA file !!!")

        #If framewise force summation is not selected
        if not framewise:               

            #Intitialise forces on atom with 0                         
            Forces_on_Atoms = []
            for i in range(0, num_atoms+1):
                Forces_on_Atoms.append(Point(0, 0, 0))
            num_frames = 0
            with open(pfa_filename, "r") as fp:
                pfa_current_line = fp.readline()
                while True:
                    pfa_current_line = fp.readline()
                    #if blank line then end of file and end the read operation
                    if len(pfa_current_line) == 0:
                        break
                    #if new frame is detected then increment frame number
                    elif pfa_current_line[:5] == "frame" or pfa_current_line[:5] == "force":
                        num_frames += 1
                        continue
                    #summation of forces from all frames are done if
                    # force is applied by atom then forces are subtracted
                    # otherwise added to previous force sum
                    else:
                        pfa_current_line = pfa_current_line.split()
                        force_applied_by = int(pfa_current_line[1])
                        force_recieved_by = int(pfa_current_line[0])
                        force_magnitude = Point(pfa_current_line[2], pfa_current_line[3], pfa_current_line[4])
                        Forces_on_Atoms[force_recieved_by] += force_magnitude
                        Forces_on_Atoms[force_applied_by] += force_magnitude.negate()
            #all forces will be written in new pfa file
            with open(summed_pfa_filename, "w") as fp:
                fp.write("frame " + str(num_frames) + "\n")
                for i in range(0, len(Forces_on_Atoms)):
                    fp.write(str(i+1) + "\t" + str(Forces_on_Atoms[i].x) + "\t" + str(Forces_on_Atoms[i].y) + "\t" + str(Forces_on_Atoms[i].z) + "\n")

        #If framewise force summation is selected
        else:
            open(summed_pfa_filename, "w").close()
            #Intitialise forces on atom with 0  
            Forces_on_Atoms_Holder = []
            for i in range(0, num_atoms+1):
                Forces_on_Atoms_Holder.append(Point(0, 0, 0))
            Forces_on_Atoms = Forces_on_Atoms_Holder

            with open(pfa_filename, "r") as fp:
                pfa_current_line = fp.readline()
                while True:
                    pfa_current_line = fp.readline()
                    #if blank line then end of file and end the read operation
                    if len(pfa_current_line) == 0:
                        break
                    #if new frame is detected then summation of forces from previous frame is written in file
                    elif pfa_current_line[:5] == "frame" or pfa_current_line[:5] == "force":
                        frame_number = int(pfa_current_line[6:])
                        if frame_number >= 0:
                            with open(summed_pfa_filename, "a") as fp_temp:
                                fp_temp.write("frame " + str(frame_number) + "\n")
                                for i in range(0, len(Forces_on_Atoms)):
                                    fp_temp.write(str(i+1) + "\t" + str(Forces_on_Atoms[i].x) + "\t" + str(Forces_on_Atoms[i].y) + "\t" + str(Forces_on_Atoms[i].z) + "\n")
                            Forces_on_Atoms = []
                            for i in range(0, num_atoms+1):
                                Forces_on_Atoms.append(Point(0, 0, 0))

                    #summation of forces from one frames are done if
                    # force is applied by atom then forces are subtracted
                    # otherwise added to previous force sum                            
                    else:
                        pfa_current_line = pfa_current_line.split()
                        force_applied_by = int(pfa_current_line[1])
                        force_recieved_by = int(pfa_current_line[0])
                        force_magnitude = Point(pfa_current_line[2], pfa_current_line[3], pfa_current_line[4])

                        Forces_on_Atoms[force_recieved_by] += force_magnitude
                        Forces_on_Atoms[force_applied_by] += force_magnitude.negate()
        
        end_time=time.time()
        
        logger.info("Parsed PFA file {} created with {} atoms in {} seconds".format(summed_pfa_filename, num_atoms, (end_time-start_time)))
    
    except:
        logger.error(traceback.format_exc())
        #sys.exit(0)
    return
    
def parse_summed_pfa(summed_pfa_file, atom_info, residue_list):
    '''
    Function to parse the summed .pfa file
    Arguments : 
                summed_pfa_file : Path to the summed .pfa file
                atom_info : vector of class 'AtomForced'
                residue_list : List of residues to be calculated (example : ["POPS","POPC"])
    Returns : 
                Dictionary of input residues containing a list of their atoms
                for example if the framewise summation pfa is selected :-
                if ["POPS","POPC"] are selected 
                [1]['POPS'][AtomForced class object, .... ]
                   ['POPC'][AtomForced class object, .... ]
                [2]['POPS'][AtomForced class object, .... ]
                   ['POPC'][AtomForced class object, .... ]
                   :
                   :
                [100]['POPS'][AtomForced class object, .... ]
                     ['POPC'][AtomForced class object, .... ]
                
                if framewise is selected :-
                last frame will content all the details if the last frame is 100 then,
                [100]['POPS'][AtomForced class object, .... ]
                     ['POPC'][AtomForced class object, .... ]
    '''
    try:
        start_time=time.time()
        current_frame = 0
        AllResidueGroupsFramewise = {}
        with open(summed_pfa_file, 'r') as fp:
            restart = True
            while restart:
                for i in atom_info:
                    # read lines from summation generated pfa file
                    summed_pfa_current_line = fp.readline().split()
                    #if blank line then end of file and end the read operation
                    if len(summed_pfa_current_line) == 0:
                        restart = False
                        break
                    #if new frame is detected then summation of forces from previous frame is written in file
                    elif summed_pfa_current_line[0] == "frame":
                        current_frame = int(summed_pfa_current_line[1])
                        AllResidueGroupsFramewise[current_frame] = {}
                        for residueName in residue_list:
                            AllResidueGroupsFramewise[current_frame][str(residueName)] = []
                        break
                    #summation of forces from one frames are done if
                    # force is applied by atom then forces are subtracted
                    # otherwise added to previous force sum           
                    else:
                        if i["Residue_Name"] in residue_list:
                            if int(summed_pfa_current_line[0]) == i["Atom_Number"]:
                                f_x = summed_pfa_current_line[1]
                                f_y = summed_pfa_current_line[2]
                                f_z = summed_pfa_current_line[3]
                                forceAtom = AtomForced(
                                            i["Residue_Number"],
                                            i["Residue_Name"],
                                            i["Atom_Name"],
                                            i["Atom_Number"],
                                            i["X_Coordinate"],
                                            i["Y_Coordinate"],
                                            i["Z_Coordinate"],
                                            f_x,
                                            f_y,
                                            f_z
                                        )
                                AllResidueGroupsFramewise[current_frame][i["Residue_Name"]].append(forceAtom)
        
        end_time = time.time()
        logger.info("Summed PFA file parsed in {} seconds".format((end_time-start_time)))

        return AllResidueGroupsFramewise

    except:
        logger.error(traceback.format_exc())
        #sys.exit(0)
    return

def bfactor_pdb(atom_dict, pdb_filename, bfactor_pdb_filename, mode = "combined"):
    '''
    Create a .pdb file with the same atoms as the .gro file but with the bfactor
    Arguments : 
                summed_pfa_file : Path to the summed .pfa file
                gro_file : Path to the .gro file
                residue_list : List of residues to be calculated (example : ["POPS","POPC"])
                mode : "combined" (atomistic or combined)
    Returns : 
                None
    Outputs : 
                .pdb file with bfactor values loaded
    '''
    try:
        start_time = time.time()
        bfactor = {}
        atom_numbers = {}
        min_force = np.finfo(float).max
        max_force = np.finfo(float).min
        num_frames = len(atom_dict)
        num_frames_orignal = list(atom_dict.keys())[0]  if num_frames == 1 else num_frames
        numKeys = len(atom_dict[list(atom_dict.keys())[0]])
        allKeys = [key for key in atom_dict[list(atom_dict.keys())[0]]]
        if mode == "combined":
            for i in range(numKeys):
                if num_frames == num_frames_orignal:
                    forced_atomGroup = [atom_dict[window][allKeys[i]] for window in range(0, num_frames)]
                else:
                    forced_atomGroup = [atom_dict[num_frames_orignal][allKeys[i]]]
                [specific_frame.sort(key= lambda x: x.Coords.z) for specific_frame in forced_atomGroup]
                forced_atomGroup = np.asarray(forced_atomGroup)
                if num_frames == num_frames_orignal:
                    forces_averaged = []
                    x=np.zeros(forced_atomGroup.shape)
                    for n1, ii in enumerate(forced_atomGroup):
                        for n2, j in enumerate(ii):
                            x[n1][n2] = forced_atomGroup[n1][n2].Force.x
                    y=np.zeros(forced_atomGroup.shape)
                    for n1, ii in enumerate(forced_atomGroup):
                        for n2, j in enumerate(ii):
                            y[n1][n2] = forced_atomGroup[n1][n2].Force.y
                    z=np.zeros(forced_atomGroup.shape)
                    for n1, ii in enumerate(forced_atomGroup):
                        for n2, j in enumerate(ii):
                            z[n1][n2] = forced_atomGroup[n1][n2].Force.z
                    forces_averaged = np.sqrt(np.square(np.sum(x, axis = 0)) + np.square(np.sum(y, axis = 0)) + np.square(np.sum(z, axis = 0)))
                else:
                    forces_averaged = [sum(map(lambda x: x.Force.mod(), atoms))/(num_frames_orignal -1) for atoms in zip(*forced_atomGroup)]
                bfactor[allKeys[i]] = sum(forces_averaged)/len(forces_averaged)
                atom_numbers[allKeys[i]] = [atom.AtomNumber for atom in forced_atomGroup[0]]
        else:
            for i in range(numKeys):
                if num_frames == num_frames_orignal:
                    forced_atomGroup = [atom_dict[window][allKeys[i]] for window in range(0, num_frames)]
                else:
                    forced_atomGroup = [atom_dict[num_frames_orignal][allKeys[i]]]
                [specific_frame.sort(key= lambda x: x.Coords.z) for specific_frame in forced_atomGroup]
                forced_atomGroup = np.asarray(forced_atomGroup)
                if num_frames == num_frames_orignal:
                    forces_averaged = []
                    x=np.zeros(forced_atomGroup.shape)
                    for n1, ii in enumerate(forced_atomGroup):
                        for n2, j in enumerate(ii):
                            x[n1][n2] = forced_atomGroup[n1][n2].Force.x
                    y=np.zeros(forced_atomGroup.shape)
                    for n1, ii in enumerate(forced_atomGroup):
                        for n2, j in enumerate(ii):
                            y[n1][n2] = forced_atomGroup[n1][n2].Force.y
                    z=np.zeros(forced_atomGroup.shape)
                    for n1, ii in enumerate(forced_atomGroup):
                        for n2, j in enumerate(ii):
                            z[n1][n2] = forced_atomGroup[n1][n2].Force.z
                    forces_averaged = np.sqrt(np.square(np.sum(x, axis = 0)) + np.square(np.sum(y, axis = 0)) + np.square(np.sum(z, axis = 0)))
                else:
                    forces_averaged = [sum(map(lambda x: x.Force.mod(), atoms))/(num_frames_orignal -1) for atoms in zip(*forced_atomGroup)]
                bfactor[allKeys[i]] = {}
                atom_numbers[allKeys[i]] = [atom.AtomNumber for atom in forced_atomGroup[0]]
                for n, atom_num in enumerate(atom_numbers[allKeys[i]]):
                    bfactor[allKeys[i]][atom_num] = forces_averaged[n]
        min_force = np.min(forces_averaged) if np.min(forces_averaged) < min_force else min_force
        max_force = np.max(forces_averaged) if np.max(forces_averaged) > max_force else max_force

        with open(pdb_filename, 'r') as fp_read:
            with open(bfactor_pdb_filename, 'w') as fp_write:
                while True:
                    line = fp_read.readline()
                    if line[:6] == "ENDMDL":
                        break
                    if line[0:4] != "ATOM":
                        fp_write.write(str(line))
                    else:
                        found = 0
                        atom_number = int(line[6:11])
                        for atom in atom_numbers:
                            if atom_number in atom_numbers[atom]:
                                fp_write.write(line[:-17] + str(round(((bfactor[atom]- min_force) / (max_force - min_force) * 9.99), 2))+ "\n") if mode == "combined" else fp_write.write(line[:-17] + str(round(((bfactor[atom][atom_number] - min_force) / (max_force - min_force) * 9.99), 2))+ "\n")
                                fp_read.readline()
                                found = True
                                break
                        if not found:
                            fp_write.write(str(line))
        end_time = time.time()
        logger.info("PDB with BFactor values created in {} seconds".format((end_time-start_time)))
        
    except:
        logger.error(traceback.format_exc(),  format(sys.exc_info()[-1].tb_lineno))
        #sys.exit(0)
    return
      
def create_average_residue_graph(atom_dict, plot_name="average"):
    try:
        num_frames = len(atom_dict)
        num_frames_orignal = list(atom_dict.keys())[0]  if num_frames == 1 else num_frames
        numKeys = len(atom_dict[list(atom_dict.keys())[0]])
        allKeys = [key for key in atom_dict[list(atom_dict.keys())[0]]]
        fig, axs = plt.subplots(numKeys)
        for i in range(numKeys):
            forced_atomGroup = [atom_dict[window][allKeys[i]] for window in range(0, num_frames)] if num_frames == num_frames_orignal else [atom_dict[num_frames_orignal][allKeys[i]]]
            [specific_frame.sort(key= lambda x: x.Coords.z) for specific_frame in forced_atomGroup]
            if num_frames == num_frames_orignal:
                forces_averaged = []
                for atoms in zip(*forced_atomGroup):
                    average_at_coordinate = Point(0,0,0)
                    for atom in atoms:
                        average_at_coordinate += atom.Force
                    forces_averaged.append(average_at_coordinate.mod()/len(atoms))
            else:
                forces_averaged = [sum(map(lambda x: x.Force.mod(), atoms))/(num_frames_orignal -1) for atoms in zip(*forced_atomGroup)]
            distances = [atom.Coords.z for atom in forced_atomGroup[0]]
            axs[i].plot(distances, forces_averaged)
            axs[i].title.set_text(str(allKeys[i]))
        axs[i].set_xlabel("Z - Coordinate", fontsize=10)
        fig.supylabel("Force (N)", fontsize=10)
        fig.tight_layout()
        fig.savefig(f"{plot_name}_{allKeys}.svg", dpi = 1000)
        plt.close()

        logger.info(f"Average Force plots created and saved")
    except:
        logger.error(traceback.format_exc())
        #sys.exit(0)

def create_specific_frame_graph(atom_dict, specific_frame, plot_name="specific_frame"):
    try:
        frame  = specific_frame
        numKeys = len(atom_dict[frame])
        allKeys = [key for key in atom_dict[frame]]
        for i in range(numKeys):
            forced_atomGroup = atom_dict[frame][allKeys[i]]
            forced_atomGroup.sort(key= lambda x: x.Coords.z)
            forces = [atom.Force.mod() for atom in forced_atomGroup]
            distances = [atom.Coords.z for atom in forced_atomGroup]
            plt.plot(distances, forces)
            plt.title(str(allKeys[i])+"_"+str(frame))
            plt.xlabel("Z - Coordinate", fontsize=10, labelpad=5)
            plt.ylabel("Force (N)", fontsize=10, labelpad=5)
            plt.savefig(f"{plot_name}_{allKeys[i]}_{frame}.svg", dpi = 1000)
            plt.close()
        logger.info(f"Force plots created and saved for frame - {specific_frame}")
    except:
        logger.error(traceback.format_exc())
        #sys.exit(0)

def create_moving_window_graph(atom_dict, moving_window, plot_name="moving_window"):
    try:
        num_frames = len(atom_dict) - 1
        numKeys = len(atom_dict[list(atom_dict.keys())[0]])
        allKeys = [key for key in atom_dict[list(atom_dict.keys())[0]]]
        for i in range(numKeys):
            for frame in range(0, num_frames, moving_window):
                forced_atomGroup = [atom_dict[window][allKeys[i]] for window in range(frame, frame + moving_window) if window <= num_frames]
                [specific_frame.sort(key= lambda x: x.Coords.z) for specific_frame in forced_atomGroup]
                forces_averaged = []
                for atoms in zip(*forced_atomGroup):
                    average_at_coordinate = Point(0,0,0)
                    for atom in atoms:
                        average_at_coordinate += atom.Force
                    forces_averaged.append(average_at_coordinate.mod()/len(atoms))
                distances = [atom.Coords.z for atom in forced_atomGroup[0]]
                plt.plot(distances, forces_averaged)
                plt.xlabel("Z - Coordinate", fontsize=10, labelpad=5)
                plt.ylabel("Force (N)", fontsize=10, labelpad=5)
                plt.title(f'{allKeys[i]}_{frame}_to_{frame+moving_window}')
                plt.savefig(f"{plot_name}_{allKeys[i]}_{frame}_to_{frame+moving_window}.svg", dpi = 1000)
                plt.close()
    except:
        logger.error(traceback.format_exc())
        #sys.exit(0)

def plot_curvature(universe, atom_dict, selection = "", num_x_bins = 10, num_y_bins = 10, split = False, gangle = None, plot_name="curvature_combined"):
    num_frames = len(atom_dict)
    num_frames_orignal = list(atom_dict.keys())[0]  if num_frames == 1 else num_frames
    numKeys = len(atom_dict[list(atom_dict.keys())[0]])
    allKeys = [key for key in atom_dict[list(atom_dict.keys())[0]]]
    binned_data = []
    binned_data_split = {"Upper": [], "Lower": []}
    upper_range = []
    lower_range = []
    for i in range(numKeys):
        forced_atomGroup = [atom_dict[window][allKeys[i]] for window in range(0, num_frames)] if num_frames == num_frames_orignal else [atom_dict[num_frames_orignal][allKeys[i]]]
        [specific_frame.sort(key= lambda x: x.Coords.z) for specific_frame in forced_atomGroup]
        forced_atomGroup = np.array(forced_atomGroup).flatten()
        if split:
            mean_z = mean([x.Coords.z for x in forced_atomGroup])
            upper_membrane = [atom for atom in forced_atomGroup if atom.Coords.z > mean_z]
            lower_membrane = [atom for atom in forced_atomGroup if atom.Coords.z < mean_z]
            temp_range = (int(min(upper_membrane, key = lambda x: x.Coords.z).ResNum), int(max(upper_membrane, key = lambda x: x.Coords.z).ResNum))
            upper_range.append((min(temp_range), max(temp_range)))
            temp_range = (int(min(lower_membrane, key = lambda x: x.Coords.z).ResNum), int(max(lower_membrane, key = lambda x: x.Coords.z).ResNum))
            lower_range.append((min(temp_range), max(temp_range)))
        if not split:
            x_min = math.floor(int(min(forced_atomGroup, key = lambda atom: atom.Coords.x).Coords.x))
            x_max = math.ceil(int(max(forced_atomGroup, key = lambda atom: atom.Coords.x).Coords.x))
            y_min = math.floor(int(min(forced_atomGroup, key = lambda atom: atom.Coords.y).Coords.y))
            y_max = math.ceil(int(max(forced_atomGroup, key = lambda atom: atom.Coords.y).Coords.y))
            x_coordinates = [float(atom.Coords.x) for atom in forced_atomGroup]
            y_coordinates = [float(atom.Coords.y) for atom in forced_atomGroup]
            x_bins = np.arange(x_min, x_max, (x_max - x_min)/num_x_bins)
            y_bins = np.arange(y_min, y_max, (y_max - y_min)/num_y_bins)
            atom_x_bin = np.digitize(x_coordinates, x_bins)
            atom_y_bin = np.digitize(y_coordinates, y_bins)
            temp_binned_data = np.zeros((num_x_bins, num_y_bins))
            for n,i in enumerate(forced_atomGroup):
                temp_binned_data[atom_x_bin[n] - 1][atom_y_bin[n] - 1] = i.Force.mod()/(num_frames_orignal -1)
            binned_data.append(temp_binned_data)
        else:
            for key in binned_data_split:
                forced_atomGroup = upper_membrane if key == "Upper" else lower_membrane
                x_min = math.floor(int(min(forced_atomGroup, key = lambda atom: atom.Coords.x).Coords.x))
                x_max = math.ceil(int(max(forced_atomGroup, key = lambda atom: atom.Coords.x).Coords.x))
                y_min = math.floor(int(min(forced_atomGroup, key = lambda atom: atom.Coords.y).Coords.y))
                y_max = math.ceil(int(max(forced_atomGroup, key = lambda atom: atom.Coords.y).Coords.y))
                x_coordinates = [float(atom.Coords.x) for atom in forced_atomGroup]
                y_coordinates = [float(atom.Coords.y) for atom in forced_atomGroup]
                x_bins = np.arange(x_min, x_max, (x_max - x_min)/num_x_bins)
                y_bins = np.arange(y_min, y_max, (y_max - y_min)/num_y_bins)
                atom_x_bin = np.digitize(x_coordinates, x_bins)
                atom_y_bin = np.digitize(y_coordinates, y_bins)
                temp_binned_data = np.zeros((num_x_bins, num_y_bins))
                for n,i in enumerate(forced_atomGroup):
                    temp_binned_data[atom_x_bin[n] - 1][atom_y_bin[n] - 1] = i.Force.mod()/(num_frames_orignal -1)
                binned_data_split[key].append(temp_binned_data)
    selected_atoms = ["P"]
    curavature_plots = ["Z_Surface", "Mean_Curvature", "Gaussian_Curvature"]
    if not split:
        selected_residues = {}
        axis_removed = {}
        for i in range(len(selected_atoms)):
            selected_residues[selected_atoms[i]] = {}
            curvature_selected = MembraneCurvature(universe, select = f"type {selected_atoms[i]}", n_x_bins = num_x_bins, n_y_bins = num_y_bins).run() #, select = 'resid 0-1023', n_x_bins=12, n_y_bins=12
            selected_residues[selected_atoms[i]]["Z_Surface"] = curvature_selected.results.average_z_surface
            selected_residues[selected_atoms[i]]["Mean_Curvature"] = curvature_selected.results.average_mean
            selected_residues[selected_atoms[i]]["Gaussian_Curvature"] = curvature_selected.results.average_gaussian
        for i in range(len(selected_atoms)):
            axis_removed[selected_atoms[i]] = {} #max,min
            axis_removed[selected_atoms[i]]["Z_Surface"] = [np.unravel_index(np.array(selected_residues[selected_atoms[i]]["Z_Surface"]).argmax(), np.array(curvature_selected.results.average_z_surface).shape)]
            axis_removed[selected_atoms[i]]["Mean_Curvature"] = [np.unravel_index(np.array(selected_residues[selected_atoms[i]]["Mean_Curvature"]).argmax(), np.array(curvature_selected.results.average_mean).shape)]
            axis_removed[selected_atoms[i]]["Gaussian_Curvature"] = [np.unravel_index(np.array(selected_residues[selected_atoms[i]]["Gaussian_Curvature"]).argmax(), np.array(curvature_selected.results.average_gaussian).shape)]
            axis_removed[selected_atoms[i]]["Z_Surface"].append(np.unravel_index(np.array(selected_residues[selected_atoms[i]]["Z_Surface"]).argmin(), np.array(curvature_selected.results.average_z_surface).shape))
            axis_removed[selected_atoms[i]]["Mean_Curvature"].append(np.unravel_index(np.array(selected_residues[selected_atoms[i]]["Mean_Curvature"]).argmin(), np.array(curvature_selected.results.average_mean).shape))
            axis_removed[selected_atoms[i]]["Gaussian_Curvature"].append(np.unravel_index(np.array(selected_residues[selected_atoms[i]]["Gaussian_Curvature"]).argmin(), np.array(curvature_selected.results.average_gaussian).shape))
            
    else:
        selected_residues_split = {"Upper": {}, "Lower": {}}
        axis_removed_split = {"Upper": {}, "Lower": {}}
        for split_key in selected_residues_split:
            lower_limit = min(lower_range)[0] if split_key == "Lower" else min(upper_range)[0]
            upper_limit = max(lower_range)[1] if split_key == "Lower" else max(upper_range)[1]
            universe_selected = universe.select_atoms(f"prop z < {universe.select_atoms('type P').center_of_mass()[2]} and type P") if split_key == "Lower" else universe.select_atoms(f"prop z > {universe.select_atoms('type P').center_of_mass()[2]} and type P")
            for i in range(len(selected_atoms)):
                selected_residues_split[split_key][selected_atoms[i]] = {}
                curvature_selected = MembraneCurvature(universe_selected, n_x_bins = num_x_bins, n_y_bins = num_y_bins).run()#, select = f"resid {lower_limit}:{upper_limit} and name {selected_atoms[i]}"
                selected_residues_split[split_key][selected_atoms[i]]["Z_Surface"] = curvature_selected.results.average_z_surface
                selected_residues_split[split_key][selected_atoms[i]]["Mean_Curvature"] = curvature_selected.results.average_mean
                selected_residues_split[split_key][selected_atoms[i]]["Gaussian_Curvature"] = curvature_selected.results.average_gaussian
            for i in range(len(selected_atoms)):
                axis_removed_split[split_key][selected_atoms[i]] = {} #max,min
                axis_removed_split[split_key][selected_atoms[i]]["Z_Surface"] = [np.unravel_index(np.array(selected_residues_split[split_key][selected_atoms[i]]["Z_Surface"]).argmax(), np.array(curvature_selected.results.average_z_surface).shape)]
                axis_removed_split[split_key][selected_atoms[i]]["Mean_Curvature"] = [np.unravel_index(np.array(selected_residues_split[split_key][selected_atoms[i]]["Mean_Curvature"]).argmax(), np.array(curvature_selected.results.average_mean).shape)]
                axis_removed_split[split_key][selected_atoms[i]]["Gaussian_Curvature"] = [np.unravel_index(np.array(selected_residues_split[split_key][selected_atoms[i]]["Gaussian_Curvature"]).argmax(), np.array(curvature_selected.results.average_gaussian).shape)]
                axis_removed_split[split_key][selected_atoms[i]]["Z_Surface"].append(np.unravel_index(np.array(selected_residues_split[split_key][selected_atoms[i]]["Z_Surface"]).argmin(), np.array(curvature_selected.results.average_z_surface).shape))
                axis_removed_split[split_key][selected_atoms[i]]["Mean_Curvature"].append(np.unravel_index(np.array(selected_residues_split[split_key][selected_atoms[i]]["Mean_Curvature"]).argmin(), np.array(curvature_selected.results.average_mean).shape))
                axis_removed_split[split_key][selected_atoms[i]]["Gaussian_Curvature"].append(np.unravel_index(np.array(selected_residues_split[split_key][selected_atoms[i]]["Gaussian_Curvature"]).argmin(), np.array(curvature_selected.results.average_gaussian).shape))
    rows = []
    if not split:
        for i in range(len(selected_atoms)):
            fig, ax = plt.subplots(ncols=3, nrows=3, constrained_layout=True)
            for n, j in enumerate(curavature_plots):
                num = 0
                im = ax[num][n].contourf(selected_residues[selected_atoms[i]][j], cmap='PuBuGn', origin='lower')
                ax[num][n].contour(selected_residues[selected_atoms[i]][j], cmap = 'PuBuGn', origin='lower', levels=10)
                ax[num][n].set_aspect('equal')
                cbar = plt.colorbar(im, ticks=[np.nanmin(selected_residues[selected_atoms[i]][j]), np.nanmax(selected_residues[selected_atoms[i]][j])], orientation='horizontal', ax = ax[num][n], shrink=0.7)
                cbar.ax.tick_params(labelsize=3, width=0.5)
                ax[num][n].xaxis.set_tick_params(labelsize=4)
                ax[num][n].yaxis.set_tick_params(labelsize=4)
                ax[num][n].set_xlabel("X - Coordinate", {'fontsize':4})
                ax[num][n].set_ylabel("Y - Coordinate", {'fontsize':4})
                ax[num][n].set_title(f"Average {j}", fontdict={'fontsize':5}, pad=2)
                cbar.set_label(f"{j} (nm$^{-1}$)", fontsize=4, labelpad=2)
            
            for n, j in enumerate(curavature_plots):
                ax[1][n].plot(range(num_x_bins), selected_residues[selected_atoms[i]][j][:,axis_removed[selected_atoms[i]][j][0][1]], linewidth= 1, color='mediumseagreen', label="Max")
                ax[1][n].plot(range(num_x_bins), selected_residues[selected_atoms[i]][j][:,axis_removed[selected_atoms[i]][j][1][1]], linewidth= 1, color='purple', label="Min")
                ax[1][n].xaxis.set_tick_params(labelsize=4)
                ax[1][n].yaxis.set_tick_params(labelsize=4)
                ax[1][n].set_title(f" Y Bin {axis_removed[selected_atoms[i]][j][1]} Average {j} (nm$^{-1}$)", fontdict={'fontsize':5}, pad=2)
                if n == 0:
                    ax[1][n].legend(loc="upper right", markerscale=0.3, fontsize='xx-small')
            for lipid in range(numKeys):
                im = ax[2][lipid].contourf(binned_data[lipid], cmap='PuBuGn', origin='lower')
                ax[2][lipid].contour(binned_data[lipid], cmap = 'PuBuGn', origin='lower', levels=10)
                ax[2][lipid].set_aspect('equal')
                cbar = plt.colorbar(im, ticks=[binned_data[lipid].min(), binned_data[lipid].max()], orientation='horizontal', ax = ax[2][lipid], shrink=0.7)
                cbar.ax.tick_params(labelsize=3, width=0.5)
                ax[2][lipid].xaxis.set_tick_params(labelsize=4)
                ax[2][lipid].yaxis.set_tick_params(labelsize=4)
                ax[2][lipid].set_title(f"Forces on {allKeys[lipid]}", fontdict={'fontsize':5}, pad=2)
                ax[2][lipid].set_xlabel("X - Coordinate", {'fontsize':4})
                ax[2][lipid].set_ylabel("Y - Coordinate", {'fontsize':4})
                cbar.set_label(f"Force Z (N)", fontsize=4, labelpad=2)
            ax[2][2].remove()
            if gangle != None:
                plt.figtext(0.85, 0.2, f"Average Angle - \n {list(gangle.keys())[0]} : {gangle[list(gangle.keys())[0]]} \n{list(gangle.keys())[1]} : {gangle[list(gangle.keys())[1]]}", ha="center", fontdict={'fontsize':6})
            fig.suptitle(f'Curvature Plots for Atom {selected_atoms[i]}', fontsize=12)
            plt.savefig(f'{plot_name}_{num_x_bins}_{num_y_bins}_{selected_atoms[i]}.svg', dpi = 1000)
            plt.close()
    else:
        for position in binned_data_split:
            for i in range(len(selected_atoms)):
                fig, ax = plt.subplots(ncols=3, nrows=3, constrained_layout=True)
                for n, j in enumerate(curavature_plots):
                    num = 0
                    im = ax[num][n].contourf(selected_residues_split[position][selected_atoms[i]][j], cmap='PuBuGn', origin='lower')
                    ax[num][n].contour(selected_residues_split[position][selected_atoms[i]][j], cmap = 'PuBuGn', origin='lower', levels=10)
                    ax[num][n].set_aspect('equal')
                    cbar = plt.colorbar(im, ticks=[np.nanmin(selected_residues_split[position][selected_atoms[i]][j]), np.nanmax(selected_residues_split[position][selected_atoms[i]][j])], orientation='horizontal', ax = ax[num][n], shrink=0.7)
                    cbar.ax.tick_params(labelsize=3, width=0.5)
                    ax[num][n].xaxis.set_tick_params(labelsize=4)
                    ax[num][n].yaxis.set_tick_params(labelsize=4)
                    ax[num][n].set_xlabel("X - Coordinate", {'fontsize':4})
                    ax[num][n].set_ylabel("Y - Coordinate", {'fontsize':4})
                    ax[num][n].set_title(f"Average {j}", fontdict={'fontsize':5}, pad=2)
                    cbar.set_label(f"{j} (nm$^{-1}$)", fontsize=4, labelpad=2)
                
                for n, j in enumerate(curavature_plots):
                    ax[1][n].plot(range(num_x_bins), selected_residues_split[position][selected_atoms[i]][j][:,axis_removed_split[position][selected_atoms[i]][j][0][1]], linewidth= 1, color='mediumseagreen', label="Max")
                    ax[1][n].plot(range(num_x_bins), selected_residues_split[position][selected_atoms[i]][j][:,axis_removed_split[position][selected_atoms[i]][j][1][1]], linewidth= 1, color='purple', label="Min")
                    ax[1][n].xaxis.set_tick_params(labelsize=4)
                    ax[1][n].yaxis.set_tick_params(labelsize=4)
                    ax[1][n].set_title(f" Y Bin {axis_removed_split[position][selected_atoms[i]][j][1]} Average {j} (nm$^{-1}$)", fontdict={'fontsize':5}, pad=2)
                    if n == 0:
                        ax[1][n].legend(loc="upper right", markerscale=0.3, fontsize='xx-small')
                for lipid in range(numKeys):
                    im = ax[2][lipid].contourf(binned_data_split[position][lipid], cmap='PuBuGn', origin='lower')
                    ax[2][lipid].contour(binned_data_split[position][lipid], cmap = 'PuBuGn', origin='lower', levels=10)
                    ax[2][lipid].set_aspect('equal')
                    cbar = plt.colorbar(im, ticks=[binned_data_split[position][lipid].min(), binned_data_split[position][lipid].max()], orientation='horizontal', ax = ax[2][lipid], shrink=0.7)
                    cbar.ax.tick_params(labelsize=3, width=0.5)
                    ax[2][lipid].xaxis.set_tick_params(labelsize=4)
                    ax[2][lipid].yaxis.set_tick_params(labelsize=4)
                    ax[2][lipid].set_title(f"Forces on {allKeys[lipid]}", fontdict={'fontsize':5}, pad=2)
                    ax[2][lipid].set_xlabel("X - Coordinate", {'fontsize':4})
                    ax[2][lipid].set_ylabel("Y - Coordinate", {'fontsize':4})
                    cbar.set_label(f"Force Z (N)", fontsize=4, labelpad=2)
                ax[2][2].remove()
                if gangle != None:
                    plt.figtext(0.85, 0.2, f"Average Angle - \n {list(gangle[position].keys())[0]} : {gangle[position][list(gangle[position].keys())[0]]} \n{list(gangle[position].keys())[1]} : {gangle[position][list(gangle[position].keys())[1]]}", ha="center", fontdict={'fontsize':7})
                fig.suptitle(f'Curvature Plots for Atom {selected_atoms[i]} {position} Membrane', fontsize=12)
                plt.savefig(f'{plot_name}_{num_x_bins}_{num_y_bins}_{selected_atoms[i]}_{position}.svg', dpi = 1000)
                plt.close()

def plot_curvature_framewise(universe, atom_dict, specific_frame=None, selection = "", num_x_bins = 10, num_y_bins = 10, split = False, gangle = None, plot_name="curvature_combined_framewise"):
    num_frames = len(atom_dict) - 1
    num_frames_orignal = list(atom_dict.keys())[0]  if num_frames == 1 else num_frames
    numKeys = len(atom_dict[list(atom_dict.keys())[0]])
    allKeys = [key for key in atom_dict[list(atom_dict.keys())[0]]]

    binned_data = {}
    binned_data_split = {}
    upper_range = []
    lower_range = []
    for i in range(numKeys):
        binned_data[allKeys[i]] = []
        binned_data_split[allKeys[i]] = {"Upper": [], "Lower": []}
        forced_atomGroup = [atom_dict[specific_frame][allKeys[i]]]
        [specific_frame.sort(key= lambda x: x.Coords.z) for specific_frame in forced_atomGroup]
        forced_atomGroup = np.array(forced_atomGroup).flatten()
        if split:
            mean_z = mean([x.Coords.z for x in forced_atomGroup])
            upper_membrane = [atom for atom in forced_atomGroup if atom.Coords.z > mean_z]
            lower_membrane = [atom for atom in forced_atomGroup if atom.Coords.z < mean_z]
            temp_range = (int(min(upper_membrane, key = lambda x: x.Coords.z).ResNum), int(max(upper_membrane, key = lambda x: x.Coords.z).ResNum))
            upper_range.append((min(temp_range), max(temp_range)))
            temp_range = (int(min(lower_membrane, key = lambda x: x.Coords.z).ResNum), int(max(lower_membrane, key = lambda x: x.Coords.z).ResNum))
            lower_range.append((min(temp_range), max(temp_range)))
        if not split:
            x_min = math.floor(int(min(forced_atomGroup, key = lambda atom: atom.Coords.x).Coords.x))
            x_max = math.ceil(int(max(forced_atomGroup, key = lambda atom: atom.Coords.x).Coords.x))
            y_min = math.floor(int(min(forced_atomGroup, key = lambda atom: atom.Coords.y).Coords.y))
            y_max = math.ceil(int(max(forced_atomGroup, key = lambda atom: atom.Coords.y).Coords.y))
            x_coordinates = [float(atom.Coords.x) for atom in forced_atomGroup]
            y_coordinates = [float(atom.Coords.y) for atom in forced_atomGroup]
            x_bins = np.arange(x_min, x_max, (x_max - x_min)/num_x_bins)
            y_bins = np.arange(y_min, y_max, (y_max - y_min)/num_y_bins)
            atom_x_bin = np.digitize(x_coordinates, x_bins)
            atom_y_bin = np.digitize(y_coordinates, y_bins)
            temp_binned_data = np.zeros((num_x_bins, num_y_bins))
            for n, atom in enumerate(forced_atomGroup):
                temp_binned_data[atom_x_bin[n] - 1][atom_y_bin[n] - 1] = atom.Force.mod()/(num_frames_orignal -1)
            binned_data[allKeys[i]].append(temp_binned_data)
        else:
            for key in binned_data_split[allKeys[i]]:
                forced_atomGroup = upper_membrane if key == "Upper" else lower_membrane
                x_min = math.floor(int(min(forced_atomGroup, key = lambda atom: atom.Coords.x).Coords.x))
                x_max = math.ceil(int(max(forced_atomGroup, key = lambda atom: atom.Coords.x).Coords.x))
                y_min = math.floor(int(min(forced_atomGroup, key = lambda atom: atom.Coords.y).Coords.y))
                y_max = math.ceil(int(max(forced_atomGroup, key = lambda atom: atom.Coords.y).Coords.y))
                x_coordinates = [float(atom.Coords.x) for atom in forced_atomGroup]
                y_coordinates = [float(atom.Coords.y) for atom in forced_atomGroup]
                x_bins = np.arange(x_min, x_max, (x_max - x_min)/num_x_bins)
                y_bins = np.arange(y_min, y_max, (y_max - y_min)/num_y_bins)
                atom_x_bin = np.digitize(x_coordinates, x_bins)
                atom_y_bin = np.digitize(y_coordinates, y_bins)
                temp_binned_data = np.zeros((num_x_bins, num_y_bins))
                for n,atom in enumerate(forced_atomGroup):
                    temp_binned_data[atom_x_bin[n] - 1][atom_y_bin[n] - 1] = atom.Force.mod()/(num_frames_orignal -1)
                binned_data_split[allKeys[i]][key].append(temp_binned_data)

    selected_atoms = ["P"]
    curavature_plots = ["Z_Surface", "Mean_Curvature", "Gaussian_Curvature"]
    if not split:
        selected_residues = {}
        axis_removed = {}
        for i in range(len(selected_atoms)):
            selected_residues[selected_atoms[i]] = {}
            curvature_selected = MembraneCurvature(universe, select = f"type {selected_atoms[i]}", n_x_bins = num_x_bins, n_y_bins = num_y_bins).run() #, select = 'resid 0-1023', n_x_bins=12, n_y_bins=12
            selected_residues[selected_atoms[i]]["Z_Surface"] = curvature_selected.results.z_surface
            selected_residues[selected_atoms[i]]["Mean_Curvature"] = curvature_selected.results.mean
            selected_residues[selected_atoms[i]]["Gaussian_Curvature"] = curvature_selected.results.gaussian
        for i in range(len(selected_atoms)):
            axis_removed[selected_atoms[i]] = {"Z_Surface" : list(range(num_frames)), "Mean_Curvature": list(range(num_frames)), "Gaussian_Curvature": list(range(num_frames))}
            axis_removed[selected_atoms[i]]["Z_Surface"][specific_frame] = [np.unravel_index(np.nanargmax(np.array(selected_residues[selected_atoms[i]]["Z_Surface"][specific_frame])), np.array(curvature_selected.results.z_surface[specific_frame]).shape)]
            axis_removed[selected_atoms[i]]["Mean_Curvature"][specific_frame] = [np.unravel_index(np.nanargmax(np.array(selected_residues[selected_atoms[i]]["Mean_Curvature"][specific_frame])), np.array(curvature_selected.results.mean[specific_frame]).shape)]
            axis_removed[selected_atoms[i]]["Gaussian_Curvature"][specific_frame] = [np.unravel_index(np.nanargmax(np.array(selected_residues[selected_atoms[i]]["Gaussian_Curvature"][specific_frame])), np.array(curvature_selected.results.gaussian[specific_frame]).shape)]
            axis_removed[selected_atoms[i]]["Z_Surface"][specific_frame].append(np.unravel_index(np.nanargmin(np.array(selected_residues[selected_atoms[i]]["Z_Surface"][specific_frame])), np.array(curvature_selected.results.z_surface[specific_frame]).shape))
            axis_removed[selected_atoms[i]]["Mean_Curvature"][specific_frame].append(np.unravel_index(np.nanargmin(np.array(selected_residues[selected_atoms[i]]["Mean_Curvature"][specific_frame])), np.array(curvature_selected.results.mean[specific_frame]).shape))
            axis_removed[selected_atoms[i]]["Gaussian_Curvature"][specific_frame].append(np.unravel_index(np.nanargmin(np.array(selected_residues[selected_atoms[i]]["Gaussian_Curvature"][specific_frame])), np.array(curvature_selected.results.gaussian[specific_frame]).shape))
    else:
        selected_residues_split = {"Upper": {}, "Lower": {}}
        axis_removed_split = {"Upper": {}, "Lower": {}}
        for split_key in selected_residues_split:
            lower_limit = min(lower_range)[0] if split_key == "Lower" else min(upper_range)[0]
            upper_limit = max(lower_range)[1] if split_key == "Lower" else max(upper_range)[1]
            universe_selected = universe.select_atoms(f"prop z < {universe.select_atoms('type P').center_of_mass()[2]} and type P") if split_key == "Lower" else universe.select_atoms(f"prop z > {universe.select_atoms('type P').center_of_mass()[2]} and type P")
            for i in range(len(selected_atoms)):
                selected_residues_split[split_key][selected_atoms[i]] = {}
                curvature_selected = MembraneCurvature(universe_selected, n_x_bins = num_x_bins, n_y_bins = num_y_bins).run()#, select = f"resid {lower_limit}:{upper_limit} and name {selected_atoms[i]}"
                selected_residues_split[split_key][selected_atoms[i]]["Z_Surface"] = curvature_selected.results.z_surface
                selected_residues_split[split_key][selected_atoms[i]]["Mean_Curvature"] = curvature_selected.results.mean
                selected_residues_split[split_key][selected_atoms[i]]["Gaussian_Curvature"] = curvature_selected.results.gaussian
            for i in range(len(selected_atoms)):
                axis_removed_split[split_key][selected_atoms[i]] = {"Z_Surface" : list(range(num_frames)), "Mean_Curvature": list(range(num_frames)), "Gaussian_Curvature": list(range(num_frames))}
                axis_removed_split[split_key][selected_atoms[i]]["Z_Surface"][specific_frame] = [np.unravel_index(np.nanargmax(np.array(selected_residues_split[split_key][selected_atoms[i]]["Z_Surface"][specific_frame])), np.array(curvature_selected.results.z_surface[specific_frame]).shape)]
                axis_removed_split[split_key][selected_atoms[i]]["Mean_Curvature"][specific_frame] = [np.unravel_index(np.nanargmax(np.array(selected_residues_split[split_key][selected_atoms[i]]["Mean_Curvature"][specific_frame])), np.array(curvature_selected.results.mean[specific_frame]).shape)]
                axis_removed_split[split_key][selected_atoms[i]]["Gaussian_Curvature"][specific_frame] = [np.unravel_index(np.nanargmax(np.array(selected_residues_split[split_key][selected_atoms[i]]["Gaussian_Curvature"][specific_frame])), np.array(curvature_selected.results.gaussian[specific_frame]).shape)]
                axis_removed_split[split_key][selected_atoms[i]]["Z_Surface"][specific_frame].append(np.unravel_index(np.nanargmin(np.array(selected_residues_split[split_key][selected_atoms[i]]["Z_Surface"][specific_frame])), np.array(curvature_selected.results.z_surface[specific_frame]).shape))
                axis_removed_split[split_key][selected_atoms[i]]["Mean_Curvature"][specific_frame].append(np.unravel_index(np.nanargmin(np.array(selected_residues_split[split_key][selected_atoms[i]]["Mean_Curvature"][specific_frame])), np.array(curvature_selected.results.mean[specific_frame]).shape))
                axis_removed_split[split_key][selected_atoms[i]]["Gaussian_Curvature"][specific_frame].append(np.unravel_index(np.nanargmin(np.array(selected_residues_split[split_key][selected_atoms[i]]["Gaussian_Curvature"][specific_frame])), np.array(curvature_selected.results.gaussian[specific_frame]).shape))
    rows = []
    if not split:
        for i in range(len(selected_atoms)):
            for frame in [specific_frame]:
                fig, ax = plt.subplots(ncols=3, nrows=3, constrained_layout=True)
                for n, j in enumerate(curavature_plots):
                    num = 0
                    im = ax[num][n].contourf(selected_residues[selected_atoms[i]][j][frame], cmap='PuBuGn', origin='lower')
                    ax[num][n].contour(selected_residues[selected_atoms[i]][j][frame], cmap = 'PuBuGn', origin='lower', levels=10)
                    ax[num][n].set_aspect('equal')
                    cbar = plt.colorbar(im, ticks=[np.nanmin(selected_residues[selected_atoms[i]][j][frame]), np.nanmax(selected_residues[selected_atoms[i]][j][frame])], orientation='horizontal', ax = ax[num][n], shrink=0.7)
                    cbar.ax.tick_params(labelsize=3, width=0.5)
                    ax[num][n].xaxis.set_tick_params(labelsize=4)
                    ax[num][n].yaxis.set_tick_params(labelsize=4)
                    ax[num][n].set_xlabel("X - Coordinate", {'fontsize':4})
                    ax[num][n].set_ylabel("Y - Coordinate", {'fontsize':4})
                    ax[num][n].set_title(f"Average {j}", fontdict={'fontsize':5}, pad=2)
                    cbar.set_label(f"{j} (nm$^{-1}$)", fontsize=4, labelpad=2)

                for n, j in enumerate(curavature_plots):
                    ax[1][n].plot(range(num_x_bins), selected_residues[selected_atoms[i]][j][frame][:,axis_removed[selected_atoms[i]][j][frame][0][1]], linewidth= 1, color='mediumseagreen', label="Max")
                    ax[1][n].plot(range(num_x_bins), selected_residues[selected_atoms[i]][j][frame][:,axis_removed[selected_atoms[i]][j][frame][1][1]], linewidth= 1, color='purple', label="Min")
                    ax[1][n].xaxis.set_tick_params(labelsize=4)
                    ax[1][n].yaxis.set_tick_params(labelsize=4)
                    ax[1][n].set_title(f"Y Bin ({axis_removed[selected_atoms[i]][j][frame][0][1]},{axis_removed[selected_atoms[i]][j][frame][1][1]}) Average {j} (nm$^{-1}$)", fontdict={'fontsize':5}, pad=2)
                    if n == 0:
                        ax[1][n].legend(loc="upper right", markerscale=0.3, fontsize='xx-small')
                for lipid in range(numKeys):
                    im = ax[2][lipid].contourf(binned_data[allKeys[lipid]][0], cmap='PuBuGn', origin='lower')
                    ax[2][lipid].contour(binned_data[allKeys[lipid]][0], cmap = 'PuBuGn', origin='lower', levels=10)
                    ax[2][lipid].set_aspect('equal')
                    cbar = plt.colorbar(im, ticks=[binned_data[allKeys[lipid]][0].min(), binned_data[allKeys[lipid]][0].max()], orientation='horizontal', ax = ax[2][lipid], shrink=0.7)
                    cbar.ax.tick_params(labelsize=3, width=0.5)
                    ax[2][lipid].xaxis.set_tick_params(labelsize=4)
                    ax[2][lipid].yaxis.set_tick_params(labelsize=4)
                    ax[2][lipid].set_title(f"Forces on {allKeys[lipid]}", fontdict={'fontsize':5}, pad=2)
                    ax[2][lipid].set_xlabel("X - Coordinate", {'fontsize':4})
                    ax[2][lipid].set_ylabel("Y - Coordinate", {'fontsize':4})
                    cbar.set_label(f"Force Z (N)", fontsize=4, labelpad=2)
                ax[2][2].remove()
                if gangle != None:
                    plt.figtext(0.85, 0.2, f"Average Angle - \n {list(gangle.keys())[0]} : {gangle[list(gangle.keys())[0]][frame]} \n{list(gangle.keys())[1]} : {gangle[list(gangle.keys())[1]][frame]}", ha="center", fontdict={'fontsize':7})
                fig.suptitle(f'Curvature Plots for Atom {selected_atoms[i]} - Frame {frame}', fontsize=12)
                plt.savefig(f'{plot_name}_{num_x_bins}_{num_y_bins}_{selected_atoms[i]}_{frame}.svg', dpi = 1000)
                plt.close()
        if gangle!=None:
            plt.plot(range(len(gangle[list(gangle.keys())[0]])), gangle[list(gangle.keys())[0]], linewidth=1, color='crimson', label=list(gangle.keys())[0])
            plt.plot(range(len(gangle[list(gangle.keys())[1]])), gangle[list(gangle.keys())[1]], linewidth=1, color='deepskyblue', label=list(gangle.keys())[1])
            plt.xlabel("Frame")
            plt.ylabel("Angle")
            plt.legend(loc='upper left')
            plt.savefig(f'{plot_name}_angles_framewise.svg', dpi = 1000)
            plt.close()
    else:
        for i in range(len(selected_atoms)):
            for position in binned_data_split[allKeys[i]]:
                for frame in [specific_frame]:
                    fig, ax = plt.subplots(ncols=3, nrows=3, constrained_layout=True)
                    for n, j in enumerate(curavature_plots):
                        num = 0
                        im = ax[num][n].contourf(selected_residues_split[position][selected_atoms[i]][j][frame], cmap='PuBuGn', origin='lower')
                        ax[num][n].contour(selected_residues_split[position][selected_atoms[i]][j][frame], cmap = 'PuBuGn', origin='lower', levels=10)
                        ax[num][n].set_aspect('equal')
                        cbar = plt.colorbar(im, ticks=[np.nanmin(selected_residues_split[position][selected_atoms[i]][j][frame]), np.nanmax(selected_residues_split[position][selected_atoms[i]][j][frame])], orientation='horizontal', ax = ax[num][n], shrink=0.7)
                        cbar.ax.tick_params(labelsize=3, width=0.5)
                        ax[num][n].xaxis.set_tick_params(labelsize=4)
                        ax[num][n].yaxis.set_tick_params(labelsize=4)
                        ax[num][n].set_xlabel("X - Coordinate", {'fontsize':4})
                        ax[num][n].set_ylabel("Y - Coordinate", {'fontsize':4})
                        ax[num][n].set_title(f"Average {j}", fontdict={'fontsize':5}, pad=2)
                        cbar.set_label(f"{j} (nm$^{-1}$)", fontsize=4, labelpad=2)
                    for n, j in enumerate(curavature_plots):
                        ax[1][n].plot(range(num_x_bins), selected_residues_split[position][selected_atoms[i]][j][frame][:,axis_removed_split[position][selected_atoms[i]][j][frame][0][1]], linewidth= 1, color='mediumseagreen', label="Max")
                        ax[1][n].plot(range(num_x_bins), selected_residues_split[position][selected_atoms[i]][j][frame][:,axis_removed_split[position][selected_atoms[i]][j][frame][1][1]], linewidth= 1, color='purple', label="Min")
                        ax[1][n].xaxis.set_tick_params(labelsize=4)
                        ax[1][n].yaxis.set_tick_params(labelsize=4)
                        ax[1][n].set_title(f"Y Bin ({axis_removed_split[position][selected_atoms[i]][j][frame][0][1]},{axis_removed_split[position][selected_atoms[i]][j][frame][1][1]}) Average {j} (nm$^{-1}$)", fontdict={'fontsize':5}, pad=2)
                        if n == 0:
                            ax[1][n].legend(loc="upper right", markerscale=0.3, fontsize='xx-small')
                    for lipid in range(numKeys):
                        im = ax[2][lipid].contourf(binned_data_split[allKeys[lipid]][position][0], cmap='PuBuGn', origin='lower')
                        ax[2][lipid].contour(binned_data_split[allKeys[lipid]][position][0], cmap = 'PuBuGn', origin='lower', levels=10)
                        ax[2][lipid].set_aspect('equal')
                        cbar = plt.colorbar(im, ticks=[binned_data_split[allKeys[lipid]][position][0].min(), binned_data_split[allKeys[lipid]][position][0].max()], orientation='horizontal', ax = ax[2][lipid], shrink=0.7)
                        cbar.ax.tick_params(labelsize=3, width=0.5)
                        ax[2][lipid].xaxis.set_tick_params(labelsize=4)
                        ax[2][lipid].yaxis.set_tick_params(labelsize=4)
                        ax[2][lipid].set_title(f"Forces on {allKeys[lipid]}", fontdict={'fontsize':5}, pad=2)
                        ax[2][lipid].set_xlabel("X - Coordinate", {'fontsize':4})
                        ax[2][lipid].set_ylabel("Y - Coordinate", {'fontsize':4})
                        cbar.set_label(f"Force Z (N)", fontsize=4, labelpad=2)
                    ax[2][2].remove()
                    plt.figtext(0.85, 0.2, f"Average Angle - \n {list(gangle[position].keys())[0]} : {gangle[position][list(gangle[position].keys())[0]][specific_frame]} \n{list(gangle[position].keys())[1]} : {gangle[position][list(gangle[position].keys())[1]][specific_frame]}", ha="center", fontdict={'fontsize':7})
                    fig.suptitle(f'Curvature Plots for Atom {selected_atoms[i]} {position} Membrane - Frame {frame}', fontsize=12)
                    plt.savefig(f'{plot_name}_{num_x_bins}_{num_y_bins}_{selected_atoms[i]}_{frame}_{position}.svg', dpi = 1000)
                    plt.close()
                plt.plot(range(len(gangle[position][list(gangle[position].keys())[0]])), gangle[position][list(gangle[position].keys())[0]], linewidth=1, color='crimson', label=list(gangle[position].keys())[0])
                plt.plot(range(len(gangle[position][list(gangle[position].keys())[1]])), gangle[position][list(gangle[position].keys())[1]], linewidth=1, color='deepskyblue', label=list(gangle[position].keys())[1])
                plt.xlabel("Frame")
                plt.ylabel("Angle")
                plt.legend(loc='upper left')
                plt.savefig(f'{plot_name}_angles_framewise_{position}.svg', dpi = 1000)
                plt.close()

def plot_curvature_window(universe, atom_dict, window_size = 10, num_x_bins = 10, num_y_bins = 10, split = False, gangle = None, plot_name="curvature_combined_window"):
    num_frames = len(atom_dict) - 1
    num_frames_orignal = list(atom_dict.keys())[0]  if num_frames == 1 else num_frames
    numKeys = len(atom_dict[list(atom_dict.keys())[0]])
    allKeys = [key for key in atom_dict[list(atom_dict.keys())[0]]]

    binned_data = {}
    binned_data_split = {}
    upper_range = []
    lower_range = []
    for i in range(numKeys):
        binned_data[allKeys[i]] = []
        binned_data_split[allKeys[i]] = {"Upper": [], "Lower": []}
        frame_list = []
        for window in range(0, num_frames, window_size):
            forced_atomGroup = [atom_dict[windowc][allKeys[i]] for windowc in range(window, window+window_size) if windowc <= num_frames]
            [specific_frame.sort(key= lambda x: x.Coords.z) for specific_frame in forced_atomGroup]
            if split:
                mean_z = mean([x.Coords.z for x in forced_atomGroup[0]])
                upper_membrane = [[atom for atom in Group if atom.Coords.z > mean_z] for Group in forced_atomGroup]
                lower_membrane = [[atom for atom in Group if atom.Coords.z < mean_z] for Group in forced_atomGroup]
                temp_range = (int(min(upper_membrane[0], key = lambda x: x.Coords.z).ResNum), int(max(upper_membrane[0], key = lambda x: x.Coords.z).ResNum))
                upper_range.append((min(temp_range), max(temp_range)))
                temp_range = (int(min(lower_membrane[0], key = lambda x: x.Coords.z).ResNum), int(max(lower_membrane[0], key = lambda x: x.Coords.z).ResNum))
                lower_range.append((min(temp_range), max(temp_range)))
            if not split:
                x_min = math.floor(int(min([min(forced_atomGroup[windowc], key = lambda atom: atom.Coords.x) for windowc in range(window_size) if windowc + window <= num_frames], key = lambda atom: atom.Coords.x).Coords.x))
                x_max = math.ceil(int(max([max(forced_atomGroup[windowc], key = lambda atom: atom.Coords.x) for windowc in range(window_size) if windowc + window <= num_frames], key = lambda atom: atom.Coords.x).Coords.x))
                y_min = math.floor(int(min([min(forced_atomGroup[windowc], key = lambda atom: atom.Coords.y) for windowc in range(window_size) if windowc + window <= num_frames], key = lambda atom: atom.Coords.y).Coords.y))
                y_max = math.ceil(int(max([max(forced_atomGroup[windowc], key = lambda atom: atom.Coords.y) for windowc in range(window_size) if windowc + window <= num_frames], key = lambda atom: atom.Coords.y).Coords.y))
                x_coordinates = [float(atom.Coords.x) for atom in forced_atomGroup[0]]
                y_coordinates = [float(atom.Coords.y) for atom in forced_atomGroup[0]]
                x_bins = np.arange(x_min, x_max, (x_max - x_min)/num_x_bins)
                y_bins = np.arange(y_min, y_max, (y_max - y_min)/num_y_bins)
                atom_x_bin = np.digitize(x_coordinates, x_bins)
                atom_y_bin = np.digitize(y_coordinates, y_bins)
                temp_binned_data = np.zeros((num_x_bins, num_y_bins))
                forces_averaged = []
                for n, atoms in enumerate(zip(*forced_atomGroup)):
                    average_at_coordinate = Point(0,0,0)
                    for atom in atoms:
                        average_at_coordinate += atom.Force
                    temp_binned_data[atom_x_bin[n] - 1][atom_y_bin[n] - 1] = average_at_coordinate.mod()/(num_frames_orignal -1)
                binned_data[allKeys[i]].append(temp_binned_data)
            else:
                for key in binned_data_split[allKeys[i]]:
                    forced_atomGroup = upper_membrane if key == "Upper" else lower_membrane
                    x_min = math.floor(int(min([min(forced_atomGroup[windowc], key = lambda atom: atom.Coords.x) for windowc in range(window_size)], key = lambda atom: atom.Coords.x).Coords.x))
                    x_max = math.ceil(int(max([max(forced_atomGroup[windowc], key = lambda atom: atom.Coords.x) for windowc in range(window_size)], key = lambda atom: atom.Coords.x).Coords.x))
                    y_min = math.floor(int(min([min(forced_atomGroup[windowc], key = lambda atom: atom.Coords.y) for windowc in range(window_size)], key = lambda atom: atom.Coords.y).Coords.y))
                    y_max = math.ceil(int(max([max(forced_atomGroup[windowc], key = lambda atom: atom.Coords.y) for windowc in range(window_size)], key = lambda atom: atom.Coords.y).Coords.y))
                    x_coordinates = [float(atom.Coords.x) for atom in forced_atomGroup[0]]
                    y_coordinates = [float(atom.Coords.y) for atom in forced_atomGroup[0]]
                    x_bins = np.arange(x_min, x_max, (x_max - x_min)/num_x_bins)
                    y_bins = np.arange(y_min, y_max, (y_max - y_min)/num_y_bins)
                    atom_x_bin = np.digitize(x_coordinates, x_bins)
                    atom_y_bin = np.digitize(y_coordinates, y_bins)
                    temp_binned_data = np.zeros((num_x_bins, num_y_bins))
                    forces_averaged = []
                    for n, atoms in enumerate(zip(*forced_atomGroup)):
                        average_at_coordinate = Point(0,0,0)
                        for atom in atoms:
                            average_at_coordinate += atom.Force
                        temp_binned_data[atom_x_bin[n] - 1][atom_y_bin[n] - 1] = average_at_coordinate.mod()/(num_frames_orignal -1)
                    binned_data_split[allKeys[i]][key].append(temp_binned_data)

    selected_atoms = ["P"]
    curavature_plots = ["Z_Surface", "Mean_Curvature", "Gaussian_Curvature"]
    if not split:
        selected_residues = {}
        axis_removed = {}
        for i in range(len(selected_atoms)):
            selected_residues[selected_atoms[i]] = {}
            curvature_selected = MembraneCurvature(universe, select = f"type {selected_atoms[i]}", n_x_bins = num_x_bins, n_y_bins = num_y_bins).run() #, select = 'resid 0-1023', n_x_bins=12, n_y_bins=12
            selected_residues[selected_atoms[i]]["Z_Surface"] = curvature_selected.results.z_surface
            selected_residues[selected_atoms[i]]["Mean_Curvature"] = curvature_selected.results.mean
            selected_residues[selected_atoms[i]]["Gaussian_Curvature"] = curvature_selected.results.gaussian
        for i in range(len(selected_atoms)):
            axis_removed[selected_atoms[i]] = {"Z_Surface" : list(range(num_frames)), "Mean_Curvature": list(range(num_frames)), "Gaussian_Curvature": list(range(num_frames))}
            for window in range(0, num_frames, window_size):
                z_sur = np.nanmean(np.array(selected_residues[selected_atoms[i]]["Z_Surface"][window : window+window_size]), axis = 0)
                meanc = np.nanmean(np.array(selected_residues[selected_atoms[i]]["Mean_Curvature"][window : window+window_size]), axis = 0)
                gausc = np.nanmean(np.array(selected_residues[selected_atoms[i]]["Gaussian_Curvature"][window : window+window_size]), axis = 0)
                axis_removed[selected_atoms[i]]["Z_Surface"][window] = [np.unravel_index(np.nanargmax(z_sur), z_sur.shape)]
                axis_removed[selected_atoms[i]]["Mean_Curvature"][window] = [np.unravel_index(np.nanargmax(meanc), meanc.shape)]
                axis_removed[selected_atoms[i]]["Gaussian_Curvature"][window] = [np.unravel_index(np.nanargmax(gausc), gausc.shape)]
                axis_removed[selected_atoms[i]]["Z_Surface"][window].append(np.unravel_index(np.nanargmin(z_sur), z_sur.shape))
                axis_removed[selected_atoms[i]]["Mean_Curvature"][window].append(np.unravel_index(np.nanargmin(meanc), meanc.shape))
                axis_removed[selected_atoms[i]]["Gaussian_Curvature"][window].append(np.unravel_index(np.nanargmin(gausc), gausc.shape))
    else:
        selected_residues_split = {"Upper": {}, "Lower": {}}
        axis_removed_split = {"Upper": {}, "Lower": {}}
        for split_key in selected_residues_split:
            lower_limit = min(lower_range)[0] if split_key == "Lower" else min(upper_range)[0]
            upper_limit = max(lower_range)[1] if split_key == "Lower" else max(upper_range)[1]
            universe_selected = universe.select_atoms(f"prop z < {universe.select_atoms('type P').center_of_mass()[2]} and type P") if split_key == "Lower" else universe.select_atoms(f"prop z > {universe.select_atoms('type P').center_of_mass()[2]} and type P")
            for i in range(len(selected_atoms)):
                selected_residues_split[split_key][selected_atoms[i]] = {}
                curvature_selected = MembraneCurvature(universe_selected, n_x_bins = num_x_bins, n_y_bins = num_y_bins).run()#, select = f"resid {lower_limit}:{upper_limit} and name {selected_atoms[i]}"
                selected_residues_split[split_key][selected_atoms[i]]["Z_Surface"] = curvature_selected.results.z_surface
                selected_residues_split[split_key][selected_atoms[i]]["Mean_Curvature"] = curvature_selected.results.mean
                selected_residues_split[split_key][selected_atoms[i]]["Gaussian_Curvature"] = curvature_selected.results.gaussian
            for i in range(len(selected_atoms)):
                axis_removed_split[split_key][selected_atoms[i]] = {"Z_Surface" : list(range(num_frames)), "Mean_Curvature": list(range(num_frames)), "Gaussian_Curvature": list(range(num_frames))}
                for window in range(0, num_frames, window_size):
                    z_sur = np.nanmean(np.array(selected_residues_split[split_key][selected_atoms[i]]["Z_Surface"][window : window+window_size]), axis = 0)
                    meanc = np.nanmean(np.array(selected_residues_split[split_key][selected_atoms[i]]["Mean_Curvature"][window : window+window_size]), axis = 0)
                    gausc = np.nanmean(np.array(selected_residues_split[split_key][selected_atoms[i]]["Gaussian_Curvature"][window : window+window_size]), axis = 0)
                    axis_removed_split[split_key][selected_atoms[i]]["Z_Surface"][window] = [np.unravel_index(np.nanargmax(z_sur), z_sur.shape)]
                    axis_removed_split[split_key][selected_atoms[i]]["Mean_Curvature"][window] = [np.unravel_index(np.nanargmax(meanc), meanc.shape)]
                    axis_removed_split[split_key][selected_atoms[i]]["Gaussian_Curvature"][window] = [np.unravel_index(np.nanargmax(gausc), gausc.shape)]
                    axis_removed_split[split_key][selected_atoms[i]]["Z_Surface"][window].append(np.unravel_index(np.nanargmin(z_sur), z_sur.shape))
                    axis_removed_split[split_key][selected_atoms[i]]["Mean_Curvature"][window].append(np.unravel_index(np.nanargmin(meanc), meanc.shape))
                    axis_removed_split[split_key][selected_atoms[i]]["Gaussian_Curvature"][window].append(np.unravel_index(np.nanargmin(gausc), gausc.shape))
    rows = []
    if not split:
        for i in range(len(selected_atoms)):
            window_angles = []
            for windown, window in enumerate(range(0, num_frames, window_size)):
                fig, ax = plt.subplots(ncols=3, nrows=3, constrained_layout=True)
                for n, j in enumerate(curavature_plots):
                    num = 0
                    im = ax[num][n].contourf(selected_residues[selected_atoms[i]][j][window], cmap='PuBuGn', origin='lower')
                    ax[num][n].contour(selected_residues[selected_atoms[i]][j][window], cmap = 'PuBuGn', origin='lower', levels=10)
                    ax[num][n].set_aspect('equal')
                    cbar = plt.colorbar(im, ticks=[np.nanmin(selected_residues[selected_atoms[i]][j][window]), np.nanmax(selected_residues[selected_atoms[i]][j][window])], orientation='horizontal', ax = ax[num][n], shrink=0.7)
                    cbar.ax.tick_params(labelsize=3, width=0.5)
                    ax[num][n].xaxis.set_tick_params(labelsize=4)
                    ax[num][n].yaxis.set_tick_params(labelsize=4)
                    ax[num][n].set_xlabel("X - Coordinate", {'fontsize':4})
                    ax[num][n].set_ylabel("Y - Coordinate", {'fontsize':4})
                    ax[num][n].set_title(f"Average {j}", fontdict={'fontsize':5}, pad=2)
                    cbar.set_label(f"{j} (nm$^{-1}$)", fontsize=4, labelpad=2)

                for n, j in enumerate(curavature_plots):
                    ax[1][n].plot(range(num_x_bins), selected_residues[selected_atoms[i]][j][window][:,axis_removed[selected_atoms[i]][j][window][0][1]], linewidth= 1, color='mediumseagreen', label="Max")
                    ax[1][n].plot(range(num_x_bins), selected_residues[selected_atoms[i]][j][window][:,axis_removed[selected_atoms[i]][j][window][1][1]], linewidth= 1, color='purple', label="Min")
                    ax[1][n].xaxis.set_tick_params(labelsize=4)
                    ax[1][n].yaxis.set_tick_params(labelsize=4)
                    ax[1][n].set_title(f"Y Bin ({axis_removed[selected_atoms[i]][j][window][0][1]},{axis_removed[selected_atoms[i]][j][window][1][1]}) Average {j} (nm$^{-1}$)", fontdict={'fontsize':5}, pad=2)
                    if n == 0:
                        ax[1][n].legend(loc="upper right", markerscale=0.3, fontsize='xx-small')

                for lipid in range(numKeys):
                    im = ax[2][lipid].contourf(binned_data[allKeys[lipid]][windown], cmap='PuBuGn', origin='lower')
                    ax[2][lipid].contour(binned_data[allKeys[lipid]][windown], cmap = 'PuBuGn', origin='lower', levels=10)
                    ax[2][lipid].set_aspect('equal')
                    cbar = plt.colorbar(im, ticks=[binned_data[allKeys[lipid]][windown].min(), binned_data[allKeys[lipid]][windown].max()], orientation='horizontal', ax = ax[2][lipid], shrink=0.7)
                    cbar.ax.tick_params(labelsize=3, width=0.5)
                    ax[2][lipid].set_title(f"Forces on {allKeys[lipid]}", fontdict={'fontsize':5}, pad=2)
                    ax[2][lipid].set_xlabel("X - Coordinate", {'fontsize':4})
                    ax[2][lipid].set_ylabel("Y - Coordinate", {'fontsize':4})
                    ax[2][lipid].xaxis.set_tick_params(labelsize=4)
                    ax[2][lipid].yaxis.set_tick_params(labelsize=4)
                    cbar.set_label(f"Force Z (N)", fontsize=4, labelpad=2)
                ax[2][2].remove()
                if gangle != None:
                    window_angles.append((mean(gangle[list(gangle.keys())[0]][window:window+window_size]), mean(gangle[list(gangle.keys())[1]][window:window+window_size])))
                    plt.figtext(0.85, 0.2, f"Average Angle - \n {list(gangle.keys())[0]} : {window_angles[-1][0]} \n{list(gangle.keys())[1]} : {window_angles[-1][1]}", ha="center", fontdict={'fontsize':7})
                fig.suptitle(f'Curvature Plots for Atom {selected_atoms[i]} - Window {window}-{window+window_size}', fontsize=12)
                plt.savefig(f'{plot_name}_{num_x_bins}_{num_y_bins}_{selected_atoms[i]}_window_{window}_{window+window_size}.svg', dpi = 1000)
                plt.close()
            if len(window_angles) > 0:
                angles_lipid1 = [x[0] for x in window_angles]
                angles_lipid2 = [x[1] for x in window_angles]
                plt.plot(range(len(angles_lipid1)), angles_lipid1, linewidth=1, color='crimson')
                plt.plot(range(len(angles_lipid2)), angles_lipid2, linewidth=1, color='deepskyblue')
                plt.xticks(range(len(window_angles)))
                plt.xlabel("Window")
                plt.ylabel("Average Angle")
                plt.savefig(f'{plot_name}_angles_windowed_{window_size}.svg', dpi = 1000)
                plt.close()
    else:
        for i in range(len(selected_atoms)):
            for position in binned_data_split[allKeys[i]]:
                window_angles = []
                for windown, window in enumerate(range(0, num_frames, window_size)):
                    fig, ax = plt.subplots(ncols=3, nrows=3, constrained_layout=True)
                    for n, j in enumerate(curavature_plots):
                        num = 0
                        im = ax[num][n].contourf(selected_residues_split[position][selected_atoms[i]][j][window], cmap='PuBuGn', origin='lower')
                        ax[num][n].contour(selected_residues_split[position][selected_atoms[i]][j][window], cmap = 'PuBuGn', origin='lower', levels=10)
                        ax[num][n].set_aspect('equal')
                        cbar = plt.colorbar(im, ticks=[np.nanmin(selected_residues_split[position][selected_atoms[i]][j][window]), np.nanmax(selected_residues_split[position][selected_atoms[i]][j][window])], orientation='horizontal', ax = ax[num][n], shrink=0.7)
                        cbar.ax.tick_params(labelsize=3, width=0.5)
                        ax[num][n].xaxis.set_tick_params(labelsize=4)
                        ax[num][n].yaxis.set_tick_params(labelsize=4)
                        ax[num][n].set_xlabel("X - Coordinate", {'fontsize':4})
                        ax[num][n].set_ylabel("Y - Coordinate", {'fontsize':4})
                        ax[num][n].set_title(f"Average {j}", fontdict={'fontsize':5}, pad=2)
                        cbar.set_label(f"{j} (nm$^{-1}$)", fontsize=4, labelpad=2)

                    for n, j in enumerate(curavature_plots):
                        ax[1][n].plot(range(num_x_bins), selected_residues_split[position][selected_atoms[i]][j][window][:,axis_removed_split[position][selected_atoms[i]][j][window][0][1]], linewidth= 1, color='mediumseagreen', label="Max")
                        ax[1][n].plot(range(num_x_bins), selected_residues_split[position][selected_atoms[i]][j][window][:,axis_removed_split[position][selected_atoms[i]][j][window][1][1]], linewidth= 1, color='purple', label="Min")
                        ax[1][n].xaxis.set_tick_params(labelsize=4)
                        ax[1][n].yaxis.set_tick_params(labelsize=4)
                        ax[1][n].set_title(f"Y Bin ({axis_removed_split[position][selected_atoms[i]][j][window][0][1]},{axis_removed_split[position][selected_atoms[i]][j][window][1][1]}) Average {j} (nm$^{-1}$)", fontdict={'fontsize':5}, pad=2)
                        if n == 0:
                            ax[1][n].legend(loc="upper right", markerscale=0.3, fontsize='xx-small')

                    for lipid in range(numKeys):
                        im = ax[2][lipid].contourf(binned_data_split[allKeys[lipid]][position][windown], cmap='PuBuGn', origin='lower')
                        ax[2][lipid].contour(binned_data_split[allKeys[lipid]][position][windown], cmap = 'PuBuGn', origin='lower', levels=10)
                        ax[2][lipid].set_aspect('equal')
                        cbar = plt.colorbar(im, ticks=[binned_data_split[allKeys[lipid]][position][windown].min(), binned_data_split[allKeys[lipid]][position][windown].max()], orientation='horizontal', ax = ax[2][lipid], shrink=0.7)
                        cbar.ax.tick_params(labelsize=3, width=0.5)
                        ax[2][lipid].xaxis.set_tick_params(labelsize=4)
                        ax[2][lipid].yaxis.set_tick_params(labelsize=4)
                        ax[2][lipid].set_title(f"Forces on {allKeys[lipid]}", fontdict={'fontsize':5}, pad=2)
                        ax[2][lipid].set_xlabel("X - Coordinate", {'fontsize':4})
                        ax[2][lipid].set_ylabel("Y - Coordinate", {'fontsize':4})
                        cbar.set_label(f"Force Z (N)", fontsize=4, labelpad=2)
                    ax[2][2].remove()
                    if gangle != None:
                        window_angles.append((mean(gangle[position][list(gangle[position].keys())[0]][window:window+window_size]), mean(gangle[position][list(gangle[position].keys())[1]][window:window+window_size])))
                        plt.figtext(0.85, 0.2, f"Average Angle - \n {list(gangle[position].keys())[0]} : {window_angles[-1][0]} \n{list(gangle[position].keys())[1]} : {window_angles[-1][1]}", ha="center", fontdict={'fontsize':7})
                    fig.suptitle(f'Curvature Plots for Atom {selected_atoms[i]} {position} Membrane - Window {window}-{window+window_size}', fontsize=12)
                    plt.savefig(f'{plot_name}_{num_x_bins}_{num_y_bins}_{selected_atoms[i]}_{position}_window_{window}_{window+window_size}.svg', dpi = 1000)
                    plt.close()
                if len(window_angles) > 0:
                    angles_lipid1 = [x[0] for x in window_angles]
                    angles_lipid2 = [x[1] for x in window_angles]
                    plt.plot(range(len(angles_lipid1)), angles_lipid1, linewidth=1, color='crimson')
                    plt.plot(range(len(angles_lipid2)), angles_lipid2, linewidth=1, color='deepskyblue')
                    plt.xticks(range(len(window_angles)))
                    plt.xlabel("Window")
                    plt.ylabel("Average Angle")
                    plt.legend(loc='upper left')
                    plt.savefig(f'{plot_name}_angles_windowed_{window_size}_{position}.svg', dpi = 1000)
                    plt.close()

def plot_msd(universe, select='all', msd_type='xyz', fft=True, timestep=1, start_index=None, end_index=None, plot_name="MSD"):
    try:
        MSD = msd.EinsteinMSD(universe, 
                              select=select, 
                              msd_type=msd_type, 
                              fft=fft)
        MSD.run()
        lagtimes = np.arange(MSD.n_frames)*timestep
        msd_result =  MSD.results.timeseries
        lagtimes = np.arange(MSD.n_frames)*timestep # make the lag-time axis
        
        # plot the actual MSD
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(lagtimes, msd_result, color='blue', linestyle="-", label=r'MSD')
        # calculate diffusion coefficient
        start_index = 0 if start_index==None else start_index
        end_index = -1 if end_index==None else end_index

        linear_model = linregress(lagtimes[start_index:end_index],
                                  msd_result[start_index:end_index])
        slope = linear_model.slope
        error = linear_model.rvalue
        D = slope * 1/(2*MSD.dim_fac)
        plt.xlabel("Frame", fontsize=10, labelpad=5)
        plt.ylabel("MSD", fontsize=10, labelpad=5)
        plt.title(f'Frame vs MSD (Diffusion coefficient : {D})')
        plt.savefig(f"{plot_name}.svg", dpi = 1000)
        plt.close()
        logger.info(f"Diffution coefficient and MSD plotted")
    except:
        logger.error(traceback.format_exc())
        #sys.exit(0)
    return

def gangle(trr_filename, tpr_filename, ndx_filename, pdb_filename, mda_universe, group1=None, group2=None, selection=None, grouping=None, c_atom_name="C4B", angles = False, g1="vector", g2="z", seltype="res_com", selrpos="res_com", mode="average", split = False):
    random_string = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    if angles == False:
        if group1 == None or group2 == None:
             raise ValueError('Enter valid lipid groups') 
        if split == False:
            avg_angles = {}
            membrane = {}
            for i in mda_universe.atoms:
                if i.resname == group1 or i.resname == group2:
                    if i.type == "P" or i.name == c_atom_name:
                        if i.resname not in membrane:
                            membrane[i.resname] = [i.id]
                        else:
                            membrane[i.resname].append(i.id)
            ndx_filename = f'custom_membrane_{random_string}.ndx'
            with open(ndx_filename, 'w') as f:
                for i in membrane:
                        f.write(f"[ {i} ]\n")
                        group_str = " ".join([str(i) for i in membrane[i]])
                        f.write("\n".join(textwrap.wrap(group_str, 15)))
                        f.write("\n")
            for group in [group1, group2]:
                filename=f"angle_{group}_{random_string}.xvg"
                subprocess.run(["gmx", "gangle", "-f", trr_filename, "-s", tpr_filename, "-n", ndx_filename, "-g1", g1, "-g2", g2, "-group1", group, "-seltype", seltype, "-selrpos", selrpos, "-oav", filename], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                if mode=="average":
                    angle = []
                    with open(filename, "r") as f:
                        for line in f:
                            if (not line[0]=="#") and (not line[0]=="@"):
                                angle.append(float(line.split()[1]))
                    subprocess.run(["rm", filename])
                    avg_angles[group] = mean(angle)
                elif mode=="framewise" or mode=="window":
                    angle = []
                    with open(filename, "r") as f:
                        for line in f:
                            if (not line[0]=="#") and (not line[0]=="@"):
                                angle.append(float(line.split()[1]))
                    subprocess.run(["rm", filename])
                    avg_angles[group] = angle
            subprocess.run(["rm", ndx_filename])
            return avg_angles
        else:
            membrane = {'Lower':{}, 'Upper':{}}
            lower_membrane = [atom.resid for atom in mda_universe.select_atoms(f"prop z < {mda_universe.select_atoms('type P').center_of_mass()[2]} and type P")]
            upper_membrane = [atom.resid for atom in mda_universe.select_atoms(f"prop z > {mda_universe.select_atoms('type P').center_of_mass()[2]} and type P")]
            for atom in mda_universe.atoms:
                if atom.resname == group1 or atom.resname == group2:
                    if atom.resid in lower_membrane:
                        if atom.type == "P" or atom.name == c_atom_name:
                            if atom.resname not in membrane["Lower"]:
                                membrane["Lower"][atom.resname] = [atom.id]
                            else:
                                membrane["Lower"][atom.resname].append(atom.id)
                    elif atom.resid in upper_membrane:
                        if atom.type == "P" or atom.name == c_atom_name:
                            if atom.resname not in membrane["Upper"]:
                                membrane["Upper"][atom.resname] = [atom.id]
                            else:
                                membrane["Upper"][atom.resname].append(atom.id)
            ndx_filename = f'custom_membrane_{random_string}.ndx'
            with open(ndx_filename, 'w') as f:
                for i in membrane:
                    for resgroup in membrane[i]:
                        f.write(f"[ {i}_{resgroup} ]\n")
                        group_str = " ".join([str(i) for i in membrane[i][resgroup]])
                        f.write("\n".join(textwrap.wrap(group_str, 15)))
                        f.write("\n")
            for group in [group1, group2]:
                filename_lower = f"angle_lower_{group}_{random_string}.xvg"
                temp_group = f"Lower_{group}"
                subprocess.run(["gmx", "gangle", "-f", trr_filename, "-s", tpr_filename, "-n", ndx_filename, "-group1", temp_group, "-seltype", "whole_res_com", "-oav", filename_lower], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            for group in [group1, group2]:
                filename_upper = f"angle_upper_{group}_{random_string}.xvg"
                temp_group = f"Upper_{group}"
                subprocess.run(["gmx", "gangle", "-f", trr_filename, "-s", tpr_filename, "-n", ndx_filename, "-group1", temp_group, "-seltype", "whole_res_com", "-oav", filename_upper], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            angle = {'Upper':{}, 'Lower':{}}
            for group in [group1, group2]:
                if mode=="average":
                    for n, i in enumerate([f"angle_upper_{group}_{random_string}.xvg", f"angle_lower_{group}_{random_string}.xvg"]):
                        angle[list(angle.keys())[n]][group] = []
                        with open(i, "r") as f:
                            for line in f:
                                if (not line[0]=="#") and (not line[0]=="@"):
                                    angle[list(angle.keys())[n]][group].append(float(line.split()[1]))
                        angle[list(angle.keys())[n]][group]=mean(angle[list(angle.keys())[n]][group])
                elif mode=="framewise" or mode=="window":
                    for n, i in enumerate([f"angle_upper_{group}_{random_string}.xvg", f"angle_lower_{group}_{random_string}.xvg"]):
                        angle[list(angle.keys())[n]][group] = []
                        with open(i, "r") as f:
                            for line in f:
                                if (not line[0]=="#") and (not line[0]=="@"):
                                    angle[list(angle.keys())[n]][group].append(float(line.split()[1]))
                subprocess.run(["rm", i])
            subprocess.run(["rm", ndx_filename])
            return angle
    else:
        if split == False:
            avg_angles = {} 
            membrane = {}
            for i in mda_universe.atoms:
                if i.resname in selection:
                    if i.type == "P" or i.name == c_atom_name:
                        if i.resname not in membrane:
                            membrane[i.resname] = [i.id]
                        else:
                            membrane[i.resname].append(i.id)
            ndx_filename = f'custom_membrane_{random_string}.ndx'
            with open(ndx_filename, 'w') as f:
                if grouping == "combine":
                    f.write(f"[ Combined ]\n")
                for i in membrane:
                    if grouping != "combine":
                        f.write(f"[ {i} ]\n")
                    group_str = " ".join([str(i) for i in membrane[i]])
                    f.write("\n".join(textwrap.wrap(group_str, 15)))
                    f.write("\n")
            if grouping != "combine":
                for group in selection:
                    filename=f"angle_{group}_{random_string}.xvg"
                    subprocess.run(["gmx", "gangle", "-f", trr_filename, "-s", tpr_filename, "-n", ndx_filename, "-g1", g1, "-g2", g2, "-group1", group, "-seltype", seltype, "-selrpos", selrpos, "-oav", filename], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                    angle = []
                    with open(filename, "r") as f:
                        for line in f:
                            if (not line[0]=="#") and (not line[0]=="@"):
                                angle.append(float(line.split()[1]))
                    subprocess.run(["rm", filename])
                    avg_angles[group] = angle
            else:
                filename=f"angle_combined_{random_string}.xvg"
                group = "Combined"
                subprocess.run(["gmx", "gangle", "-f", trr_filename, "-s", tpr_filename, "-n", ndx_filename, "-g1", g1, "-g2", g2, "-group1", group, "-seltype", seltype, "-selrpos", selrpos, "-oav", filename], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                angle = []
                with open(filename, "r") as f:
                    for line in f:
                        if (not line[0]=="#") and (not line[0]=="@"):
                            angle.append(float(line.split()[1]))
                subprocess.run(["rm", filename])
                avg_angles[group] = angle
            subprocess.run(["rm", ndx_filename])
            return avg_angles
        else:
            membrane = {'Lower':{}, 'Upper':{}}
            lower_membrane = [atom.resid for atom in mda_universe.select_atoms(f"prop z < {mda_universe.select_atoms('type P').center_of_mass()[2]} and type P")]
            upper_membrane = [atom.resid for atom in mda_universe.select_atoms(f"prop z > {mda_universe.select_atoms('type P').center_of_mass()[2]} and type P")]
            for atom in mda_universe.atoms:
                if atom.resname in selection:
                    if atom.resid in lower_membrane:
                        if atom.type == "P" or atom.name == c_atom_name:
                            if atom.resname not in membrane["Lower"]:
                                membrane["Lower"][atom.resname] = [atom.id]
                            else:
                                membrane["Lower"][atom.resname].append(atom.id)
                    elif atom.resid in upper_membrane:
                        if atom.type == "P" or atom.name == c_atom_name:
                            if atom.resname not in membrane["Upper"]:
                                membrane["Upper"][atom.resname] = [atom.id]
                            else:
                                membrane["Upper"][atom.resname].append(atom.id)
            ndx_filename = f'custom_membrane_{random_string}.ndx'
            with open(ndx_filename, 'w') as f:
                for i in membrane:
                    if grouping == "combine":
                        f.write(f"[ {i}_Combined ]\n")
                    for resgroup in membrane[i]:
                        if grouping != "combine":
                            f.write(f"[ {i}_{resgroup} ]\n")
                        group_str = " ".join([str(i) for i in membrane[i][resgroup]])
                        f.write("\n".join(textwrap.wrap(group_str, 15)))
                        f.write("\n")
            if grouping != "combine":
                for group in selection:
                    filename_lower = f"angle_lower_{group}_{random_string}.xvg"
                    temp_group = f"Lower_{group}"
                    subprocess.run(["gmx", "gangle", "-f", trr_filename, "-s", tpr_filename, "-n", ndx_filename, "-group1", temp_group, "-seltype", "whole_res_com", "-oav", filename_lower], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                for group in selection:
                    filename_upper = f"angle_upper_{group}_{random_string}.xvg"
                    temp_group = f"Upper_{group}"
                    subprocess.run(["gmx", "gangle", "-f", trr_filename, "-s", tpr_filename, "-n", ndx_filename, "-group1", temp_group, "-seltype", "whole_res_com", "-oav", filename_upper], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            else:
                group = "Combined"
                filename_lower = f"angle_lower_{group}_{random_string}.xvg"
                temp_group = f"Lower_{group}"
                subprocess.run(["gmx", "gangle", "-f", trr_filename, "-s", tpr_filename, "-n", ndx_filename, "-group1", temp_group, "-seltype", "whole_res_com", "-oav", filename_lower], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

                filename_upper = f"angle_upper_{group}_{random_string}.xvg"
                temp_group = f"Upper_{group}"
                subprocess.run(["gmx", "gangle", "-f", trr_filename, "-s", tpr_filename, "-n", ndx_filename, "-group1", temp_group, "-seltype", "whole_res_com", "-oav", filename_upper], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

            angle = {'Upper':{}, 'Lower':{}}
            if grouping != "combine":
                for group in selection:
                    for n, i in enumerate([f"angle_upper_{group}_{random_string}.xvg", f"angle_lower_{group}_{random_string}.xvg"]):
                        angle[list(angle.keys())[n]][group] = []
                        with open(i, "r") as f:
                            for line in f:
                                if (not line[0]=="#") and (not line[0]=="@"):
                                    angle[list(angle.keys())[n]][group].append(float(line.split()[1]))
                    subprocess.run(["rm", i])
            else:
                group = "Combined"
                for n, i in enumerate([f"angle_upper_{group}_{random_string}.xvg", f"angle_lower_{group}_{random_string}.xvg"]):
                    angle[list(angle.keys())[n]][group] = []
                    with open(i, "r") as f:
                        for line in f:
                            if (not line[0]=="#") and (not line[0]=="@"):
                                angle[list(angle.keys())[n]][group].append(float(line.split()[1]))
                    subprocess.run(["rm", i])
            subprocess.run(["rm", ndx_filename])
            return angle

def clustering_plots(pdb_file, top_bottom, protein_residue_names, lipids_to_cluster, attached_ligands, mode="pair", plot_name="Cluster", box_side_length = 6):
    class PDB_Atom():
        def __init__(self, pdbline):
            self.Atom_serial_number = str(pdbline[7:12])
            self.Res_name = pdbline[18:22].strip()
            self.chain_identifies = pdbline[22]
            self.Res_number = int(pdbline[23:27])
            self.xcoord = float(pdbline[31:39])
            self.ycoord = float(pdbline[39:47])
            self.zcoord = float(pdbline[47:55])
            self.Temp_factor = pdbline[61:67]
            self.PDBLINE = pdbline
            self.Inner_surface = False
            self.Residue_ID = str(self.Res_number)+"."+str(self.chain_identifies)
            self.Selected_this_run = False
            self.atom_name = pdbline[13:17].strip()

    class Coordinate:
        def __init__(self,x,y,z):
            self.x = x
            self.y = y
            self.z = z
    try:
        # open the PDB_file and read all the data
        with open(pdb_file, 'r') as f:
            all_atoms_list = []
            while True:
                line = f.readline()
                if line[:6] == "ENDMDL":
                    break
                # Parse the PDB data to get all the atomic information in the file
                line = " "+line
                if(line[1:5]=="ATOM" or line[1:5]=="HETA"):
                    a = PDB_Atom(line)
                    all_atoms_list.append(a)
            f.close()
        # Now divide the atoms into lipid and protein atoms
        protein_atoms_list = []
        lipid_atoms_list = []
        ALL_LIPIDS = []
        for atom in all_atoms_list:
            ALL_LIPIDS.append(atom)
            if(atom.Res_name in protein_residue_names):
                protein_atoms_list.append(atom)
            elif(atom.Res_name not in attached_ligands):
                lipid_atoms_list.append(atom)
        min_z = 10000
        max_z =  -10000
        mean_z = 0
        for atom in lipid_atoms_list:
            mean_z += atom.zcoord
            if(atom.zcoord > max_z):
                max_z = atom.zcoord
            if(atom.zcoord < min_z):
                min_z = atom.zcoord

        protein_coordinates_list = []
        lipid_coordinates_list = []

        prev_prot_resnum = protein_atoms_list[0].Res_number
        x, y, z = 0, 0, 0
        count = 0

        for atom in protein_atoms_list:
            if(atom.Res_number == prev_prot_resnum):
                x += atom.xcoord
                y += atom.ycoord
                z += atom.zcoord
                count += 1
            else:
                protein_coordinates_list.append( Coordinate(x/count, y/count, z/count) )
                x, y, z = atom.xcoord, atom.ycoord, atom.zcoord
                prev_prot_resnum = atom.Res_number
                count = 1
        protein_coordinates_list.append(Coordinate(x/count, y/count, z/count))
        x, y, z = 0, 0, 0 
        count = 0
        temp_list = []
        for atom in lipid_atoms_list:
            if(atom.Res_name in lipids_to_cluster):
                temp_list.append(atom)

        lipid_atoms_list = temp_list

        prev_lipid_resnum = lipid_atoms_list[0].Res_number
        for atom in lipid_atoms_list:
            if (atom.Res_number == prev_lipid_resnum):
                x += atom.xcoord
                y += atom.ycoord
                z += atom.zcoord
                count += 1
            else:
                lipid_coordinates_list.append( Coordinate(x/count, y/count, z/count) )
                x, y, z = atom.xcoord, atom.ycoord, atom.zcoord
                prev_lipid_resnum = atom.Res_number
                count = 1
        lipid_coordinates_list.append(Coordinate(x/count, y/count, z/count))
        min_memb_x = min(lipid_atoms_list, key=lambda x: x.xcoord).xcoord
        min_memb_y = min(lipid_atoms_list, key= lambda x: x.ycoord).ycoord
        mean_z = 0
        for coord in lipid_coordinates_list:
            mean_z += coord.z

        for coord in protein_coordinates_list:
            mean_z += coord.z

        mean_z /= (len(protein_coordinates_list)+ len(lipid_coordinates_list))

        z_min_lip = min(lipid_coordinates_list, key=lambda x: x.z).z
        z_max_lip = max(lipid_coordinates_list, key=lambda x: x.z).z

        new_lip_coords = []
        new_prot_coords = []

        for coord in lipid_coordinates_list:
            if(top_bottom == "top"):
                if(coord.z >= mean_z):
                    new_lip_coords.append(coord)
            elif(top_bottom == "bottom"):
                if(coord.z <= mean_z):
                    new_lip_coords.append(coord)
            else:
                new_lip_coords.append(coord)

        lipid_coordinates_list = new_lip_coords

        for coord in protein_coordinates_list:
            if(top_bottom == "top"):
                if(coord.z >= mean_z and coord.z < z_max_lip):
                    new_prot_coords.append(coord)
            elif(top_bottom == "bottom"):
                if(coord.z <= mean_z and coord.z >= z_min_lip):
                    new_prot_coords.append(coord)
            else:
                if(coord.z >= z_min_lip and coord.z <= z_max_lip):
                    new_prot_coords.append(coord)
        protein_coordinates_list = new_prot_coords
        all_coords_list = []
        for coord in lipid_coordinates_list:
            all_coords_list.append(coord)
        for coord in protein_coordinates_list:
            all_coords_list.append(coord)
        min_x = min(ALL_LIPIDS, key= lambda x: x.xcoord).xcoord
        min_y = min(ALL_LIPIDS, key= lambda x: x.ycoord).ycoord
        min_z = min(all_coords_list, key= lambda x: x.z).z

        max_x = max(ALL_LIPIDS, key= lambda x: x.xcoord).xcoord
        max_y = max(ALL_LIPIDS, key= lambda x: x.ycoord).ycoord
        max_z = max(all_coords_list, key= lambda x: x.z).z

        num_partX = math.ceil((max_x - min_x)/box_side_length)
        num_partY = math.ceil((max_y - min_y)/box_side_length)

        protein_cluster_grid = np.zeros((num_partX, num_partY))
        lipid_cluster_grid = np.zeros((num_partX, num_partY))
        for coord in lipid_coordinates_list:
            x_ind = math.floor(coord.x / box_side_length)
            y_ind = math.floor(coord.y / box_side_length)
            lipid_cluster_grid[x_ind][y_ind] += 1

        for coord in protein_coordinates_list:
            x_ind = math.floor(coord.x / box_side_length)
            y_ind = math.floor(coord.y / box_side_length)
            protein_cluster_grid[x_ind][y_ind] += 10
        protein_grid_x = []
        protein_grid_y = []

        x = 0
        y = 0

        for i in protein_cluster_grid:
            y = 0
            for j in i:
                if j!= 0:
                    protein_grid_x.append(x)
                    protein_grid_y.append(y)
                y += 1
            x += 1
        lipid_cluster_grid = np.array(lipid_cluster_grid)
        protein_cluster_grid = np.array(protein_cluster_grid)

        cols = {
            1: 'white',
            2: 'green',
            3: 'purple',
            4: 'blue',
            5: 'red'
        }

        cvr = colors.ColorConverter()
        tmp = sorted(cols.keys())
        cols_rgb = [cvr.to_rgb(cols[k]) for k in tmp]
        intervals = np.array([0, 1, 2, 3, 4, 5]) 
        cmap, norm = colors.from_levels_and_colors(intervals, cols_rgb)

        x, y = np.meshgrid(range(lipid_cluster_grid.shape[0]), range(lipid_cluster_grid.shape[1]))

        s = ","
        fig, ax = plt.subplots()
        p = plt.imshow(lipid_cluster_grid, cmap='YlOrRd' )
        plt.clim(0, 3)
        cbar = plt.colorbar(p, ticks=[0,1,2,3,4])
        ax.scatter(protein_grid_y, protein_grid_x, marker="s", s=5)
        ax.xaxis.tick_top()
        #fig.supxlabel(f"X Bins (nm$^{-1}$)", verticalalignment = 'top')
        plt.xlabel(f"X Bin (nm$^{-1}$)")
        fig.supylabel(f"Y Bin (nm$^{-1}$)",)
        ax.tick_params(labelbottom=False,labeltop=True)
        #fig.tight_layout()
        if mode == "pair":
            plt.savefig(plot_name+"_"+lipids_to_cluster+"_"+attached_ligands[0]+".svg", format="svg", dpi = 1000)
        else:
            plt.savefig(plot_name+"_"+lipids_to_cluster+".svg", format="svg", dpi = 1000)
        plt.close()
    except:
        logger.error(traceback.format_exc(), " at line", format(sys.exc_info()[-1].tb_lineno))
        #sys.exit(0)

def create_graph_1(atom_dict, specific_frame = False, moving_window = False, graph_filename = "", MEMBRANE_PARTITION_THRESHOLD_FRACTION = 0.01):
    '''
    Does some preprocess but dunno what for
    Returns : 
                None
    Outputs : 
                Save a .svg file of the graph created with the specified name
    '''
    if specific_frame:
        frame  = specific_frame
        numKeys = len(atom_dict[frame])
        allKeys = [key for key in atom_dict[frame]]
        for i in range(numKeys):
            forced_atomGroup = atom_dict[frame][allKeys[i]]
            forced_atomGroup.sort(key= lambda x: x.Coords.z)
            forces = [atom.Force.mod() for atom in forced_atomGroup]
            distances = [atom.Coords.z for atom in forced_atomGroup]
            plt.plot(distances, forces)
            plt.title(str(allKeys[i])+"_"+str(frame))
            plt.savefig(f"plots/{allKeys[i]}_{frame}.svg", dpi = 1000)
            plt.close()
    elif moving_window:
        num_frames = len(atom_dict)
        numKeys = len(atom_dict[list(atom_dict.keys())[0]])
        allKeys = [key for key in atom_dict[list(atom_dict.keys())[0]]]
        for i in range(numKeys):
            for frame in range(0, num_frames, moving_window):
                forced_atomGroup = [atom_dict[window][allKeys[i]] for window in range(frame, frame + moving_window)]
                [specific_frame.sort(key= lambda x: x.Coords.z) for specific_frame in forced_atomGroup]
                forces_averaged = []
                for atoms in zip(*forced_atomGroup):
                    average_at_coordinate = Point(0,0,0)
                    for atom in atoms:
                        average_at_coordinate += atom.Force
                    forces_averaged.append(average_at_coordinate.mod()/len(atoms))
                distances = [atom.Coords.z for atom in forced_atomGroup[0]]
                plt.plot(distances, forces_averaged)
                plt.title(f'{allKeys[i]}_{frame}_to_{frame+moving_window}')
                plt.savefig(f"plots/{allKeys[i]}_{frame}_to_{frame+moving_window}.svg", dpi = 1000)
                plt.close()
    else:
        num_frames = len(atom_dict)
        num_frames_orignal = list(atom_dict.keys())[0]  if num_frames == 1 else num_frames
        numKeys = len(atom_dict[list(atom_dict.keys())[0]])
        allKeys = [key for key in atom_dict[list(atom_dict.keys())[0]]]
        fig, axs = plt.subplots(numKeys)
        for i in range(numKeys):
            forced_atomGroup = [atom_dict[window][allKeys[i]] for window in range(0, num_frames)] if num_frames == num_frames_orignal else [atom_dict[num_frames_orignal][allKeys[i]]]
            [specific_frame.sort(key= lambda x: x.Coords.z) for specific_frame in forced_atomGroup]
            if num_frames == num_frames_orignal:
                forces_averaged = []
                for atoms in zip(*forced_atomGroup):
                    average_at_coordinate = Point(0,0,0)
                    for atom in atoms:
                        average_at_coordinate += atom.Force
                    forces_averaged.append(average_at_coordinate.mod()/len(atoms))
            else:
                forces_averaged = [sum(map(lambda x: x.Force.mod(), atoms))/(num_frames_orignal -1) for atoms in zip(*forced_atomGroup)]
            distances = [atom.Coords.z for atom in forced_atomGroup[0]]
            axs[i].plot(distances, forces_averaged)
            axs[i].title.set_text(str(allKeys[i]))
        fig.savefig(f"plots/Average_{allKeys}.svg", dpi = 1000)
        fig.show()
        plt.close()
    

def create_graph_2(atom_dict, graph_filename = "", MEMBRANE_PARTITION_THRESHOLD_FRACTION = 0.01):
    '''
    Does some preprocess but dunno what for
    Returns : 
                None
    Outputs : 
                Save a .svg file of the graph created with the specified name
    '''
    all_atoms_membrane_non_zero_force = []
    for key in atom_dict:
        for atom in atom_dict[key]:
            if(atom.Force.mod()!=0):
                all_atoms_membrane_non_zero_force.append(atom)
    x_min = float(min(all_atoms_membrane_non_zero_force, key= lambda t: t.Coords.x).Coords.x)
    y_min = float(min(all_atoms_membrane_non_zero_force, key= lambda t: t.Coords.y).Coords.y)
    
    for atom in all_atoms_membrane_non_zero_force:
        atom.Coords.x = float(atom.Coords.x) + abs(x_min)
        atom.Coords.y = float(atom.Coords.y) + abs(y_min)

    x_min = float(min(all_atoms_membrane_non_zero_force, key= lambda t: t.Coords.x).Coords.x)
    y_min = float(min(all_atoms_membrane_non_zero_force, key= lambda t: t.Coords.y).Coords.y)
    x_max = float(max(all_atoms_membrane_non_zero_force, key= lambda t: t.Coords.x).Coords.x)
    y_max = float(max(all_atoms_membrane_non_zero_force, key= lambda t: t.Coords.y).Coords.y)
    f_min = float(min(all_atoms_membrane_non_zero_force, key= lambda t: t.Force.mod()).Force.mod())
    f_max = float(max(all_atoms_membrane_non_zero_force, key= lambda t: t.Force.mod()).Force.mod())
    num_partitions = math.ceil(MEMBRANE_PARTITION_THRESHOLD_FRACTION * len(all_atoms_membrane_non_zero_force))
    num_partitions_x = math.ceil((abs(x_max) - abs(x_min)))*10 
    num_partitions_y = math.ceil((abs(y_max) - abs(y_min)))*10

    # X_part_width = math.ceil((x_max - x_min)/num_partitions)
    # Y_part_width = math.ceil((y_max - y_min)/num_partitions)
    #initializing force array for the non zero force atoms of the membrane  
    part = int(max(num_partitions_x, num_partitions_y))
    z = np.zeros((part, part))   
    for atom in all_atoms_membrane_non_zero_force:
        x_index = math.floor(float(float(atom.Coords.x))*10)
        y_index = math.floor(float(float(atom.Coords.y))*10)
        z[x_index][y_index] += atom.Force.mod()

    x, y = np.meshgrid(range(z.shape[0]), range(z.shape[1]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z)
    plt.title('z as 3d height map')
    plt.savefig("test_2.svg")
    plt.show()
    # show hight map in 2d

    plt.figure()
    plt.title('z as 2d heat map')
    p = plt.imshow(z)
    plt.colorbar(p)
    plt.savefig("test_3.svg")
    plt.show()