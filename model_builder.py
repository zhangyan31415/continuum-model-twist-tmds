import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import eigh  
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import json
import os

# Define Pauli matrices globally for accessibility in projection functions
PAULI_MATRICES = [
    np.array([[1, 0], [0, 1]]),    # Identity matrix
    np.array([[0, 1], [1, 0]]),    # Pauli X
    np.array([[0, -1j], [1j, 0]]), # Pauli Y
    np.array([[1, 0], [0, -1]])    # Pauli Z
]

class ContinuousModel:
    def __init__(self, material, angle, model_type, path_model, kpath):
        """
        Initialize a ContinuousModel instance.
        
        Parameters:
        material (str): Name of the material, e.g., "tMoTe2", "tWSe2"
        angle (str): Angle parameter, e.g., '2.13'
        model_type (str): Type of model, e.g., 'full', 'reduced'
        path_model (str): Base path to the model data
        kpath (np.ndarray): Fractional k-path
        """
        self.material = material
        self.angle = angle
        self.model_type = model_type
        self.path_model = path_model
        self.kpath_frac = kpath
        
        
        # Initialize attributes
        self.bM1 = None
        self.bM2 = None
        self.q1 = None
        self.q2 = None
        self.q3 = None
        self.Qlayer1 = None
        self.Qlayer2 = None
        self.Qset = None
        self.C2yT_proj_matrix = None
        self.C3_proj_matrix = None

        self.Fk0xmat = None
        self.Fk0ymat = None

        self.kpath_cart = None

        self.eigenvalues = None
        self.eigenvectors = None
        
        # Load data
        self.load_data()
        
        # Define constants
        self.define_constants()
        
        # Generate projection matrices
        self.generate_projections()
    
    def rot(self, vec, theta):
        """
        Rotate a 2D or 3D vector by a specified angle in degrees.
        
        Parameters:
        vec (array-like): The vector to rotate.
        theta (float): The rotation angle in degrees.
        
        Returns:
        np.ndarray: The rotated vector.
        """
        theta_rad = np.deg2rad(theta)
        rot_mat = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad),  np.cos(theta_rad)]
        ])
        if len(vec) == 2:
            return np.dot(rot_mat, vec)
        elif len(vec) == 3:
            temp = np.zeros(3)
            temp[:2] = np.dot(rot_mat, vec[:2])
            temp[2] = vec[2]
            return temp
    
    def load_json_data(self, filepath):
        """
        Load data from a JSON file, converting complex numbers and tuple-like strings back to their original format.
        
        Parameters:
        filepath (str): Path to the JSON file.
        
        Returns:
        dict: The loaded data with converted types.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        def convert_back(data):
            if isinstance(data, dict):
                # Convert keys back to tuples if possible and recursively process values
                return {
                    eval(k) if ',' in k else k: convert_back(v)
                    for k, v in data.items()
                }
            elif isinstance(data, str) and ('+' in data or 'j' in data):
                # Convert strings that represent complex numbers back to complex type
                try:
                    return complex(data)
                except ValueError:
                    return data  # Return the original string if conversion fails
            return data

        return convert_back(data)
    
    def load_data(self):
        """
        Load necessary data files, including Qlayer1, Qlayer2, coefficients, and kpath.
        """
        Qlayer1_path = os.path.join(self.path_model, self.material, self.model_type, f"{self.angle}_Qlayer1.txt")
        Qlayer2_path = os.path.join(self.path_model, self.material, self.model_type, f"{self.angle}_Qlayer2.txt")
        coeff_path = os.path.join(self.path_model, self.material, self.model_type, f"coeff_{self.angle}.json")
        
        # Check if files exist
        for file_path in [Qlayer1_path, Qlayer2_path, coeff_path]:
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Error: The file {file_path} does not exist.")
        
        # Load data
        try:
            self.Qlayer1 = np.loadtxt(Qlayer1_path)
            self.Qlayer2 = np.loadtxt(Qlayer2_path)
            coeff = self.load_json_data(coeff_path)
            self.m_coeff = coeff['diag']
            self.intra = coeff['intra']
            self.inter = coeff['inter']
            self.Qset = np.concatenate((self.Qlayer1, self.Qlayer2), axis=0)
        except Exception as e:
            raise RuntimeError(f"Error loading data: {e}")
    
    def define_constants(self):
        """
        Define constants based on the loaded data, including bM1, bM2, q1, q2, q3.
        """
        q_norm = np.min(np.linalg.norm(self.Qset, axis=1))
        self.q1 = np.array([0, q_norm])
        self.q2 = self.rot(self.q1, 120)
        self.q3 = self.rot(self.q1, 240)
        self.bM1 = np.array([np.sqrt(3) * q_norm, 0])
        self.bM2 = self.rot(self.bM1, 60)
        self.BM_mat = np.array([self.bM1, self.bM2])

        # Compute Fk0 matrices
        self.Fk0xmat = np.array([
            [-Q2[0] if np.allclose(Q1, Q2, atol=1e-5) else 0 for Q2 in self.Qset] 
            for Q1 in self.Qset
        ])
        self.Fk0ymat = np.array([
            [-Q2[1] if np.allclose(Q1, Q2, atol=1e-5) else 0 for Q2 in self.Qset] 
            for Q1 in self.Qset
        ])

        # Convert kpath_frac to Cartesian coordinates
        self.kpath_cart = np.dot(self.kpath_frac[:, :2], self.BM_mat)

    
    def generate_projections(self):
        """
        Generate C2yT and C3 projection matrices.
        """
        self.C2yT_proj_matrix = self.C2yT_K1_valley_spin_up_proj(self.Qset)
        self.C3_proj_matrix = self.C3K1_valley_spin_up_proj(self.Qset)
    
    def rotation_matrix(self, theta):
        """
        Return a 2D rotation matrix for a given angle in radians.
        
        Parameters:
        theta (float): Rotation angle in radians.
        
        Returns:
        np.ndarray: The rotation matrix.
        """
        return np.array([
            [np.cos(theta), -np.sin(theta)], 
            [np.sin(theta),  np.cos(theta)]
        ])
    
    def C3K1_valley_spin_up_proj(self, Qset):
        """
        Generate the C3K1 valley spin-up projection matrix.
        
        Parameters:
        Qset (np.ndarray): Set of Q vectors.
        rotation_func (callable): Function to perform rotation.
        
        Returns:
        np.ndarray: The projection matrix.
        """
        q1norm = np.linalg.norm(Qset[0])
        value = np.exp(1j * np.pi / 3)
    
        C3_proj_matrix = np.array([
            [value if np.linalg.norm(Q1 - self.rotation_matrix(np.deg2rad(120)) @ Q2) < q1norm / 10 else 0 
             for Q2 in Qset] 
            for Q1 in Qset
        ])
    
        return C3_proj_matrix
    
    def C2yT_K1_valley_spin_up_proj(self, Qset):
        """
        Generate the C2yT K1 valley spin-up projection matrix.
        
        Parameters:
        Qset (np.ndarray): Set of Q vectors.
        pauli_matrix (np.ndarray): Pauli matrix to use in projection.
        
        Returns:
        np.ndarray: The projection matrix.
        """
        q1norm = np.linalg.norm(Qset[0])
        C2yT_proj_matrix = np.array([
            [1 if np.linalg.norm(Q1 - PAULI_MATRICES[3] @ Q2) < q1norm / 10 else 0 
             for Q2 in Qset] 
            for Q1 in Qset
        ])
    
        return C2yT_proj_matrix
    
    def C2yT_proj(self, matrix):
        """
        Apply the C2yT projection to a given matrix.
        
        Parameters:
        matrix (np.ndarray): The matrix to project.
        
        Returns:
        np.ndarray: The projected matrix.
        """
        C2yT_G_rep_conj = self.C2yT_proj_matrix.T.conj()
        return self.C2yT_proj_matrix @ matrix.conj() @ C2yT_G_rep_conj
    
    def C3_proj(self, matrix):
        """
        Apply the C3 projection to a given matrix.
        
        Parameters:
        matrix (np.ndarray): The matrix to project.
        
        Returns:
        np.ndarray: The projected matrix.
        """
        return self.C3_proj_matrix @ matrix @ self.C3_proj_matrix.T.conj()
    
    def symmetrize_term(self, term):
        """
        Symmetrize a given term using C3 and C2yT projections.
        
        Parameters:
        term (list or array): The terms to symmetrize.
        
        Returns:
        np.ndarray: The symmetrized term.
        """
        term_symm = (
            term[0] + 
            self.C3_proj(term[1]) + 
            self.C3_proj(self.C3_proj(term[2])) + 
            self.C2yT_proj(term[3]) + 
            self.C2yT_proj(self.C3_proj(term[4])) + 
            self.C2yT_proj(self.C3_proj(self.C3_proj(term[5])))
        )
    
        term_symm += term_symm.T.conj()
    
        return term_symm
    
    def kinetic(self, k, m, n):
        """
        Calculate the kinetic matrix element for given k, Qset, and indices m, n.
        
        Parameters:
        k (array-like): The k-point vector.
        m (int): Power index.
        n (int): Power index.
        
        Returns:
        np.ndarray: The kinetic matrix.
        """
        kinetic_matrix = np.zeros((len(self.Qset), len(self.Qset)), dtype=complex)
        
        def calculate_term(k, m, n, Q1, Q2):
            term1 = np.dot(k - Q1, np.array([1, 1j]))
            term2 = np.dot(k - Q1, np.array([1, -1j]))
            if np.allclose(Q1, Q2, atol=1e-5):
                if m == 0:
                    return term1 ** n + term2 ** n
                elif n == 0:
                    return term1 ** m + term2 ** m
                elif m != n:
                    return term1 ** n * term2 ** m + term1 ** m * term2 ** n
                else:
                    return term1 ** n * term2 ** n
            return 0
        
        for i, Q1 in enumerate(self.Qset):
            kinetic_matrix[i, i] += calculate_term(k, m, n, Q1, Q1)
        
        return kinetic_matrix
    
    def generate_C3invariant_terms(self, term, kplusterm, kminusterm, m, n):
        """
        Generate C3 invariant terms for the Hamiltonian.
        
        Parameters:
        term (np.ndarray): The term matrix.
        kplusterm (list of np.ndarray): List of k+ term matrices.
        kminusterm (list of np.ndarray): List of k- term matrices.
        m (int): Power index.
        n (int): Power index.
        
        Returns:
        list of np.ndarray: Symmetrized terms.
        """
        factor = (
            term +
            np.exp((m - n) * 1j * 2 * np.pi / 3) * self.C3_proj(term) +
            np.exp((m - n) * 1j * 4 * np.pi / 3) * self.C3_proj(self.C3_proj(term))
        )
        terms = []
        term1 = []
        term2 = []
        for i in range(6):
            temp = np.linalg.matrix_power(kplusterm[i], n) @ np.linalg.matrix_power(kminusterm[i], m)
            term1.append(factor @ temp)
            term2.append(temp @ factor)

        term1 = np.array(term1)
        term2 = np.array(term2)
        terms.extend([term1, 1j * term1])

        terms = np.array(terms)
    
        terms = [self.symmetrize_term(term) for term in terms]
    
        return terms
    
    def GenTermQ0plusQ1(self, k, harmonic, m, n, hoppingtype):
        """
        Generate terms for Q0 + Q1 in the Hamiltonian.
        
        Parameters:
        k (array-like): The k-point vector.
        harmonic (int): Harmonic index.
        m (int): Power index.
        n (int): Power index.
        hoppingtype (str): 'intra' or 'inter'.
        
        Returns:
        list of np.ndarray: Generated terms.
        """
        intra_harmonics_map = {
            1: self.bM1,
            2: self.bM1 + self.bM2,
            3: 2 * self.bM1,
            4: 2 * self.bM1 + self.bM2,
            5: 3 * self.bM1
        }
    
        inter_harmonics_map = {
            1: self.q1,
            2: -2 * self.q1,
            3: self.q1 + self.bM2,
            4: 2 * self.bM2 + self.q3,
            5: 4 * self.q1,
            6: self.q1 + 2 * self.bM2
        }
        harmonic = int(harmonic)
        if hoppingtype == "inter":
            p = inter_harmonics_map.get(harmonic)
            term = np.array([
                [1 if np.linalg.norm(Q1 - Q2 - p) < 1e-2 else 0 for Q2 in self.Qset] 
                for Q1 in self.Qset
            ])
        else:
            p = intra_harmonics_map.get(harmonic)
            term = np.array([
                [1 if (np.linalg.norm(Q1 - Q2 - p) < 1e-2 and 
                       np.min([np.linalg.norm(Q1 - q) for q in self.Qlayer1]) < 1e-3) else 0 
                 for Q2 in self.Qset] 
                for Q1 in self.Qset
            ])
    
        kx, ky = k
        k_symm = [
            [kx, ky], 
            self.rot([kx, ky], -120), 
            self.rot([kx, ky], -240),
            [kx, -ky], 
            self.rot([kx, -ky], -120), 
            self.rot([kx, -ky], -240)
        ]
        
        kplusterm = []
        kminusterm = []
    
        for ks in k_symm:
            kx_symm, ky_symm = ks
            kplusterm.append(
                (kx_symm + 1j * ky_symm) * np.eye(len(self.Fk0xmat)) + self.Fk0xmat + 1j * self.Fk0ymat
            )
            kminusterm.append(
                (kx_symm - 1j * ky_symm) * np.eye(len(self.Fk0xmat)) + self.Fk0xmat - 1j * self.Fk0ymat
            )
    
        initial_terms = self.generate_C3invariant_terms(term, kplusterm, kminusterm, m, n)
    
        return initial_terms
    
    def H_model(self, k, coeff_list_diag, coeff_list_offdiag_intra, coeff_list_offdiag_inter):
        """
        Generate the full Hamiltonian matrix.
        
        Parameters:
        k (array-like): The k-point vector.
        coeff_list_diag (dict): Diagonal coefficients.
        coeff_list_offdiag_intra (dict): Intra hopping coefficients.
        coeff_list_offdiag_inter (dict): Inter hopping coefficients.
        
        Returns:
        np.ndarray: The Hamiltonian matrix.
        """
        Hamk = np.zeros((len(self.Qset), len(self.Qset)), dtype=complex)
        for key, coeff in coeff_list_diag.items():
            m, n = map(int, key)
            Hamk += coeff * self.kinetic(k, m, n)
    
        for harmonic, terms in coeff_list_offdiag_intra.items():
            for key, coeff in terms.items():
                m, n = map(int, key)
                hoppingtype = 'intra'
                terms_generated = self.GenTermQ0plusQ1(
                    k, harmonic, m, n, hoppingtype
                )
                Hamk += np.real(coeff) * terms_generated[0] 
                Hamk += np.imag(coeff) * terms_generated[1]
    
        for harmonic, terms in coeff_list_offdiag_inter.items():
            for key, coeff in terms.items():
                m, n = map(int, key)
                hoppingtype = 'inter'
                terms_generated = self.GenTermQ0plusQ1(
                    k, harmonic, m, n, hoppingtype
                )
                Hamk += np.real(coeff) * terms_generated[0] 
                Hamk += np.imag(coeff) * terms_generated[1]
    
        return Hamk
    

    def get_hamiltonian(self, k_point):
        """
        Build Hamiltonian for a single k-point.

        Parameters:
        k_point (array-like): The k-point vector.

        Returns:
        np.ndarray: The Hamiltonian matrix at the given k-point.
        """
        Hk = self.H_model(
            k=k_point,
            coeff_list_diag=self.m_coeff,
            coeff_list_offdiag_intra=self.intra,
            coeff_list_offdiag_inter=self.inter,
        )
        return Hk
    
    def build_hamiltonians1(self, n_jobs=-1):
        """
        Build Hamiltonians for all k-points along the kpath in parallel with a progress bar.
        
        Parameters:
        n_jobs (int): Number of parallel jobs.
                    -1 means using all available CPU cores.
                    Positive integers specify the exact number of cores.
        """
        # Define the function to build a single Hamiltonian
        def build_single_hamiltonian(k_point):
            """
            Construct the Hamiltonian matrix for a given k-point.
            
            Parameters:
            k_point (array-like): The k-point vector.
            
            Returns:
            np.ndarray: The Hamiltonian matrix for the k-point.
            """
            return self.get_hamiltonian(k_point)
        
        # Wrap the Parallel processing within tqdm_joblib to display a progress bar
        with tqdm_joblib(tqdm(desc="Building Hamiltonians", total=len(self.kpath_cart))):
            # Parallelize the Hamiltonian construction across all k-points
            self.Hamiltonians = Parallel(n_jobs=n_jobs)(
                delayed(build_single_hamiltonian)(k) for k in self.kpath_cart
            )
        
        # Confirmation message upon completion
        print("All Hamiltonians built successfully.")
  
    def diagonalize_hamiltonians1(self, n_jobs=-1):
        """
        Diagonalize all Hamiltonian matrices to obtain eigenvalues and eigenvectors with a progress bar.
        
        Parameters:
        n_jobs (int): Number of parallel jobs. 
                    -1 means using all available CPU cores. 
                    Positive integers specify the exact number of cores.
        """
        # Define the diagonalization function for a single Hamiltonian matrix
        def diagonalize(Hk):
            """
            Diagonalize a single Hamiltonian matrix.
            
            Parameters:
            Hk (np.ndarray): Hamiltonian matrix for a specific k-point.
            
            Returns:
            tuple: Sorted eigenvalues and corresponding eigenvectors.
            """
            eigenvals, eigenvecs = eigh(Hk)  # Compute eigenvalues and eigenvectors
            idx = np.argsort(eigenvals)      # Get indices that would sort the eigenvalues
            sorted_eigenvals = eigenvals[idx]
            sorted_eigenvecs = eigenvecs[:, idx]
            return sorted_eigenvals, sorted_eigenvecs
        
        
        # Use tqdm_joblib to add a progress bar to joblib's Parallel
        with tqdm_joblib(tqdm(desc="Diagonalizing Hamiltonians", total=len(self.Hamiltonians))):
            # Parallelize the diagonalization across all Hamiltonian matrices
            results = Parallel(n_jobs=n_jobs)(
                delayed(diagonalize)(Hk) for Hk in self.Hamiltonians
            )
        
        # Unzip the results into eigenvalues and eigenvectors
        self.eigenvalues, self.eigenvectors = zip(*results)
        
        # Convert to NumPy arrays for easier manipulation
        self.eigenvalues = np.array(self.eigenvalues) - np.max(self.eigenvalues)  # Shift energies
        self.eigenvectors = np.array(self.eigenvectors)
        
        # Confirmation message upon completion
        print("All Hamiltonians diagonalized successfully.")

    def build_hamiltonians(self, n_jobs=-1, show_progress=False):
        """
        Build Hamiltonians for all k-points along the kpath in parallel with an optional progress bar.
        
        Parameters:
        n_jobs (int): Number of parallel jobs.
                     -1 means using all available CPU cores.
                     Positive integers specify the exact number of cores.
        show_progress (bool): If True, display a progress bar. Defaults to True.
                              Note: Progress bars display correctly when running Python scripts (.py files).
                              They may not render properly in interactive environments like Jupyter Notebooks.
        """
        # Define the function to build a single Hamiltonian
        def build_single_hamiltonian(k_point):
            """
            Construct the Hamiltonian matrix for a given k-point.
            
            Parameters:
            k_point (array-like): The k-point vector.
            
            Returns:
            np.ndarray: The Hamiltonian matrix for the k-point.
            """
            return self.get_hamiltonian(k_point)
        
        # Ensure that kpath_cart is defined
        if not hasattr(self, 'kpath_cart') or self.kpath_cart is None:
            raise ValueError("kpath_cart is not defined. Please ensure kpath_cart is set before building Hamiltonians.")
        
        # Conditional progress bar setup
        if show_progress:
            with tqdm_joblib(tqdm(desc="Building Hamiltonians", total=len(self.kpath_cart), unit="Hk")):
                # Parallelize the Hamiltonian construction across all k-points
                self.Hamiltonians = Parallel(n_jobs=n_jobs)(
                    delayed(build_single_hamiltonian)(k) for k in self.kpath_cart
                )
        else:
            # Parallelize without progress bar
            self.Hamiltonians = Parallel(n_jobs=n_jobs)(
                delayed(build_single_hamiltonian)(k) for k in self.kpath_cart
            )
        
        # Confirmation message upon completion
        print("All Hamiltonians built successfully.")

    def diagonalize_hamiltonians(self, n_jobs=-1, show_progress=False):
        """
        Diagonalize all Hamiltonian matrices to obtain eigenvalues and eigenvectors with an optional progress bar.
        
        Parameters:
        n_jobs (int): Number of parallel jobs. 
                     -1 means using all available CPU cores. 
                     Positive integers specify the exact number of cores.
        show_progress (bool): If True, display a progress bar. Defaults to True.
                              Note: Progress bars display correctly when running Python scripts (.py files).
                              They may not render properly in interactive environments like Jupyter Notebooks.
        """
        # Define the diagonalization function for a single Hamiltonian matrix
        def diagonalize(Hk):
            """
            Diagonalize a single Hamiltonian matrix.
            
            Parameters:
            Hk (np.ndarray): Hamiltonian matrix for a specific k-point.
            
            Returns:
            tuple: Sorted eigenvalues and corresponding eigenvectors.
            """
            eigenvals, eigenvecs = eigh(Hk)  # Compute eigenvalues and eigenvectors
            idx = np.argsort(eigenvals)      # Get indices that would sort the eigenvalues
            sorted_eigenvals = eigenvals[idx]
            sorted_eigenvecs = eigenvecs[:, idx]
            return sorted_eigenvals, sorted_eigenvecs
        
        # Ensure that Hamiltonians are built before diagonalization
        if not hasattr(self, 'Hamiltonians') or self.Hamiltonians is None:
            raise ValueError("Hamiltonians not built. Please build Hamiltonians before diagonalization.")
        
        # Conditional progress bar setup
        if show_progress:
            with tqdm_joblib(tqdm(desc="Diagonalizing Hamiltonians", total=len(self.Hamiltonians), unit="Diag")):
                # Parallelize the diagonalization across all Hamiltonian matrices
                results = Parallel(n_jobs=n_jobs)(
                    delayed(diagonalize)(Hk) for Hk in self.Hamiltonians
                )
        else:
            # Parallelize without progress bar
            results = Parallel(n_jobs=n_jobs)(
                delayed(diagonalize)(Hk) for Hk in self.Hamiltonians
            )
        
        # Unzip the results into eigenvalues and eigenvectors
        self.eigenvalues, self.eigenvectors = zip(*results)
        
        # Convert to NumPy arrays for easier manipulation
        self.eigenvalues = np.array(self.eigenvalues) - np.max(self.eigenvalues)  # Shift energies
        self.eigenvectors = np.array(self.eigenvectors)
        
        # Confirmation message upon completion
        print("All Hamiltonians diagonalized successfully.")

    def plot_band_structure(self, high_symmetry_kpoints, high_symmetry_labels, ymin=None, ymax=None,
                            title='Band Structure', ylabel='Energy (meV)',
                            line_color='black', line_width=1.0):
        """
        Plot the band structure with customizable options.

        Parameters:
        high_symmetry_kpoints (list of int): Positions of high-symmetry points along the kpath.
        high_symmetry_labels (list of str): Labels for the high-symmetry points.
        ymin (float, optional): Minimum energy to display on y-axis.
        ymax (float, optional): Maximum energy to display on y-axis.
        title (str, optional): Title of the plot. Defaults to 'Band Structure'.
        xlabel (str, optional): Label for the x-axis. Defaults to 'Path'.
        ylabel (str, optional): Label for the y-axis. Defaults to 'Energy (meV)'.
        line_color (str, optional): Color of the energy bands. Defaults to 'black'.
        line_width (float, optional): Width of the energy band lines. Defaults to 1.0.
        """
        # Check if eigenvalues and kpath are available
        if self.eigenvalues is None:
            raise ValueError("Eigenvalues not computed. Please diagonalize the Hamiltonian first.")

        # Compute cumulative distance for x-axis
        distances = [0]
        kpath_cart = self.kpath_frac[:, :2] @ self.BM_mat
        for i in range(1, len(kpath_cart)):
            dk = kpath_cart[i] - kpath_cart[i - 1]
            distance = np.linalg.norm(dk)
            distances.append(distances[-1] + distance)
        x_axis = np.array(distances)

        # Validate high_symmetry_kpoints and labels
        if len(high_symmetry_kpoints) != len(high_symmetry_labels):
            raise ValueError("The number of high symmetry k-points must match the number of labels.")

        # Set y-axis limits
        if ymin is None:
            ymin = np.min(self.eigenvalues[:,-6]) * 1.1  # Automatically set based on data
        if ymax is None:
            ymax = - np.min(self.eigenvalues[:,-6]) * 0.1  # Automatically set based on data


        # Create plot
        fig, ax = plt.subplots(figsize=(3.5, 5))  # Adjust figure size as needed

        # Plot each band
        for band_index in range(self.eigenvalues.shape[1]):
            ax.plot(x_axis, self.eigenvalues[:, band_index], color=line_color, linewidth=line_width)

        # Add vertical lines at high-symmetry points
        for point in high_symmetry_kpoints:
            ax.axvline(x=x_axis[point], color='gray', linestyle='-', linewidth=0.5)
        # Add horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

        # Set x-ticks and labels
        ax.set_xticks(x_axis[high_symmetry_kpoints])
        ax.set_xticklabels(high_symmetry_labels)

        # Set axis labels and limits
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_xlim(0, x_axis[-1])
        ax.set_ylim(ymin, ymax)

        # Set title
        ax.set_title(title, fontsize=16)

        # Adjust layout
        plt.tight_layout()

        # Show plot
        plt.show()
