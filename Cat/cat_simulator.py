from typing import Optional

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

class CatSimulator:
    def __init__(self,
                 memory_N: int,
                 buffer_N: Optional[int] = None):
        self.memory_N = memory_N
        self.buffer_N = buffer_N

        # initialize operators
        if self.buffer_N is not None:
            self.a_memory = qt.tensor(qt.destroy(self.memory_N), qt.qeye(self.buffer_N))
            self.a_buffer = qt.tensor(qt.qeye(self.memory_N), qt.destroy(self.buffer_N))
        else:
            self.a_memory = qt.destroy(self.memory_N)

        self.all_states = []
        self.all_times = []
    
    def get_current_state(self):
        if len(self.all_states) == 0:
            if self.buffer_N is not None:
                return qt.tensor(qt.fock_dm(self.memory_N, 0), qt.fock_dm(self.buffer_N, 0))
            return qt.fock_dm(self.memory_N, 0)
        return self.all_states[-1]
    
    def get_current_time(self):
        if len(self.all_times) == 0:
            return 0
        return self.all_times[-1]

    def simulate_stabalization(self, T, dt,
                 alpha_stab: float = 0,
                 kappa_2photon: float = 0,
                 kappa_1photon_memory: float = 0,
                 kappa_1photon_buffer: float = 0,
                 epsilon_drive_memory: float = 0,
                 epsilon_squeezing_memory: float = 0,
                 epsilon_drive_buffer: float = 0,
                 g_2: float = 0,
                 delta_memory: float = 0,
                 delta_buffer: float = 0,
                 initial_state: Optional[qt.Qobj] = None):

        tlist = np.arange(self.get_current_time(), self.get_current_time() + T, dt)

        # if not use buffer mode
        if self.buffer_N is None:
            # define the Hamiltonian and collapse operators
            H = delta_memory * self.a_memory.dag() * self.a_memory + epsilon_drive_memory * self.a_memory + np.conj(epsilon_drive_memory) * self.a_memory.dag() + epsilon_squeezing_memory * (self.a_memory**2) + np.conj(epsilon_squeezing_memory) * (self.a_memory**2).dag()
            c_ops = [np.sqrt(kappa_2photon) * (self.a_memory**2 - alpha_stab**2),
                     np.sqrt(kappa_1photon_memory) * self.a_memory]

            initial_state = self.get_current_state() if initial_state is None else initial_state
            result = qt.mesolve(H, initial_state, tlist, c_ops)

            self.all_states.extend(result.states)
            self.all_times.extend(result.times)
        else:
            # define the Hamiltonian and collapse operators
            H = delta_memory * self.a_memory.dag() * self.a_memory + epsilon_drive_memory * self.a_memory + np.conj(epsilon_drive_memory) * self.a_memory.dag() + epsilon_squeezing_memory * (self.a_memory**2) + np.conj(epsilon_squeezing_memory) * (self.a_memory**2).dag() + delta_buffer * self.a_buffer.dag() * self.a_buffer + epsilon_drive_buffer * self.a_buffer + np.conj(epsilon_drive_buffer) * self.a_buffer.dag() + g_2 * self.a_memory**2 * self.a_buffer.dag() + np.conj(g_2) * (self.a_memory**2).dag() * self.a_buffer
            c_ops = [np.sqrt(kappa_1photon_memory) * self.a_memory,
                     np.sqrt(kappa_1photon_buffer) * self.a_buffer]

            initial_state = self.get_current_state() if initial_state is None else initial_state
            result = qt.mesolve(H, initial_state, tlist, c_ops)

            self.all_states.extend(result.states)
            self.all_times.extend(result.times)

def plot_wigner(rho, title_str, xmax, npts):
    """
    Plot Wigner function of density matrix rho.
    rho : input density matrix
    xmax : maximum x and p value to plot (in coherent state units)
    npts : number of points along each axis
    title_str : string for plot title
    """

    # Define phase-space grid in physical units
    xlist = np.linspace(-xmax * np.sqrt(2), xmax * np.sqrt(2), npts)
    ylist = np.linspace(-xmax * np.sqrt(2), xmax * np.sqrt(2), npts)

    W = qt.wigner(rho, xlist, ylist) # get Wigner function values

    # make color scale symmetric around 0
    Wmax = np.max(np.abs(W))

    plt.figure(figsize=(6, 5))
    plt.imshow(
        W,
        extent=[xlist[0] / np.sqrt(2), xlist[-1] / np.sqrt(2),
                ylist[0] / np.sqrt(2), ylist[-1] / np.sqrt(2)],
        origin='lower',
        aspect='equal',
        cmap= 'bwr',
        vmin=-Wmax,
        vmax=+Wmax
    )

    # put ticks at every integer (…,-3,-2,-1,0,1,2,3,…)
    xmin = xlist[0] / np.sqrt(2)
    xmax = xlist[-1] / np.sqrt(2)
    ymin = ylist[0] / np.sqrt(2)
    ymax = ylist[-1] / np.sqrt(2)

    xticks = np.arange(np.ceil(xmin), np.floor(xmax) + 1, 1)
    yticks = np.arange(np.ceil(ymin), np.floor(ymax) + 1, 1)

    plt.xticks(xticks)
    plt.yticks(yticks)

    plt.xlabel(r'$\Re(\alpha)$')
    plt.ylabel(r'$\Im(\alpha)$')
    plt.title(title_str)
    plt.colorbar(label='W(q, p)')
    plt.tight_layout()
    plt.show()


def plot_photon_number(rho, title_str):
    """
    Plot photon number distribution of density matrix rho.
    rho        : input density matrix
    title_str  : string for plot title
    """
    if rho.isket:
        rho = qt.ket2dm(rho)
    N = rho.shape[0]
    populations = np.real(rho.diag())  # P(n) = <n|rho|n>

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(N), populations, color='steelblue', edgecolor='white', linewidth=0.5)
    ax.set_xlabel(r'Photon number $n$', fontsize=13)
    ax.set_ylabel(r'$P(n)$', fontsize=13)
    ax.set_title(title_str, fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=11)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.6, alpha=0.4)
    fig.tight_layout()
    plt.show()
