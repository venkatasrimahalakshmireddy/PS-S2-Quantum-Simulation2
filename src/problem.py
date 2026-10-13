import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

N_QUBITS = 4
J = 1.0

def tfim_operator(g):
    terms = []
    for i in range(N_QUBITS - 1):
        label = ["I"] * N_QUBITS
        label[N_QUBITS - 1 - i] = "Z"
        label[N_QUBITS - 2 - i] = "Z"
        terms.append(("".join(label), -J))
    for i in range(N_QUBITS):
        label = ["I"] * N_QUBITS
        label[N_QUBITS - 1 - i] = "X"
        terms.append(("".join(label), -float(g)))
    return SparsePauliOp.from_list(terms)

def exact_hamiltonian_matrix(g):
    # Matrix built from Qiskit's Pauli operator to keep bit ordering consistent.
    return np.asarray(tfim_operator(g).to_matrix(), dtype=complex)

def initial_state_vector():
    # Qiskit circuit qubit 0 is the rightmost bit in displayed ket notation.
    # Prepare |0101> by applying X to qubits 0 and 2.
    state = np.zeros(2**N_QUBITS, dtype=complex)
    state[int("0101", 2)] = 1.0
    return state

def exact_z0(g, t, initial="0101"):
    psi0 = np.zeros(2**N_QUBITS, dtype=complex)
    psi0[int(initial, 2)] = 1.0
    psi = expm(-1j * exact_hamiltonian_matrix(g) * t) @ psi0
    # Qubit 0 is least-significant bit in computational basis indexing.
    z = np.array([1.0 if ((i >> 0) & 1) == 0 else -1.0 for i in range(2**N_QUBITS)])
    return float(np.real(np.vdot(psi, z * psi)))

def trotter_circuit(g, t=2.0, steps=1, measure=True):
    qc = QuantumCircuit(N_QUBITS, N_QUBITS if measure else 0)
    # |0101> in displayed ket order -> X on q0 and q2.
    qc.x(0)
    qc.x(2)
    dt = float(t) / int(steps)
    for _ in range(int(steps)):
        # H_Z=-J ZZ; exp(-i dt H_Z)=exp(+i J dt ZZ).
        # RZZ(theta)=exp(-i theta ZZ/2), hence theta=-2 J dt.
        for i in range(N_QUBITS - 1):
            qc.rzz(-2.0 * J * dt, i, i + 1)
        # H_X=-g X; RX(theta)=exp(-i theta X/2), theta=-2 g dt.
        for i in range(N_QUBITS):
            qc.rx(-2.0 * float(g) * dt, i)
    if measure:
        qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc

def z0_from_counts(counts, shots=None):
    total = sum(counts.values()) if shots is None else shots
    if total == 0:
        return float("nan")
    # Classical bit c0 is the rightmost character in Qiskit count strings.
    return sum((1 if key.replace(" ", "")[-1] == "0" else -1) * value
               for key, value in counts.items()) / total
