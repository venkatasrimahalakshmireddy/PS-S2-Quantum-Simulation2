import json
from pathlib import Path
from qiskit.transpiler import CouplingMap
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

ROOT = Path(__file__).resolve().parents[1]

def load_processor(name):
    path = ROOT / "processors" / f"processor_{name}.json"
    config = json.loads(path.read_text())
    coupling = CouplingMap(couplinglist=config["coupling_edges"])
    noise_cfg = config["noise"]
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(
        depolarizing_error(noise_cfg["one_qubit_depolarizing_p"], 1),
        ["rz", "sx", "x"]
    )
    noise.add_all_qubit_quantum_error(
        depolarizing_error(noise_cfg["cx_depolarizing_p"], 2), ["cx"]
    )
    p = noise_cfg["readout_error_p"]
    noise.add_all_qubit_readout_error(ReadoutError([[1-p, p], [p, 1-p]]))
    return config, coupling, noise
