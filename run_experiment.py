import sys
from pathlib import Path
import pandas as pd
from qiskit import transpile
from qiskit_aer import AerSimulator
from src.problem import trotter_circuit, exact_z0
from src.processors import load_processor, ROOT
from src.metrics import circuit_metrics, absolute_error
from src.problem import z0_from_counts
from src.visualization import plot_topology, plot_comparison

def main():
    rows = []
    shots = 4096
    for proc_name in ("A", "B"):
        config, coupling, noise = load_processor(proc_name)
        plot_topology(config, ROOT / "results" / "figures" / f"processor_{proc_name}_topology.png")
        for g in (0.5, 1.0, 1.5):
            exact = exact_z0(g, 2.0)
            for steps in (1, 2, 4):
                logical = trotter_circuit(g, t=2.0, steps=steps, measure=True)
                compiled = transpile(
                    logical, basis_gates=config["basis_gates"], coupling_map=coupling,
                    optimization_level=1, seed_transpiler=42
                )
                # Compare with an unconstrained compile to estimate routing overhead.
                # SWAPs are decomposed into CX gates in the required basis, so report
                # the excess-CX equivalent as a transparent estimate rather than
                # pretending the transpiled circuit still contains explicit SWAP ops.
                unconstrained = transpile(
                    logical, basis_gates=config["basis_gates"],
                    optimization_level=1, seed_transpiler=42
                )
                ideal_backend = AerSimulator()
                ideal_result = ideal_backend.run(compiled, shots=shots, seed_simulator=42).result()
                ideal_value = z0_from_counts(ideal_result.get_counts())
                noisy_backend = AerSimulator(noise_model=noise)
                noisy_result = noisy_backend.run(compiled, shots=shots, seed_simulator=42).result()
                noisy_value = z0_from_counts(noisy_result.get_counts())
                metrics = circuit_metrics(compiled)
                baseline_cx = int(unconstrained.count_ops().get("cx", 0))
                routing_cx_overhead = max(0, metrics["cx_count"] - baseline_cx)
                metrics["unconstrained_cx_count"] = baseline_cx
                metrics["routing_cx_overhead"] = routing_cx_overhead
                metrics["swap_equivalent_overhead_estimate"] = routing_cx_overhead / 3.0
                rows.append({
                    "processor": config["name"], "g": g, "t": 2.0, "steps": steps,
                    "exact_z0": exact, "ideal_measured_z0": ideal_value,
                    "ideal_abs_error": absolute_error(ideal_value, exact),
                    "noisy_measured_z0": noisy_value,
                    "noisy_abs_error": absolute_error(noisy_value, exact),
                    **metrics
                })
    out_csv = ROOT / "results" / "tables" / "benchmark_results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    plot_comparison(out_csv, ROOT / "results" / "figures" / "trotter_error_comparison.png")
    print(f"Saved {len(rows)} benchmark rows to {out_csv}")
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    main()
