def circuit_metrics(circuit):
    ops = circuit.count_ops()
    return {
        "depth": int(circuit.depth() or 0),
        "cx_count": int(ops.get("cx", 0)),
        "swap_count": int(ops.get("swap", 0)),
        "operation_count": int(sum(ops.values())),
        "num_qubits": int(circuit.num_qubits),
    }

def absolute_error(estimate, reference):
    return abs(float(estimate) - float(reference))
