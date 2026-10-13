from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

def plot_topology(config, out_path):
    graph = nx.Graph()
    graph.add_nodes_from(range(config["num_qubits"]))
    graph.add_edges_from(map(tuple, config["coupling_edges"]))
    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(6, 4))
    nx.draw_networkx(graph, pos, node_size=650, font_size=10)
    plt.title(f'{config["name"]} coupling graph')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_comparison(csv_path, out_path):
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    grouped = df.groupby(["processor", "steps"], as_index=False)["ideal_abs_error"].mean()
    pivot = grouped.pivot(index="steps", columns="processor", values="ideal_abs_error")
    ax = pivot.plot(kind="bar", figsize=(7, 4))
    ax.set_ylabel("Absolute error in <Z0>")
    ax.set_title("Trotter error by processor and step count")
    ax.figure.tight_layout()
    ax.figure.savefig(out_path, dpi=160)
    plt.close(ax.figure)
