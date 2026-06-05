## Overview

ChipGAT is an end-to-end machine learning pipeline for pre-routing timing analysis of digital circuits using Graph Neural Networks.

The project transforms RTL designs into graph-based representations and learns to predict per-pin timing behavior (slack, slew, and criticality) directly from netlist structure, enabling early-stage timing risk estimation before placement and routing.

## Pipeline Summary

### 1. Design Compilation (SiliconCompiler)

Verilog designs with arbitrary SDC constraints are parsed and synthesized using SiliconCompiler, producing an OpenDB (ODB) representation of the circuit for downstream processing. 
Multiple clock speeds and operating points can be sampled to improve robustness across different timing scenarios.

### 2. Liberty Parsing

The Liberty file is parsed to extract pin capacitance and cell drive strength. In addition, a cell to index mapping is created.

### 3. Feature Extraction (OpenROAD + Tcl)

Using OpenROAD scripts, per-pin features are extracted from the ODB database:

- pin direction
- min slack 
- max rise/fall slew 
- fanout 
- clock pin indicators  
- cell type 

### 4. Dataset Construction (Graph Generation)

A dataset builder merges all extracted information into a graph representation of the design, where:

- nodes: pins  
- edges: intra-cell and inter-cell connections from the netlist  

- node features:
  - cell ID
  - log2(cell drive strength)
  - pin direction  
  - log1p(fanout)  
  - input/output depth  
  - clock indicator  
  - pin capacitance  

- targets:
  - min slack / clock period 
  - slack-derived criticality; exp(-8 * (min slack / clk period)) 
  - max rise/fall slew
    
- edge attribute:
  - intra-cell / inter-cell 
  
- clock period

The resulting graphs are converted into PyTorch Geometric (.pt) format for training.

### 5. Model Training (BiGAT)

A bi-directional graph attention network (BiGAT) is trained on the constructed graphs to learn pre-routing timing relationships in digital circuits. 

Per-pin numerical features are embedded using a shared linear projection, while categorical cell-level attributes (cell type and drive strength) and the clock period are encoded separately and added as learned embeddings.

The model performs node-level prediction of the minimal slack, the maximum rise/fall slew and a slack-derived criticality score. 
The criticality is computed as exp(-8 * (min slack / clk period)), which maps timing slack to a risk metric that increases as the available timing margin decreases. 
It serves as the primary target for identifying timing-critical regions of the design and prioritizing pins with the highest likelihood of timing violations.

### 6. Prediction

## How to Use
### 1. Setup

### 2. Design Compilation

All paths are relative to the repository root.

For each design that should be part of the dataset, put all its Rtl files in a directory. 
A generic sdc file is already included at Dataset/lib_sdc . 

First, define the target clock in the SDC file:

```tcl
create_clock -name clk -period 10 [get_ports clk]
```

where `clk` is the clock input of the top-level module and the period is specified in nanoseconds. Note that the pipeline also supports sampling different clock periods for the same design.


Next, run the chip_run.py file:

```bash
python dataset/chip_run.py \
  --rtl <rtl_files> \
  --sdc <clk.sdc> \
  --clk_period <clock_period_ns> \
  --design <design_name> \
  --top_module <top_module>
```

Example:

```bash
python3 dataset/chip_run.py \
  --rtl dataset/designs/aes/rtl/*.v \
  --sdc dataset/lib_sdc/generic_clk.sdc \
  --clk_period 10 \
  --design aes \
  --top_module aes
```

### 3. Liberty Parsing

```bash
python dataset/lib_scd/parse_lib.py \
  --lib <path_to_lib_file>
```

Example:

```bash
python dataset/lib_scd/parse_lib.py \
  --lib build/aes_10/job0/synthesis/0/inputs/sc_sky130hd_sky130_fd_sc_hd__ss_n40C_1v40.lib

This creates three cell dictionarys.














































