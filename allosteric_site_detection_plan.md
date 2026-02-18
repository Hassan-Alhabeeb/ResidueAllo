# Allosteric / Cryptic Binding Site Detection via Per-Residue ML Classification
## Complete Implementation Plan

---

## TABLE OF CONTENTS

1. [Data Strategy](#1-data-strategy)
2. [Feature Engineering](#2-feature-engineering)
3. [Model Architecture Options](#3-model-architecture-options)
4. [Baseline Comparisons](#4-baseline-comparisons)
5. [Validation Strategy](#5-validation-strategy)
6. [Detailed Timeline](#6-detailed-timeline)
7. [Risk Assessment](#7-risk-assessment)
8. [Publication Strategy](#8-publication-strategy)
9. [Environment Setup](#9-environment-setup)

---

## 1. DATA STRATEGY

### 1.1 Primary Data Source: AlloSteric Database (ASD)

**URL:** http://mdl.shsmu.edu.cn/ASD/
**Current version:** ASD2023 (published in Nucleic Acids Res. 2024, 52, D376-D383)
**Maintained by:** Shanghai Jiao Tong University School of Medicine

**What ASD contains:**
- ~1,949 allosteric proteins from 11 protein families (ASD v4.0 / ASD2023)
- 66,589 predicted allosteric sites covering ~80% of the human proteome
- Experimentally validated allosteric modulators and their binding site residues
- For each allosteric site: PDB ID, chain ID, allosteric modulator identity, and site-defining residues

**How to download ASD data:**
1. Navigate to http://mdl.shsmu.edu.cn/ASD/
2. Click the "DOWNLOAD" menu on the navigation bar
3. ASD provides bulk download of allosteric proteins and modulators
4. The data comes as structured text files and PDB-format coordinate files
5. Each entry links to a PDB structure with allosteric modulator bound, from which you extract contacting residues

**Critical limitation of raw ASD:** The downloaded data does NOT come as a clean CSV with per-residue binary labels. You get PDB IDs, ligand identifiers, and site descriptions. You must parse the PDB files yourself to extract which residues are within a distance cutoff (typically 4.5 or 5.0 Angstroms) of the allosteric modulator to generate residue-level labels.

### 1.2 Much Better Option: AlloBench (Use This Instead)

**URL:** https://pubs.acs.org/doi/10.1021/acsomega.5c01263
**Figshare:** https://acs.figshare.com/articles/journal_contribution/AlloBench/28847898
**Published:** ACS Omega, 2025

AlloBench is a pipeline that already solved the data curation problem for you:
- 2,141 allosteric sites from 2,034 protein structures with 418 unique protein chains
- Integrates ASD + UniProt + Mechanism and Catalytic Site Atlas (M-CSA) + PDB
- Outputs a **CSV file** with allosteric and active site residue lists per protein
- Fixes obsolete PDB/UniProt IDs, filters for high-resolution structures
- Provides a Jupyter notebook pipeline you can re-run and customize

**How to use AlloBench:**
```python
# 1. Download the AlloBench repository and supplementary data from Figshare
# 2. The pipeline outputs CSV with columns including:
#    - PDB ID, Chain, UniProt ID
#    - Allosteric site residue numbers (comma-separated list)
#    - Active site residue numbers
#    - Resolution, organism, etc.

# 3. Convert to per-residue binary labels:
import pandas as pd
from Bio.PDB import PDBParser

df = pd.read_csv("allobench_dataset.csv")

for _, row in df.iterrows():
    pdb_id = row["pdb_id"]
    chain = row["chain_id"]
    allo_residues = set(map(int, row["allosteric_residues"].split(",")))
    # Download PDB, iterate all residues in chain
    # Label 1 if residue number in allo_residues, else 0
```

### 1.3 Supplementary Benchmark Datasets

**ASBench** (http://mdl.shsmu.edu.cn/asbench)
- 235 unique allosteric sites (Core set) / 147 structurally diverse (Core-Diversity set)
- Available since 2015, well-established benchmark
- Residue info on web interface but must be extracted from structure files
- Use as a held-out test set since many prior methods were NOT trained on it

**CASBench** (https://biokinet.belozersky.msu.ru/casbench)
- 91 proteins with BOTH catalytic AND allosteric sites annotated
- Text files and PyMOL session files with residue-level annotations
- Useful for distinguishing allosteric sites from catalytic sites
- Residue numbering synchronized with PDB structures

### 1.4 Generating Per-Residue Labels

For each protein chain in the dataset:

```python
import numpy as np
from Bio.PDB import PDBParser, NeighborSearch

def label_allosteric_residues(pdb_file, chain_id, ligand_resnames, distance_cutoff=5.0):
    """
    Label residues as allosteric (1) if any atom is within distance_cutoff
    of any atom in the specified allosteric ligand.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]
    chain = model[chain_id]

    # Collect ligand atoms
    ligand_atoms = []
    for residue in chain:
        if residue.get_resname() in ligand_resnames:
            ligand_atoms.extend(residue.get_atoms())

    # Also check HETATM in other chains or as separate entities
    for c in model:
        for residue in c:
            if residue.get_resname() in ligand_resnames:
                ligand_atoms.extend(residue.get_atoms())

    if not ligand_atoms:
        return None  # No ligand found

    # Build neighbor search on ligand atoms
    ns = NeighborSearch(ligand_atoms)

    labels = {}
    for residue in chain:
        if residue.id[0] != " ":  # Skip HETATM
            continue
        resnum = residue.id[1]
        # Check if any protein atom is within cutoff of any ligand atom
        is_allosteric = False
        for atom in residue:
            neighbors = ns.search(atom.get_vector().get_array(), distance_cutoff)
            if neighbors:
                is_allosteric = True
                break
        labels[resnum] = 1 if is_allosteric else 0

    return labels
```

### 1.5 Negative Labels (Non-Allosteric Residues)

**Do NOT use separate "non-allosteric proteins" as negatives.** Instead:

1. **Within-protein negatives:** For each allosteric protein, all residues NOT in the allosteric site are negatives. This is standard practice (used by DeepAllo, AlloPED, AlloFusion).

2. **Rationale:** This avoids the confound of learning protein identity rather than allosteric site properties. A model trained on "allosteric proteins vs. random proteins" would learn protein family signatures, not site-level features.

3. **Expected class ratio:** Only ~5% of residues in a typical allosteric protein are in the allosteric site (DeepAllo reports 5.12% positive at residue level). This means ~19:1 negative-to-positive ratio.

### 1.6 Handling Class Imbalance

The ~19:1 class imbalance is severe. Use a multi-pronged approach:

**Strategy A: Weighted loss function (primary)**
```python
# For XGBoost:
n_pos = sum(y_train == 1)
n_neg = sum(y_train == 0)
scale_pos_weight = n_neg / n_pos  # ~19.0
model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight)

# For PyTorch (ESM-2 fine-tuning):
pos_weight = torch.tensor([n_neg / n_pos])
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

**Strategy B: Focal loss (for deep learning models)**
```python
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
```

**Strategy C: Neighborhood-aware sampling**
Instead of random undersampling (which loses information), oversample by including nearby residues (within 8 Angstroms of allosteric site) as "soft positives" with reduced weight. This captures the transition zone.

**Strategy D: Evaluation metric selection**
Never use accuracy. Use:
- MCC (Matthews Correlation Coefficient) -- single best metric for imbalanced binary classification
- AUROC -- for ranking ability
- AUPRC (Precision-Recall AUC) -- more informative than AUROC when positives are rare
- F1 score -- harmonic mean of precision and recall
- Precision@k -- practical: "of the top k residues predicted, how many are truly allosteric?"

### 1.7 Preventing Data Leakage in Train/Test Splits

**The problem:** Proteins in the same family share conserved allosteric sites. If a kinase appears in both train and test, the model memorizes kinase allosteric site patterns rather than learning generalizable features.

**Solution: Sequence-identity-based clustering with MMseqs2**

```bash
# Install MMseqs2 (Windows: download binary from GitHub releases)
# https://github.com/soedinglab/MMseqs2/releases

# 1. Create FASTA file of all protein sequences in the dataset
# 2. Cluster at 30% sequence identity (standard threshold):
mmseqs easy-cluster sequences.fasta clust tmp --min-seq-id 0.3 -c 0.8

# Output: clust_cluster.tsv (representative -> member mapping)
# Each cluster is a group of similar proteins
```

```python
import pandas as pd

# Parse MMseqs2 clustering output
clusters = pd.read_csv("clust_cluster.tsv", sep="\t", header=None,
                       names=["representative", "member"])

# Group by cluster representative
cluster_groups = clusters.groupby("representative")["member"].apply(list)

# Split CLUSTERS (not individual proteins) into train/val/test
# 70% train, 15% val, 15% test -- at the cluster level
from sklearn.model_selection import GroupShuffleSplit

cluster_ids = clusters["representative"].values
# Assign each protein its cluster ID
protein_to_cluster = dict(zip(clusters["member"], clusters["representative"]))

# Use cluster ID as the group for GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, temp_idx = next(gss.split(X, y, groups=cluster_labels))

# Further split temp into val and test
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(gss2.split(X[temp_idx], y[temp_idx],
                                     groups=cluster_labels[temp_idx]))
```

**Expected dataset sizes after clustering:**
- Starting: ~418 unique chains from AlloBench
- After 30% identity clustering: ~200-250 clusters
- Train: ~140-175 clusters (~300 proteins)
- Val: ~30-37 clusters (~60 proteins)
- Test: ~30-37 clusters (~60 proteins)

This is small. To increase data, also incorporate ASD2023 entries not in AlloBench.

### 1.8 Dataset Size Reality Check

| Source | Proteins | With residue labels | Notes |
|--------|----------|-------------------|-------|
| AlloBench | 2,034 structures / 418 unique chains | Yes (CSV) | Primary training data |
| ASBench Core | 235 allosteric sites | Must extract | Good held-out test |
| CASBench | 91 proteins | Yes (text files) | Has catalytic + allosteric |
| ASD2023 raw | ~1,949 proteins | Must extract | Supplement AlloBench |

**Total usable proteins with residue-level labels: ~400-600 after deduplication and quality filters.** This is enough for classical ML and fine-tuning, but tight for training a GNN from scratch.

---

## 2. FEATURE ENGINEERING

### 2.a B-factors from PDB files

**What it is biologically:** B-factors (temperature factors / Debye-Waller factors) quantify atomic displacement / thermal motion. High B-factors indicate flexible regions. Allosteric sites often overlap with regions of intermediate flexibility -- not too rigid (active site), not too floppy (disordered loops).

**How to compute:**
```python
from Bio.PDB import PDBParser
import numpy as np

def extract_bfactors(pdb_file, chain_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_file)
    chain = structure[0][chain_id]

    residue_bfactors = {}
    for residue in chain:
        if residue.id[0] != " ":
            continue
        atoms = list(residue.get_atoms())
        # Mean B-factor across all atoms in the residue
        mean_b = np.mean([a.get_bfactor() for a in atoms])
        # Also compute CA B-factor specifically
        ca_b = None
        for a in atoms:
            if a.get_name() == "CA":
                ca_b = a.get_bfactor()
        residue_bfactors[residue.id[1]] = {
            "mean_bfactor": mean_b,
            "ca_bfactor": ca_b,
            "max_bfactor": max(a.get_bfactor() for a in atoms),
            "min_bfactor": min(a.get_bfactor() for a in atoms),
        }
    return residue_bfactors
```

**Output format:** Dictionary mapping residue number to B-factor statistics
**Time:** <1 second per protein (pure parsing, no computation)
**Gotchas:**
- B-factors are NOT comparable across different PDB entries (different refinement procedures)
- Normalize B-factors within each protein: z-score normalization `(b - mean) / std`
- AlphaFold structures do NOT have B-factors; the "B-factor" column stores pLDDT confidence scores instead (0-100 scale, higher = more confident)
- For AlphaFold structures, use pLDDT as a separate feature (it correlates inversely with disorder)
- Some PDB entries have uniform B-factors (e.g., all 0.0) -- discard or flag these

### 2.b Conservation Scores

**What it is biologically:** Evolutionary conservation indicates functional importance. Allosteric sites are often moderately conserved -- more than surface loops, less than catalytic residues. Conservation across orthologous sequences is a strong predictor.

**Option 1: ConSurf web server (small scale, <50 proteins)**
- URL: https://consurf.tau.ac.il/
- Upload PDB or provide PDB ID + chain
- Takes 20-60 minutes per protein (builds MSA, runs Rate4Site)
- Output: "consurf_grades.txt" file with per-residue conservation scores (1-9 scale)
- Limit: one job at a time, no batch mode on web server

**Option 2: Local ConSurf installation (batch processing)**
```bash
# Requires: Perl, BLAST+, MUSCLE/ClustalW, Rate4Site
# GitHub: https://github.com/Rostlab/ConSurf

# Installation (Linux/WSL -- does NOT run natively on Windows):
git clone https://github.com/Rostlab/ConSurf.git
cd ConSurf
# Follow INSTALL file for Perl module dependencies

# Run for a single protein:
perl ConSurf.pl -PDB protein.pdb -CHAIN A -Out_Dir ./output/ \
    -MSAprogram MUSCLE -DB SWISS-PROT -MaxHomol 150 -Algorithm Bayesian
```
**Time:** 15-45 minutes per protein locally (bottleneck: PSI-BLAST search)
**For 500 proteins:** 5-15 days of compute (can parallelize across cores)

**Option 3: PSSM from PSI-BLAST (fastest, most practical for >100 proteins)**
```bash
# Install BLAST+ from NCBI: https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/
# Download UniRef50 or SwissProt database

# Generate PSSM:
psiblast -query sequence.fasta -db uniref50 -num_iterations 3 \
    -out_ascii_pssm output.pssm -num_threads 4

# Parse PSSM in Python:
def parse_pssm(pssm_file):
    """Returns numpy array of shape (seq_len, 20) with PSSM scores."""
    scores = []
    with open(pssm_file) as f:
        lines = f.readlines()
        for line in lines[3:]:  # Skip header
            parts = line.split()
            if len(parts) >= 22:
                try:
                    row = [int(x) for x in parts[2:22]]
                    scores.append(row)
                except ValueError:
                    break
    return np.array(scores)
```
**Time:** 2-5 minutes per protein (PSI-BLAST against UniRef50)
**For 500 proteins:** ~1-2 days (parallelizable)
**Output:** 20-dimensional vector per residue (log-odds for each amino acid)

**Recommendation:** Use PSSM as the primary conservation feature. It is what AlloFusion and other recent methods use, it is fast, and the 20-dimensional encoding is richer than a single ConSurf score.

### 2.c Solvent Accessible Surface Area (SASA)

**What it is biologically:** SASA measures how exposed each residue is to solvent. Allosteric sites are typically partially buried -- accessible enough for small molecules to reach, but not fully exposed surface.

**Tool: FreeSASA**
```bash
pip install freesasa
```

```python
import freesasa

def compute_sasa(pdb_file):
    """Compute per-residue SASA using FreeSASA."""
    structure = freesasa.Structure(pdb_file)
    result = freesasa.calc(structure)

    # Get per-residue SASA
    area_classes = freesasa.classifyResults(result, structure)

    # Alternative: iterate residues
    residue_sasa = {}
    for i in range(structure.nAtoms()):
        resnum = structure.residueNumber(i)
        chain = structure.chainLabel(i)
        key = (chain, resnum)
        if key not in residue_sasa:
            residue_sasa[key] = 0.0
        residue_sasa[key] += result.atomArea(i)

    return residue_sasa
```

**Also compute relative SASA (rSASA):**
```python
# Maximum SASA values for each amino acid (Gly-X-Gly tripeptide, Angstrom^2)
MAX_SASA = {
    'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0,
    'CYS': 167.0, 'GLN': 225.0, 'GLU': 223.0, 'GLY': 104.0,
    'HIS': 224.0, 'ILE': 197.0, 'LEU': 201.0, 'LYS': 236.0,
    'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0, 'SER': 155.0,
    'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0,
}

def relative_sasa(abs_sasa, resname):
    return abs_sasa / MAX_SASA.get(resname, 200.0)
```

**Output:** Absolute SASA (Angstrom^2) and relative SASA (0-1 fraction) per residue
**Time:** <1 second per protein
**Gotchas:**
- FreeSASA v2.x pip install may fail on Windows. If so, use `conda install -c conda-forge freesasa`
- Alternative: use BioPython's `ShrakeRupley` SASA calculator:
```python
from Bio.PDB.SASA import ShrakeRupley
sr = ShrakeRupley()
sr.compute(structure[0], level="R")  # R = residue level
for residue in structure[0].get_residues():
    print(residue.id[1], residue.sasa)
```

### 2.d Contact Density / Coordination Number

**What it is biologically:** How many other residues each residue contacts. Core residues have high contact density; surface residues have low. Allosteric sites tend to have intermediate contact density and are at interfaces between structural domains.

```python
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist
import numpy as np

def compute_contact_features(pdb_file, chain_id, distance_cutoff=8.0):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_file)
    chain = structure[0][chain_id]

    # Get CA coordinates
    residues = []
    ca_coords = []
    for res in chain:
        if res.id[0] != " ":
            continue
        for atom in res:
            if atom.get_name() == "CA":
                residues.append(res.id[1])
                ca_coords.append(atom.get_vector().get_array())
                break

    ca_coords = np.array(ca_coords)
    dist_matrix = cdist(ca_coords, ca_coords)

    features = {}
    for i, resnum in enumerate(residues):
        contacts = np.sum(dist_matrix[i] < distance_cutoff) - 1  # Exclude self
        # Also compute at different cutoffs
        contacts_6 = np.sum(dist_matrix[i] < 6.0) - 1
        contacts_10 = np.sum(dist_matrix[i] < 10.0) - 1
        contacts_12 = np.sum(dist_matrix[i] < 12.0) - 1

        features[resnum] = {
            "contact_count_8A": contacts,
            "contact_count_6A": contacts_6,
            "contact_count_10A": contacts_10,
            "contact_count_12A": contacts_12,
            "contact_density": contacts / max(len(residues), 1),
        }

    return features
```

**Output:** Integer contact count and float density per residue
**Time:** <1 second per protein
**Gotchas:** Some PDB files have missing CA atoms; handle gracefully. Use CB for glycine fallback.

### 2.e Secondary Structure (DSSP)

**What it is biologically:** Secondary structure (helix, sheet, coil) context of each residue. Allosteric sites often occur at secondary structure boundaries (e.g., loop-helix junctions).

**Installation on Windows:**
```bash
# Option 1: conda (recommended for Windows)
conda install -c salilab dssp

# Option 2: Download mkdssp binary
# https://github.com/PDB-REDO/dssp/releases
# Download the Windows executable, rename to dssp.exe, add to PATH

# Option 3: Use BioPython's built-in DSSP wrapper (needs external dssp binary)
pip install biopython
```

```python
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

def compute_dssp(pdb_file, chain_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_file)
    model = structure[0]

    # dssp_executable must be in PATH or specify full path
    dssp = DSSP(model, pdb_file, dssp="mkdssp")

    features = {}
    for key in dssp:
        chain, resinfo = key
        if chain != chain_id:
            continue
        resnum = resinfo[1]
        dssp_data = dssp[key]
        # dssp_data: (index, aa, ss, rasa, phi, psi, nh_o_1, ...)

        ss = dssp_data[2]  # H, B, E, G, I, T, S, -
        rasa = dssp_data[3]  # Relative ASA

        # One-hot encode secondary structure
        ss_types = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']
        ss_onehot = [1 if ss == s else 0 for s in ss_types]

        features[resnum] = {
            "ss_type": ss,
            "ss_onehot": ss_onehot,
            "phi": dssp_data[4],
            "psi": dssp_data[5],
            "rasa_dssp": rasa,
        }

    return features
```

**Output:** SS type (8 classes), phi/psi angles, relative ASA per residue
**Time:** <2 seconds per protein
**Gotchas:**
- DSSP requires the PDB file to have proper ATOM records (not just CA)
- Some PDB files cause DSSP to crash (unusual residues, chain breaks). Wrap in try/except.
- The mkdssp (DSSP v4) binary name differs from old dssp. BioPython parameter: `dssp="mkdssp"`

### 2.f Amino Acid Physicochemical Properties

**What it is biologically:** Intrinsic properties of each amino acid (hydrophobicity, charge, size, etc.) that determine local chemical environment.

```python
# Pre-defined property dictionaries (no external tools needed)
AA_PROPERTIES = {
    # Kyte-Doolittle hydrophobicity
    'hydrophobicity': {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
    },
    # Molecular weight (Da)
    'mol_weight': {
        'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
        'Q': 146.2, 'E': 147.1, 'G': 75.0, 'H': 155.2, 'I': 131.2,
        'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
        'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1,
    },
    # Charge at pH 7
    'charge': {
        'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
        'Q': 0, 'E': -1, 'G': 0, 'H': 0.1, 'I': 0,
        'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
        'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0,
    },
    # van der Waals volume (A^3)
    'vdw_volume': {
        'A': 67, 'R': 148, 'N': 96, 'D': 91, 'C': 86,
        'Q': 114, 'E': 109, 'G': 48, 'H': 118, 'I': 124,
        'L': 124, 'K': 135, 'M': 124, 'F': 135, 'P': 90,
        'S': 73, 'T': 93, 'W': 163, 'Y': 141, 'V': 105,
    },
    # Flexibility (Normalized B-factor from Vihinen & Mantsala 1989)
    'flexibility': {
        'A': 0.984, 'R': 1.008, 'N': 1.048, 'D': 1.068, 'C': 0.906,
        'Q': 1.037, 'E': 1.094, 'G': 1.031, 'H': 0.950, 'I': 0.927,
        'L': 0.935, 'K': 1.102, 'M': 0.952, 'F': 0.915, 'P': 1.049,
        'S': 1.046, 'T': 0.997, 'W': 0.904, 'Y': 0.929, 'V': 0.931,
    },
    # Polarity (Grantham 1974)
    'polarity': {
        'A': 8.1, 'R': 10.5, 'N': 11.6, 'D': 13.0, 'C': 5.5,
        'Q': 10.5, 'E': 12.3, 'G': 9.0, 'H': 10.4, 'I': 5.2,
        'L': 4.9, 'K': 11.3, 'M': 5.7, 'F': 5.2, 'P': 8.0,
        'S': 9.2, 'T': 8.6, 'W': 5.4, 'Y': 6.2, 'V': 5.9,
    },
}

def get_aa_features(sequence):
    """Return array of shape (seq_len, n_properties)."""
    props = list(AA_PROPERTIES.keys())
    features = []
    for aa in sequence:
        row = [AA_PROPERTIES[prop].get(aa, 0.0) for prop in props]
        features.append(row)
    return np.array(features)  # shape: (L, 5)
```

**Output:** 5-dimensional vector per residue (expandable with more scales)
**Time:** Negligible (dictionary lookup)
**Gotchas:** Handle non-standard amino acids (selenomethionine MSE -> MET, etc.)

### 2.g Residue Depth

**What it is biologically:** Average distance of a residue's atoms from the molecular surface. Deeper residues are more buried. Allosteric sites often have intermediate depth.

**Requires:** MSMS program by Michel Sanner

```bash
# Download MSMS from: https://ccsb.scripps.edu/msms/downloads/
# Windows binary available. Add to PATH.
```

```python
from Bio.PDB import PDBParser
from Bio.PDB.ResidueDepth import ResidueDepth, get_surface

def compute_residue_depth(pdb_file, chain_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_file)
    model = structure[0]

    # This requires MSMS to be installed and in PATH
    try:
        rd = ResidueDepth(model)
    except Exception as e:
        print(f"MSMS failed: {e}")
        return None

    features = {}
    for key, (res_depth, ca_depth) in rd.property_dict.items():
        chain, resinfo = key
        if chain != chain_id:
            continue
        resnum = resinfo[1]
        features[resnum] = {
            "residue_depth": res_depth,  # Mean depth of all atoms
            "ca_depth": ca_depth,        # Depth of CA atom
        }
    return features
```

**Alternative without MSMS (simpler, less accurate):**
```python
def approximate_depth(pdb_file, chain_id, probe_radius=1.4):
    """Approximate residue depth using distance to nearest surface residue.
    Surface residues defined as those with rSASA > 0.25."""
    # Compute SASA first, identify surface residues
    # Then for each residue, compute distance to nearest surface residue CA
    # This avoids the MSMS dependency
    pass
```

**Output:** Depth in Angstroms per residue (typically 0-15 A)
**Time:** 1-3 seconds per protein (MSMS is fast)
**Gotchas:**
- MSMS occasionally fails on structures with unusual atoms or missing residues
- Clean PDB files first: remove waters, ligands, non-standard residues
- MSMS Windows binary may have path issues; use raw strings: `r"C:\path\to\msms.exe"`

### 2.h ESM-2 Protein Language Model Embeddings

**What it is biologically:** Learned representations from evolutionary patterns across millions of protein sequences. Captures long-range co-evolutionary signals that correlate with functional sites, including allosteric sites.

**Model selection for 8GB VRAM:**

| Model | Parameters | Hidden dim | VRAM (inference) | VRAM (LoRA fine-tune) |
|-------|-----------|------------|------------------|----------------------|
| esm2_t6_8M_UR50D | 8M | 320 | ~1 GB | ~2 GB |
| esm2_t12_35M_UR50D | 35M | 480 | ~2 GB | ~3 GB |
| esm2_t30_150M_UR50D | 150M | 640 | ~3 GB | ~5 GB |
| esm2_t33_650M_UR50D | 650M | 1280 | ~6 GB | ~10 GB (need Colab) |

**Recommendation:** Use `esm2_t33_650M_UR50D` for feature extraction (inference only, fits in 8GB). For fine-tuning with LoRA, use `esm2_t30_150M_UR50D` locally or `esm2_t33_650M_UR50D` on Colab.

```python
import torch
from transformers import AutoTokenizer, AutoModel

def extract_esm2_embeddings(sequence, model_name="facebook/esm2_t33_650M_UR50D"):
    """Extract per-residue embeddings from ESM-2."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Tokenize
    inputs = tokenizer(sequence, return_tensors="pt", padding=False)

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.last_hidden_state shape: (1, seq_len+2, hidden_dim)
    # Remove [CLS] and [EOS] tokens
    embeddings = outputs.last_hidden_state[0, 1:-1, :]  # (seq_len, hidden_dim)

    return embeddings.numpy()
```

**For batch processing with GPU:**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).half()  # FP16 to save memory

def batch_extract(sequences, batch_size=4):
    all_embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = model(**inputs)
        # Process each sequence individually (different lengths)
        for j, seq in enumerate(batch):
            seq_len = len(seq)
            emb = outputs.last_hidden_state[j, 1:seq_len+1, :]
            all_embeddings.append(emb.cpu().float().numpy())
    return all_embeddings
```

**Output:** Array of shape (seq_len, 1280) for the 650M model
**Time:** ~2-5 seconds per protein on GPU; ~30-60 seconds on CPU for a 500-residue protein
**For 500 proteins:** ~30 minutes on GPU, ~8 hours on CPU
**Gotchas:**
- Maximum sequence length is 1024 tokens for ESM-2. Proteins longer than ~1022 residues must be split into overlapping windows (stride 512, average overlapping regions).
- ESM-2 tokenizer adds special tokens; always remove them: `[1:-1]` indexing.
- On Windows, ensure CUDA toolkit version matches PyTorch CUDA version.

### 2.i Graph-Based Features (Residue Contact Graph)

**What it is biologically:** The protein structure as a graph where nodes are residues and edges connect residues within a spatial distance cutoff. Graph-theoretic properties (betweenness centrality, clustering coefficient) identify residues at structural "communication hubs" -- often allosteric sites.

```python
import networkx as nx
from scipy.spatial.distance import cdist

def build_residue_graph(pdb_file, chain_id, cutoff=8.0):
    """Build residue contact graph from CA-CA distances."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_file)
    chain = structure[0][chain_id]

    residues = []
    coords = []
    for res in chain:
        if res.id[0] != " ":
            continue
        for atom in res:
            if atom.get_name() == "CA":
                residues.append(res.id[1])
                coords.append(atom.get_vector().get_array())
                break

    coords = np.array(coords)
    dist_matrix = cdist(coords, coords)

    G = nx.Graph()
    for i, resnum in enumerate(residues):
        G.add_node(resnum)

    for i in range(len(residues)):
        for j in range(i+1, len(residues)):
            if dist_matrix[i][j] < cutoff:
                G.add_edge(residues[i], residues[j],
                          weight=1.0/dist_matrix[i][j])

    # Compute graph features
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    clustering = nx.clustering(G)
    degree = dict(G.degree())
    pagerank = nx.pagerank(G)

    features = {}
    for resnum in residues:
        features[resnum] = {
            "betweenness_centrality": betweenness.get(resnum, 0),
            "closeness_centrality": closeness.get(resnum, 0),
            "clustering_coefficient": clustering.get(resnum, 0),
            "degree": degree.get(resnum, 0),
            "pagerank": pagerank.get(resnum, 0),
        }
    return features, G
```

**Output:** 5 graph features per residue
**Time:** <2 seconds per protein (graph construction + centrality computation)
**Gotchas:** Cutoff choice matters. 8 A for CA-CA is standard. For all-atom, use 4.5 A.

### 2.j Pocket Detection Features (FPocket)

**What it is biologically:** FPocket identifies geometric pockets on the protein surface using Voronoi tessellation and alpha spheres. Each pocket gets a "druggability score." Allosteric sites are usually detected as pockets, but not ranked #1 (the active site is).

**Installation on Windows:**
```bash
# FPocket does NOT compile natively on Windows.
# Three options:

# Option 1: WSL (Windows Subsystem for Linux) -- RECOMMENDED
wsl --install  # If WSL not yet set up
# Then inside WSL:
sudo apt-get update
sudo apt-get install -y fpocket
# Or from source:
git clone https://github.com/Discngine/fpocket.git
cd fpocket
make
sudo make install

# Option 2: Conda (works on Windows)
conda install -c conda-forge fpocket

# Option 3: Docker
docker pull quay.io/biocontainers/fpocket:4.0.2--h9f5acd7_0
```

**Running FPocket:**
```bash
# Command line:
fpocket -f protein.pdb

# Output directory: protein_out/
# Key output files:
#   protein_out/protein_pockets.pqr  -- all pockets with alpha sphere info
#   protein_out/protein_info.txt     -- pocket summary statistics
#   protein_out/pockets/             -- individual pocket PDB files
```

**Parsing FPocket output in Python:**
```python
import os
import re

def parse_fpocket_output(fpocket_dir, chain_id):
    """Parse FPocket output to get per-residue pocket features."""
    info_file = os.path.join(fpocket_dir, [f for f in os.listdir(fpocket_dir)
                                            if f.endswith("_info.txt")][0])

    pocket_residues = {}  # pocket_id -> list of residue numbers
    pocket_scores = {}    # pocket_id -> druggability score

    # Parse pocket info
    with open(info_file) as f:
        content = f.read()

    # Parse individual pocket PDB files
    pockets_dir = os.path.join(fpocket_dir, "pockets")
    for pocket_file in sorted(os.listdir(pockets_dir)):
        if not pocket_file.endswith(".pdb"):
            continue
        pocket_id = int(re.search(r'pocket(\d+)', pocket_file).group(1))

        residues = set()
        with open(os.path.join(pockets_dir, pocket_file)) as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    chain = line[21]
                    if chain == chain_id:
                        resnum = int(line[22:26].strip())
                        residues.add(resnum)
        pocket_residues[pocket_id] = residues

    # Assign per-residue features
    residue_features = {}
    for pocket_id, residues in pocket_resid.items():
        for resnum in residues:
            if resnum not in residue_features:
                residue_features[resnum] = {
                    "in_pocket": 1,
                    "best_pocket_rank": pocket_id,
                    "n_pockets": 0,
                }
            residue_features[resnum]["n_pockets"] += 1
            if pocket_id < residue_features[resnum]["best_pocket_rank"]:
                residue_features[resnum]["best_pocket_rank"] = pocket_id

    return residue_features
```

**Output:** Pocket membership, rank, druggability score per residue
**Time:** 5-30 seconds per protein (depends on size)
**For 500 proteins:** ~2-4 hours
**Gotchas:**
- FPocket needs clean PDB files: remove waters, remove ligands, remove alternate conformations
- FPocket version 4.x output format differs from version 2.x
- Some very small proteins (<50 residues) produce no pockets

### 2.k Normal Mode Analysis / Flexibility Prediction

**What it is biologically:** Normal mode analysis (NMA) using elastic network models (ANM/GNM) predicts large-scale conformational motions. Residues with high mobility in low-frequency modes are often involved in allosteric transitions. This is one of the most informative features for allosteric site prediction -- the AllosES method showed it improves MCC significantly.

**Tool: ProDy**
```bash
pip install prody
```

```python
import prody

def compute_nma_features(pdb_file, chain_id, n_modes=20):
    """Compute ANM/GNM flexibility features using ProDy."""
    # Parse structure
    structure = prody.parsePDB(pdb_file, chain=chain_id)
    calphas = structure.select('calpha')

    if calphas is None or len(calphas) < 10:
        return None

    # Anisotropic Network Model (ANM)
    anm = prody.ANM('protein')
    anm.buildHessian(calphas, cutoff=15.0)
    anm.calcModes(n_modes=n_modes)

    # Gaussian Network Model (GNM)
    gnm = prody.GNM('protein')
    gnm.buildKirchhoff(calphas, cutoff=10.0)
    gnm.calcModes(n_modes=n_modes)

    # Per-residue features
    residue_ids = calphas.getResnums()

    # Squared fluctuations (mobility) from ANM
    anm_sqflucts = prody.calcSqFlucts(anm[:n_modes])

    # GNM squared fluctuations
    gnm_sqflucts = prody.calcSqFlucts(gnm[:n_modes])

    # Slow mode contributions (modes 1-3 capture global motions)
    slow_flucts = prody.calcSqFlucts(anm[:3])

    # Fast mode contributions (modes 18-20 capture local motions)
    fast_flucts = prody.calcSqFlucts(anm[n_modes-3:n_modes])

    # Collectivity of each residue
    # (how much each residue participates in collective motions)

    # Stiffness: inverse of fluctuation
    stiffness = 1.0 / (anm_sqflucts + 1e-8)

    features = {}
    for i, resnum in enumerate(residue_ids):
        features[int(resnum)] = {
            "anm_fluctuation": float(anm_sqflucts[i]),
            "gnm_fluctuation": float(gnm_sqflucts[i]),
            "slow_mode_fluct": float(slow_flucts[i]),
            "fast_mode_fluct": float(fast_flucts[i]),
            "stiffness": float(stiffness[i]),
            "slow_fast_ratio": float(slow_flucts[i] / (fast_flucts[i] + 1e-8)),
        }

    return features
```

**Output:** 6 flexibility features per residue
**Time:** 2-10 seconds per protein (eigenvalue decomposition)
**For 500 proteins:** ~1 hour
**Gotchas:**
- ProDy works on Windows with pip install. No special dependencies.
- Very large proteins (>2000 residues) can be slow for ANM. Use GNM only for those.
- Missing residues cause gaps in the residue numbering; handle with care when aligning features.

### 2.l Perturbation Response Scanning (PRS) — From Existing GNM

**What it does:** "Poke each residue, measure what happens." PRS applies a unit force to residue i and measures how much the rest of the protein displaces. This directly models allosteric signal propagation.

**Tool: ProDy** (already installed)
```python
from prody.dynamics import calcPerturbResponse

# After building GNM and computing modes:
prs_matrix, effectiveness, sensitivity = calcPerturbResponse(gnm)
# effectiveness[i] = how strongly residue i propagates signals outward (allosteric drivers)
# sensitivity[i] = how strongly residue i responds to perturbations (allosteric receivers)
```

**Output:** 2 features per residue (effectiveness, sensitivity)
**Time:** <1 second per protein (matrix algebra on existing GNM modes)
**Why it matters:** Allosteric sites should have high effectiveness (they send conformational signals). This is the core feature of AlloPred (BMC Bioinf. 2015). Uses existing GNM — zero new dependencies.

### 2.m Essential Site Scanning Analysis (ESSA) — From Existing GNM

**What it does:** "What if something binds here?" ESSA adds a virtual ligand mass at each residue and measures how much the protein's vibration patterns change. Large shift = binding here would alter global dynamics = likely allosteric site.

**Tool: ProDy** (already installed)
```python
from prody import ESSA

essa = ESSA()
essa.setSystem(calphas)  # AtomGroup of CA atoms, NOT a GNM/ANM object
essa.scanResidues()
essa_zscores = essa.getESSAZscores()
# High z-score = binding at this residue would significantly shift global dynamics
```

**Output:** 1 feature per residue (ESSA z-score)
**Time:** UNKNOWN — MUST BENCHMARK FIRST. ESSA internally rebuilds an ENM N times (once per residue). If it uses ANM, a 500-residue protein = 500 ANM builds, potentially minutes-to-hours per protein. **Before committing: time ESSA on 3 proteins of different sizes (200, 500, 1500 residues).** If too slow, implement a GNM-based approximation (perturb Kirchhoff matrix, measure spectral shift).
**Why it matters:** Directly simulates "what happens if a small molecule binds here?" Captures functional impact that FPocket's geometry misses. Validated in Krieger et al. (PNAS 2020).

### 2.n GNM Mutual Information — Information Coupling (Renamed from "Transfer Entropy")

**What it does:** Measures how much knowing one residue's displacement reduces uncertainty about another's. High MI = tightly coupled residues.

**IMPORTANT NOTE (from Opus review):** The original plan called this "transfer entropy," but true TE requires time-lagged data from MD simulations (PMC5283753 uses MD, not GNM). GNM gives equilibrium (equal-time) correlations only. The formula below computes **Gaussian mutual information**, which is symmetric: MI(i→j) = MI(j→i). Therefore TE_net would be identically zero. We compute only 1 meaningful aggregate feature, not 3.

**Computed from:** GNM covariance matrix (analytical, no MD needed)
```python
# From GNM covariance matrix C (pseudo-inverse of Kirchhoff):
# MI(i,j) = 0.5 * log(C_ii * C_jj / (C_ii * C_jj - C_ij^2))
# Per-residue aggregate:
# MI_coupling(i) = mean MI(i,j) over all j≠i  (how info-coupled is this residue?)
```

**Output:** 1 feature per residue (MI_coupling) — NOT 3 as originally planned
**Time:** <1 second per protein (matrix operations on existing GNM)
**Why it matters:** Measures how "information-connected" each residue is to the rest of the protein. Allosteric sites should be highly coupled. Complementary to PRS (which measures mechanical propagation, not information content).

### 2.o GNM Cross-Correlation Summaries

**What it does:** Measures how strongly each residue's motions correlate with other residues. Allosteric and active sites have highly correlated motions (JCIM 2016 showed 92% overlap).

```python
from prody import calcCrossCorr
cross_corr = calcCrossCorr(gnm)  # NxN matrix
# Per-residue features:
mean_abs_corr = np.mean(np.abs(cross_corr), axis=1)  # global coupling
max_corr = np.max(np.abs(cross_corr - np.eye(N)), axis=1)  # strongest partner
mean_neg_corr = np.mean(np.minimum(cross_corr, 0), axis=1)  # anti-correlated motions
```

**Output:** 3 features per residue
**Time:** <1 second per protein
**Why it matters:** Allosteric sites are communication hubs — high global coupling.

### 2.p AAindex Physicochemical Properties — Lookup Table

**What it does:** For each amino acid type, look up physical/chemical properties. VdW volume was a top-3 discriminator in AlloPED (2025).

```python
AAINDEX = {
    'ALA': {'vdw_volume': 67, 'hydrophobicity': 1.8, 'polarizability': 0.046, 'flexibility': 0.357, 'charge': 0},
    'VAL': {'vdw_volume': 105, 'hydrophobicity': 4.2, 'polarizability': 0.057, 'flexibility': 0.386, 'charge': 0},
    # ... 20 amino acids
}
```

**Output:** 5-8 features per residue (vdw_volume, hydrophobicity, polarizability, flexibility_index, isoelectric_point, residue_mass)
**Time:** Zero compute (pure dictionary lookup)
**Why it matters:** Current features describe WHERE a residue is (distance, contacts) and HOW it moves (GNM), but not WHAT it is chemically. These fill that gap.

### 2.q P2Rank Pocket Propensity — ML-Based Pocket Detection

**What it does:** ML-trained pocket detector (Random Forest on 35 surface features). Alternative to FPocket with better benchmark performance. Outputs per-residue pocket propensity scores.

**Tool:** P2Rank (Java CLI, works on Windows natively)
```bash
# Download from https://github.com/rdk/p2rank
prank.bat predict -f protein.pdb
# Output: {protein}_residues.csv with columns: residue_label, score, probability, pocket_id
```

**Output:** 2 features per residue (pocket_score, pocket_probability)
**Time:** <1 second per protein (pre-trained model inference)
**Why it matters:** FPocket uses Voronoi geometry, P2Rank uses ML. Disagreements between them are informative. A residue flagged by both is more likely a real pocket.

### 2.r PSSM from PSI-BLAST — Evolutionary Conservation Profile

**What it does:** PSI-BLAST searches for homologs and builds a 20-number profile per residue encoding evolutionary substitution patterns. Allosteric residues show intermediate conservation (not ultra-conserved like active sites, not variable like surface loops).

**Tool:** BLAST+ (Windows native, no WSL needed)
```bash
# Download UniRef50 (~24 GB uncompressed), format with makeblastdb
psiblast -query protein.fasta -db uniref50_db -num_iterations 3 -out_ascii_pssm pssm.txt
```

**Output:** 20 features per residue (log-odds for each amino acid) + 1 scalar conservation score
**Time:** 2-10 min per protein. **Realistic estimate: 40-60 hrs parallel** (Opus review: UniRef50 formatted DB is ~120 GB not 24 GB; Windows BLAST+ is 20-40% slower than Linux; large proteins >1000 residues take 20-30 min). **Run via WSL for better I/O.** Benchmark on 20 proteins first.
**Why it matters:** Used in AlloFusion (JCIM 2025), AllosES (JCIM 2024), STINGAllo (2025). Encodes which positions are under evolutionary selection pressure. Complements ESM-2 embeddings.

### 2.s Spatial Neighbor Evolutionary Info (SNEI) — Conservation Context

**What it does:** For each residue, average the conservation scores of its spatial neighbors within 8A/12A. Captures whether a residue sits in a conserved microenvironment even if the residue itself is variable.

```python
# Requires PSSM entropy (from 2.r) + CA-CA distance matrix (already computed)
# Step 1: Reduce 20-dim PSSM row to scalar conservation = Shannon entropy of PSSM row
#   conservation[i] = -sum(softmax(pssm[i]) * log(softmax(pssm[i])))
#   Low entropy = conserved (few substitutions tolerated)
# Step 2: Average over spatial neighbors, EXCLUDING self and sequence neighbors ±2
for i in range(n_residues):
    spatial_8A = (dist_matrix[i] < 8.0) & (dist_matrix[i] > 0) & (np.abs(np.arange(n_residues) - i) > 2)
    spatial_12A = (dist_matrix[i] < 12.0) & (dist_matrix[i] > 0) & (np.abs(np.arange(n_residues) - i) > 2)
    snei_8A[i] = np.mean(conservation[spatial_8A]) if spatial_8A.any() else 0.0
    snei_12A[i] = np.mean(conservation[spatial_12A]) if spatial_12A.any() else 0.0
```

**Output:** 2-3 features per residue
**Time:** Seconds (after PSSM is computed)
**Why it matters:** AllosES identified this as one of its most discriminative features. Allosteric sites tend to sit at interfaces between conserved and variable regions.
**Note:** Exclude self (distance 0) and sequence neighbors (±2 positions, always within 8A due to backbone geometry ~3.8A per residue) to capture truly spatial neighborhood conservation.

### 2.t Local Frustration Index — Energetic Frustration

**What it does:** Measures whether each residue's contacts are energetically optimal. "Frustrated" residues have suboptimal contacts — they're spring-loaded for conformational change, exactly what allosteric sites need.

**Tool:** frustrapy (pip install frustrapy)
```python
import frustrapy
result = frustrapy.calculate_frustration(pdb_file, mode='singleresidue')
# Output: per-residue frustration index (continuous), + classification (min/neutral/high)
```

**Output:** 2-3 features per residue (frustration_index, frustrated_contact_density, frustrated_proportion)
**Time:** ~30-60 sec per protein, ~2 hrs parallel for all 2043
**Why it matters:** Nature Communications 2023 showed frustration conservation is enriched at allosteric sites. PNAS 2024 showed frustration predicts conformational motions with AlphaFold2.

### 2.u Miyazawa-Jernigan Contact Energy

**What it does:** For each residue, sum up statistical contact potentials with all neighbors. Energetically suboptimal contacts indicate regions prone to conformational switching.

```python
MJ_MATRIX = {('ALA','ALA'): -2.72, ('ALA','ARG'): -2.57, ...}  # 20x20 lookup (Miyazawa & Jernigan 1996)
for i in range(n_residues):
    contacts = get_contacts_within_8A(i)
    total_mj = sum(MJ_MATRIX[restype_i, restype_j] for j in contacts)
    n_contacts = len(contacts)
    mj_energy_per_contact[i] = total_mj / max(n_contacts, 1)  # NORMALIZE by contact count
```

**Output:** 1 feature per residue (mj_energy_per_contact — normalized to avoid protein-size bias)
**Time:** <1 second per protein (pure lookup)
**Note:** Raw MJ energy correlates with protein size (more contacts = lower total energy). Always normalize by contact count to get a meaningful per-residue signal.

### 2.v IMPORTANT: Collinearity Warning for GNM-Derived Features (Opus Review)

PRS, cross-correlations, mutual information, and GNM fluctuation are ALL derived from the same Kirchhoff/covariance matrix. Adding 7+ correlated features from one source creates problems:
- **Feature importance dilution:** SHAP/gain splits across correlated features, each appears less important than the signal warrants
- **Overfitting risk:** More ways to memorize training data (train-test gap already 0.123)

**Required step before training:** After extracting all GNM-derived features, run pairwise Pearson correlation. Drop any feature with r > 0.90 with another GNM feature, keeping the one with higher univariate AUROC. Alternatively, PCA all GNM-derived features into 3-4 components.

### 2.w Additional Features for Future Investigation

**Coevolution (DCA/EVcouplings):** Highest theoretical impact but infeasible at scale (~hours/protein for MSA+DCA). Consider after PSSM if time allows.

**ProtT5-XL embeddings:** AlloPED found ProtT5 > ESM-2 for allosteric prediction. Same pipeline as ESM-2, 3B params, fits RTX 4070 in FP16. PCA→128 dim. Try as ESM-2 replacement or concatenate for 256-dim LM features. If concatenating with ESM-2, PCA the joint raw embeddings (1280+2560=3840 → 128-256 components) rather than concatenating two separate PCAs.

**AlphaFold pLDDT scores:** Per-residue confidence from AlphaFold DB. Requires downloading AF structures + residue mapping. Lower priority.

**Disorder prediction (IUPred):** `pip install iupred3`. Allosteric sites sometimes overlap with disordered regions.

**Hydrogen-Deuterium Exchange (HDX) — Investigated, DEFERRED:** Computational HDX prediction (Best-Vendruscolo formula) computes `ln(PF) = 0.35 × contacts_6.5A + 2.0 × H-bonds`, which is a linear combination of features we already have (contact counts + DSSP secondary structure). XGBoost can learn this combination implicitly. Experimental HDX databases are too sparse (~dozens of overlapping proteins). The genuinely useful HDX signal (differential HDX apo vs holo, local unfolding energetics) requires MD simulations or COREX/BEST (impractical at scale). **Revisit only if:** (1) we run MD simulations for a subset of proteins, or (2) a scalable computational HDX tool emerges.

**AAindex note (Opus review):** Section 2.f already computes hydrophobicity, molecular weight, charge, VdW volume, flexibility, polarity. Plus 20-dim AA one-hot lets XGBoost learn any amino acid property trivially. **Before adding AAindex features, audit against existing 64-dim feature list and only add genuinely novel properties** (polarizability, isoelectric_point if not already present). Consider AAindex PCA (all 566 scales → top 5-8 PCA components per AA type) instead of cherry-picking individual properties.

### 2.x Feature Summary Table (Updated)

**Note:** Structural subtotal = 64 includes AA one-hot (20), physicochemical (5), and other sub-features not individually listed. See extract_features.py for the definitive list. After adding Tier 1 GNM features, run correlation analysis and drop r > 0.90 pairs (see 2.v).

| Feature Group | # Features | Compute Time | Dependencies | Importance | Status |
|--------------|-----------|-------------|-------------|------------|--------|
| B-factors | 4 | <1s | BioPython | Medium | DONE |
| SASA (abs + rel) | 3 | <1s | FreeSASA | High | DONE (abs), TODO (rel) |
| Contact density | 5 | <1s | SciPy | Medium | DONE |
| DSSP secondary structure | 3 | <2s | mkdssp | Medium | DONE |
| HSE half-sphere | 2 | <1s | BioPython | Medium | DONE |
| AA one-hot + physicochemical | 25 | <1s | None | Medium | DONE |
| Sequence position, dist, phi/psi | 4 | <1s | BioPython/NumPy | Medium | DONE |
| Other structural (see script) | 18 | <1s | Various | Medium | DONE |
| **Structural subtotal** | **64** | | | | **DONE** |
| GNM fluctuation + NMA features | 6 | 2-10s | ProDy | Very High | DONE |
| Graph centrality (5 types incl. clustering) | 5 | <2s | NetworkX | High | DONE |
| **NMA/Graph subtotal** | **11** | | | | **DONE (GNM only)** |
| FPocket pocket features | 8 | 5-30s/protein | FPocket (WSL) | High | RUNNING |
| ESM-2 PCA embeddings | 128 | 2-5s GPU | PyTorch | Very High | DONE (650M) |
| **Current total** | **211** | | | | |
| | | | | | |
| **--- NEW: TIER 1 (free wins) ---** | | | | | |
| PRS effectiveness/sensitivity | 2 | <1s | ProDy (existing) | Very High | TODO |
| ESSA z-scores | 1 | BENCHMARK FIRST | ProDy (existing) | Very High | TODO |
| GNM mutual information | 1 | <1s | NumPy (existing) | High | TODO |
| GNM cross-correlation | 3 | <1s | ProDy (existing) | High | TODO |
| AAindex (non-redundant only) | 2-4 | 0s | None (lookup) | Medium | TODO |
| MJ contact energy (normalized) | 1 | <1s | None (lookup) | Medium | TODO |
| Relative SASA | 1 | 0s | FreeSASA (existing) | Medium | TODO |
| **Tier 1 subtotal** | **~11-13** | **minutes total** | **no new deps** | | |
| | | | | | |
| **--- NEW: TIER 2 (new tools) ---** | | | | | |
| P2Rank pocket propensity | 2 | <1s/protein | Java 11+ CLI | High | TODO |
| PSSM (PSI-BLAST) | 21 | 2-10 min/protein | BLAST+, UniRef50 (~120GB) | Very High | TODO |
| SNEI (spatial conservation) | 3 | seconds | After PSSM | High | TODO |
| Local frustration | 3 | 30-60s/protein | frustrapy (test first!) | High | TODO |
| **Tier 2 subtotal** | **~29** | **~40-60 hrs total** | | | |
| | | | | | |
| **--- NEW: TIER 3 (month 3) ---** | | | | | |
| ProtT5-XL PCA embeddings | 128 | 2-5s GPU | transformers | High | TODO |
| MSA entropy (JSD) | 3 | ~28 hrs (MSA build) | HMMER, UniRef50 | High | TODO |
| | | | | | |
| **GRAND TOTAL (all tiers)** | **~385-400** | | | | |

**Post-extraction step:** Run pairwise correlation on all GNM-derived features (PRS, MI, cross-corr, fluctuation). Drop any with r > 0.90. Expected final Tier 1 count after filtering: ~8-10 features.

---

## 3. MODEL ARCHITECTURE OPTIONS

### 3.a Classical ML: XGBoost on Hand-Crafted Features

**Architecture:** Concatenate all ~62 hand-crafted features per residue into a flat vector. Feed into XGBoost binary classifier.

**Implementation:**
```python
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score

# Feature matrix: (n_residues_total, 62)
# Label vector: (n_residues_total,)
# Group vector: protein_id per residue (for GroupKFold)

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=19.0,  # Handle class imbalance
    tree_method="gpu_hist",  # GPU acceleration
    eval_metric="aucpr",
    early_stopping_rounds=50,
)

# Cross-validation with GroupKFold (groups = protein_id)
gkf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    model.fit(X[train_idx], y[train_idx],
              eval_set=[(X[val_idx], y[val_idx])],
              verbose=50)
```

**Window context features (important trick):**
```python
def add_window_features(X_residue, sequence_lengths, window=5):
    """Add features from neighboring residues (sliding window)."""
    X_windowed = []
    idx = 0
    for seq_len in sequence_lengths:
        seq_features = X_residue[idx:idx+seq_len]
        for i in range(seq_len):
            neighbors = []
            for offset in range(-window, window+1):
                j = i + offset
                if 0 <= j < seq_len:
                    neighbors.append(seq_features[j])
                else:
                    neighbors.append(np.zeros_like(seq_features[0]))
            X_windowed.append(np.concatenate(neighbors))
        idx += seq_len
    return np.array(X_windowed)
# This expands features from 62 to 62 * 11 = 682 with window=5
```

**Expected performance:**
- MCC: 0.35-0.50 (based on PASSer2.0 and AlloPED-pocket results)
- AUROC: 0.80-0.88
- F1 (allosteric class): 0.45-0.60

**Pros:**
- Fast to train (minutes, not hours)
- Feature importance is directly interpretable
- No GPU needed
- Easy to iterate and debug
- Strong baseline that often beats deep learning when data is limited

**Cons:**
- Cannot capture long-range dependencies beyond the window
- Feature engineering is manual and domain-knowledge-intensive
- No end-to-end learning

**Libraries:** `xgboost`, `scikit-learn`, `lightgbm` (alternative)
**Compute:** CPU only, trains in 5-15 minutes for 500 proteins

### 3.b Fine-Tuned ESM-2 with Classification Head

**Architecture:** ESM-2 encoder -> per-token linear classification head. Fine-tune with LoRA to keep memory manageable.

**Implementation:**
```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch

# Load ESM-2 with token classification head
model_name = "facebook/esm2_t30_150M_UR50D"
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=2,  # allosteric vs. non-allosteric
)

# Apply LoRA
lora_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "key", "value"],  # Attention layers
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Trainable: ~0.5% of total parameters

# Training loop
tokenizer = AutoTokenizer.from_pretrained(model_name)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Weighted loss for class imbalance
pos_weight = torch.tensor([19.0]).to(device)
criterion = torch.nn.CrossEntropyLoss(
    weight=torch.tensor([1.0, 19.0]).to(device)
)

for epoch in range(20):
    for batch in train_loader:
        inputs = tokenizer(batch["sequences"], return_tensors="pt",
                          padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = batch["labels"].to(device)  # (batch, seq_len)

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**Expected performance:**
- MCC: 0.40-0.55 (based on fine-tuning PLMs -- Nat Comms 2024 shows consistent improvement)
- AUROC: 0.85-0.92
- F1 (allosteric class): 0.50-0.65

**Pros:**
- Leverages massive pre-training on evolutionary data
- Captures long-range sequence dependencies
- State-of-the-art approach (DeepAllo, AlloFusion both use PLMs)
- LoRA makes fine-tuning memory-efficient

**Cons:**
- Sequence-only: no structural information (SASA, depth, contacts, NMA)
- Limited to 1024 tokens (proteins >1022 residues need windowing)
- Needs GPU for reasonable training speed
- Risk of overfitting on small dataset (~400 proteins)

**Libraries:** `transformers`, `peft`, `accelerate`, `torch`
**Compute:** 150M model with LoRA fits in 5GB VRAM. Training: ~2-4 hours on single GPU.

### 3.c Graph Neural Network (GNN) on Residue Contact Graph

**Architecture:** Build a residue contact graph from 3D structure. Node features = hand-crafted features per residue. Edge features = distance, contact type. Use Graph Attention Network (GAT) or GVP for per-node classification.

**Implementation:**
```python
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

class AllostericGNN(torch.nn.Module):
    def __init__(self, in_channels=62, hidden_channels=128, num_layers=4):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels, heads=4, concat=False))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=4, concat=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(GATConv(hidden_channels, hidden_channels, heads=4, concat=False))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 2),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)

        # Per-node classification
        out = self.classifier(x)
        return out

# Data preparation
def protein_to_graph(features_dict, contact_graph, labels):
    """Convert a protein to a PyTorch Geometric Data object."""
    residues = sorted(features_dict.keys())
    node_features = torch.tensor([features_dict[r] for r in residues], dtype=torch.float)

    edges = []
    for u, v in contact_graph.edges():
        i = residues.index(u)
        j = residues.index(v)
        edges.append([i, j])
        edges.append([j, i])  # Undirected

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    y = torch.tensor([labels.get(r, 0) for r in residues], dtype=torch.long)

    return Data(x=node_features, edge_index=edge_index, y=y)
```

**Expected performance:**
- MCC: 0.40-0.55 (GNNs capture structural context well)
- AUROC: 0.85-0.90
- F1 (allosteric class): 0.50-0.60

**Pros:**
- Naturally encodes protein structure as a graph
- Captures spatial neighborhoods and long-range structural contacts
- Inductive: can generalize to unseen protein folds
- Message passing propagates information across the structure

**Cons:**
- Requires 3D structure (not applicable to sequence-only predictions)
- Small dataset may cause overfitting
- Graph construction choices (cutoff, edge features) significantly affect results
- More complex to implement and debug than XGBoost

**Libraries:** `torch_geometric`, `torch`, `networkx`
**Compute:** Fits easily in 8GB VRAM. Training: ~1-3 hours for 500 proteins.

### 3.d Hybrid: ESM-2 Embeddings + Structural Features into XGBoost

**Architecture:** Extract ESM-2 per-residue embeddings (1280-dim), concatenate with hand-crafted structural features (62-dim), feed the 1342-dim vector into XGBoost.

**This is the recommended approach for this project.** It combines the best of both worlds:
- ESM-2 captures evolutionary/sequence patterns
- Structural features capture 3D geometry
- XGBoost handles tabular data well and is robust to small datasets

**Implementation:**
```python
import numpy as np
import xgboost as xgb

# 1. Extract ESM-2 embeddings for all proteins (offline, save to disk)
# Shape per protein: (seq_len, 1280)

# 2. Extract structural features for all proteins
# Shape per protein: (seq_len, 62)

# 3. Concatenate
# Shape per protein: (seq_len, 1342)

# 4. Optionally reduce ESM-2 dimensionality with PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=128)
esm_reduced = pca.fit_transform(esm_embeddings)  # (n_residues, 128)
# Now total features: 128 + 62 = 190

# 5. Train XGBoost
X = np.hstack([esm_reduced, structural_features])
model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.03,
    scale_pos_weight=19.0,
    colsample_bytree=0.6,
    subsample=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
)
```

**Expected performance (best expected for this project):**
- MCC: 0.45-0.60
- AUROC: 0.88-0.93
- F1 (allosteric class): 0.55-0.70

**Pros:**
- Best expected performance with limited data
- ESM-2 embeddings are extracted once and cached (no GPU needed for training)
- XGBoost training is fast and interpretable
- Easy to ablate features and understand contributions
- Robust to small datasets

**Cons:**
- Two-stage pipeline (not end-to-end)
- PCA on ESM-2 embeddings may lose information
- Window context must be manually engineered

**Libraries:** `transformers`, `xgboost`, `scikit-learn`
**Compute:** ESM-2 extraction: ~30 min GPU. XGBoost training: ~15 min CPU.

### 3.e 3D CNN on Voxelized Protein Structure

**Architecture:** Voxelize the protein structure into a 3D grid. Each voxel contains atom type/density information. Apply 3D convolutions to predict per-residue labels.

**NOT recommended for this project:**
- Requires significant GPU memory for 3D convolutions
- Rotation invariance is not guaranteed (need data augmentation)
- Resolution vs. grid size tradeoff is difficult
- Per-residue labels from voxels requires post-processing
- Much worse data efficiency than graph-based methods
- Published methods using this approach (e.g., DeepSite) are older and outperformed by sequence+graph methods

**Skip this approach. Included only for completeness.**

### 3.f Model Comparison Summary

| Approach | Expected MCC | Training Time | GPU Required? | Implementation Effort | Recommended? |
|----------|-------------|---------------|---------------|----------------------|-------------|
| XGBoost + handcrafted | 0.35-0.50 | 15 min | No | Low | Yes (baseline) |
| ESM-2 + LoRA fine-tune | 0.40-0.55 | 2-4 hrs | Yes | Medium | Yes (if data enough) |
| GNN (GAT) | 0.40-0.55 | 1-3 hrs | Optional | Medium-High | Optional |
| **Hybrid (ESM-2 + struct -> XGB)** | **0.45-0.60** | **45 min total** | **GPU for extraction only** | **Medium** | **PRIMARY** |
| 3D CNN | 0.30-0.45 | 6+ hrs | Yes | High | No |

---

## 4. BASELINE COMPARISONS

### 4.1 Tools You MUST Compare Against

**Tier 1 -- Essential baselines (must include):**

| Tool | Type | How to run | URL |
|------|------|-----------|-----|
| FPocket | Geometry-based pocket detection | `fpocket -f protein.pdb` | https://github.com/Discngine/fpocket |
| PASSer (Ensemble) | XGBoost + GCN ensemble | Web: https://passer.smu.edu/ or API | https://passer.smu.edu/ |
| PASSer2.0 | AutoML (AutoGluon) | Web server or downloadable code | Same as above |

**Tier 2 -- Recent state-of-the-art (include if possible):**

| Tool | Type | How to run | URL |
|------|------|-----------|-----|
| DeepAllo | ProtBERT + XGBoost + AutoML | GitHub: https://github.com/MoaazK/deepallo | Bioinformatics 2025 |
| AlloPED | ProtT5 + DCNN | GitHub: https://github.com/mjcoo/AlloPED | bioRxiv 2025 |
| AlloFusion | ProtT5 + PSSM + multi-feature | GitHub: https://github.com/hjb-001/AlloFusion | JCIM 2025 |

**Tier 3 -- Older methods (include for historical context):**

| Tool | Type | How to run | URL |
|------|------|-----------|-----|
| Allosite / AllositePro | SVM-based | Web server | http://mdl.shsmu.edu.cn/AST/ |
| PARS | Normal mode analysis | Web server | http://bioinf.uab.cat/cgi-bin/pars-cgi/pars.pl |
| Ohm | Physics-based | Web server | http://dokhlab.med.psu.edu/ohm/ |

### 4.2 How to Run Key Baselines

**FPocket baseline:**
```bash
# Run FPocket on all test proteins
for pdb in test_proteins/*.pdb; do
    fpocket -f $pdb
done

# Evaluate: For each protein, check if FPocket's top-N pockets
# overlap with the known allosteric site
```

```python
def evaluate_fpocket(fpocket_pockets, true_allosteric_residues, top_n=3):
    """Compute Jaccard Index between FPocket top-N pockets and true site."""
    predicted_residues = set()
    for pocket_id in range(1, top_n + 1):
        if pocket_id in fpocket_pockets:
            predicted_residues.update(fpocket_pockets[pocket_id])

    intersection = predicted_residues & true_allosteric_residues
    union = predicted_residues | true_allosteric_residues

    jaccard = len(intersection) / len(union) if union else 0
    precision = len(intersection) / len(predicted_residues) if predicted_residues else 0
    recall = len(intersection) / len(true_allosteric_residues) if true_allosteric_residues else 0

    return {"jaccard": jaccard, "precision": precision, "recall": recall}
```

**PASSer baseline (via API):**
```python
import requests

def run_passer(pdb_file):
    """Submit a PDB to PASSer web server."""
    url = "https://passer.smu.edu/api/predict"
    with open(pdb_file, 'rb') as f:
        response = requests.post(url, files={"file": f})
    return response.json()  # Returns ranked pockets with residue lists
```

**DeepAllo baseline (local):**
```bash
git clone https://github.com/MoaazK/deepallo.git
cd deepallo
pip install -r requirements.txt
python pipeline.py --pdb protein.pdb
```

### 4.3 Evaluation Metrics

**Primary metrics (report ALL of these):**

```python
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, confusion_matrix
)

def full_evaluation(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUROC": roc_auc_score(y_true, y_pred_proba),
        "AUPRC": average_precision_score(y_true, y_pred_proba),
        "F1": f1_score(y_true, y_pred, pos_label=1),
        "Precision": precision_score(y_true, y_pred, pos_label=1),
        "Recall": recall_score(y_true, y_pred, pos_label=1),
        "Specificity": recall_score(y_true, y_pred, pos_label=0),
    }

    # Precision@k (top-k residues per protein)
    # This requires protein-level evaluation
    return metrics

def precision_at_k(y_true_per_protein, y_proba_per_protein, k_values=[10, 20, 50]):
    """For each protein, rank residues by predicted probability.
    Report what fraction of the top-k are truly allosteric."""
    results = {k: [] for k in k_values}
    for y_true, y_proba in zip(y_true_per_protein, y_proba_per_protein):
        ranking = np.argsort(-y_proba)
        for k in k_values:
            top_k = ranking[:k]
            prec = np.mean(y_true[top_k])
            results[k].append(prec)
    return {f"P@{k}": np.mean(v) for k, v in results.items()}
```

**Pocket-level metrics (for comparison with FPocket/PASSer):**
- Top-1 accuracy: Is the highest-ranked pocket the allosteric site?
- Top-3 accuracy: Is the allosteric site among the top-3 ranked pockets?
- Site overlap Jaccard Index: Residue overlap between predicted and true site

**Current state of the art (approximate, from AlloBench benchmarking):**
- Best pocket-level top-3 accuracy: ~82.7% (PASSer2.0)
- Best residue-level MCC: ~0.544 (AlloPED)
- Best residue-level AUROC: ~0.920 (AlloPED)
- Best pocket-level F1: ~89.66% (DeepAllo)
- AlloBench comprehensive benchmark: ALL tools below 60% accuracy on the cleaned dataset

**To be competitive, your model should aim for:**
- Residue-level MCC > 0.50
- AUROC > 0.90
- Something these existing tools cannot do (see Publication Strategy)

---

## 5. VALIDATION STRATEGY

### 5.1 Cross-Validation Protocol

```python
# Cluster-aware 5-fold cross-validation
# Step 1: Cluster proteins at 30% sequence identity using MMseqs2
# Step 2: Assign fold IDs at the cluster level
# Step 3: All proteins in the same cluster go to the same fold

from sklearn.model_selection import GroupKFold

def cluster_aware_cv(X, y, protein_ids, cluster_assignments, n_splits=5):
    """
    X: feature matrix (n_residues, n_features)
    y: labels (n_residues,)
    protein_ids: which protein each residue belongs to (n_residues,)
    cluster_assignments: which cluster each protein belongs to
    """
    # Map each residue to its protein's cluster
    residue_clusters = [cluster_assignments[pid] for pid in protein_ids]

    gkf = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=residue_clusters)):
        # Verify: no cluster appears in both train and val
        train_clusters = set(np.array(residue_clusters)[train_idx])
        val_clusters = set(np.array(residue_clusters)[val_idx])
        assert len(train_clusters & val_clusters) == 0, "Data leakage detected!"

        yield fold, train_idx, val_idx
```

### 5.2 Held-Out Test Sets

**Test Set 1: ASBench Core-Diversity (147 sites)**
- Rationale: Well-established benchmark, many methods do NOT train on it
- Exclude any ASBench proteins that overlap with AlloBench training data (check by UniProt ID)

**Test Set 2: CASBench subset (91 proteins)**
- Rationale: Tests ability to distinguish allosteric from catalytic sites
- Unique because catalytic site labels are also available

**Test Set 3: Recently deposited PDB structures (post-2024)**
- Download proteins deposited after Jan 2024 that have allosteric modulators
- These cannot have been in any training set
- Query PDB: `release_date:[2024-01-01 TO *] AND ligand_type:allosteric`

### 5.3 Case Study Proteins

**Case Study 1: COX-2 (PDB: 4PH9)**
- You already have this structure
- COX-2 is a conformational heterodimer with well-characterized allosteric monomer (Eallo)
- Known allosteric site involves fatty acid binding to Eallo subunit
- Predict allosteric residues and compare to published mutagenesis data
- Compelling because COX-2 is a major drug target

**Case Study 2: Protein Kinase A (PKA, PDB: 1ATP / 4WB5)**
- Classic allosteric protein; allosteric site well-characterized
- In ASD database; multiple structures with different allosteric modulators

**Case Study 3: Hemoglobin (PDB: 2HHB)**
- Textbook allosteric protein (T/R state transition)
- Allosteric site (2,3-BPG binding) is far from active site
- Great for illustrating that the model detects distal regulatory sites

**Case Study 4: GPCRs (e.g., Beta-2 adrenergic receptor, PDB: 4LDO)**
- Allosteric modulators are a hot topic in GPCR drug discovery
- Multiple known allosteric sites

**Case Study 5: A protein NOT in any database (novel prediction)**
- Choose a medically relevant enzyme with suspected but unconfirmed allosteric regulation
- Predict allosteric sites; propose validation experiments
- This is what makes the paper exciting

### 5.4 Validating Novel Predictions

For predicted sites NOT in any database:

1. **Literature cross-check:** Search PubMed for mutagenesis studies on the predicted residues. If mutations at predicted sites alter activity without being in the active site, that is evidence.

2. **Conservation analysis:** Predicted allosteric sites should be more conserved than random surface residues but less than catalytic residues.

3. **Molecular dynamics (optional):** Run short (100 ns) MD simulations. If perturbing predicted site residues (in silico mutagenesis) alters dynamics of the active site, this supports the prediction.

4. **Docking validation:** Dock known allosteric modulators into predicted pockets using AutoDock Vina (you already know how to do this). If the modulator binds with reasonable affinity, that is supportive.

5. **Structural comparison:** Compare predicted sites across homologous proteins. If the same structural location is predicted across the family, it is more likely to be a true allosteric site.

---

## 6. DETAILED TIMELINE (8 MONTHS)

### Month 1: Data Curation and Environment Setup (Weeks 1-4)

**Week 1: Environment and data acquisition**
- [ ] Set up Python environment with all packages (see Section 9)
- [ ] Download AlloBench dataset from Figshare
- [ ] Download ASBench and CASBench datasets
- [ ] Download ASD2023 supplementary data
- [ ] Install FPocket via conda, test on a single protein
- [ ] Install mkdssp, test on a single protein
- Deliverable: Working environment, all raw data downloaded

**Week 2: Data processing pipeline**
- [ ] Write script to extract per-residue allosteric labels from AlloBench CSV
- [ ] Write script to download PDB files for all proteins in the dataset
- [ ] Handle missing structures, obsolete PDB IDs
- [ ] Run sequence extraction for all proteins
- Deliverable: Clean dataset of PDB files with per-residue labels

**Week 3: Dataset splitting**
- [ ] Run MMseqs2 clustering at 30% identity
- [ ] Create train/val/test splits at the cluster level
- [ ] Verify no data leakage
- [ ] Generate dataset statistics (class balance, protein size distribution)
- Deliverable: Final train/val/test splits saved as JSON/CSV

**Week 4: Feature pipeline scaffold**
- [ ] Write modular feature extraction framework
- [ ] Each feature type = one Python module with a common interface
- [ ] Test on 10 proteins end-to-end
- [ ] Set up result logging and experiment tracking (Weights & Biases or MLflow)
- Deliverable: Feature extraction pipeline that processes one protein end-to-end

### Month 2: Feature Engineering (Weeks 5-8)

**Week 5: Structural features**
- [ ] Compute B-factors, SASA, DSSP, contact density for all proteins
- [ ] Compute residue depth (handle MSMS failures gracefully)
- [ ] Compute amino acid physicochemical properties
- [ ] Quality check: visualize features for 5 proteins in PyMOL
- Deliverable: Structural features for all proteins

**Week 6: Graph and NMA features**
- [ ] Build residue contact graphs for all proteins
- [ ] Compute graph centrality features
- [ ] Run ProDy ANM/GNM analysis for all proteins
- [ ] Handle edge cases (small proteins, missing coordinates)
- Deliverable: Graph and flexibility features for all proteins

**Week 7: Conservation and FPocket features**
- [ ] Run PSI-BLAST and generate PSSMs for all sequences (can parallelize)
- [ ] Run FPocket on all proteins (batch script)
- [ ] Parse FPocket output, extract per-residue pocket features
- Deliverable: Conservation and pocket features for all proteins

**Week 8: ESM-2 embeddings**
- [ ] Extract ESM-2 (650M) embeddings for all proteins
- [ ] Save embeddings to disk (HDF5 or NumPy format)
- [ ] Verify alignment between embedding positions and PDB residue numbering
- [ ] Apply PCA to reduce to 128 dimensions; save transform
- Deliverable: ESM-2 embeddings for all proteins, complete feature matrix

### Month 3: Baseline Model and Initial Results (Weeks 9-12)

**Week 9: XGBoost baseline (structural features only)**
- [ ] Assemble feature matrix (62 features, no ESM-2)
- [ ] Train XGBoost with GroupKFold CV
- [ ] Hyperparameter tuning (Optuna or grid search)
- [ ] Record MCC, AUROC, AUPRC, F1 on validation set
- Deliverable: Baseline XGBoost model with structural features

**Week 10: Hybrid model (ESM-2 + structural + XGBoost)**
- [ ] Concatenate ESM-2 (PCA-reduced) with structural features
- [ ] Train XGBoost hybrid model
- [ ] Compare to structural-only baseline
- [ ] Feature importance analysis (SHAP values)
- Deliverable: Hybrid model; comparison table

**Week 11: External baselines**
- [ ] Run FPocket on all test proteins; compute metrics
- [ ] Submit test proteins to PASSer web server (or use API)
- [ ] Run DeepAllo on test proteins (if code available)
- [ ] Compile comparison table
- Deliverable: Head-to-head comparison of your model vs. 3 baselines

**Week 12: Analysis and iteration**
- [ ] Error analysis: which proteins does the model fail on? Why?
- [ ] Feature ablation study: remove one feature group at a time
- [ ] Window size ablation (5, 7, 9, 11 residues)
- [ ] Decide whether to pursue GNN or ESM-2 fine-tuning next
- Deliverable: Ablation results; decision on next model

### *** MINIMUM PUBLISHABLE UNIT CHECKPOINT ***
At this point (Week 12), you have enough for a paper IF:
- Your hybrid model outperforms all baselines on at least one metric
- You have a compelling feature importance story
- You apply it to COX-2 and get biologically interpretable results

### Month 4: Advanced Models (Weeks 13-16)

**Week 13-14: ESM-2 LoRA fine-tuning (if data supports it)**
- [ ] Implement token classification with LoRA
- [ ] Train on AlloBench training set
- [ ] Monitor for overfitting (small dataset risk)
- [ ] Compare to frozen ESM-2 + XGBoost hybrid
- Deliverable: Fine-tuned ESM-2 model

**Week 15-16: GNN model (optional, if time permits)**
- [ ] Implement GAT-based per-residue classifier
- [ ] Train with structural features as node features
- [ ] Try: GNN with ESM-2 embeddings as additional node features
- [ ] Compare all architectures
- Deliverable: GNN model; comprehensive architecture comparison

### Month 5: Validation and Case Studies (Weeks 17-20)

**Week 17: Held-out test set evaluation**
- [ ] Evaluate all models on ASBench Core-Diversity
- [ ] Evaluate on CASBench
- [ ] Evaluate on recently deposited structures (post-2024)
- Deliverable: Final test results on three held-out sets

**Week 18-19: COX-2 case study**
- [ ] Predict allosteric sites on COX-2 (4PH9)
- [ ] Compare predictions to known Eallo binding site
- [ ] Cross-reference with mutagenesis literature
- [ ] Dock a known allosteric modulator into predicted sites
- [ ] Generate PyMOL visualizations
- Deliverable: COX-2 case study with figures

**Week 20: Additional case studies**
- [ ] Hemoglobin (2HHB) -- classic allosteric protein
- [ ] PKA or a kinase
- [ ] One novel prediction target
- Deliverable: 2-3 additional case studies

### Month 6: Paper Writing and Web Server (Weeks 21-24)

**Week 21-22: Paper drafting**
- [ ] Write Methods section (most technical, do first)
- [ ] Write Results section with all figures and tables
- [ ] Write Introduction and Discussion
- [ ] Prepare all supplementary material
- Deliverable: Complete paper draft

**Week 23: Web server or tool packaging (optional but recommended)**
- [ ] Create a simple Flask/Streamlit web app for predictions
- [ ] Or: package as a pip-installable Python tool
- [ ] Deploy on a free platform (Streamlit Cloud, HuggingFace Spaces)
- Deliverable: Publicly accessible prediction tool

**Week 24: Revision and submission**
- [ ] Internal review, revise paper
- [ ] Prepare GitHub repository with code and data
- [ ] Submit to target journal
- Deliverable: Submitted paper

### Months 7-8: Buffer and Revision (Weeks 25-32)

- Respond to reviewer comments
- Additional experiments if requested
- Possible second submission if rejected

---

## 7. RISK ASSESSMENT

### Risk 1: ASD/AlloBench data is insufficient

**Probability:** Medium (30%)
**Impact:** High
**Details:** ~400 unique protein chains may be too few, especially after clustering. Some feature combinations may overfit.

**Mitigation:**
- Start with XGBoost (robust to small data) not deep learning
- Use strong regularization (dropout, weight decay, early stopping)
- Use data augmentation: randomly mask 10% of features during training
- Augment with predicted allosteric sites from ASD2023 (66,589 predicted sites, use as "soft labels" with lower weight)
- Consider multi-task learning: predict pocket membership AND allosteric site simultaneously

**Backup plan:** If dataset is too small, pivot to a transfer learning approach:
1. Pre-train on a large pocket detection dataset (scPDB has ~17,000 structures)
2. Fine-tune on allosteric sites

### Risk 2: ESM-2 embeddings don't help

**Probability:** Low (15%)
**Impact:** Medium
**Details:** If structural features alone are sufficient, ESM-2 adds complexity without benefit.

**Mitigation:**
- This is actually a publishable finding: "structural features suffice for allosteric site prediction"
- Ablation study will clearly demonstrate this
- The XGBoost + structural features baseline is still a valid paper if it beats FPocket/PASSer

**Backup plan:** Replace ESM-2 with simpler sequence features (PSSM + one-hot encoding). If even those don't help, the paper's angle becomes "structure > sequence for allosteric site prediction."

### Risk 3: Cannot beat FPocket + PASSer baselines

**Probability:** Medium (25%)
**Impact:** High
**Details:** FPocket is surprisingly hard to beat for pocket detection. PASSer2.0 already uses XGBoost on FPocket features.

**Mitigation:**
- Your model adds features that PASSer2.0 does NOT use: ESM-2 embeddings, NMA flexibility, graph centrality
- Evaluate at the RESIDUE level (PASSer operates at the pocket level) -- different granularity may favor your approach
- Focus on a specific protein family where your model excels (e.g., enzymes, GPCRs)

**Backup plan:** If you cannot beat PASSer on overall metrics, find a subset where you do. For example:
- "Our model outperforms PASSer on proteins with cryptic allosteric sites"
- "Our model provides residue-level resolution, unlike pocket-based methods"
- Frame as complementary rather than competitive

### Risk 4: Feature computation pipeline is too slow

**Probability:** Low (10%)
**Impact:** Low-Medium
**Details:** PSI-BLAST for PSSMs is the bottleneck (~3-5 min per protein * 500 = ~42 hours).

**Mitigation:**
- Parallelize PSI-BLAST across CPU cores
- Use pre-computed PSSMs from databases like ProteinNet
- For initial experiments, skip PSSM and use ESM-2 as the sole evolutionary feature

### Risk 5: DSSP/MSMS/FPocket installation fails on Windows

**Probability:** Medium (30%)
**Impact:** Low
**Details:** These tools are designed for Linux. Windows compatibility is spotty.

**Mitigation:**
- Use WSL (Windows Subsystem for Linux) for all Linux-only tools
- Use conda for FPocket and DSSP
- For MSMS failure: use the approximate depth calculation (distance to nearest surface residue)
- For DSSP failure: use BioPython's built-in secondary structure prediction or PyMOL's dss command

### Risk 6: Protein sequence length > 1024 (ESM-2 limit)

**Probability:** Medium (20% of proteins)
**Impact:** Low
**Details:** ESM-2 has a 1024-token limit. Some proteins exceed this.

**Mitigation:**
- Use sliding window: split into overlapping windows of 1022 residues with stride 512
- Average embeddings in overlapping regions
- This is standard practice and well-documented

### Risk 7: Model predicts catalytic sites instead of allosteric sites

**Probability:** Medium-High (35%)
**Impact:** High
**Details:** Both catalytic and allosteric sites are functionally important, conserved, and partially buried. The model may learn "important residue" rather than "allosteric residue."

**Mitigation:**
- Include CASBench in training: provides both catalytic and allosteric labels
- Add a "distance to catalytic site" feature (if catalytic site is known)
- Multi-class classification: background / catalytic / allosteric
- Explicitly evaluate false positive rate near catalytic sites
- This is actually a known problem in the field and addressing it would be novel

---

## 8. PUBLICATION STRATEGY

### 8.1 Target Journals (Ranked by Fit)

| Rank | Journal | IF | Why | Timeline |
|------|---------|-----|-----|----------|
| 1 | **Bioinformatics** | 5.8 | DeepAllo was published here; perfect fit for methods paper | 4-8 weeks review |
| 2 | **J. Chem. Inf. Model. (JCIM)** | 5.6 | AlloFusion, AllosES published here; chemistry + informatics focus | 6-10 weeks review |
| 3 | **Briefings in Bioinformatics** | 9.5 | Higher impact, accepts comprehensive methods papers | 8-12 weeks review |
| 4 | **Nucleic Acids Research (Web Server)** | 14.9 | IF you build a web server; NAR web server issue is very prestigious | Strict deadline (usually April) |
| 5 | **PLOS Computational Biology** | 3.8 | Good for more biology-focused story (COX-2 case study) | 6-10 weeks review |
| 6 | **Proteins: Structure, Function, Bioinformatics** | 2.9 | Lower bar, good backup option | 4-8 weeks review |

**Recommended strategy:** Submit to JCIM first (good fit, reasonable review time). If rejected, revise and submit to Bioinformatics. If writing a strong COX-2 story with biological insights, consider Briefings in Bioinformatics.

### 8.2 Framing the Novelty

The field is active with multiple recent publications (DeepAllo, AlloPED, AlloFusion all in 2025-2026). You MUST differentiate. Options:

**Angle A: "Unified multi-modal framework with systematic feature analysis"**
- No existing method combines ALL of: ESM-2 embeddings + NMA flexibility + graph centrality + FPocket + PSSM
- The ablation study showing which features matter most is itself a contribution
- Story: "We systematically evaluate 12 feature categories for allosteric site prediction and identify the optimal combination"

**Angle B: "Residue-level prediction with structural interpretability"**
- Most existing tools predict pockets (clusters of residues), not individual residues
- Per-residue predictions enable: identifying individual key residues for mutagenesis
- Story: "From pocket to residue: fine-grained allosteric site prediction enables rational protein engineering"

**Angle C: "Distinguishing allosteric from catalytic sites"**
- This is an unsolved problem that no existing tool addresses well
- Multi-class classification (background / catalytic / allosteric) would be novel
- Story: "Beyond pocket detection: discriminating allosteric from catalytic sites using machine learning"

**Angle D: "Application to a specific therapeutic target"**
- Deep case study on COX-2 or another drug target
- Predict novel allosteric sites and validate computationally
- Story: "Discovery of novel allosteric sites in COX-2 using interpretable machine learning"

**Recommended: Combine Angle A + B.** The systematic feature analysis is unique and the residue-level granularity differentiates from DeepAllo/PASSer. Add Angle C as a secondary contribution if CASBench results are good.

### 8.3 Required Figures and Tables

**Main figures (6-8):**

1. **Method overview figure** (graphical abstract / pipeline diagram)
   - Input: PDB structure -> Feature extraction (show all 12 categories) -> Model -> Per-residue predictions
   - Tools: BioRender or Adobe Illustrator

2. **Dataset statistics**
   - Panel A: Distribution of protein lengths
   - Panel B: Class balance (pie chart: allosteric vs. non-allosteric residues)
   - Panel C: Clustering dendrogram showing sequence diversity
   - Panel D: Train/val/test split sizes

3. **Feature importance analysis** (SHAP summary plot)
   - Top 20 most important features from XGBoost
   - Color by feature value (high/low)
   - This is often reviewers' favorite figure

4. **Performance comparison bar chart**
   - Grouped bar chart: Your model vs. FPocket vs. PASSer vs. DeepAllo
   - Metrics: MCC, AUROC, AUPRC, F1
   - Error bars from cross-validation

5. **ROC and PR curves**
   - Panel A: ROC curves for all methods (one plot)
   - Panel B: Precision-Recall curves (more informative for imbalanced data)

6. **Feature ablation heatmap**
   - Rows: feature groups removed (one at a time)
   - Columns: metrics (MCC, AUROC, F1)
   - Color: red = performance drop (feature is important)

7. **Case study: COX-2 structural visualization**
   - Panel A: True allosteric site colored on structure (blue)
   - Panel B: Predicted probabilities mapped onto structure (gradient)
   - Panel C: Overlay showing agreement
   - Panel D: False positives -- are they biologically meaningful?
   - Generate with PyMOL using B-factor column for coloring

8. **Architecture comparison**
   - Performance of XGBoost vs. ESM-2 fine-tuned vs. GNN vs. Hybrid
   - Scatter plot: Performance (MCC) vs. Training time

**Main tables (3-4):**

1. **Feature descriptions** -- all 12 categories with tool, dimension, compute time
2. **Cross-validation results** -- per-fold metrics for primary model
3. **Head-to-head comparison** -- all methods, all metrics, all test sets
4. **Case study residues** -- predicted residues vs. known, with literature references

**Supplementary material:**
- Complete feature list with mathematical definitions
- Hyperparameter optimization results
- Additional case studies
- Per-protein performance breakdown
- Code availability statement with GitHub link

### 8.4 Things NOT to Say

- Do NOT claim "first ML method for allosteric site prediction" (PASSer was in 2021)
- Do NOT ignore DeepAllo, AlloPED, AlloFusion -- they are all from 2025
- Do NOT use accuracy as the primary metric (meaningless with 95% negative class)
- Do NOT claim novelty for using ESM-2 (DeepAllo already uses ProtBERT; AlloFusion uses ProtT5)
- DO claim novelty for the specific COMBINATION of features + model + evaluation

---

## 9. ENVIRONMENT SETUP

### 9.1 Complete pip Install Commands

```bash
# Create a fresh virtual environment
python -m venv allosteric_env
# On Windows:
allosteric_env\Scripts\activate

# ============================================================
# CORE SCIENTIFIC COMPUTING
# ============================================================
pip install numpy==1.26.4
pip install scipy==1.13.1
pip install pandas==2.2.2
pip install scikit-learn==1.5.1
pip install matplotlib==3.9.2
pip install seaborn==0.13.2

# ============================================================
# STRUCTURAL BIOLOGY
# ============================================================
pip install biopython==1.84
pip install prody==2.4.1

# SASA calculation
pip install freesasa
# If freesasa fails on Windows, use BioPython's built-in ShrakeRupley instead
# (already included in biopython above)

# ============================================================
# MACHINE LEARNING
# ============================================================
pip install xgboost==2.1.1
pip install lightgbm==4.4.0
pip install optuna==3.6.1          # Hyperparameter optimization
pip install shap==0.45.1           # Feature importance

# ============================================================
# DEEP LEARNING (ESM-2, GNN)
# ============================================================
# Install PyTorch with CUDA support (adjust cuda version to match your GPU)
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.42.4   # For ESM-2
pip install peft==0.11.1           # For LoRA fine-tuning
pip install accelerate==0.31.0     # For distributed training
pip install datasets==2.20.0       # HuggingFace datasets

# For GNN (PyTorch Geometric)
pip install torch-geometric==2.5.3
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.3.1+cu121.html

# ============================================================
# GRAPH ANALYSIS
# ============================================================
pip install networkx==3.3

# ============================================================
# SEQUENCE ANALYSIS
# ============================================================
# MMseqs2: Download binary from https://github.com/soedinglab/MMseqs2/releases
# For Windows: download mmseqs-win64.zip, extract, add to PATH

# BLAST+: Download from https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/
# Install ncbi-blast-2.16.0+-x64-win64.exe

# ============================================================
# EXPERIMENT TRACKING
# ============================================================
pip install wandb==0.17.5          # Weights & Biases (optional)
pip install mlflow==2.14.3         # Alternative to W&B

# ============================================================
# DATA FORMATS
# ============================================================
pip install h5py==3.11.0           # For storing large embedding arrays
pip install pyarrow==16.1.0        # For efficient data storage
pip install tqdm==4.66.4           # Progress bars

# ============================================================
# VISUALIZATION
# ============================================================
pip install pymol-open-source       # May fail on Windows; use conda instead
# conda install -c conda-forge pymol-open-source
# Or use standalone PyMOL (already installed on your system)

# ============================================================
# JUPYTER
# ============================================================
pip install jupyterlab==4.2.4
pip install ipywidgets==8.1.3
```

### 9.2 Non-Python Tool Installation (Windows)

**DSSP (mkdssp):**
```bash
# Option A: conda (easiest)
conda install -c salilab dssp

# Option B: Download binary
# Go to https://github.com/PDB-REDO/dssp/releases
# Download dssp-4.4.7-win64-mingw.zip (or latest)
# Extract mkdssp.exe to a directory in your PATH
# Test:
mkdssp --version
```

**FPocket:**
```bash
# Option A: conda (recommended for Windows)
conda install -c conda-forge fpocket

# Option B: WSL
# In WSL terminal:
sudo apt-get install fpocket

# Test:
fpocket -f test_protein.pdb
```

**MSMS (for residue depth):**
```
# Download from: https://ccsb.scripps.edu/msms/downloads/
# Windows binary: msms_win.zip
# Extract msms.exe to a directory in your PATH
# Test:
msms -help

# If MSMS is unavailable or fails, skip residue depth feature.
# Use the approximate_depth() function from Section 2.g instead.
```

**BLAST+ (for PSSM computation):**
```
# Download installer from:
# https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/
# Run ncbi-blast-2.16.0+-x64-win64.exe
# Default install to C:\Program Files\NCBI\blast-2.16.0+
# Add C:\Program Files\NCBI\blast-2.16.0+\bin to PATH

# Download sequence database (UniRef50, ~12 GB compressed):
# https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/
# Or smaller: SwissProt (~90 MB compressed)

# Format database:
makeblastdb -in uniref50.fasta -dbtype prot -out uniref50

# Test:
psiblast -query test.fasta -db uniref50 -num_iterations 3 -out_ascii_pssm test.pssm
```

**MMseqs2:**
```
# Download from: https://github.com/soedinglab/MMseqs2/releases
# Windows: mmseqs-win64.zip
# Extract mmseqs.exe and mmseqs.bat
# Add to PATH

# Test:
mmseqs version
```

### 9.3 Google Colab Setup (for ESM-2 training with more GPU)

```python
# At the top of your Colab notebook:
!pip install transformers peft accelerate torch-geometric
!pip install xgboost shap prody biopython freesasa

# Check GPU:
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
# Colab free: typically T4 with 15 GB VRAM
# Sufficient for ESM-2 650M fine-tuning with LoRA
```

### 9.4 Project Directory Structure

```
allosteric_project/
|-- data/
|   |-- raw/
|   |   |-- allobench/          # AlloBench CSV and metadata
|   |   |-- asbench/            # ASBench PDB files
|   |   |-- casbench/           # CASBench annotations
|   |   |-- pdb_files/          # Downloaded PDB structures
|   |-- processed/
|   |   |-- labels/             # Per-residue labels (JSON per protein)
|   |   |-- features/           # Extracted features (HDF5/NPY)
|   |   |-- splits/             # Train/val/test assignments
|   |   |-- sequences.fasta     # All protein sequences
|   |   |-- clusters.tsv        # MMseqs2 clustering output
|-- src/
|   |-- data/
|   |   |-- download.py         # Download PDB files
|   |   |-- labeling.py         # Generate per-residue labels
|   |   |-- splitting.py        # Cluster-aware train/test split
|   |-- features/
|   |   |-- bfactors.py
|   |   |-- sasa.py
|   |   |-- dssp.py
|   |   |-- contacts.py
|   |   |-- depth.py
|   |   |-- physicochemical.py
|   |   |-- graph.py
|   |   |-- nma.py
|   |   |-- fpocket.py
|   |   |-- esm2.py
|   |   |-- pssm.py
|   |   |-- assemble.py         # Combine all features
|   |-- models/
|   |   |-- xgboost_model.py
|   |   |-- esm2_finetune.py
|   |   |-- gnn_model.py
|   |   |-- hybrid_model.py
|   |-- evaluation/
|   |   |-- metrics.py
|   |   |-- baselines.py        # Run FPocket, PASSer, etc.
|   |   |-- visualization.py
|   |-- case_studies/
|   |   |-- cox2.py
|   |   |-- hemoglobin.py
|-- notebooks/
|   |-- 01_data_exploration.ipynb
|   |-- 02_feature_analysis.ipynb
|   |-- 03_model_training.ipynb
|   |-- 04_case_studies.ipynb
|-- results/
|   |-- models/                 # Saved model checkpoints
|   |-- figures/                # Paper figures
|   |-- tables/                 # Paper tables (CSV)
|-- paper/
|   |-- manuscript.tex
|   |-- figures/
|   |-- supplementary/
|-- environment.yml             # Conda environment file
|-- requirements.txt            # Pip requirements
|-- README.md
```

---

## APPENDIX A: QUICK-START RECIPE (DO THIS FIRST)

If you want to see results FAST, do this minimal pipeline first (1-2 weeks):

```python
# 1. Download 50 proteins from AlloBench (subset)
# 2. Compute only: SASA + B-factors + contact_count + AA_properties (10 min total)
# 3. Train XGBoost with scale_pos_weight
# 4. Evaluate MCC on a 10-protein held-out set

# This gives you a working baseline in ~3 days.
# Then incrementally add features and compare.
```

## APPENDIX B: KEY REFERENCES

1. ASD2023: Nucleic Acids Res. 2024, 52, D376-D383
2. AlloBench: ACS Omega 2025 (DOI: 10.1021/acsomega.5c01263)
3. PASSer2.0: Front. Mol. Biosci. 2022, 9, 879251
4. DeepAllo: Bioinformatics 2025, 41(6), btaf294
5. AlloPED: bioRxiv 2025, 2025.03.28.645953
6. AlloFusion: JCIM 2025, 65(16), 8858-8870
7. ASBench: Bioinformatics 2015, 31(15), 2598-2600
8. CASBench: Acta Naturae 2019, 11(1), 74-80
9. ESM-2: bioRxiv 2022, Science 2023
10. Fine-tuning PLMs: Nature Communications 2024
11. ESMBind LoRA tutorial: HuggingFace Blog (Amelie Schreiber)
12. COX-2 allostery: PNAS 2015, 112(40), 12366-12371
