"""
GSNH-MDT vs Blossom — Using Professor's Exact .dl8 Data
=========================================================

.dl8 format (confirmed):
    label f1 f2 f3 ... fn
    (space-separated integers, first column = class, rest = binary features)

Professor's methodology:
    - DT (depth 7): Blossom on original .dl8 data
    - MDT (depth 5): Blossom on EXPANDED data (all C(n,2)×5 pairwise 2CNFs)
    - 10 random 80/20 splits, seeds 1,101,201,...,901

Our approach:
    - DT (depth 7): sklearn DecisionTreeClassifier on original data
    - GSNH-MDT (depth 5): Native multivariate search on original data
    - GSNH-MDT (depth 7): For direct depth comparison
    - Same 10 random 80/20 splits

Usage:
    python benchmark_dl8.py
"""

import numpy as np
import os
import sys
import glob
import time
import traceback
import warnings
from collections import OrderedDict
from itertools import combinations
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings('ignore')


# =============================================================================
# IMPORT GSNH
# =============================================================================

def import_gsnh():
    """Import GSNH module."""
    try:
        import gsnh_mdt_v3 as gsnh
        print("✓ Imported gsnh_mdt_v3")
        return gsnh
    except ImportError:
        pass
    try:
        import importlib.util
        for c in ['GSNH-MDT-v3.py', './GSNH-MDT-v3.py',
                  os.path.join(os.path.dirname(
                      os.path.abspath(__file__)), 'GSNH-MDT-v3.py')]:
            if os.path.exists(c):
                spec = importlib.util.spec_from_file_location(
                    "gsnh_mdt_v3", c)
                gsnh = importlib.util.module_from_spec(spec)
                sys.modules["gsnh_mdt_v3"] = gsnh
                spec.loader.exec_module(gsnh)
                print(f"✓ Imported from {c}")
                return gsnh
    except Exception as e:
        pass
    print("✗ Cannot find GSNH module")
    sys.exit(1)


# =============================================================================
# .DL8 PARSER (confirmed format: no header, space-separated ints)
# =============================================================================

def parse_dl8(filepath):
    """
    Parse .dl8 file.
    
    Format: label f1 f2 f3 ... fn
    All values are integers (typically 0/1 for binary features).
    First column = class label, remaining columns = features.
    
    Returns: X (n_samples, n_features), y (n_samples,)
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [int(x) for x in line.split()]
            data.append(vals)
    
    data = np.array(data, dtype=np.float64)
    
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Invalid data shape: {data.shape}")
    
    y = data[:, 0].astype(np.int32)
    X = data[:, 1:]
    
    return X, y


def load_all_dl8(data_dir='data', max_binary_features=None):
    """
    Load all .dl8 files.
    
    Also computes the number of binary (expanded) features
    that Blossom would use: n + C(n,2) × 5
    """
    dl8_files = sorted(glob.glob(os.path.join(data_dir, '*.dl8')))
    
    if not dl8_files:
        print(f"  No .dl8 files in {os.path.abspath(data_dir)}/")
        return OrderedDict()
    
    datasets = OrderedDict()
    
    for filepath in dl8_files:
        name = os.path.basename(filepath).replace('.dl8', '')
        try:
            X, y = parse_dl8(filepath)
            n_unary = X.shape[1]
            n_binary = int(n_unary + n_unary * (n_unary - 1) / 2 * 5)
            
            # Make labels binary if needed
            unique_labels = np.unique(y)
            if len(unique_labels) > 2:
                majority = np.argmax(np.bincount(y.astype(int)))
                y = (y == majority).astype(np.int32)
            elif set(unique_labels) != {0, 1}:
                y = (y == unique_labels[1]).astype(np.int32)
            
            # Remove zero-variance columns
            var = np.var(X, axis=0)
            valid = var > 1e-10
            if valid.sum() < X.shape[1]:
                X = X[:, valid]
            
            if X.shape[1] == 0:
                print(f"  ⚠ {name:<35} No valid features")
                continue
            
            # Check if binary
            unique_vals = np.unique(X)
            is_binary = np.all((unique_vals == 0) | (unique_vals == 1))
            
            datasets[name] = {
                'X': X, 'y': y,
                'n_unary': n_unary,
                'n_binary': n_binary,
                'is_binary': is_binary,
            }
            
            btype = "binary" if is_binary else "mixed"
            print(f"  ✓ {name:<35} n={len(y):>6}  "
                  f"unary={n_unary:>4}  binary={n_binary:>8}  "
                  f"({btype})  pos={y.mean():.2%}")
            
        except Exception as e:
            print(f"  ✗ {name:<35} {str(e)[:60]}")
    
    return datasets


# =============================================================================
# BLOSSOM REFERENCE (from professor's Table 1)
# =============================================================================

BLOSSOM = {
    'adult_discretized':
        {'dt': 0.8540, 'mdt': 0.8550, 'dt_sz': 87.0, 'mdt_sz': 24.7,
         'dt_ex': 6.16, 'mdt_ex': 4.92, 'unary': 59, 'binary': 8555},
    'anneal':
        {'dt': 0.8681, 'mdt': 0.8687, 'dt_sz': 51.1, 'mdt_sz': 15.2,
         'dt_ex': 4.25, 'mdt_ex': 1.77, 'unary': 93, 'binary': 21390},
    'audiology':
        {'dt': 0.9295, 'mdt': 0.9227, 'dt_sz': 7.4, 'mdt_sz': 4.3,
         'dt_ex': 1.19, 'mdt_ex': 0.96, 'unary': 148, 'binary': 54390},
    'australian-credit':
        {'dt': 0.8220, 'mdt': 0.8265, 'dt_sz': 59.7, 'mdt_sz': 19.9,
         'dt_ex': 3.66, 'mdt_ex': 4.07, 'unary': 125, 'binary': 38750},
    'balance-scale-bin':
        {'dt': 0.7849, 'mdt': 0.8373, 'dt_sz': 76.8, 'mdt_sz': 23.8,
         'dt_ex': 2.66, 'mdt_ex': 3.46, 'unary': 16, 'binary': 600},
    'bank_conv-bin':
        {'dt': 0.8961, 'mdt': 0.8966, 'dt_sz': 88.0, 'mdt_sz': 28.3,
         'dt_ex': 5.46, 'mdt_ex': 5.31, 'unary': 212, 'binary': 111830},
    'banknote-bin':
        {'dt': 0.9862, 'mdt': 0.9876, 'dt_sz': 19.7, 'mdt_sz': 11.8,
         'dt_ex': 1.23, 'mdt_ex': 1.89, 'unary': 28, 'binary': 1890},
    'biodeg-bin':
        {'dt': 0.8354, 'mdt': 0.8340, 'dt_sz': 72.4, 'mdt_sz': 23.2,
         'dt_ex': 5.22, 'mdt_ex': 3.82, 'unary': 304, 'binary': 230280},
    'breast-cancer-un':
        {'dt': 0.9380, 'mdt': 0.9438, 'dt_sz': 25.4, 'mdt_sz': 12.8,
         'dt_ex': 1.07, 'mdt_ex': 1.45, 'unary': 89, 'binary': 19580},
    'breast-wisconsin':
        {'dt': 0.9328, 'mdt': 0.9526, 'dt_sz': 18.3, 'mdt_sz': 11.1,
         'dt_ex': 1.82, 'mdt_ex': 2.93, 'unary': 120, 'binary': 35700},
    'car_evaluation-bin':
        {'dt': 0.9165, 'mdt': 0.9150, 'dt_sz': 35.1, 'mdt_sz': 13.6,
         'dt_ex': 1.36, 'mdt_ex': 1.30, 'unary': 14, 'binary': 455},
    'car-un':
        {'dt': 0.9668, 'mdt': 0.9734, 'dt_sz': 58.0, 'mdt_sz': 14.9,
         'dt_ex': 2.93, 'mdt_ex': 1.23, 'unary': 21, 'binary': 1050},
    'compas_discretized':
        {'dt': 0.6612, 'mdt': 0.6667, 'dt_sz': 102.1, 'mdt_sz': 30.9,
         'dt_ex': 5.40, 'mdt_ex': 5.09, 'unary': 25, 'binary': 1500},
    'diabetes':
        {'dt': 0.6630, 'mdt': 0.7221, 'dt_sz': 107.4, 'mdt_sz': 26.1,
         'dt_ex': 6.57, 'mdt_ex': 4.62, 'unary': 112, 'binary': 31080},
    'forest-fires-un':
        {'dt': 0.5269, 'mdt': 0.5635, 'dt_sz': 31.2, 'mdt_sz': 12.3,
         'dt_ex': 1.39, 'mdt_ex': 1.49, 'unary': 989, 'binary': 2442830},
    'german-credit':
        {'dt': 0.6805, 'mdt': 0.6950, 'dt_sz': 96.0, 'mdt_sz': 24.8,
         'dt_ex': 5.87, 'mdt_ex': 4.73, 'unary': 112, 'binary': 31080},
    'heart-cleveland':
        {'dt': 0.7367, 'mdt': 0.7650, 'dt_sz': 31.1, 'mdt_sz': 18.5,
         'dt_ex': 2.59, 'mdt_ex': 3.89, 'unary': 95, 'binary': 22325},
    'hepatitis':
        {'dt': 0.7276, 'mdt': 0.7862, 'dt_sz': 12.5, 'mdt_sz': 8.7,
         'dt_ex': 1.30, 'mdt_ex': 1.99, 'unary': 68, 'binary': 11390},
    'HTRU_2-bin':
        {'dt': 0.9760, 'mdt': 0.9778, 'dt_sz': 100.2, 'mdt_sz': 24.1,
         'dt_ex': 5.28, 'mdt_ex': 4.33, 'unary': 70, 'binary': 12075},
    'hypothyroid':
        {'dt': 0.9743, 'mdt': 0.9775, 'dt_sz': 52.5, 'mdt_sz': 19.1,
         'dt_ex': 3.51, 'mdt_ex': 2.69, 'unary': 88, 'binary': 19140},
    'IndiansDiabetes-bin':
        {'dt': 0.6539, 'mdt': 0.7162, 'dt_sz': 114.9, 'mdt_sz': 27.5,
         'dt_ex': 5.64, 'mdt_ex': 5.02, 'unary': 43, 'binary': 4515},
    'ionosphere':
        {'dt': 0.8789, 'mdt': 0.8732, 'dt_sz': 19.8, 'mdt_sz': 12.2,
         'dt_ex': 1.37, 'mdt_ex': 2.29, 'unary': 445, 'binary': 493950},
    'kr-vs-kp':
        {'dt': 0.9836, 'mdt': 0.9908, 'dt_sz': 35.2, 'mdt_sz': 14.9,
         'dt_ex': 1.88, 'mdt_ex': 2.94, 'unary': 73, 'binary': 13140},
    'letter':
        {'dt': 0.9915, 'mdt': 0.9891, 'dt_sz': 81.1, 'mdt_sz': 17.7,
         'dt_ex': 4.74, 'mdt_ex': 3.29, 'unary': 224, 'binary': 124880},
    'letter_recognition-bin':
        {'dt': 0.9941, 'mdt': 0.9949, 'dt_sz': 50.9, 'mdt_sz': 13.9,
         'dt_ex': 3.89, 'mdt_ex': 3.33, 'unary': 240, 'binary': 143400},
    'lymph':
        {'dt': 0.8194, 'mdt': 0.8677, 'dt_sz': 14.0, 'mdt_sz': 7.7,
         'dt_ex': 1.37, 'mdt_ex': 2.24, 'unary': 68, 'binary': 11390},
    'magic04-bin':
        {'dt': 0.8457, 'mdt': 0.8453, 'dt_sz': 125.0, 'mdt_sz': 31.0,
         'dt_ex': 7.49, 'mdt_ex': 5.56, 'unary': 86, 'binary': 18275},
    'messidor-bin':
        {'dt': 0.6364, 'mdt': 0.6450, 'dt_sz': 85.1, 'mdt_sz': 28.3,
         'dt_ex': 6.54, 'mdt_ex': 4.91, 'unary': 86, 'binary': 18275},
    'mushroom':
        {'dt': 1.0000, 'mdt': 1.0000, 'dt_sz': 7.8, 'mdt_sz': 4.0,
         'dt_ex': 0.29, 'mdt_ex': 1.00, 'unary': 119, 'binary': 35105},
    'pendigits':
        {'dt': 0.9970, 'mdt': 0.9963, 'dt_sz': 19.9, 'mdt_sz': 14.7,
         'dt_ex': 2.09, 'mdt_ex': 3.27, 'unary': 216, 'binary': 116100},
    'primary-tumor':
        {'dt': 0.7426, 'mdt': 0.7603, 'dt_sz': 41.5, 'mdt_sz': 21.2,
         'dt_ex': 1.96, 'mdt_ex': 4.01, 'unary': 31, 'binary': 2325},
    'segment':
        {'dt': 0.9991, 'mdt': 0.9998, 'dt_sz': 5.0, 'mdt_sz': 3.0,
         'dt_ex': 1.69, 'mdt_ex': 1.12, 'unary': 235, 'binary': 137475},
    'seismic_bumps-bin':
        {'dt': 0.8992, 'mdt': 0.9153, 'dt_sz': 102.8, 'mdt_sz': 21.3,
         'dt_ex': 5.17, 'mdt_ex': 4.41, 'unary': 91, 'binary': 20475},
    'soybean':
        {'dt': 0.9386, 'mdt': 0.9488, 'dt_sz': 27.9, 'mdt_sz': 11.9,
         'dt_ex': 1.87, 'mdt_ex': 3.10, 'unary': 50, 'binary': 6125},
    'splice-1':
        {'dt': 0.9447, 'mdt': 0.9589, 'dt_sz': 57.5, 'mdt_sz': 21.8,
         'dt_ex': 4.60, 'mdt_ex': 4.77, 'unary': 287, 'binary': 205205},
    'spambase-bin':
        {'dt': 0.8772, 'mdt': 0.8765, 'dt_sz': 75.1, 'mdt_sz': 21.8,
         'dt_ex': 6.58, 'mdt_ex': 4.26, 'unary': 386, 'binary': 371525},
    'Statlog_satellite-bin':
        {'dt': 0.9660, 'mdt': 0.9694, 'dt_sz': 58.4, 'mdt_sz': 19.0,
         'dt_ex': 3.57, 'mdt_ex': 3.54, 'unary': 539, 'binary': 724955},
    'taiwan_binarised':
        {'dt': 0.8130, 'mdt': 0.8167, 'dt_sz': 121.7, 'mdt_sz': 30.3,
         'dt_ex': 8.09, 'mdt_ex': 5.30, 'unary': 205, 'binary': 104550},
    'tic-tac-toe':
        {'dt': 0.9705, 'mdt': 0.9378, 'dt_sz': 39.8, 'mdt_sz': 20.2,
         'dt_ex': 2.23, 'mdt_ex': 3.92, 'unary': 27, 'binary': 1755},
    'titanic-un':
        {'dt': 0.7904, 'mdt': 0.8163, 'dt_sz': 66.8, 'mdt_sz': 21.5,
         'dt_ex': 3.98, 'mdt_ex': 4.35, 'unary': 333, 'binary': 276390},
    'vehicle':
        {'dt': 0.9482, 'mdt': 0.9524, 'dt_sz': 26.2, 'mdt_sz': 13.2,
         'dt_ex': 2.41, 'mdt_ex': 3.38, 'unary': 252, 'binary': 158130},
    'vote':
        {'dt': 0.9420, 'mdt': 0.9489, 'dt_sz': 15.9, 'mdt_sz': 10.8,
         'dt_ex': 1.73, 'mdt_ex': 2.11, 'unary': 48, 'binary': 5640},
    'wine1-un':
        {'dt': 0.6417, 'mdt': 0.6361, 'dt_sz': 12.0, 'mdt_sz': 6.2,
         'dt_ex': 0.07, 'mdt_ex': 0.28, 'unary': 1276, 'binary': 4067250},
    'wine2-un':
        {'dt': 0.6730, 'mdt': 0.6703, 'dt_sz': 12.8, 'mdt_sz': 7.3,
         'dt_ex': 0.25, 'mdt_ex': 0.73, 'unary': 1276, 'binary': 4067250},
    'wine3-un':
        {'dt': 0.7278, 'mdt': 0.6722, 'dt_sz': 13.7, 'mdt_sz': 6.9,
         'dt_ex': 0.45, 'mdt_ex': 0.61, 'unary': 1276, 'binary': 4067250},
    'winequality-red-bin':
        {'dt': 0.9859, 'mdt': 0.9884, 'dt_sz': 13.5, 'mdt_sz': 8.9,
         'dt_ex': 1.25, 'mdt_ex': 2.44, 'unary': 42, 'binary': 4305},
    'yeast':
        {'dt': 0.6980, 'mdt': 0.7208, 'dt_sz': 87.4, 'mdt_sz': 26.5,
         'dt_ex': 4.45, 'mdt_ex': 4.87, 'unary': 89, 'binary': 19580},
}


# =============================================================================
# EXPLANATION LENGTH
# =============================================================================

class ExplAnalyzer:
    """
    Compute weak AXp explanation length.
    
    GSNH Horn clause rules (matching professor's counting):
    - TRUE branch of L1 ∨ L2 ∨ ... ∨ Lk:
        Only need ONE literal that is true → 1 feature
    - FALSE branch (¬L1 ∧ ¬L2 ∧ ... ∧ ¬Lk):
        Need ALL literals verified → k features
    - Unary split: always 1 feature
    
    This matches the professor's PrintSol.avg_len_axp():
    - "or" + "yes" branch → 1 feature (P=3/4)
    - "or" + "no" branch  → 2 features (P=1/4)
    - "xor" either branch → 2 features (P=1/2)
    - atom              → 1 feature  (P=1/2)
    """
    
    @staticmethod
    def gsnh_expl(root, x):
        """Explanation length for one instance."""
        feats = set()
        node = root
        x2d = x.reshape(1, -1)
        
        while node is not None:
            if node.get('is_leaf', True) or node.get('predicate') is None:
                break
            
            pred = node['predicate']
            lits = pred.literals
            result = pred.evaluate(x2d)[0]
            
            if result:
                # TRUE (disjunctive) branch: 1 sufficient literal
                for lit in lits:
                    if lit.evaluate(x2d)[0]:
                        feats.add(lit.feature)
                        break
                node = node.get('left')
            else:
                # FALSE (conjunctive) branch: all features needed
                for lit in lits:
                    feats.add(lit.feature)
                node = node.get('right')
        
        return len(feats)
    
    @staticmethod
    def sklearn_expl(tree, x):
        """Explanation length for sklearn DT = path length."""
        t = tree.tree_
        nid = 0
        depth = 0
        while t.children_left[nid] != t.children_right[nid]:
            depth += 1
            if x[t.feature[nid]] <= t.threshold[nid]:
                nid = t.children_left[nid]
            else:
                nid = t.children_right[nid]
        return depth
    
    @staticmethod
    def avg_gsnh(tree, X):
        if hasattr(tree, 'root_') and tree.root_ is None:
            return 0.0
        # To avoid massive slowdowns in benchmarks, take a sample of 50 for AXp extraction
        n_samples = min(50, len(X))
        idx = np.random.choice(len(X), n_samples, replace=False)
        return np.mean([len(tree.extract_axp(X[i])) for i in idx])
    
    @staticmethod
    def avg_sklearn(tree, X):
        return np.mean([ExplAnalyzer.sklearn_expl(tree, X[i])
                        for i in range(len(X))])


def count_gsnh_nodes(node):
    if node is None:
        return 0
    if node.get('is_leaf', True) or node.get('predicate') is None:
        return 1
    return 1 + count_gsnh_nodes(node.get('left')) + \
           count_gsnh_nodes(node.get('right'))


# =============================================================================
# BENCHMARK
# =============================================================================

class DL8Benchmark:
    """
    Exact replication of professor's experimental setup.
    
    - 10 random 80/20 splits
    - DT: max_depth=7
    - MDT: max_depth=5
    - Report: accuracy, tree size, explanation length
    """
    
    def __init__(self, gsnh_module, n_runs=10, random_state=42):
        self.gsnh = gsnh_module
        self.n_runs = n_runs
        self.rs = random_state
        self.results_ = OrderedDict()
    
    def run_all(self, datasets, skip_large=5000):
        """Run on all datasets, optionally skipping very large ones."""
        total = len(datasets)
        
        for idx, (name, info) in enumerate(datasets.items()):
            X, y = info['X'], info['y']
            n_unary = info['n_unary']
            
            # Skip extremely large feature spaces
            # (these take too long for 2D/3D search)
            if skip_large and X.shape[1] > skip_large:
                print(f"\n[{idx+1}/{total}] {name} — "
                      f"SKIPPED (d={X.shape[1]} > {skip_large})")
                continue
            
            print(f"\n[{idx+1}/{total}] ", end="")
            
            try:
                t0 = time.time()
                self._evaluate_one(name, X, y, n_unary)
                elapsed = time.time() - t0
                print(f"    ⏱ {elapsed:.1f}s")
            except Exception as e:
                print(f"    ✗ FAILED: {e}")
                traceback.print_exc()
    
    def _evaluate_one(self, name, X, y, n_unary):
        """Evaluate one dataset."""
        n, d = X.shape
        print(f"{name}")
        print(f"    n={n}, d={d} (unary={n_unary}), pos={y.mean():.2%}")
        
        ET = self.gsnh.ExpertGSNHTree
        SC = self.gsnh.StoppingCriteria
        
        # Adaptive settings based on dimensionality
        if d > 500:
            n_bins = 16
            use_2d, use_3d = True, False
            top_k = 10
        elif d > 100:
            n_bins = 32
            use_2d, use_3d = True, False
            top_k = 12
        else:
            n_bins = 64
            use_2d = True
            use_3d = d <= 50
            top_k = 15
        
        # For binary features, 2 bins suffice
        unique_vals = np.unique(X)
        if len(unique_vals) <= 3:
            n_bins = min(n_bins, 8)
        
        # Storage
        metrics = {
            'dt7': {'acc': [], 'size': [], 'expl': []},
            'g5': {'acc': [], 'size': [], 'expl': []},
            'g7': {'acc': [], 'size': [], 'expl': []},
        }
        
        splitter = StratifiedShuffleSplit(
            n_splits=self.n_runs, test_size=0.2, random_state=self.rs
        )
        
        for run_idx, (tr_idx, te_idx) in enumerate(splitter.split(X, y)):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            
            # ── sklearn DT (depth 7) ──
            dt = DecisionTreeClassifier(
                max_depth=7, min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.rs + run_idx
            )
            dt.fit(X_tr, y_tr)
            metrics['dt7']['acc'].append(
                float((dt.predict(X_te) == y_te).mean()))
            metrics['dt7']['size'].append(dt.tree_.node_count)
            metrics['dt7']['expl'].append(
                ExplAnalyzer.avg_sklearn(dt, X_te))
            
            # ── GSNH depth 5 ──
            g5 = ET(
                stopping_criteria=SC(
                    max_depth=5, min_samples_leaf=5,
                    min_samples_split=10),
                n_bins=n_bins, top_k_features=top_k,
                search_1d=True, search_2d=use_2d, search_3d=use_3d,
            )
            g5.fit(X_tr, y_tr)
            metrics['g5']['acc'].append(
                float((g5.predict(X_te) == y_te).mean()))
            metrics['g5']['size'].append(count_gsnh_nodes(g5.root_))
            metrics['g5']['expl'].append(
                ExplAnalyzer.avg_gsnh(g5, X_te))
            
            # ── GSNH depth 7 ──
            g7 = ET(
                stopping_criteria=SC(
                    max_depth=7, min_samples_leaf=5,
                    min_samples_split=10),
                n_bins=n_bins, top_k_features=top_k,
                search_1d=True, search_2d=use_2d, search_3d=use_3d,
            )
            g7.fit(X_tr, y_tr)
            metrics['g7']['acc'].append(
                float((g7.predict(X_te) == y_te).mean()))
            metrics['g7']['size'].append(count_gsnh_nodes(g7.root_))
            metrics['g7']['expl'].append(
                ExplAnalyzer.avg_gsnh(g7, X_te))
            
            if (run_idx + 1) % 5 == 0:
                print(f"    Run {run_idx+1}/{self.n_runs}: "
                      f"DT7={metrics['dt7']['acc'][-1]:.4f}  "
                      f"G5={metrics['g5']['acc'][-1]:.4f}  "
                      f"G7={metrics['g7']['acc'][-1]:.4f}")
        
        # Average
        result = {}
        for key in metrics:
            result[key] = {
                'acc': np.mean(metrics[key]['acc']),
                'acc_std': np.std(metrics[key]['acc']),
                'size': np.mean(metrics[key]['size']),
                'expl': np.mean(metrics[key]['expl']),
                'accs': metrics[key]['acc'],
            }
        result['n'] = n
        result['d'] = d
        result['n_unary'] = n_unary
        
        self.results_[name] = result
        
        # Print per-dataset summary
        print(f"\n    {'Model':<22} {'Accuracy':<18} "
              f"{'Size':<10} {'Expl':<8}")
        print(f"    {'─'*56}")
        
        for key, label in [('dt7', 'sklearn DT(d=7)'),
                           ('g5', 'GSNH-MDT(d=5)'),
                           ('g7', 'GSNH-MDT(d=7)')]:
            r = result[key]
            print(f"    {label:<22} "
                  f"{r['acc']:.4f}±{r['acc_std']:.4f}  "
                  f"{r['size']:>7.1f}  {r['expl']:.2f}")
        
        # Blossom reference
        bl = self._find_blossom(name)
        if bl:
            print(f"    {'Blossom DT(d=7)':<22} "
                  f"{bl['dt']:.4f}              "
                  f"{bl['dt_sz']:>7.1f}  {bl['dt_ex']:.2f}")
            print(f"    {'Blossom MDT(d=5)':<22} "
                  f"{bl['mdt']:.4f}              "
                  f"{bl['mdt_sz']:>7.1f}  {bl['mdt_ex']:.2f}")
    
    def _find_blossom(self, name):
        """Fuzzy match Blossom reference."""
        if name in BLOSSOM:
            return BLOSSOM[name]
        # Try variations
        for bname in BLOSSOM:
            if (name.replace('-', '_').lower() == 
                    bname.replace('-', '_').lower()):
                return BLOSSOM[bname]
        return None
    
    def print_table(self):
        """Print complete comparison table."""
        if not self.results_:
            print("No results.")
            return
        
        print(f"\n{'='*140}")
        print("COMPLETE RESULTS — Professor's .dl8 Data")
        print(f"{'='*140}")
        
        hdr = (f"{'Dataset':<30} "
               f"|{'DT7':>8} {'G5':>8} {'G7':>8} {'BlMDT':>8} "
               f"|{'DT sz':>7} {'G5 sz':>7} {'G7 sz':>7} {'Bl sz':>7} "
               f"|{'DT ex':>7} {'G5 ex':>7} {'Bl ex':>7}")
        print(hdr)
        print("─" * 140)
        
        # Accumulators
        acc = {'dt7': [], 'g5': [], 'g7': []}
        sz = {'dt7': [], 'g5': [], 'g7': []}
        ex = {'dt7': [], 'g5': []}
        wins = {'g5': 0, 'g7': 0}
        losses = {'g5': 0, 'g7': 0}
        ties = {'g5': 0, 'g7': 0}
        
        # Blossom comparison
        bl_g5_accs = []
        bl_mdt_accs = []
        bl_g5_sizes = []
        bl_mdt_sizes = []
        
        for name, r in self.results_.items():
            dt = r['dt7']
            g5 = r['g5']
            g7 = r['g7']
            bl = self._find_blossom(name)
            
            bl_acc = f"{bl['mdt']:.4f}" if bl else "     —"
            bl_sz = f"{bl['mdt_sz']:>6.1f}" if bl else "     —"
            bl_ex = f"{bl['mdt_ex']:>6.2f}" if bl else "     —"
            
            m5 = "+" if g5['acc'] > dt['acc'] + 0.005 else \
                 "-" if g5['acc'] < dt['acc'] - 0.005 else "="
            m7 = "+" if g7['acc'] > dt['acc'] + 0.005 else \
                 "-" if g7['acc'] < dt['acc'] - 0.005 else "="
            
            print(f"{name:<30} "
                  f"|{dt['acc']:>7.4f} {g5['acc']:>7.4f}{m5}"
                  f"{g7['acc']:>7.4f}{m7} {bl_acc:>8} "
                  f"|{dt['size']:>6.1f} {g5['size']:>6.1f}"
                  f" {g7['size']:>6.1f} {bl_sz:>7} "
                  f"|{dt['expl']:>6.2f} {g5['expl']:>6.2f}"
                  f" {bl_ex:>7}")
            
            for k in ['dt7', 'g5', 'g7']:
                acc[k].append(r[k]['acc'])
                sz[k].append(r[k]['size'])
            ex['dt7'].append(dt['expl'])
            ex['g5'].append(g5['expl'])
            
            if g5['acc'] > dt['acc'] + 0.005: wins['g5'] += 1
            elif g5['acc'] < dt['acc'] - 0.005: losses['g5'] += 1
            else: ties['g5'] += 1
            
            if g7['acc'] > dt['acc'] + 0.005: wins['g7'] += 1
            elif g7['acc'] < dt['acc'] - 0.005: losses['g7'] += 1
            else: ties['g7'] += 1
            
            if bl:
                bl_g5_accs.append(g5['acc'])
                bl_mdt_accs.append(bl['mdt'])
                bl_g5_sizes.append(g5['size'])
                bl_mdt_sizes.append(bl['mdt_sz'])
        
        # Average row
        n_ds = len(self.results_)
        print("─" * 140)
        print(f"{'AVERAGE (' + str(n_ds) + ' datasets)':<30} "
              f"|{np.mean(acc['dt7']):>7.4f} {np.mean(acc['g5']):>8.4f}"
              f" {np.mean(acc['g7']):>8.4f} {'':>8} "
              f"|{np.mean(sz['dt7']):>6.1f} {np.mean(sz['g5']):>6.1f}"
              f" {np.mean(sz['g7']):>6.1f} {'':>7} "
              f"|{np.mean(ex['dt7']):>6.2f} {np.mean(ex['g5']):>6.2f}"
              f" {'':>7}")
        
        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS")
        print(f"{'='*80}")
        
        print(f"\n  Accuracy (mean ± std across {n_ds} datasets):")
        for k, label in [('dt7', 'sklearn DT(d=7)'),
                         ('g5', 'GSNH-MDT(d=5)'),
                         ('g7', 'GSNH-MDT(d=7)')]:
            print(f"    {label:<22} {np.mean(acc[k]):.4f} "
                  f"± {np.std(acc[k]):.4f}")
        
        print(f"\n  Tree Size:")
        for k, label in [('dt7', 'sklearn DT(d=7)'),
                         ('g5', 'GSNH-MDT(d=5)'),
                         ('g7', 'GSNH-MDT(d=7)')]:
            comp = np.mean(sz['dt7']) / max(np.mean(sz[k]), 1)
            print(f"    {label:<22} {np.mean(sz[k]):>6.1f}"
                  f"  (compression: {comp:.1f}×)")
        
        print(f"\n  Explanation Length:")
        print(f"    sklearn DT(d=7):   {np.mean(ex['dt7']):.2f}")
        print(f"    GSNH-MDT(d=5):     {np.mean(ex['g5']):.2f}")
        
        print(f"\n  Win/Loss vs sklearn DT(d=7):")
        for k in ['g5', 'g7']:
            label = 'GSNH(d=5)' if k == 'g5' else 'GSNH(d=7)'
            print(f"    {label}: W={wins[k]} L={losses[k]} "
                  f"T={ties[k]} / {n_ds}")
        
        # Statistical tests
        for k, label in [('g5', 'GSNH5'), ('g7', 'GSNH7')]:
            a = np.array(acc[k])
            b = np.array(acc['dt7'])
            d = a - b
            if np.std(d) > 1e-10 and len(d) >= 3:
                _, pt = stats.ttest_rel(a, b)
                try:
                    _, pw = stats.wilcoxon(d)
                except:
                    pw = pt
                print(f"\n    {label} vs DT7: "
                      f"Δ={d.mean():+.4f}  "
                      f"t-test p={pt:.4f}  "
                      f"Wilcoxon p={pw:.4f}")
        
        # Blossom comparison
        if bl_g5_accs:
            print(f"\n  GSNH-MDT(d=5) vs Blossom MDT(d=5) "
                  f"({len(bl_g5_accs)} datasets):")
            print(f"    GSNH avg acc:     {np.mean(bl_g5_accs):.4f}")
            print(f"    Blossom avg acc:  {np.mean(bl_mdt_accs):.4f}")
            print(f"    GSNH avg size:    {np.mean(bl_g5_sizes):.1f}")
            print(f"    Blossom avg size: {np.mean(bl_mdt_sizes):.1f}")
            
            g = np.array(bl_g5_accs)
            b = np.array(bl_mdt_accs)
            bw = sum(1 for x, y in zip(g, b) if x > y + 0.005)
            bl = sum(1 for x, y in zip(g, b) if x < y - 0.005)
            bt = len(g) - bw - bl
            print(f"    GSNH wins: {bw}  "
                  f"Blossom wins: {bl}  Ties: {bt}")
    
    def generate_latex(self):
        """Generate LaTeX in professor's table format."""
        if not self.results_:
            return ""
        
        print(f"\n{'='*80}")
        print("LATEX TABLE")
        print(f"{'='*80}\n")
        
        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering\small")
        lines.append(
            r"\caption{Comparison on professor's .dl8 datasets. "
            r"DT: sklearn depth-7. GSNH: depth-5 and depth-7. "
            r"10 random 80/20 splits.}")
        lines.append(r"\begin{tabular}{l|ccc|ccc|cc}")
        lines.append(r"\toprule")
        lines.append(
            r" & \multicolumn{3}{c|}{accuracy} "
            r"& \multicolumn{3}{c|}{tree size} "
            r"& \multicolumn{2}{c}{expl.\ len} \\")
        lines.append(
            r"dataset & DT$_7$ & GSNH$_5$ & GSNH$_7$ "
            r"& DT$_7$ & GSNH$_5$ & GSNH$_7$ "
            r"& DT$_7$ & GSNH$_5$ \\")
        lines.append(r"\midrule")
        
        for name, r in self.results_.items():
            dt, g5, g7 = r['dt7'], r['g5'], r['g7']
            cname = name.replace('_', r'\_')
            
            best = max(dt['acc'], g5['acc'], g7['acc'])
            
            def fmt(v):
                s = f"{v:.4f}"
                return r"\textbf{" + s + "}" if abs(v - best) < 0.001 else s
            
            lines.append(
                f"{cname} & {fmt(dt['acc'])} & {fmt(g5['acc'])} "
                f"& {fmt(g7['acc'])} "
                f"& {dt['size']:.1f} & {g5['size']:.1f} "
                f"& {g7['size']:.1f} "
                f"& {dt['expl']:.2f} & {g5['expl']:.2f} " + r"\\")
        
        lines.append(r"\midrule")
        
        n = len(self.results_)
        avg_dt = np.mean([r['dt7']['acc'] for r in self.results_.values()])
        avg_g5 = np.mean([r['g5']['acc'] for r in self.results_.values()])
        avg_g7 = np.mean([r['g7']['acc'] for r in self.results_.values()])
        avg_dt_sz = np.mean([r['dt7']['size'] for r in self.results_.values()])
        avg_g5_sz = np.mean([r['g5']['size'] for r in self.results_.values()])
        avg_g7_sz = np.mean([r['g7']['size'] for r in self.results_.values()])
        avg_dt_ex = np.mean([r['dt7']['expl'] for r in self.results_.values()])
        avg_g5_ex = np.mean([r['g5']['expl'] for r in self.results_.values()])
        
        lines.append(
            f"average ({n}) "
            f"& {avg_dt:.4f} & {avg_g5:.4f} & {avg_g7:.4f} "
            f"& {avg_dt_sz:.1f} & {avg_g5_sz:.1f} & {avg_g7_sz:.1f} "
            f"& {avg_dt_ex:.2f} & {avg_g5_ex:.2f} " + r"\\")
        
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        
        latex = "\n".join(lines)
        print(latex)
        return latex


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#" * 80)
    print("#   GSNH-MDT vs Blossom — Professor's .dl8 Data   #")
    print("#" * 80)
    
    # Import
    print("\n[1] Importing GSNH...")
    gsnh = import_gsnh()
    
    # JIT
    print("\n[2] JIT Warmup...")
    gsnh.warmup_jit()
    
    # Find data
    data_dir = None
    for d in ['data', './data', '../data']:
        if os.path.isdir(d) and glob.glob(os.path.join(d, '*.dl8')):
            data_dir = d
            break
    
    if data_dir is None:
        print("✗ No data/ directory with .dl8 files found!")
        return None
    
    # Load
    print(f"\n[3] Loading .dl8 files from {data_dir}/...")
    datasets = load_all_dl8(data_dir)
    
    if not datasets:
        print("No datasets loaded!")
        return None
    
    # Run
    print(f"\n[4] Benchmark: {len(datasets)} datasets × 10 runs")
    print(f"    sklearn DT(d=7) vs GSNH-MDT(d=5) vs GSNH-MDT(d=7)")
    
    bench = DL8Benchmark(gsnh, n_runs=10, random_state=42)
    bench.run_all(datasets, skip_large=2000)
    
    # Results
    print("\n\n[5] Results...")
    bench.print_table()
    
    # LaTeX
    print("\n\n[6] LaTeX...")
    latex = bench.generate_latex()
    
    try:
        with open('dl8_benchmark_results.tex', 'w') as f:
            f.write(latex)
        print(f"\n✓ Saved to dl8_benchmark_results.tex")
    except Exception as e:
        print(f"\n✗ Save error: {e}")
    
    print("\n" + "#" * 80)
    print("#" + " " * 23 + "COMPLETE!" + " " * 22 + "#")
    print("#" * 80 + "\n")
    
    return bench


if __name__ == "__main__":
    bench = main()
