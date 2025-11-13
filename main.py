import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from HyperParameter import HyperParameter
from micro_cluster import MicroCluster
from metric_stream import MetricStream
from collections import Counter, deque
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    cohen_kappa_score, balanced_accuracy_score
)
from imblearn.metrics import geometric_mean_score
import warnings

warnings.filterwarnings("ignore")


class RDMLStream(object):
    def __init__(self, metric_stream, args):
        self.ms = metric_stream
        self.args = args

        self.mcs = []
        self.avg_radius = 0
        self.counter = {-1: 0}
        self.predict_label = []
        self.next_mc_id = 0

        self.initialization()

        self.true_label = []
        self.labeled_num = 0
        self.create_num = 0

        self.weight_update_count = 0
        self.high_confidence_count = 0
        self.weight_applied_count = 0

        self.current_step = 0
        self.trusted_label_window = deque(maxlen=self.args.sliding_window_size)
        for label in self.ms.init_labels:
            self.trusted_label_window.append(label)

        self.class_weights = {cls: 1.0 for cls in self.ms.classes}
        self.update_class_weights()

        self.propagate_history = []
        self.propagate_attempts = 0
        self.propagate_success = 0
        self.current_density_threshold = self.args.density_threshold_base

    def initialization(self):
        """Initialize micro-clusters."""
        for cls in self.ms.classes:
            index = self.ms.init_labels == cls
            data = self.ms.init_data[index]
            self.counter[cls] = 0

            if len(data) > self.args.init_k_per_class:
                kmeans = KMeans(n_clusters=self.args.init_k_per_class, random_state=self.args.seed)
                kmeans.fit(data)
                kmeans_labels = kmeans.labels_
                for cls_ in range(self.args.init_k_per_class):
                    data_cls_ = data[kmeans_labels == cls_]
                    if len(data_cls_) == 0:
                        continue
                    mc = MicroCluster(data_cls_[0], timestamp=0, label=cls, lmbda=self.args.lmbda)
                    mc.mc_id = self.next_mc_id
                    self.next_mc_id += 1
                    for d in data_cls_[1:]:
                        mc.insert(d, timestamp=0, labeled=True)
                    self.mcs.append(mc)
                    self.counter[cls] += 1
            else:
                if len(data) > 0:
                    mc = MicroCluster(data[0], timestamp=0, label=cls, lmbda=self.args.lmbda)
                    mc.mc_id = self.next_mc_id
                    self.next_mc_id += 1
                    for d in data[1:]:
                        mc.insert(d, timestamp=0, labeled=True)
                    self.mcs.append(mc)
                    self.counter[cls] += 1

        self.avg_radius = np.average([mc.radius for mc in self.mcs if mc.n > 1])
        for mc in self.mcs:
            if mc.n <= 1:
                mc.radius = self.avg_radius

    def start(self):
        """Main loop."""
        for i, data in tqdm(enumerate(self.ms.metric_stream), total=len(self.ms.metric_stream)):
            self.current_step = i
            semi_label = self.ms.semi_labels[i]
            true_label = self.ms.true_labels[i]
            known = (semi_label != -1)

            pred_label, cls_re = self.classify(data, known, semi_label)

            self.true_label.append(true_label)
            self.predict_label.append(pred_label)
            self.labeled_num += 1 if known else 0

            if known:
                self.trusted_label_window.append(semi_label)
            elif cls_re > 0.8:
                self.trusted_label_window.append(pred_label)
                self.high_confidence_count += 1

            if i % 50 == 0:
                self.update_class_weights()

            if known:
                self.validate_propagation_history(semi_label, true_label)

            self.add_data(data, pred_label, semi_label, cls_re, known)

            self.decay_mcs()

            if (i + 1) % self.args.logging_steps == 0:
                print(f'\n=== Step {i + 1} ===')
                self.evaluation()

    def classify(self, data, known=False, semi_label=-1):
        """Classify sample (applies class weights)."""
        labeled_mcs = [mc for mc in self.mcs if mc.label != -1]
        if not labeled_mcs:
            return 0, 0.0

        mc_centers = np.stack([mc.get_center() for mc in labeled_mcs])
        mc_labels = np.array([mc.label for mc in labeled_mcs])
        mc_reliability = np.array([mc.re for mc in labeled_mcs])

        dis = euclidean_distances(mc_centers, data.reshape([1, -1])).flatten()
        k = min(self.args.k, len(labeled_mcs))

        if k == 1:
            topk_idx = np.array([0])
        else:
            topk_idx = np.argpartition(dis, k - 1)[:k]

        topk_dis = dis[topk_idx] + 1e-10
        topk_dis = topk_dis / (np.min(topk_dis) + 1e-10)
        topk_cls = mc_labels[topk_idx]
        topk_res = mc_reliability[topk_idx]

        ret_class = np.zeros(len(self.ms.classes))
        ret_reliability = np.zeros(len(self.ms.classes))

        probabilities = softmax(topk_res / topk_dis)

        for i, cls in enumerate(topk_cls):
            index = self.ms.classes.index(cls)
            # 1. Apply class weight (alpha)
            weight = self.get_adjusted_weight(cls)
            ret_class[index] += weight
            ret_reliability[index] += probabilities[i] * weight
            self.weight_applied_count += 1

        pred_label = self.ms.classes[np.argmax(ret_class)]
        cls_reliability = np.max(ret_reliability) / (np.sum(ret_reliability) + 1e-10)

        return pred_label, cls_reliability

    def get_adjusted_weight(self, cls):
        """Get adjusted class weight (applies alpha smoothing)."""
        base_weight = self.class_weights.get(cls, 1.0)
        # 2. alpha=0: no adjustment, alpha=1: full weight
        adjusted_weight = 1.0 + self.args.class_balance_alpha * (base_weight - 1.0)
        return max(0.1, adjusted_weight)

    def add_data(self, data, pred_label, semi_label, re, known=False):
        """Add data point (includes density propagation)."""
        if len(self.mcs) == 0:
            label = semi_label if known else pred_label
            mc = MicroCluster(data, timestamp=self.current_step,
                              label=label, radius=1.0, lmbda=self.args.lmbda)
            mc.mc_id = self.next_mc_id
            self.next_mc_id += 1
            self.mcs.append(mc)
            self.counter[label] += 1
            return

        mcs_centers = np.stack([mc.get_center() for mc in self.mcs])
        dis = euclidean_distances(mcs_centers, data.reshape([1, -1])).flatten()
        min_idx = np.argmin(dis)
        nearest_mc = self.mcs[min_idx]

        need_new = False
        if dis[min_idx] >= nearest_mc.radius:
            need_new = True
        elif re < self.args.minRE:
            need_new = True
        elif known and nearest_mc.label != semi_label and nearest_mc.label != -1:
            need_new = True
        elif not known and nearest_mc.label != pred_label and nearest_mc.label != -1:
            need_new = True

        if not need_new:
            nearest_mc.insert(data, timestamp=self.current_step, labeled=known)
            if nearest_mc.label == -1:
                label = semi_label if known else pred_label
                nearest_mc.label = label
                self.counter[label] += 1
                self.counter[-1] -= 1
        else:
            # 3. Create new micro-cluster
            if self.counter[-1] > self.args.maxUMC:
                self.drop(unlabeled=True)
            if len(self.mcs) >= self.args.maxMC:
                self.drop()

            if known:
                initial_label = semi_label
                initial_re = 1.0
            else:
                initial_label = -1
                initial_re = re

            mc = MicroCluster(data, timestamp=self.current_step,
                              re=initial_re, label=initial_label,
                              radius=self.avg_radius, lmbda=self.args.lmbda)
            mc.mc_id = self.next_mc_id
            self.next_mc_id += 1

            if not known and self.args.enable_density_propagate:
                propagated_label = self.enhanced_density_propagate(mc, data)
                if propagated_label != -1:
                    mc.label = propagated_label
                    self.propagate_history.append({
                        'step': self.current_step,
                        'mc_id': mc.mc_id,
                        'pred_label': propagated_label,
                        'validated': False,
                        'correct': None
                    })
                    self.propagate_attempts += 1
                else:
                    mc.label = pred_label

            self.mcs.append(mc)
            self.counter[mc.label] += 1
            self.create_num += 1

    def enhanced_density_propagate(self, umc, data):
        """Enhanced density propagation (applies alpha and beta)."""
        umc_center = umc.get_center()
        neighbor_radius = self.avg_radius * self.args.density_radius_factor

        nearby_labeled = []
        for mc in self.mcs:
            if mc != umc and mc.label != -1:
                dist = np.linalg.norm(mc.get_center() - umc_center)
                if dist < neighbor_radius:
                    nearby_labeled.append((mc, dist))

        if not nearby_labeled:
            return -1

        # 4. Apply alpha: inverse frequency density compensation
        density_dict = {cls: 0.0 for cls in self.ms.classes}

        for mc, distance in nearby_labeled:
            kernel = np.exp(-0.5 * (distance / (self.avg_radius + 1e-10)) ** 2)
            base_density = mc.n * kernel * mc.re
            weight = self.get_adjusted_weight(mc.label)
            weighted_density = base_density * weight
            density_dict[mc.label] += weighted_density

        total_density = sum(density_dict.values())
        if total_density < 1e-10:
            return -1

        for cls in density_dict:
            density_dict[cls] /= total_density

        max_cls = max(density_dict, key=density_dict.get)
        max_density_ratio = density_dict[max_cls]

        # 5. Apply beta: non-linear threshold adjustment
        base_threshold = self.current_density_threshold
        avg_weight = np.mean(list(self.class_weights.values()))
        class_freq_ratio = self.class_weights[max_cls] / (avg_weight + 1e-10)
        threshold_adjustment = np.tanh(self.args.class_balance_beta * (class_freq_ratio - 1.0))

        if class_freq_ratio > 1.0:
            adjusted_threshold = base_threshold * (1.0 - 0.3 * threshold_adjustment)
            adjusted_threshold *= self.args.minority_boost_factor
        else:
            adjusted_threshold = base_threshold * (1.0 + 0.2 * abs(threshold_adjustment))

        if max_density_ratio > adjusted_threshold:
            return max_cls

        return -1

    def update_class_weights(self):
        """Update class weights (inverse frequency)."""
        if len(self.trusted_label_window) < 50:
            return

        counter = Counter(self.trusted_label_window)
        total = sum(counter.values())

        for cls in self.ms.classes:
            # 6. Calculate inverse frequency
            freq = (counter.get(cls, 0) + 1) / (total + len(self.ms.classes))
            self.class_weights[cls] = 1.0 / (freq + 0.01)

        min_weight = min(self.class_weights.values())
        max_weight = max(self.class_weights.values())
        if max_weight - min_weight > 1e-10:
            for cls in self.ms.classes:
                normalized = (self.class_weights[cls] - min_weight) / (max_weight - min_weight)
                self.class_weights[cls] = 0.5 + 1.5 * normalized

        self.weight_update_count += 1

    def validate_propagation_history(self, semi_label, true_label):
        """Validate propagation history (when labeled data arrives)."""
        for record in self.propagate_history:
            if not record['validated']:
                # 7. Check if the propagation was correct
                is_correct = (record['pred_label'] == true_label)
                record['validated'] = True
                record['correct'] = is_correct

                if is_correct:
                    self.propagate_success += 1
                    self.trusted_label_window.append(record['pred_label'])

    def decay_mcs(self):
        """Decay micro-clusters."""
        mcs_to_remove = []
        for i, mc in enumerate(self.mcs):
            re = mc.update()
            if re < self.args.minRE:
                mcs_to_remove.append(i)

        for i in reversed(mcs_to_remove):
            self.counter[self.mcs[i].label] -= 1
            self.mcs.pop(i)

        radii = [mc.radius for mc in self.mcs if mc.n > 1]
        if radii:
            self.avg_radius = np.mean(radii)
            for mc in self.mcs:
                if mc.n <= 1:
                    mc.radius = self.avg_radius

        validated_count = sum(1 for r in self.propagate_history if r['validated'])
        if validated_count >= 20:
            success_count = sum(1 for r in self.propagate_history if r['correct'])
            success_rate = success_count / validated_count

            # 8. Update dynamic threshold (based on validation)
            target_threshold = self.args.density_threshold_base + 0.2 * (success_rate - 0.5)
            self.current_density_threshold = (
                    self.args.decay_rate * target_threshold +
                    (1 - self.args.decay_rate) * self.current_density_threshold
            )
            self.current_density_threshold = np.clip(self.current_density_threshold, 0.4, 0.9)

        if len(self.propagate_history) > self.args.sliding_window_size * 2:
            self.propagate_history = [r for r in self.propagate_history
                                      if r['step'] > self.current_step - self.args.sliding_window_size]

    def drop(self, unlabeled=False):
        """Drop micro-clusters."""
        if unlabeled:
            candidates = [mc for mc in self.mcs if mc.label == -1]
        else:
            candidates = self.mcs.copy()

        if len(candidates) > 5:
            candidates.sort(key=lambda x: x.t, reverse=False)
            for mc in candidates[-5:]:
                if mc in self.mcs:
                    self.counter[mc.label] -= 1
                    self.mcs.remove(mc)

    def evaluation(self):
        """Evaluate performance."""
        if not self.true_label or not self.predict_label:
            return {}

        balanced_acc = balanced_accuracy_score(self.true_label, self.predict_label)
        f1 = f1_score(self.true_label, self.predict_label, average='macro', zero_division=0)
        gmean = geometric_mean_score(self.true_label, self.predict_label, average='macro')

        print(f"\n--- Performance Metrics ---")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"F1 (Macro): {f1:.4f}")
        print(f"G-mean (Macro): {gmean:.4f}")

        return {
            'balanced_accuracy': balanced_acc,
            'f1': f1,
            'gmean': gmean
        }


if __name__ == '__main__':
    start_time = time.time()
    args = HyperParameter()

    config = {'alpha': 0.35, 'beta': 1.0, 'uratio': 0.8, 'name': 'Low class weighting (alpha=0.3)'}

    print(f"\n{'=' * 80}")
    print(f"Running test: {config['name']}")
    print(f"Alpha={config['alpha']}, Beta={config['beta']}, Unlabeled Ratio={config['uratio']}")
    print('=' * 80)

    args.class_balance_alpha = config['alpha']
    args.class_balance_beta = config['beta']
    args.unlabeled_ratio = config['uratio']

    try:
        metric_stream = MetricStream(args)
        model = RDMLStream(metric_stream, args)
        model.start()

        results = model.evaluation()
        results.update({
            'config': config['name'],
            'alpha': config['alpha'],
            'beta': config['beta'],
            'uratio': config['uratio'],
        })

        print(f"\nResults:\n")
        print(f"  Balanced Accuracy: {results['balanced_accuracy']:.4f}")
        print(f"  F1 (Macro): {results['f1']:.4f}")
        print(f"  G-mean: {results['gmean']:.4f}\n")

    except Exception as e:
        print(f"Error in config {config['name']}: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'=' * 80}")
    print(f"Test completed!")
    print(f"Total time: {time.time() - start_time:.2f}s")
    print('=' * 80)